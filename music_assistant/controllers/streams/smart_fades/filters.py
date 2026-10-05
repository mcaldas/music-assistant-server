"""
Smart Fades - the FFmpeg filter toolset.

Each ``Filter`` is one tool a transition can apply to the fade-out/fade-in
stream pair; the renderer picks and orders them to realize a ``TransitionPlan``.
"""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum


class Filter(ABC):
    """Abstract base class for audio filters."""

    output_fadeout_label: str
    output_fadein_label: str

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize filter base class."""
        self.logger = logger

    @abstractmethod
    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Apply the filter and return the FFmpeg filter strings."""


class GradualTimeStretchFilter(Filter):
    """Gradual tempo change using asendcmd + rubberband with S-curve steps."""

    output_fadeout_label: str = "fadeout_gradstretch"
    output_fadein_label: str = "fadein_unchanged"

    def __init__(self, logger: logging.Logger, tempo_steps: list[tuple[float, float]]) -> None:
        """Initialize with tempo steps from compute_gradual_tempo_steps."""
        super().__init__(logger)
        # each tempo step is a tuple of (timestamp, tempo_ratio)
        self.tempo_steps = tempo_steps

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Build FFmpeg filter string for gradual time stretching."""
        if not self.tempo_steps:
            self.output_fadeout_label = input_fadeout_label.strip("[]")
            self.output_fadein_label = input_fadein_label.strip("[]")
            return []

        cmd_parts = [f"{ts:.3f} rubberband@rb tempo {ratio:.6f}" for ts, ratio in self.tempo_steps]
        cmd_string = "; ".join(cmd_parts)
        initial_ratio = self.tempo_steps[0][1]

        return [
            f"{input_fadeout_label} asendcmd=c='{cmd_string}',"
            f"rubberband@rb=tempo={initial_ratio:.6f}"
            f":transients=crisp:detector=compound:pitchq=quality"
            f" [{self.output_fadeout_label}]",
            f"{input_fadein_label} acopy [{self.output_fadein_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation."""
        n = len(self.tempo_steps)
        start = self.tempo_steps[0][1] if self.tempo_steps else 1.0
        end = self.tempo_steps[-1][1] if self.tempo_steps else 1.0
        return f"GradualTimeStretch(steps={n}, {start:.4f}->{end:.4f})"


class FadeInTrimFilter(Filter):
    """Filter that trims incoming track to align with downbeats."""

    output_fadeout_label: str = "fadeout_beatalign"
    output_fadein_label: str = "fadein_beatalign"

    def __init__(self, logger: logging.Logger, fadein_start_pos: float):
        """
        Initialize beat align filter.

        :param fadein_start_pos: Position in seconds to trim the incoming track to.
        """
        self.fadein_start_pos = fadein_start_pos
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Trim the incoming track to align with downbeats."""
        return [
            f"{input_fadeout_label}anull[{self.output_fadeout_label}]",  # codespell:ignore anull
            f"{input_fadein_label}atrim=start={self.fadein_start_pos},asetpts=PTS-STARTPTS[{self.output_fadein_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation of FadeInTrimFilter."""
        return f"FadeInTrim(start={self.fadein_start_pos:.2f}s)"


class FadeOutTrimFilter(Filter):
    """Filter that trims trailing (silent) audio off the outgoing track's tail."""

    output_fadeout_label: str = "fadeout_tailtrim"
    output_fadein_label: str = "fadein_tailtrim"

    def __init__(self, logger: logging.Logger, fadeout_end_pos: float, trimmed_seconds: float):
        """
        Initialize fade-out trim filter.

        :param fadeout_end_pos: Position in seconds where the outgoing track's
            audible content ends; everything after it is dropped.
            Measured on the untrimmed input timeline, so this filter must precede
            any time-stretching filter in the chain.
        :param trimmed_seconds: Amount of trailing audio in seconds that the trim
            drops, for logging/debugging purposes.
        """
        self.fadeout_end_pos = fadeout_end_pos
        self.trimmed_seconds = trimmed_seconds
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Trim the outgoing track's tail at the effective audio end."""
        return [
            f"{input_fadeout_label}atrim=end={self.fadeout_end_pos:.3f},"
            f"asetpts=PTS-STARTPTS[{self.output_fadeout_label}]",
            f"{input_fadein_label}anull[{self.output_fadein_label}]",  # codespell:ignore anull
        ]

    def __repr__(self) -> str:
        """Return string representation of FadeOutTrimFilter."""
        return f"FadeOutTrim(end={self.fadeout_end_pos:.2f}s, trimmed={self.trimmed_seconds:.2f}s)"


class ShelfType(StrEnum):
    """EQ band for a scheduled-gain filter; values are the ffmpeg filter names."""

    LOW = "lowshelf"
    HIGH = "highshelf"
    PEAK = "equalizer"


class ShelfFilter(Filter):
    """Shelving EQ whose gain follows a scheduled ramp (asendcmd-driven)."""

    def __init__(
        self,
        logger: logging.Logger,
        shelf_type: ShelfType,
        frequency: int,
        gain_steps: list[tuple[float, float]],
        stream_type: str,
    ):
        """
        Initialize shelf filter.

        :param shelf_type: Which shelving band to process.
        :param frequency: Shelf corner frequency in Hz.
        :param gain_steps: Schedule of (time_seconds, gain_db); the first step at
            t=0 sets the initial gain.
        :param stream_type: 'fadeout' or 'fadein' - which stream to process.
        """
        self.shelf_type = shelf_type
        self.frequency = frequency
        self.gain_steps = gain_steps
        self.stream_type = stream_type
        band = "low" if shelf_type is ShelfType.LOW else "high"
        if stream_type == "fadeout":
            self.output_fadeout_label = f"fadeout_{band}shelf"
            self.output_fadein_label = f"fadein_pt_{band}_out"
        else:
            self.output_fadeout_label = f"fadeout_pt_{band}_in"
            self.output_fadein_label = f"fadein_{band}shelf"
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Generate the shelf chain on this filter's stream and passthrough on the other."""
        if self.stream_type == "fadeout":
            input_label, output_label = input_fadeout_label, self.output_fadeout_label
            pass_in, pass_out = input_fadein_label, self.output_fadein_label
        else:
            input_label, output_label = input_fadein_label, self.output_fadein_label
            pass_in, pass_out = input_fadeout_label, self.output_fadeout_label
        band = "low" if self.shelf_type is ShelfType.LOW else "high"
        instance = f"{self.shelf_type}@{self.stream_type}_{band}"
        cmd = "; ".join(f"{t:.3f} {instance} g {g:.2f}" for t, g in self.gain_steps)
        initial = self.gain_steps[0][1]
        return [
            f"{pass_in}anull[{pass_out}]",  # codespell:ignore anull
            f"{input_label}asendcmd=c='{cmd}',"
            f"{instance}=g={initial:.2f}:f={self.frequency}:width_type=q:width=0.707"
            f"[{output_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation of ShelfFilter."""
        gains = f"{self.gain_steps[0][1]:.0f}->{self.gain_steps[-1][1]:.0f}dB"
        return f"Shelf({self.shelf_type}@{self.frequency}Hz {self.stream_type} {gains})"


class PeakFilter(Filter):
    """Parametric peak EQ (mid swap) whose gain follows a scheduled ramp (asendcmd-driven)."""

    def __init__(
        self,
        logger: logging.Logger,
        frequency: int,
        width_oct: float,
        gain_steps: list[tuple[float, float]],
        stream_type: str,
    ):
        """
        Initialize peak filter.

        :param frequency: Peak center frequency in Hz.
        :param width_oct: Peak bandwidth in octaves.
        :param gain_steps: Schedule of (time_seconds, gain_db); the first step at
            t=0 sets the initial gain.
        :param stream_type: 'fadeout' or 'fadein' - which stream to process.
        """
        self.frequency = frequency
        self.width_oct = width_oct
        self.gain_steps = gain_steps
        self.stream_type = stream_type
        if stream_type == "fadeout":
            self.output_fadeout_label = "fadeout_midswap"
            self.output_fadein_label = "fadein_pt_midswap"
        else:
            self.output_fadeout_label = "fadeout_pt_midswap"
            self.output_fadein_label = "fadein_midswap"
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Generate the peak EQ chain on this filter's stream and passthrough on the other."""
        if self.stream_type == "fadeout":
            input_label, output_label = input_fadeout_label, self.output_fadeout_label
            pass_in, pass_out = input_fadein_label, self.output_fadein_label
        else:
            input_label, output_label = input_fadein_label, self.output_fadein_label
            pass_in, pass_out = input_fadeout_label, self.output_fadeout_label
        instance = f"{ShelfType.PEAK}@{self.stream_type}_mid"
        cmd = "; ".join(f"{t:.3f} {instance} g {g:.2f}" for t, g in self.gain_steps)
        initial = self.gain_steps[0][1]
        return [
            f"{pass_in}anull[{pass_out}]",  # codespell:ignore anull
            f"{input_label}asendcmd=c='{cmd}',"
            f"{instance}=g={initial:.2f}:f={self.frequency}:width_type=o:width={self.width_oct}"
            f"[{output_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation of PeakFilter."""
        gains = f"{self.gain_steps[0][1]:.0f}->{self.gain_steps[-1][1]:.0f}dB"
        return f"Peak({self.frequency}Hz {self.stream_type} {gains})"


class SweepFilter(Filter):
    """High- or low-pass whose cutoff (and wet share) follows a schedule (asendcmd-driven)."""

    def __init__(
        self,
        logger: logging.Logger,
        kind: str,
        steps: list[tuple[float, float]],
        mix_steps: list[tuple[float, float]],
        stream_type: str,
        sample_rate: int,
    ):
        """
        Initialize the filter sweep.

        :param kind: The ffmpeg filter: "highpass" or "lowpass".
        :param steps: Schedule of (time_seconds, cutoff_hz); the first step sets the
            initial cutoff.
        :param mix_steps: Schedule of (time_seconds, wet share 0..1); empty keeps the
            filter fully wet.
        :param stream_type: 'fadeout' or 'fadein' - which stream to process.
        :param sample_rate: Sample rate of the stream; cutoffs stay below its Nyquist.
        """
        self.kind = kind
        # a cutoff at or past Nyquist makes the biquad blow up (16 kHz streams)
        self.steps = [(t, min(hz, 0.45 * sample_rate)) for t, hz in steps]
        self.sample_rate = sample_rate
        self.mix_steps = mix_steps
        self.stream_type = stream_type
        if stream_type == "fadeout":
            self.output_fadeout_label = "fadeout_sweep"
            self.output_fadein_label = "fadein_pt_sweep_out"
        else:
            self.output_fadeout_label = "fadeout_pt_sweep_in"
            self.output_fadein_label = "fadein_sweep"
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Generate the sweep on this filter's stream and passthrough on the other."""
        if self.stream_type == "fadeout":
            input_label, output_label = input_fadeout_label, self.output_fadeout_label
            pass_in, pass_out = input_fadein_label, self.output_fadein_label
        else:
            input_label, output_label = input_fadein_label, self.output_fadein_label
            pass_in, pass_out = input_fadeout_label, self.output_fadeout_label
        instance = f"{self.kind}@{self.stream_type}_sweep"
        # asendcmd orders the commands by time itself
        cmd = "; ".join(
            [f"{t:.3f} {instance} f {hz:.1f}" for t, hz in self.steps]
            + [f"{t:.3f} {instance} m {wet:.3f}" for t, wet in self.mix_steps]
        )
        # asendcmd applies a command once per frame: 10 ms frames keep the steps fine enough
        # not to zipper on a one-bar sweep
        return [
            f"{pass_in}anull[{pass_out}]",  # codespell:ignore anull
            f"{input_label}asetnsamples=n={self.sample_rate // 100}:p=0,asendcmd=c='{cmd}',"
            f"{instance}=f={self.steps[0][1]:.1f}:width_type=q:width=0.707[{output_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation of SweepFilter."""
        cutoffs = f"{self.steps[0][1]:.0f}->{self.steps[-1][1]:.0f}Hz"
        return f"Sweep({self.kind} {self.stream_type} {cutoffs})"


class EchoOutFilter(Filter):
    """
    Echo the outgoing stream's last beat on into the incoming one, past the outgoing's end.

    A copy of the outgoing stream keeps only the beat before the point where the crossfade
    cuts it, low-cut and with click-free edges; ``aecho`` repeats it every beat, each repeat
    at half the level of the one before. The copy is moved onto the incoming stream's
    timeline and mixed into it, so the echo rings on under the incoming track while the
    crossfade, the timing and the output length stay as they are.
    """

    output_fadeout_label: str = "fadeout_echo_dry"
    output_fadein_label: str = "fadein_echo"
    # the first repeat 9 dB below the beat (headroom: nothing limits the mix), each next one
    # 6 dB below the one before; the low cut keeps the outgoing kick out of the incoming one's
    first: float = 0.35
    decay: float = 0.5
    low_cut_hz: int = 300

    def __init__(
        self,
        logger: logging.Logger,
        pre_crossfade_samples: int,
        crossfade_samples: int,
        beat_samples: int,
        repeats: int,
        sample_rate: int,
    ):
        """
        Initialize the echo out.

        :param pre_crossfade_samples: Where the incoming stream starts in the mix.
        :param crossfade_samples: Overlap; the outgoing stream is cut at pre + overlap.
        :param beat_samples: One outgoing beat: the echoed slice and the gap between repeats.
        :param repeats: How many times the beat repeats.
        :param sample_rate: Sample rate of both streams.
        """
        self.pre_crossfade_samples = pre_crossfade_samples
        self.crossfade_samples = crossfade_samples
        self.beat_samples = beat_samples
        self.repeats = repeats
        self.sample_rate = sample_rate
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Split the beat off the outgoing stream, echo it, and mix it into the incoming one."""
        end = self.pre_crossfade_samples + self.crossfade_samples
        edge = self.sample_rate // 100
        beats = range(1, self.repeats + 1)
        # aecho truncates each delay to whole samples: aim at the middle of the sample
        delays = "|".join(
            f"{(k * self.beat_samples + 0.5) * 1000 / self.sample_rate:.6f}" for k in beats
        )
        decays = "|".join(f"{self.first * self.decay ** (k - 1):.6f}" for k in beats)
        return [
            f"{input_fadeout_label}asplit=2[{self.output_fadeout_label}][echo_send]",
            f"[echo_send]highpass=f={self.low_cut_hz},"
            f"afade=t=in:start_sample={max(0, end - self.beat_samples)}:nb_samples={edge},"
            # ends 1 ms early: the outgoing trim rounds its end to the millisecond
            f"afade=t=out:start_sample={end - edge - self.sample_rate // 1000}:nb_samples={edge},"
            f"aecho=0:1:{delays}:{decays},"
            f"atrim=start_sample={self.pre_crossfade_samples},asetpts=PTS-STARTPTS[echo_wet]",
            f"{input_fadein_label}[echo_wet]amix=inputs=2:normalize=0:duration=first"
            f"[{self.output_fadein_label}]",
        ]

    def __repr__(self) -> str:
        """Return string representation of EchoOutFilter."""
        return f"EchoOut({self.repeats}x{self.beat_samples} samples)"


class StreamingCrossfadeFilter(Filter):
    """
    Crossfade that emits blended output while the fade-in input is still arriving.

    Same math as ffmpeg's acrossfade (a faded-out and a faded-in stream, summed),
    but built from afade+adelay+amix, which produce a frame as soon as both inputs
    have one — acrossfade holds all output back until its second input hits EOF,
    which stalls a fade against a realtime source for the whole overlap.

    With ``pre_crossfade_samples`` the blend is positioned: the outgoing stream
    plays that long untouched (the incoming side is delayed silence there), fades
    over the overlap, and is cut hard at the planned end — a time-stretched
    branch may land slightly off its planned length, and the cut keeps such
    drift out of the incoming track's audio. Without it, both inputs must hold
    exactly the overlap.
    """

    output_fadeout_label: str = "crossfade"
    output_fadein_label: str = "crossfade"

    def __init__(
        self,
        logger: logging.Logger,
        crossfade_samples: int,
        *,
        pre_crossfade_samples: int = 0,
        fade_samples: int = 0,
        fadeout_curve: str = "qsin",
        fadein_curve: str = "qsin",
    ):
        """
        Initialize streaming crossfade filter.

        :param crossfade_samples: Overlap length in PCM samples.
        :param pre_crossfade_samples: Samples of the outgoing stream played
            untouched before the overlap begins.
        :param fade_samples: How long each stream fades: the outgoing over the
            overlap's last samples, the incoming over its first, both at full
            between; 0 fades both over the whole overlap.
        :param fadeout_curve: afade curve applied to the outgoing stream.
        :param fadein_curve: afade curve applied to the incoming stream.
        """
        self.crossfade_samples = crossfade_samples
        self.pre_crossfade_samples = pre_crossfade_samples
        self.fade_samples = fade_samples
        self.fadeout_curve = fadeout_curve
        self.fadein_curve = fadein_curve
        super().__init__(logger)

    def apply(self, input_fadein_label: str, input_fadeout_label: str) -> list[str]:
        """Apply the afade+adelay+amix filter chain."""
        ns = self.crossfade_samples
        pre = self.pre_crossfade_samples
        fade = min(self.fade_samples, ns) or ns
        fadeout_chain = f"afade=t=out:start_sample={pre + ns - fade}:nb_samples={fade}:curve={self.fadeout_curve}"
        fadein_chain = f"afade=t=in:start_sample=0:nb_samples={fade}:curve={self.fadein_curve}"
        if pre:
            fadeout_chain += f",atrim=end_sample={pre + ns}"
            fadein_chain += f",adelay={pre}S:all=1"
        # equal-power qsin curves; the default tri/tri dips ~3dB mid-fade on uncorrelated
        # material. The final output stays unlabeled: this filter ends the chain and an
        # unconnected named output fails the whole graph.
        return [
            f"{input_fadeout_label}{fadeout_chain}[xfade_out]",
            f"{input_fadein_label}{fadein_chain}[xfade_in]",
            "[xfade_out][xfade_in]amix=inputs=2:normalize=0",
        ]

    def __repr__(self) -> str:
        """Return string representation of StreamingCrossfadeFilter."""
        if self.pre_crossfade_samples:
            return (
                f"StreamingCrossfade(pre={self.pre_crossfade_samples}, ns={self.crossfade_samples})"
            )
        return f"StreamingCrossfade(ns={self.crossfade_samples})"
