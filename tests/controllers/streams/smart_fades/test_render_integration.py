"""End-to-end render check: the built chain actually swaps the bass in ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator

import numpy as np
import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.smart_fades.fades import (
    SmartCrossFade,
    StandardCrossFade,
    _feed_ffmpeg_stdin,
)
from music_assistant.controllers.streams.smart_fades.planner.requested import (
    CUT_SECONDS,
    RequestedTransitionPlanner,
    TransitionRequest,
)
from music_assistant.helpers.process import AsyncProcess
from music_assistant.models.audio_analysis import AudioAnalysisData

PCM = AudioFormat(content_type=ContentType.PCM_F32LE, sample_rate=44100, bit_depth=32, channels=2)
SR = 44100


def _tone(freq: float, seconds: float, level: float = 0.2) -> np.ndarray:
    """Return a stereo-interleaved sine tone."""
    t = np.arange(int(SR * seconds)) / SR
    mono = (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.repeat(mono, 2)


def _analysis(bpm: float, duration: float) -> AudioAnalysisData:
    """Synthetic flat-energy analysis with a steady beat grid."""
    interval = 60.0 / bpm
    beats = np.arange(0.0, duration, interval, dtype=np.float32)
    return AudioAnalysisData(
        duration=duration,
        bpm=bpm,
        beats=beats.tolist(),
        downbeats=beats[::4].tolist(),
        rms_energy=np.full(1800, 0.5, dtype=np.float32).tolist(),
        key="A",
        mode="minor",
    )


def _with_bands(
    analysis: AudioAnalysisData, low: float, low_mid: float, mid: float, high: float
) -> AudioAnalysisData:
    """Attach flat ``band_rms`` envelopes at the given amplitudes."""
    analysis.extra_data = {
        "band_rms": {
            "low": np.full(1800, low, dtype=np.float32).tolist(),
            "low_mid": np.full(1800, low_mid, dtype=np.float32).tolist(),
            "mid": np.full(1800, mid, dtype=np.float32).tolist(),
            "high": np.full(1800, high, dtype=np.float32).tolist(),
        }
    }
    return analysis


def _analysis_with_mid_bands(bpm: float, duration: float) -> AudioAnalysisData:
    """Analysis with a mid-heavy, bass-light ``band_rms`` profile that clears the mid gate."""
    # bass-light so the low swap stays out of the way; mid-heavy and constant
    # so duty_mid saturates to 1.0 and F_mid clears the 0.18-0.30 gate corridor
    return _with_bands(_analysis(bpm, duration), 0.05, 0.3, 0.7, 0.3)


def _analysis_with_instrumental_bands(bpm: float, duration: float) -> AudioAnalysisData:
    """Analysis with a bass-light, mid-light profile: every measured EQ gate bypasses."""
    # f_low ~0.014 and f_mid ~0.13 sit below their gate corridors, so both the
    # low and mid swap bypass while anchors/entry stay on the full-band paths
    return _with_bands(_analysis(bpm, duration), 0.1, 0.55, 0.3, 0.55)


def _band_rms(x: np.ndarray, lo: float, hi: float) -> float:
    """RMS of one frequency band of the (interleaved stereo) signal's left channel."""
    mono = x[0::2]
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(spec[mask] ** 2)))


async def _render(
    out_analysis: AudioAnalysisData,
    in_analysis: AudioAnalysisData,
    fade_out: bytes,
    fade_in: bytes,
) -> tuple[np.ndarray, SmartCrossFade]:
    """Build and apply a SmartCrossFade, returning the rendered mix and the fade."""
    fade = SmartCrossFade(logging.getLogger(), out_analysis, in_analysis)
    fade.build(len(fade_out), len(fade_in), PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out, fade_in, PCM)]
    return np.frombuffer(b"".join(chunks), dtype=np.float32), fade


def _cf_slice(mix: np.ndarray, fade: SmartCrossFade, frac0: float, frac1: float) -> np.ndarray:
    """Slice the rendered crossfade window between two fractions of its span."""
    timing = fade.timing_info
    start_s = timing.pre_crossfade_duration + frac0 * timing.crossfade_duration
    end_s = timing.pre_crossfade_duration + frac1 * timing.crossfade_duration
    return mix[int(start_s * SR) * 2 : int(end_s * SR) * 2]


@pytest.mark.asyncio
async def test_bass_swaps_between_tracks() -> None:
    """The low shelves attenuate A's bass and duck B's entrance vs an EQ-bypassed render."""
    fade_out = (_tone(60.0, 45.0) + _tone(3000.0, 45.0)).tobytes()  # A: 60Hz bass
    fade_in = (_tone(90.0, 45.0) + _tone(5000.0, 45.0)).tobytes()  # B: 90Hz bass
    # differential render: identical PCM, one plan with the shipped full-depth
    # kill (no band data) and one whose measured gates bypass all low shelves --
    # any energy difference is then attributable to the low EQ, not acrossfade
    killed_mix, killed = await _render(
        _analysis(120.0, 240.0), _analysis(120.0, 240.0), fade_out, fade_in
    )
    open_mix, open_ = await _render(
        _analysis_with_instrumental_bands(120.0, 240.0),
        _analysis_with_instrumental_bands(120.0, 240.0),
        fade_out,
        fade_in,
    )
    assert killed.plan is not None
    assert killed.plan.eq_plan.low_out is not None
    assert open_.plan is not None
    assert open_.plan.eq_plan.low_out is None
    assert open_.plan.eq_plan.low_in is None
    # identical geometry: the band data must only change EQ, never the timing
    assert len(killed_mix) == len(open_mix)
    # measure inside the crossfade window itself: A's bass is killed where the
    # swap completes (late); B enters bass-ducked (early); -26dB kill leaves
    # well under 30% of the bypassed render's energy
    killed_late = _cf_slice(killed_mix, killed, 0.7, 0.95)
    open_late = _cf_slice(open_mix, open_, 0.7, 0.95)
    killed_early = _cf_slice(killed_mix, killed, 0.05, 0.3)
    open_early = _cf_slice(open_mix, open_, 0.05, 0.3)
    assert _band_rms(killed_late, 55, 65) < 0.3 * _band_rms(open_late, 55, 65)
    assert _band_rms(killed_early, 85, 95) < 0.3 * _band_rms(open_early, 85, 95)
    # sanity on the killed render alone: A's bass dominates early, B's late
    assert _band_rms(killed_early, 55, 65) > 3 * _band_rms(killed_early, 85, 95)
    assert _band_rms(killed_late, 85, 95) > 3 * _band_rms(killed_late, 55, 65)


@pytest.mark.asyncio
async def test_mid_swaps_between_tracks() -> None:
    """The mid peaks trade A's 1kHz for B's 2kHz vs an EQ-bypassed render of the same PCM."""
    fade_out = _tone(1000.0, 45.0).tobytes()  # A: 1kHz "vocal"
    fade_in = _tone(2000.0, 45.0).tobytes()  # B: 2kHz "vocal"
    # differential render: identical PCM, one plan whose band data engages the
    # mid gate and one whose band data bypasses every measured EQ gate -- the
    # 1k/2k energy difference is then attributable to the mid EQ alone
    gated_mix, gated = await _render(
        _analysis_with_mid_bands(120.0, 240.0),
        _analysis_with_mid_bands(120.0, 240.0),
        fade_out,
        fade_in,
    )
    open_mix, open_ = await _render(
        _analysis_with_instrumental_bands(120.0, 240.0),
        _analysis_with_instrumental_bands(120.0, 240.0),
        fade_out,
        fade_in,
    )
    assert gated.plan is not None
    assert gated.plan.eq_plan.mid_out is not None
    assert gated.plan.eq_plan.mid_in is not None
    assert open_.plan is not None
    assert open_.plan.eq_plan.mid_out is None
    assert open_.plan.eq_plan.mid_in is None
    # identical geometry: the band data must only change EQ, never the timing
    assert len(gated_mix) == len(open_mix)
    # the -8dB depth is modest, so assert a measurable drop (not dominance):
    # A's 1kHz is attenuated where the swap completes (late); B's 2kHz enters
    # ducked (early); both measured against the EQ-bypassed render, inside
    # the crossfade window itself
    gated_late = _cf_slice(gated_mix, gated, 0.7, 0.95)
    open_late = _cf_slice(open_mix, open_, 0.7, 0.95)
    gated_early = _cf_slice(gated_mix, gated, 0.05, 0.3)
    open_early = _cf_slice(open_mix, open_, 0.05, 0.3)
    assert _band_rms(gated_late, 950, 1050) < 0.7 * _band_rms(open_late, 950, 1050)
    assert _band_rms(gated_early, 1950, 2050) < 0.7 * _band_rms(open_early, 1950, 2050)


@pytest.mark.asyncio
async def test_a_failing_fade_in_ends_the_mix_instead_of_hanging() -> None:
    """An incoming stream that dies mid-overlap must not leave ffmpeg waiting for input."""
    fade_out = _tone(220.0, 6.0).tobytes()
    delivered = _tone(440.0, 1.0).tobytes()

    async def _dying_fade_in() -> AsyncGenerator[bytes]:
        yield delivered
        raise RuntimeError("incoming source died")

    fade = StandardCrossFade(logging.getLogger(), crossfade_duration=2)
    fade.build(len(fade_out), len(_tone(440.0, 4.0).tobytes()), PCM)

    async def _drain_mix() -> None:
        # the timeout only bounds the failure: without the EOF the mix hangs here
        async with asyncio.timeout(30):
            async for _chunk in fade.apply(fade_out, _dying_fade_in(), PCM):
                pass

    started = asyncio.get_event_loop().time()
    with pytest.raises(RuntimeError, match="incoming source died"):
        await _drain_mix()
    assert asyncio.get_event_loop().time() - started < 10


@pytest.mark.asyncio
async def test_cancelled_feed_does_not_hang_writing_eof() -> None:
    """A feeder cancelled while blocked on a full stdin pipe must return, not stall on the EOF."""
    proc = AsyncProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=True,
        name="stdin-blackhole",
    )
    await proc.start()

    async def _endless_zeros() -> AsyncGenerator[bytes]:
        chunk = b"\x00" * (1024 * 1024)
        while True:
            yield chunk

    try:
        feed_task = asyncio.create_task(_feed_ffmpeg_stdin(proc, _endless_zeros()))
        # let the feeder fill the pipe and the transport's write buffer, then block
        await asyncio.sleep(0.5)
        assert not feed_task.done()
        feed_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(feed_task, timeout=2)
    finally:
        # the child never reads its stdin, so a graceful close would only wait out its timeout
        await proc.kill()


@pytest.mark.asyncio
async def test_source_cancelled_feed_still_ends_the_input() -> None:
    """A fade-in that raises CancelledError itself still ends the mixer's input with an EOF."""
    # the child exits only once its stdin reaches EOF
    proc = AsyncProcess(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=True,
        name="stdin-reader",
    )
    await proc.start()

    async def _cancelled_fade_in() -> AsyncGenerator[bytes]:
        yield b"\x00" * 1024
        raise asyncio.CancelledError

    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(_feed_ffmpeg_stdin(proc, _cancelled_fade_in()))
        assert await proc.wait_with_timeout(2) == 0
    finally:
        await proc.close()


@pytest.mark.asyncio
async def test_a_requested_cut_switches_tracks_at_the_exit_without_a_gap() -> None:
    """A cut plays A up to its exit downbeat and B right after, with no silence between."""
    fade_out, fade_in = _tone(440.0, 45.0), _tone(1760.0, 45.0)
    planner = RequestedTransitionPlanner(
        logging.getLogger(), TransitionRequest("cut", "n", 0, 224.0)
    )
    fade = SmartCrossFade(
        logging.getLogger(), _analysis(120.0, 240.0), _analysis(120.0, 240.0), planner
    )
    fade.build(fade_out.nbytes, fade_in.nbytes, PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out.tobytes(), fade_in.tobytes(), PCM)]
    mix = np.frombuffer(b"".join(chunks), dtype=np.float32)
    timing = fade.timing_info
    cut = timing.pre_crossfade_duration + timing.crossfade_duration

    assert planner.outcome == "applied"
    assert timing.crossfade_duration == pytest.approx(CUT_SECONDS, abs=1e-4)
    # A's audio ends at its exit downbeat, 224 s into the song (29 s into the tail)
    assert cut == pytest.approx(29.0, abs=0.01)
    assert len(mix) // 2 == pytest.approx(int((cut + timing.post_crossfade_duration) * SR), abs=2)

    def window(start: float, end: float) -> np.ndarray:
        return mix[int(start * SR) * 2 : int(end * SR) * 2]

    before, after = window(cut - 1.0, cut - 0.05), window(cut + 0.05, cut + 1.0)
    assert _band_rms(before, 430, 450) > 10 * _band_rms(before, 1750, 1770)
    assert _band_rms(after, 1750, 1770) > 10 * _band_rms(after, 430, 450)
    # no 10 ms stretch around the cut drops more than 30 dB below the tones' level
    around = window(cut - 0.5, cut + 0.5)[0::2]
    rms = np.sqrt(np.mean(around[: len(around) // 441 * 441].reshape(-1, 441) ** 2, axis=1))
    assert rms.min() > 0.2 / np.sqrt(2) * 10 ** (-30 / 20)


@pytest.mark.asyncio
async def test_a_requested_cut_never_lands_on_silence_in_the_incoming_track() -> None:
    """B falls silent in the bar after its one: the cut lands on B's next one, never on the gap."""
    fade_out, fade_in = _tone(440.0, 45.0), _tone(1760.0, 45.0)
    # B's one 1 s in, then 0.4 s of silence, as a pickup that dies away before the song starts
    fade_in[int(1.6 * SR) * 2 : int(2.0 * SR) * 2] = 0.0
    inc = _analysis(120.0, 240.0)
    assert inc.beats is not None
    assert inc.downbeats is not None
    assert inc.rms_energy is not None
    inc.beats = [beat + 1.0 for beat in inc.beats]
    inc.downbeats = [downbeat + 1.0 for downbeat in inc.downbeats]
    inc.rms_energy[int(1.6 / 240.0 * 1800) : int(2.0 / 240.0 * 1800)] = [0.001] * 3
    planner = RequestedTransitionPlanner(
        logging.getLogger(), TransitionRequest("cut", "n", 0, 224.0)
    )
    fade = SmartCrossFade(logging.getLogger(), _analysis(120.0, 240.0), inc, planner)
    fade.build(fade_out.nbytes, fade_in.nbytes, PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out.tobytes(), fade_in.tobytes(), PCM)]
    mix = np.frombuffer(b"".join(chunks), dtype=np.float32)
    timing = fade.timing_info
    cut = timing.pre_crossfade_duration + timing.crossfade_duration

    assert planner.outcome == "applied"
    assert cut == pytest.approx(29.0, abs=0.01)
    # no 10 ms stretch of the bar after the cut falls under -50 dBFS
    after = mix[int(cut * SR) * 2 : int((cut + 2.0) * SR) * 2][0::2]
    rms = np.sqrt(np.mean(after[: len(after) // 441 * 441].reshape(-1, 441) ** 2, axis=1))
    assert 20 * np.log10(rms.min() + 1e-12) > -50.0


def _ticking(
    beats: list[float], start: float, seconds: float, tick_hz: float, pad_hz: float, pad_from: float
) -> np.ndarray:
    """Return a stereo track ticking at ``tick_hz`` on each beat, over a quiet pad from ``pad_from``."""
    t = start + np.arange(int(SR * seconds)) / SR
    mono = np.where(t >= pad_from, 0.1 * np.sin(2 * np.pi * pad_hz * t), 0.0)
    width = int(0.004 * SR)
    tick = 0.5 * np.hanning(width) * np.sin(2 * np.pi * tick_hz * np.arange(width) / SR)
    for beat in beats:
        at = round((beat - start) * SR)
        if 0 <= at <= len(mono) - width:
            mono[at : at + width] += tick
    return np.repeat(mono.astype(np.float32), 2)


def _ticks_heard(mono: np.ndarray, tick_hz: float, start: float, end: float) -> list[float]:
    """Return where the mix carries a tick at ``tick_hz`` between two times, within 30 dB of the loudest."""
    smooth = np.hanning(int(0.004 * SR))
    level = np.abs(
        np.convolve(
            mono * np.exp(-2j * np.pi * tick_hz * np.arange(len(mono)) / SR),
            smooth / smooth.sum(),
            mode="same",
        )
    )
    floor = level.max() * 10 ** (-30 / 20)
    lo, hi = int(start * SR), int(end * SR)
    window = level[lo:hi]
    peaks = np.flatnonzero(
        (window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:]) & (window[1:-1] > floor)
    )
    return [(lo + 1 + peak) / SR for peak in peaks]


@pytest.mark.asyncio
@pytest.mark.parametrize("pickup", [0, 2], ids=["no-pickup", "pickup"])
async def test_a_requested_quick_fade_between_far_tempos_never_doubles_a_beat(pickup: int) -> None:
    """At 100 vs 128 BPM the beats heard together stay 40 ms apart at most, and B lands on A's end."""
    out, inc = _analysis(100.0, 240.0), _analysis(128.0, 240.0)
    assert out.beats is not None
    assert inc.beats is not None
    # B is silent until its first beat; with a pickup its bars start two beats later
    inc.beats = [beat + 0.1 for beat in inc.beats]
    inc.downbeats = inc.beats[pickup::4]
    fade_out = _ticking(out.beats, 240.0 - 45.0, 45.0, 2000.0, 220.0, 0.0)
    fade_in = _ticking(inc.beats, 0.0, 45.0, 6000.0, 330.0, 0.1)
    planner = RequestedTransitionPlanner(
        logging.getLogger(), TransitionRequest("quick_fade", "n", 0, 224.0)
    )
    fade = SmartCrossFade(logging.getLogger(), out, inc, planner)
    fade.build(fade_out.nbytes, fade_in.nbytes, PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out.tobytes(), fade_in.tobytes(), PCM)]
    mix = np.frombuffer(b"".join(chunks), dtype=np.float32)[0::2]
    timing = fade.timing_info
    start = timing.pre_crossfade_duration
    end = start + timing.crossfade_duration

    a_ticks = _ticks_heard(mix, 2000.0, start - 0.01, end + 0.01)
    b_ticks = _ticks_heard(mix, 6000.0, start - 0.01, end + 0.01)
    apart = [min(abs(a - b) for a in a_ticks) for b in b_ticks if a_ticks]
    # a beat of one deck heard within a quarter second of the other's is one beat played twice
    assert [gap for gap in apart if 0.045 < gap < 0.25] == []
    # A ends on its downbeat 223.2 s into the song; B's one ticks right there
    assert end == pytest.approx(223.2 - 195.0, abs=0.001)
    assert (
        min(abs(tick - end) for tick in _ticks_heard(mix, 6000.0, end - 0.05, end + 0.05)) < 0.005
    )
    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    # no 10 ms stretch around the switch drops 30 dB under the pads
    around = mix[int((end - 0.5) * SR) : int((end + 0.5) * SR)]
    rms = np.sqrt(np.mean(around[: len(around) // 441 * 441].reshape(-1, 441) ** 2, axis=1))
    assert rms.min() > 0.1 / np.sqrt(2) * 10 ** (-30 / 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["cut", "quick_fade"])
async def test_a_sung_pickup_pre_rolls_under_the_outgoing_track(style: str) -> None:
    """
    B sung 0.7 s before its one: it plays under A's last beats, its one landing on A's exit.

    A cut holds both tracks at full over the pickup; a quick fade between these tempos (they
    drift apart within a bar) fades over it.
    """
    fade_out, fade_in = _tone(440.0, 45.0), np.zeros(int(45.0 * SR) * 2, dtype=np.float32)
    # B (Pepas): silent, its voice (1760 Hz) from 0.4 s, its one 1.1 s in at 130 bpm
    fade_in[int(0.4 * SR) * 2 :] = _tone(1760.0, 45.0)[int(0.4 * SR) * 2 :]
    out, inc = _analysis(120.0, 240.0), _analysis(130.0, 240.0)
    assert out.beats is not None
    assert inc.beats is not None
    assert inc.rms_energy is not None
    # a click on A's downbeats, B's one, and B's later downbeats
    for downbeat in np.arange(1.0, 45.0, 2.0):
        fade_out[int(downbeat * SR) * 2 : int(downbeat * SR) * 2 + 2] += 0.8
    inc.beats = [beat + 0.18 for beat in inc.beats]
    inc.downbeats = inc.beats[2::4]
    for downbeat in inc.downbeats:
        if downbeat < 45.0:
            fade_in[int(downbeat * SR) * 2 : int(downbeat * SR) * 2 + 2] += 0.8
    inc.rms_energy[0] = 0.0
    inc.extra_data = {"vocal_activity": [0.9 if i >= 2 else 0.05 for i in range(1800)]}
    planner = RequestedTransitionPlanner(
        logging.getLogger(), TransitionRequest(style, "n", 0, 224.0)
    )
    fade = SmartCrossFade(logging.getLogger(), out, inc, planner)
    fade.build(fade_out.nbytes, fade_in.nbytes, PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out.tobytes(), fade_in.tobytes(), PCM)]
    mix = np.frombuffer(b"".join(chunks), dtype=np.float32)
    timing = fade.timing_info
    cut = timing.pre_crossfade_duration + timing.crossfade_duration

    def window(start: float, end: float) -> np.ndarray:
        return mix[int(start * SR) * 2 : int(end * SR) * 2]

    assert (planner.outcome, planner.reason) == (
        "applied",
        None if style == "cut" else "shortened",
    )
    assert cut == pytest.approx(29.0, abs=0.001)
    assert timing.crossfade_duration > 0.7
    # one click within 60 ms of A's exit: B's one, on it to the millisecond; A's is cut away
    edges = np.flatnonzero(np.abs(np.diff(window(cut - 0.06, cut + 0.06)[0::2])) > 0.3)
    assert len(edges) > 0
    assert (edges.max() - edges.min()) / SR < 0.003
    assert (edges.min() + 1) / SR - 0.06 == pytest.approx(0.0, abs=0.002)
    pre, alone_a, alone_b = (
        window(cut - 0.5, cut - 0.03),
        window(cut - 3.0, cut - 2.53),
        window(cut + 0.03, cut + 0.5),
    )
    a_level, b_level = _band_rms(pre, 430, 450), _band_rms(pre, 1750, 1770)
    if style == "cut":
        # A at full to its exit, B's pickup at full under it, as each plays alone
        assert a_level == pytest.approx(_band_rms(alone_a, 430, 450), rel=0.1)
        assert b_level == pytest.approx(_band_rms(alone_b, 1750, 1770), rel=0.1)
    else:
        # A fades out under B's pickup, heard before its one
        assert a_level < 0.7 * _band_rms(alone_a, 430, 450)
        assert b_level > 0.7 * _band_rms(alone_b, 1750, 1770)
    # no 10 ms stretch from A's last bar to B's first drops more than 30 dB below a tone
    around = window(cut - 2.0, cut + 2.0)[0::2]
    rms = np.sqrt(np.mean(around[: len(around) // 441 * 441].reshape(-1, 441) ** 2, axis=1))
    assert rms.min() > 0.2 / np.sqrt(2) * 10 ** (-30 / 20)


@pytest.mark.asyncio
async def test_a_requested_cut_keeps_the_one_when_the_incoming_track_stops_later() -> None:
    """B's own stop 2.5 beats into its bar plays after B has started: B enters on its one, the stop where B has it."""
    fade_out, fade_in = _tone(440.0, 45.0), _tone(1760.0, 45.0)
    # B's one 1 s in (120 bpm), its own stop from 2.25 s to 2.5 s: beats 2.5 to 3 of the bar
    fade_in[int(2.25 * SR) * 2 : int(2.5 * SR) * 2] = 0.0
    inc = _analysis(120.0, 240.0)
    assert inc.beats is not None
    assert inc.downbeats is not None
    assert inc.rms_energy is not None
    inc.beats = [beat + 1.0 for beat in inc.beats]
    inc.downbeats = [downbeat + 1.0 for downbeat in inc.downbeats]
    inc.rms_energy[int(2.25 / 240.0 * 1800) : int(2.5 / 240.0 * 1800)] = [0.001] * 2
    planner = RequestedTransitionPlanner(
        logging.getLogger(), TransitionRequest("cut", "n", 0, 224.0)
    )
    fade = SmartCrossFade(logging.getLogger(), _analysis(120.0, 240.0), inc, planner)
    fade.build(fade_out.nbytes, fade_in.nbytes, PCM)
    chunks = [chunk async for chunk in fade.apply(fade_out.tobytes(), fade_in.tobytes(), PCM)]
    mix = np.frombuffer(b"".join(chunks), dtype=np.float32)
    timing = fade.timing_info
    cut = timing.pre_crossfade_duration + timing.crossfade_duration

    def levels(start_s: float, end_s: float) -> np.ndarray:
        part = mix[int(start_s * SR) * 2 : int(end_s * SR) * 2][0::2]
        part = part[: len(part) // 441 * 441].reshape(-1, 441)
        return 20 * np.log10(np.sqrt(np.mean(part**2, axis=1)) + 1e-12)

    assert planner.outcome == "applied"
    assert cut == pytest.approx(29.0, abs=0.01)
    # B sounds from the switch on; its stop comes 1.25 s after it, as B has it after its one
    assert levels(cut + 0.03, cut + 1.2).min() > -50.0
    assert levels(cut + 1.3, cut + 1.45).max() < -60.0
