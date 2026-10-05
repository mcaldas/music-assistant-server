"""Tests for the smart fades FFmpeg filter builders (the toolset)."""

from __future__ import annotations

import logging

from music_assistant.controllers.streams.smart_fades.filters import (
    EchoOutFilter,
    FadeOutTrimFilter,
    PeakFilter,
    ShelfFilter,
    ShelfType,
    StreamingCrossfadeFilter,
    SweepFilter,
)

LOGGER = logging.getLogger(__name__)


def test_fadeout_trim_trims_fadeout_and_passes_fadein_through() -> None:
    """The fadeout stream is end-trimmed; the fadein stream is untouched."""
    fadeout_trim = FadeOutTrimFilter(logger=LOGGER, fadeout_end_pos=35.0, trimmed_seconds=10.0)
    filter_strings = fadeout_trim.apply("[fadein]", "[fadeout]")
    assert len(filter_strings) == 2
    assert repr(fadeout_trim) == "FadeOutTrim(end=35.00s, trimmed=10.00s)"

    trim_chain = next(f for f in filter_strings if "atrim" in f)
    assert trim_chain.startswith("[fadeout]")
    assert "atrim=end=35.000" in trim_chain
    assert "asetpts=PTS-STARTPTS" in trim_chain
    assert trim_chain.endswith(f"[{fadeout_trim.output_fadeout_label}]")

    passthrough = next(f for f in filter_strings if "anull" in f)  # codespell:ignore anull
    assert passthrough.startswith("[fadein]")
    assert passthrough.endswith(f"[{fadeout_trim.output_fadein_label}]")


def test_streaming_crossfade_blends_the_exact_overlap() -> None:
    """
    The blend fades both streams over the sample-exact overlap and sums them.

    afade+amix instead of acrossfade on purpose: acrossfade holds all output
    back until its second input hits EOF, which would stall a fade whose
    incoming side is still arriving. Equal-power qsin curves, not ffmpeg's
    default tri/tri. The final output stays unlabeled: an unconnected named
    output fails the whole graph.
    """
    crossfade = StreamingCrossfadeFilter(logger=LOGGER, crossfade_samples=441000)
    filter_strings = crossfade.apply("[fadein]", "[fadeout]")
    assert filter_strings == [
        "[fadeout]afade=t=out:start_sample=0:nb_samples=441000:curve=qsin[xfade_out]",
        "[fadein]afade=t=in:start_sample=0:nb_samples=441000:curve=qsin[xfade_in]",
        "[xfade_out][xfade_in]amix=inputs=2:normalize=0",
    ]


def test_streaming_crossfade_emits_the_given_curves() -> None:
    """Explicit fadeout/fadein curves override the default qsin:qsin pair."""
    crossfade = StreamingCrossfadeFilter(
        logger=LOGGER, crossfade_samples=441000, fadeout_curve="nofade", fadein_curve="tri"
    )
    filter_strings = crossfade.apply("[fadein]", "[fadeout]")
    assert "curve=nofade" in filter_strings[0]
    assert "curve=tri" in filter_strings[1]


def test_streaming_crossfade_positions_the_blend() -> None:
    """
    A positioned blend delays the incoming stream and hard-cuts the outgoing one.

    The pre-point places the fade on the outgoing stream, adelay (sample-exact,
    ``S`` suffix) aligns the incoming stream under it, and the trim at the
    planned end keeps any time-stretch drift out of the incoming audio.
    """
    crossfade = StreamingCrossfadeFilter(
        logger=LOGGER, crossfade_samples=441000, pre_crossfade_samples=882000
    )
    filter_strings = crossfade.apply("[fadein]", "[fadeout]")
    assert filter_strings == [
        "[fadeout]afade=t=out:start_sample=882000:nb_samples=441000:curve=qsin,"
        "atrim=end_sample=1323000[xfade_out]",
        "[fadein]afade=t=in:start_sample=0:nb_samples=441000:curve=qsin,"
        "adelay=882000S:all=1[xfade_in]",
        "[xfade_out][xfade_in]amix=inputs=2:normalize=0",
    ]


def test_streaming_crossfade_holds_both_streams_between_short_fades() -> None:
    """
    With a fade length the outgoing stream fades over the overlap's last samples only.

    The incoming one fades in over its first samples: both play at full in between, as
    for a pickup coming in under the outgoing track's last beats before a cut.
    """
    crossfade = StreamingCrossfadeFilter(
        logger=LOGGER, crossfade_samples=44100, pre_crossfade_samples=882000, fade_samples=882
    )
    filter_strings = crossfade.apply("[fadein]", "[fadeout]")
    assert filter_strings == [
        "[fadeout]afade=t=out:start_sample=925218:nb_samples=882:curve=qsin,"
        "atrim=end_sample=926100[xfade_out]",
        "[fadein]afade=t=in:start_sample=0:nb_samples=882:curve=qsin,"
        "adelay=882000S:all=1[xfade_in]",
        "[xfade_out][xfade_in]amix=inputs=2:normalize=0",
    ]


class TestShelfFilter:
    """asendcmd-driven shelving EQ on one stream, passthrough on the other."""

    def test_fadeout_lowshelf_strings(self) -> None:
        """A fadeout lowshelf emits an asendcmd gain schedule and a fadein passthrough."""
        f = ShelfFilter(
            LOGGER,
            ShelfType.LOW,
            100,
            [(0.0, 0.0), (30.0, -13.0), (31.0, -26.0)],
            "fadeout",
        )
        strings = f.apply("[1]", "[0]")
        assert len(strings) == 2
        # passthrough for the untouched stream
        assert any("anull" in s and "[1]" in s for s in strings)  # codespell:ignore anull
        chain = next(s for s in strings if "[0]" in s)
        assert "asendcmd=" in chain
        assert "lowshelf@fadeout_low" in chain
        assert "g=0.00" in chain  # initial gain from the first step
        assert "f=100" in chain
        assert "30.000 lowshelf@fadeout_low g -13.00" in chain

    def test_fadein_highshelf_processes_other_stream(self) -> None:
        """A fadein highshelf processes the incoming stream and passes the outgoing through."""
        f = ShelfFilter(LOGGER, ShelfType.HIGH, 13000, [(0.0, -20.0), (5.0, 0.0)], "fadein")
        strings = f.apply("[1]", "[0]")
        chain = next(s for s in strings if "[1]" in s)
        assert "highshelf@fadein_high" in chain
        assert "g=-20.00" in chain
        passthrough = next(s for s in strings if "[0]" in s)
        assert "anull" in passthrough  # codespell:ignore anull

    def test_labels_unique_per_band_and_stream(self) -> None:
        """Output labels differ per band and stream so four instances can coexist."""
        a = ShelfFilter(LOGGER, ShelfType.LOW, 100, [(0.0, -26.0)], "fadein")
        b = ShelfFilter(LOGGER, ShelfType.HIGH, 13000, [(0.0, -20.0)], "fadein")
        assert a.output_fadein_label != b.output_fadein_label
        assert a.output_fadeout_label != b.output_fadeout_label
        c = ShelfFilter(LOGGER, ShelfType.LOW, 100, [(0.0, -26.0)], "fadeout")
        assert a.output_fadein_label != c.output_fadein_label


class TestPeakFilter:
    """asendcmd-driven parametric peak EQ (mid swap) on one stream, passthrough on the other."""

    def test_fadeout_peak_strings(self) -> None:
        """A fadeout peak emits an asendcmd gain schedule and a fadein passthrough."""
        f = PeakFilter(
            LOGGER,
            1200,
            2.5,
            [(0.0, 0.0), (30.0, -4.0), (31.0, -8.0)],
            "fadeout",
        )
        strings = f.apply("[1]", "[0]")
        assert len(strings) == 2
        assert any("anull" in s and "[1]" in s for s in strings)  # codespell:ignore anull
        chain = next(s for s in strings if "[0]" in s)
        assert "asendcmd=" in chain
        assert "equalizer@fadeout_mid" in chain
        assert "g=0.00" in chain
        assert "f=1200:width_type=o:width=2.5" in chain
        assert "30.000 equalizer@fadeout_mid g -4.00" in chain

    def test_fadein_peak_processes_other_stream(self) -> None:
        """A fadein peak processes the incoming stream and passes the outgoing through."""
        f = PeakFilter(LOGGER, 1200, 2.5, [(0.0, -8.0), (5.0, 0.0)], "fadein")
        strings = f.apply("[1]", "[0]")
        chain = next(s for s in strings if "[1]" in s)
        assert "equalizer@fadein_mid" in chain
        assert "g=-8.00" in chain
        passthrough = next(s for s in strings if "[0]" in s)
        assert "anull" in passthrough  # codespell:ignore anull

    def test_labels_unique_and_no_collision_with_shelf(self) -> None:
        """Peak labels differ per stream and don't collide with ShelfFilter's labels."""
        a = PeakFilter(LOGGER, 1200, 2.5, [(0.0, -8.0)], "fadein")
        b = PeakFilter(LOGGER, 1200, 2.5, [(0.0, -8.0)], "fadeout")
        assert a.output_fadein_label != b.output_fadein_label
        low = ShelfFilter(LOGGER, ShelfType.LOW, 100, [(0.0, -26.0)], "fadein")
        high = ShelfFilter(LOGGER, ShelfType.HIGH, 13000, [(0.0, -20.0)], "fadein")
        assert a.output_fadein_label not in {low.output_fadein_label, high.output_fadein_label}
        assert a.output_fadeout_label not in {low.output_fadeout_label, high.output_fadeout_label}


def test_sweep_filter_drives_the_cutoff_and_the_wet_share() -> None:
    """A sweep is an asendcmd schedule on a named high/low-pass; the other stream passes."""
    high = SweepFilter(
        LOGGER, "highpass", [(0.0, 20.0), (13.0, 20.0), (29.0, 8000.0)], [], "fadeout", 44100
    )
    assert high.apply("[fadein]", "[fadeout]") == [
        "[fadein]anull[fadein_pt_sweep_out]",  # codespell:ignore anull
        "[fadeout]asetnsamples=n=441:p=0,asendcmd=c='0.000 highpass@fadeout_sweep f 20.0; "
        "13.000 highpass@fadeout_sweep f 20.0; 29.000 highpass@fadeout_sweep f 8000.0',"
        "highpass@fadeout_sweep=f=20.0:width_type=q:width=0.707[fadeout_sweep]",
    ]
    low = SweepFilter(
        LOGGER,
        "lowpass",
        [(0.0, 250.0), (12.0, 8000.0)],
        [(12.0, 1.0), (14.4, 0.0)],
        "fadein",
        44100,
    )
    assert low.apply("[fadein]", "[fadeout]") == [
        "[fadeout]anull[fadeout_pt_sweep_in]",  # codespell:ignore anull
        "[fadein]asetnsamples=n=441:p=0,asendcmd=c='0.000 lowpass@fadein_sweep f 250.0; "
        "12.000 lowpass@fadein_sweep f 8000.0; 12.000 lowpass@fadein_sweep m 1.000; "
        "14.400 lowpass@fadein_sweep m 0.000',"
        "lowpass@fadein_sweep=f=250.0:width_type=q:width=0.707[fadein_sweep]",
    ]
    # a 16 kHz stream: no cutoff at or past Nyquist, where the biquad blows up
    capped = SweepFilter(LOGGER, "lowpass", [(0.0, 250.0), (12.0, 8000.0)], [], "fadein", 16000)
    assert capped.steps == [(0.0, 250.0), (12.0, 7200.0)]


def test_echo_out_filter_echoes_the_last_beat_into_the_incoming_stream() -> None:
    """
    The beat before the cut is split off, echoed wet-only and mixed into the incoming stream.

    The gates are sample-exact around pre + overlap (where the crossfade cuts the outgoing
    stream); the delays aim at mid-sample because aecho truncates them; the trim by the
    pre-point moves the echo onto the incoming stream's timeline, and ``duration=first``
    keeps the incoming stream's length.
    """
    echo = EchoOutFilter(LOGGER, 1278018, 882, 22050, 22050, 0, 2, 44100)
    assert echo.apply("[fadein]", "[fadeout]") == [
        "[fadeout]asplit=2[fadeout_echo_dry][echo_send]",
        "[echo_send]highpass=f=300,"
        "afade=t=in:start_sample=1256850:nb_samples=441,"
        "afade=t=out:start_sample=1278415:nb_samples=441,"
        "aecho=0:1:500.011338|1000.011338:0.350000|0.175000,"
        "atrim=start_sample=1278018,asetpts=PTS-STARTPTS[echo_wet]",
        "[fadein][echo_wet]amix=inputs=2:normalize=0:duration=first[fadein_echo]",
    ]


def test_echo_out_filter_repeats_on_the_incoming_beat_from_its_one() -> None:
    """
    Into a faster track the repeats land on its beat; a one in the overlap takes the first.

    The outgoing beat (0.5 s) still lands on the incoming one, here 882 samples before the
    cut, and the rest follow 0.4 s apart; each repeat plays only 0.4 s of the outgoing beat,
    so it ends where the next starts.
    """
    echo = EchoOutFilter(LOGGER, 1278018, 882, 22050, 17640, 882, 3, 44100)
    assert echo.apply("[fadein]", "[fadeout]")[1] == (
        "[echo_send]highpass=f=300,"
        "afade=t=in:start_sample=1256850:nb_samples=441,"
        "afade=t=out:start_sample=1274049:nb_samples=441,"
        "aecho=0:1:480.011338|880.011338|1280.011338:0.350000|0.175000|0.087500,"
        "atrim=start_sample=1278018,asetpts=PTS-STARTPTS[echo_wet]"
    )
