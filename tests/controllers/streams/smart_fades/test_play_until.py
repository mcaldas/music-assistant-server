"""Tests for planning a transition on an outgoing track that ends early (analysis_until)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from music_assistant.controllers.streams.smart_fades.bands import build_band_profile
from music_assistant.controllers.streams.smart_fades.helpers import analysis_until
from music_assistant.controllers.streams.smart_fades.planner import SmartCrossFadePlanner
from music_assistant.controllers.streams.smart_fades.planner.context import (
    build_transition_context,
)
from music_assistant.controllers.streams.smart_fades.planner.requested import (
    BLEND_STYLES,
    REQUEST_STYLES,
    RequestedTransitionPlanner,
    TransitionRequest,
)
from music_assistant.controllers.streams.smart_fades.vocal import parse_vocal_probabilities
from music_assistant.models.audio_analysis import AudioAnalysisData
from tests.controllers.streams.smart_fades.conftest import _analysis_with_bands
from tests.controllers.streams.smart_fades.test_planner import (
    _analysis,
    _vocal_probabilities,
    _with_vocal_activity,
)
from tests.controllers.streams.smart_fades.test_requested import _outro_from

LOGGER = logging.getLogger(__name__)
# every requested style, an 8-bar one where it takes bars
STYLES = [(style, 8 if style in BLEND_STYLES else 0) for style in REQUEST_STYLES]
# an end three quarters into a bar of a 120 BPM track (2 s bars, a downbeat at 120 s)
MID_BAR_END = 121.5
# a downbeat of a 128 BPM grid on the beat tracker's 20 ms frames: float32 reads it as
# 103.1200027, and its bar from 101.24 s runs a frame longer than the bpm's 1.875 s
FRAMED_DOWNBEAT = 103.12
# 128 vs 122 BPM drift apart within a bar: a quick fade between them is shorter than one
_REASON_128 = {"quick_fade": "shortened"}


def test_analysis_until_keeps_the_envelopes_usable() -> None:
    """Grids end at the end; band and vocal envelopes stay 1800 bins and still parse."""
    full = _analysis_with_bands(0.5, 0.4, 0.3, 0.2)
    assert full.extra_data is not None
    full.extra_data["vocal_activity"] = _vocal_probabilities(240.0, [(30.0, 60.0), (100.0, 110.0)])
    cut = analysis_until(full, 120.0)
    assert cut.duration == 120.0
    assert max(cut.beats or []) <= 120.0
    assert max(cut.downbeats or []) <= 120.0
    assert build_band_profile(cut) is not None
    timeline = parse_vocal_probabilities(cut)
    assert timeline is not None
    # the vocal at 100-110 s sits at the same seconds on the new bins
    first = next(
        i
        for i, p in enumerate(timeline.probabilities)
        if p >= 0.5 and i * timeline.frame_duration > 80
    )
    assert first * timeline.frame_duration == pytest.approx(100.0, abs=0.2)
    assert full.duration == 240.0  # the stored row is untouched
    assert analysis_until(full, 300.0) is full


@pytest.mark.parametrize("end", [75.3, 120.0, 180.0])
def test_both_planners_hold_the_tail_that_ends_at_the_end(end: float) -> None:
    """The plan sits in the 45 s before the end; a requested cut exits near its exit_at."""
    out = analysis_until(_analysis(120.0), end)
    # the planners place their exit in the held tail, which is the 45 s before the end
    tail_start = build_transition_context(out, _analysis(122.0), 45.0, LOGGER).buffer_offset
    assert tail_start == pytest.approx(end - 45.0)
    plan = SmartCrossFadePlanner(LOGGER).plan(out, _analysis(122.0), 45.0)
    assert tail_start + plan.fade_out_window <= end + 1e-6

    exit_at = end - 6.0
    requested = RequestedTransitionPlanner(LOGGER, TransitionRequest("cut", "n", 0, exit_at), 45.0)
    plan = requested.plan(out, _analysis(122.0), 45.0)
    assert requested.outcome == "applied"
    # a cut ends on the downbeat nearest exit_at (2 s bars at 120 BPM)
    assert tail_start + plan.fade_out_window == pytest.approx(exit_at, abs=1.0)


async def test_the_mixer_cuts_the_outgoing_analysis_only_when_told() -> None:
    """Without fade_out_end the planner reads the stored row itself; with it, the cut copy."""
    from music_assistant_models.enums import CrossfadeMode  # noqa: PLC0415

    from music_assistant.controllers.streams.smart_fades.fades import (  # noqa: PLC0415
        SmartCrossFade,
    )
    from tests.controllers.streams.test_smartfade_transition_timings import (  # noqa: PLC0415
        PCM,
        _make_mixer,
        _seconds,
        _streamdetails,
    )

    out, inc = _analysis(120.0), _analysis(122.0)
    mixer = _make_mixer({"out": out, "in": inc})
    kwargs: dict[str, Any] = {
        "fade_in_streamdetails": _streamdetails("in"),
        "fade_out_streamdetails": _streamdetails("out"),
        "pcm_format": PCM,
        "standard_crossfade_duration": 10,
        "mode": CrossfadeMode.SMART_CROSSFADE,
        "fade_out_data": b"\x00" * _seconds(45),
        "fade_in_bytes_len": _seconds(45),
    }
    plain = await mixer.build(**kwargs)
    unset = await mixer.build(**kwargs, fade_out_end=None)
    cut = await mixer.build(**kwargs, fade_out_end=150.0)
    assert isinstance(plain, SmartCrossFade)
    assert isinstance(unset, SmartCrossFade)
    assert isinstance(cut, SmartCrossFade)
    assert plain.fade_out_analysis is out
    assert unset.fade_out_analysis is out
    assert plain._get_ffmpeg_filters() == unset._get_ffmpeg_filters()
    assert plain.timing_info == unset.timing_info
    assert cut.fade_out_analysis.duration == 150.0
    assert out.duration == 240.0


def _requested(
    out: AudioAnalysisData, style: str, bars: int, exit_at: float, cut_at_end: bool = True
) -> tuple[RequestedTransitionPlanner, float]:
    """Plan a request on a 45 s tail; return the planner and the exit in song seconds."""
    planner = RequestedTransitionPlanner(
        LOGGER, TransitionRequest(style, "n", bars, exit_at), 45.0, cut_at_end=cut_at_end
    )
    plan = planner.plan(out, _analysis(122.0), 45.0)
    assert out.duration is not None
    return planner, out.duration - 45.0 + plan.fade_out_window


@pytest.mark.parametrize(("style", "bars"), STYLES)
def test_an_exit_at_an_end_past_the_last_downbeat_takes_it(style: str, bars: int) -> None:
    """An exit_at at a mid-bar end: the next downbeat is past the end, so A leaves at 120 s."""
    out = analysis_until(_analysis(120.0), MID_BAR_END)
    planner, exit_s = _requested(out, style, bars, MID_BAR_END)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert exit_s == pytest.approx(120.0)


def _framed(bpm: float) -> AudioAnalysisData:
    """Return a 240 s track whose beats sit on the beat tracker's 20 ms frames."""
    analysis = _analysis(bpm)
    assert analysis.beats is not None
    analysis.beats = [round(beat * 50) / 50 for beat in analysis.beats]
    analysis.downbeats = analysis.beats[::4]
    return analysis


@pytest.mark.parametrize(("style", "bars"), STYLES)
@pytest.mark.parametrize("exit_at", [FRAMED_DOWNBEAT, 0.0], ids=["exit_at=end", "no-exit_at"])
def test_an_end_on_a_downbeat_leaves_on_it(style: str, bars: int, exit_at: float) -> None:
    """An end on a downbeat, as audio_analysis/bar_grid reports it, is where A leaves."""
    out = analysis_until(_framed(128.0), FRAMED_DOWNBEAT)
    planner, exit_s = _requested(out, style, bars, exit_at)

    assert (planner.outcome, planner.reason) == ("applied", _REASON_128.get(style))
    assert exit_s == pytest.approx(FRAMED_DOWNBEAT)


@pytest.mark.parametrize(("style", "bars"), STYLES)
def test_an_end_just_before_a_downbeat_takes_the_one_before(style: str, bars: int) -> None:
    """3 ms before a downbeat closing a bar a frame longer than the bpm's: A leaves at 101.24 s."""
    end = FRAMED_DOWNBEAT - 0.003
    planner, exit_s = _requested(analysis_until(_framed(128.0), end), style, bars, end)

    assert (planner.outcome, planner.reason) == ("applied", _REASON_128.get(style))
    assert exit_s == pytest.approx(101.24)


@pytest.mark.parametrize(("style", "bars"), STYLES)
def test_a_phrase_sung_through_the_end_does_not_hold_the_exit_back(style: str, bars: int) -> None:
    """A vocal running through a mid-bar end is cut there whatever the exit: every style plays."""
    out = analysis_until(_with_vocal_activity(_analysis(120.0), [(60.0, 160.0)]), MID_BAR_END)
    for exit_at in (MID_BAR_END, 0.0):
        planner, exit_s = _requested(out, style, bars, exit_at)

        assert planner.outcome == "applied"
        assert 120.0 - 1e-6 <= exit_s <= MID_BAR_END + 1e-6


@pytest.mark.parametrize(("style", "bars"), STYLES)
def test_without_an_exit_a_phrase_sung_into_the_end_leaves_the_exit_to_the_others(
    style: str, bars: int
) -> None:
    """
    Smart Fades' exit (108 s) is inside a phrase; the next phrase runs into a mid-bar end.

    The end cuts that later phrase whatever the exit, so A leaves on the first downbeat
    after the earlier phrase instead of falling back.
    """
    out = _with_vocal_activity(
        _analysis(120.0, rms_energy=_outro_from(108.0)), [(100.0, 111.5), (113.0, 160.0)]
    )
    planner, exit_s = _requested(analysis_until(out, MID_BAR_END), style, bars, 0.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert exit_s == pytest.approx(112.0)


def test_a_phrase_that_ends_before_the_end_still_holds_the_exit_back() -> None:
    """A phrase ending after the exit but before the end, or at the song's own end, still counts."""
    before = analysis_until(_with_vocal_activity(_analysis(120.0), [(60.0, 120.9)]), MID_BAR_END)
    through = analysis_until(_with_vocal_activity(_analysis(120.0), [(60.0, 160.0)]), MID_BAR_END)
    for out, cut_at_end in ((before, True), (through, False)):
        planner, _ = _requested(out, "cut", 0, MID_BAR_END, cut_at_end)

        assert (planner.outcome, planner.reason) == ("fallback", "vocal")


async def test_the_mixer_tells_a_request_that_its_tail_is_cut_at_the_end() -> None:
    """Through the mixer, an end position spares the phrase it cuts, as the planner is told."""
    from music_assistant_models.enums import CrossfadeMode  # noqa: PLC0415

    from music_assistant.controllers.streams.smart_fades.fades import (  # noqa: PLC0415
        SmartCrossFade,
    )
    from tests.controllers.streams.test_smartfade_transition_timings import (  # noqa: PLC0415
        PCM,
        _make_mixer,
        _seconds,
        _streamdetails,
    )

    out = _with_vocal_activity(_analysis(120.0), [(60.0, 160.0)])
    mixer = _make_mixer({"out": out, "in": _analysis(122.0)})
    fade = await mixer.build(
        fade_in_streamdetails=_streamdetails("in"),
        fade_out_streamdetails=_streamdetails("out"),
        pcm_format=PCM,
        standard_crossfade_duration=10,
        mode=CrossfadeMode.SMART_CROSSFADE,
        fade_out_data=b"\x00" * _seconds(45),
        fade_in_bytes_len=_seconds(45),
        request=TransitionRequest("echo_out", "in", 0, MID_BAR_END),
        fade_out_end=MID_BAR_END,
    )
    assert isinstance(fade, SmartCrossFade)
    assert isinstance(fade.planner, RequestedTransitionPlanner)
    assert (fade.planner.outcome, fade.planner.reason) == ("applied", None)


@pytest.mark.parametrize("mode", ["standard_crossfade", "smart_crossfade"])
@pytest.mark.parametrize("end", [None, 150.0])
async def test_a_standard_fade_plays_a_tail_cut_at_its_end_to_that_end(
    mode: str, end: float | None
) -> None:
    """
    Rendered: a tail cut at its end position is heard up to it, faded into the next track.

    Without an end the fade still drops the tail's last 0.2 s as it always has (a smart
    fade without analysis falls back to the standard one).
    """
    import numpy as np  # noqa: PLC0415
    from music_assistant_models.enums import ContentType, CrossfadeMode  # noqa: PLC0415
    from music_assistant_models.media_items import AudioFormat  # noqa: PLC0415

    from music_assistant.controllers.streams.audio import END_POSITION_FADE  # noqa: PLC0415
    from music_assistant.controllers.streams.smart_fades.fades import (  # noqa: PLC0415
        StandardCrossFade,
    )
    from music_assistant.helpers.audio import fade_out_pcm  # noqa: PLC0415
    from tests.controllers.streams.test_smartfade_transition_timings import (  # noqa: PLC0415
        _make_mixer,
        _streamdetails,
    )

    pcm = AudioFormat(
        content_type=ContentType.PCM_F32LE, sample_rate=44100, bit_depth=32, channels=2
    )
    sr, frame = 44100, 8

    def tone(freq: float) -> bytes:
        # 6 s at full level, two channels; A and B are orthogonal over 20 ms windows
        mono = (0.5 * np.sin(2 * np.pi * freq * np.arange(6 * sr) / sr)).astype(np.float32)
        return np.repeat(mono, 2).tobytes()

    # A's tail as the reader hands it over at an end: full level up to a 20 ms fade-out
    tail = fade_out_pcm(tone(200.0), pcm, 6 * sr * frame, int(END_POSITION_FADE * sr) * frame)
    mixer = _make_mixer()
    fade = await mixer.build(
        fade_in_streamdetails=_streamdetails("in"),
        fade_out_streamdetails=_streamdetails("out"),
        pcm_format=pcm,
        standard_crossfade_duration=4,
        mode=CrossfadeMode(mode),
        fade_out_data=tail,
        fade_in_bytes_len=6 * sr * frame,
        fade_out_end=end,
    )
    assert isinstance(fade, StandardCrossFade)
    heard = 6.0 if end else 5.8  # where A's audio ends in the render
    timing = fade.timing_info
    assert timing.pre_crossfade_duration + timing.crossfade_duration == pytest.approx(
        heard, abs=0.001
    )
    mix = b"".join([chunk async for chunk in mixer.mix(fade, tone(400.0), tail, pcm)])
    out = np.frombuffer(mix, dtype=np.float32)[0::2]
    # A runs untouched up to the overlap, then fades out under B; B then plays whole
    assert len(out) == pytest.approx((heard + 2.0) * sr, abs=sr / 1000)
    assert mix[: int((heard - 4.0) * sr) * frame] == tail[: int((heard - 4.0) * sr) * frame]

    def a_level(at: float) -> float:
        window = out[int(at * sr) : int(at * sr) + 882]
        phase = np.exp(-2j * np.pi * 200.0 * np.arange(len(window)) / sr)
        return float(2 * abs(np.dot(window, phase)) / len(window))

    # the room hears A until its end: 100 ms before it still sounds, and nothing after it
    assert a_level(heard - 0.1) > 0.005
    assert a_level(heard + 0.01) < 0.001
