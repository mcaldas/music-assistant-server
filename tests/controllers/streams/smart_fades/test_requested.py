"""Tests for RequestedTransitionPlanner: the transition an API client asked for."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import numpy.typing as npt
import pytest

from music_assistant.controllers.streams.smart_fades.models import (
    EchoOut,
    TransitionPlan,
    TransitionTier,
)
from music_assistant.controllers.streams.smart_fades.planner import SmartCrossFadePlanner
from music_assistant.controllers.streams.smart_fades.planner.requested import (
    CUT_SECONDS,
    REQUEST_PREFIX,
    RequestedTransitionPlanner,
    TransitionRequest,
)
from music_assistant.models.audio_analysis import AudioAnalysisData
from tests.controllers.streams.smart_fades.test_assembly import _mastered_fade_pair
from tests.controllers.streams.smart_fades.test_planner import _analysis, _with_vocal_activity

LOGGER = logging.getLogger(__name__)
# where a 45 s held tail starts in the 240 s test tracks
TAIL_START = 240.0 - 45.0


def _plan(
    out: AudioAnalysisData,
    inc: AudioAnalysisData,
    style: str,
    bars: int = 0,
    exit_at: float = 0.0,
    fade_in_seconds: float = 45.0,
    buffer: float = 45.0,
    audible_from: float | None = None,
    head: npt.NDArray[np.bool_] | None = None,
) -> tuple[RequestedTransitionPlanner, TransitionPlan]:
    """
    Plan a requested transition over a held tail of ``buffer`` seconds.

    B's PCM head is ``head`` (its audible 10 ms windows), else 3 s audible from ``audible_from``.
    """
    if head is None and audible_from is not None:
        head = _head(audible_from)
    planner = RequestedTransitionPlanner(
        LOGGER, TransitionRequest(style, "next", bars, exit_at), fade_in_seconds, incoming_head=head
    )
    return planner, planner.plan(out, inc, buffer)


def _head(audible_from: float, silent: tuple[float, float] | None = None) -> npt.NDArray[np.bool_]:
    """Return 3 s of B's head as the mixer reads it: audible from ``audible_from``, except ``silent``."""
    head = np.arange(300) >= round(audible_from * 100)
    if silent is not None:
        head[round(silent[0] * 100) : round(silent[1] * 100)] = False
    return head


def _shifted(analysis: AudioAnalysisData, seconds: float) -> AudioAnalysisData:
    """Move a track's grid later, so its first downbeat lands at ``seconds``."""
    assert analysis.beats is not None
    assert analysis.downbeats is not None
    analysis.beats = [beat + seconds for beat in analysis.beats]
    analysis.downbeats = [downbeat + seconds for downbeat in analysis.downbeats]
    return analysis


def _grid_ending_at(analysis: AudioAnalysisData, seconds: float) -> AudioAnalysisData:
    """Drop the beats after ``seconds``, as for an outro the beat tracker hears no beat in."""
    assert analysis.beats is not None
    assert analysis.downbeats is not None
    analysis.beats = [beat for beat in analysis.beats if beat < seconds]
    analysis.downbeats = [downbeat for downbeat in analysis.downbeats if downbeat < seconds]
    return analysis


def _silent_after_its_one(level: float = 0.001) -> AudioAnalysisData:
    """Return a track whose one is 1 s in, its energy at ``level`` from 1.6 s to 2 s."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int(1.6 / 240.0 * 1800) : int(2.0 / 240.0 * 1800)] = level
    return _shifted(_analysis(120.0, rms_energy=rms), 1.0)


def _soft_intro(bars: int, level: float = 0.015) -> AudioAnalysisData:
    """Return a track whose one is 1 s in, its energy at ``level`` for ``bars`` bars after it."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[: int((1.0 + bars * 2.0) / 240.0 * 1800)] = level
    return _shifted(_analysis(120.0, rms_energy=rms), 1.0)


def _bars_of_outgoing(plan: TransitionPlan, bpm: float) -> float:
    """Return the overlap's length in bars of the outgoing track (4/4)."""
    ratio = plan.tempo_plan.steps[-1][1] if plan.tempo_plan else 1.0
    return plan.crossfade_duration * ratio / (4 * 60.0 / bpm)


def test_blend_plays_the_requested_bars_ending_on_the_exit() -> None:
    """A blend ends on the requested downbeat, with the requested bars, ramp and EQ."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "blend", bars=4, exit_at=230.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert TAIL_START + plan.fade_out_window == pytest.approx(230.0)
    assert _bars_of_outgoing(plan, 120.0) == pytest.approx(4.0, abs=0.01)
    assert plan.tempo_plan
    assert plan.eq_plan.low_out is not None


def test_blend_without_exit_ends_where_smart_fades_would() -> None:
    """Without an exit the blend ends at Smart Fades' own exit, here the end of the tail."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "blend", bars=8)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert TAIL_START + plan.fade_out_window == pytest.approx(240.0)
    assert _bars_of_outgoing(plan, 120.0) == pytest.approx(8.0, abs=0.01)


def test_a_blend_shorter_than_asked_says_so() -> None:
    """Sixteen bars are only earned by a near-instrumental pair; eight play, reported shortened."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "blend", bars=16)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert _bars_of_outgoing(plan, 120.0) == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize(
    ("incoming_bpm", "exit_at", "reason"),
    [(150.0, 0.0, "not_blendable"), (122.0, 205.0, "no_room")],
    ids=["tempo-gap", "too-few-bars-before-the-exit"],
)
def test_a_blend_smart_fades_will_not_beatmatch_ships_the_default_plan(
    incoming_bpm: float, exit_at: float, reason: str
) -> None:
    """A pair that won't blend, or an exit too early in the tail to blend into, falls back."""
    out, inc = _analysis(120.0), _analysis(incoming_bpm)
    planner, plan = _plan(out, inc, "blend", bars=8, exit_at=exit_at)

    assert (planner.outcome, planner.reason) == ("fallback", reason)
    assert plan == SmartCrossFadePlanner(LOGGER).plan(out, inc, 45.0)


@pytest.mark.parametrize(("exit_at", "bars"), [(228.0, 8), (222.0, 4), (215.0, 4)])
def test_a_blend_always_keeps_room_for_its_tempo_ramp(exit_at: float, bars: int) -> None:
    """An early exit shortens the blend rather than squeeze its tempo ramp."""
    planner, plan = _plan(_analysis(92.0), _analysis(97.0), "blend", bars=8, exit_at=exit_at)

    assert (planner.outcome, planner.reason) == ("applied", None if bars == 8 else "shortened")
    steps = plan.tempo_plan.steps
    assert steps[-1][0] - steps[0][0] >= 6.0
    assert _bars_of_outgoing(plan, 92.0) == pytest.approx(bars, abs=0.01)


@pytest.mark.parametrize("bars", [1, 2, 4])
@pytest.mark.parametrize("exit_at", [0.0, 237.0])
def test_a_blend_past_the_end_of_the_beat_grid_never_ships_short(bars: int, exit_at: float) -> None:
    """On an exit past A's last detected beat no blend ships shorter than its bars, nor empty."""
    out = _grid_ending_at(_analysis(120.0), 226.0)
    planner, plan = _plan(out, _analysis(124.0), "blend", bars=bars, exit_at=exit_at)

    if planner.outcome == "applied":
        assert plan.crossfade_duration > 0.0
        assert _bars_of_outgoing(plan, 120.0) >= bars - 0.5
        steps = plan.tempo_plan.steps
        assert steps[-1][0] - steps[0][0] >= 6.0
    else:
        assert planner.reason == "no_room"


def test_a_blend_stays_inside_the_head_of_the_incoming_track_the_mix_receives() -> None:
    """With 12 s of B in hand an 8-bar blend (15.7 s) shortens; with 1.5 s none fits."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "blend", bars=8, fade_in_seconds=12.0)
    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert (plan.fadein_trim_start or 0.0) + plan.crossfade_duration <= 12.0

    planner, _ = _plan(_analysis(120.0), _analysis(122.0), "blend", bars=8, fade_in_seconds=1.5)
    assert (planner.outcome, planner.reason) == ("fallback", "no_room")


def test_an_exit_with_no_downbeat_near_it_falls_back() -> None:
    """An exit before a short held tail is not moved bars away: the default plan ships."""
    planner, _ = _plan(_analysis(120.0), _analysis(120.0), "cut", exit_at=205.0, buffer=30.0)

    assert (planner.outcome, planner.reason) == ("fallback", "no_room")


def test_quick_fade_is_unramped_flat_and_short_enough_to_stay_in_time() -> None:
    """Between close tempos the unramped fade lasts a bar, so the kicks don't double."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "quick_fade", exit_at=224.3)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.tier is TransitionTier.QUICK_FADE
    assert not plan.tempo_plan
    assert plan.eq_plan.low_out is None
    assert plan.eq_plan.low_in is None
    assert TAIL_START + plan.fade_out_window == pytest.approx(224.0)
    assert plan.crossfade_duration == pytest.approx(2.0)


def test_quick_fade_between_equal_tempos_takes_smart_fades_length() -> None:
    """With nothing to drift apart, the quick fade lasts Smart Fades' own four bars."""
    _, plan = _plan(_analysis(120.0), _analysis(120.0), "quick_fade", exit_at=224.3)

    assert plan.crossfade_duration == pytest.approx(8.0)


def test_quick_fade_near_the_start_of_the_tail_shortens_and_never_cuts() -> None:
    """With one bar before the exit the fade lasts one bar; with none it falls back."""
    planner, plan = _plan(_analysis(120.0), _analysis(120.0), "quick_fade", exit_at=199.0)
    assert planner.outcome == "applied"
    assert plan.crossfade_duration == pytest.approx(2.0)

    planner, _ = _plan(_analysis(120.0), _analysis(120.0), "quick_fade", exit_at=196.5)
    assert (planner.outcome, planner.reason) == ("fallback", "no_room")


def test_quick_fade_keeps_a_long_intro_as_todays_quick_fade_does() -> None:
    """B's first downbeat 20 s in: the quick fade plays B from its head, skipping nothing."""
    planner, plan = _plan(
        _analysis(120.0), _shifted(_analysis(120.0), 20.0), "quick_fade", exit_at=224.0
    )

    assert planner.outcome == "applied"
    assert plan.fadein_trim_start is None


def _with_pickup(analysis: AudioAnalysisData, beats: int) -> AudioAnalysisData:
    """Start a track's bars ``beats`` beats into its grid, as a pickup before its first one."""
    assert analysis.beats is not None
    analysis.downbeats = analysis.beats[beats::4]
    return analysis


@pytest.mark.parametrize(
    ("incoming_bpm", "pickup", "overlap"),
    [
        # 100 vs 128 BPM drift 0.525 s a bar: 40 ms of it is 0.183 s
        (128.0, 0, CUT_SECONDS),
        (128.0, 2, 0.04 / 0.525 * 2.4),
        # 100 vs 104 BPM drift 0.092 s a bar: 40 ms of it is 1.04 s, more than a beat of pickup
        (104.0, 1, 60.0 / 104.0),
        (104.0, 3, 0.04 / (4 * (0.6 - 60.0 / 104.0)) * 2.4),
    ],
    ids=["far-no-pickup", "far-pickup", "near-short-pickup", "near-long-pickup"],
)
def test_a_quick_fade_that_would_drift_in_a_bar_meets_on_the_one_within_the_drift(
    incoming_bpm: float, pickup: int, overlap: float
) -> None:
    """Under a bar, B's pickup fades in under A's last beats and B's one lands on A's exit."""
    inc = _with_pickup(_shifted(_analysis(incoming_bpm), 0.1), pickup)
    assert inc.downbeats is not None
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert TAIL_START + plan.fade_out_window == pytest.approx(223.2)
    assert plan.crossfade_duration == pytest.approx(overlap)
    # A ends where B's one plays
    assert plan.fadein_trim_start == pytest.approx(inc.downbeats[0] - overlap)
    assert plan.fadeout_curve == "qsin"
    assert not plan.tempo_plan


def test_a_quick_fade_under_a_bar_never_fades_a_out_over_silence_of_b() -> None:
    """A pickup that falls silent before B's one is no fade-in: the switch is a cut's."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int(0.6 / 240.0 * 1800) : int(1.0 / 240.0 * 1800)] = 0.001
    inc = _with_pickup(_shifted(_analysis(128.0, rms_energy=rms), 0.1), 2)
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadein_trim_start == pytest.approx(0.1 + 2 * 60.0 / 128.0 - CUT_SECONDS)


def test_a_quick_fade_under_a_bar_hears_a_gap_anywhere_in_the_pickup_it_fades_over() -> None:
    """A gap late in a pickup longer than a landing's beats is still under A's fade: a cut's switch."""
    # 100 vs 103 BPM: the fade lasts 1.37 s, 2.4 of B's beats; B falls silent 0.3 s before its one
    one = 0.1 + 3 * 60.0 / 103.0
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int((one - 0.3) / 240.0 * 1800) : int((one - 0.1) / 240.0 * 1800)] = 0.001
    inc = _with_pickup(_shifted(_analysis(103.0, rms_energy=rms), 0.1), 3)
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadein_trim_start == pytest.approx(one - CUT_SECONDS)


def test_a_quick_fade_under_a_bar_onto_a_later_one_fades_in_the_bar_before_it() -> None:
    """B's first bar falls silent: the fade lands on its next one, over the end of that bar."""
    _, plan = _plan(_analysis(150.0), _silent_after_its_one(), "quick_fade", exit_at=224.0)

    # 150 vs 120 BPM drift 0.4 s a bar of 1.6 s: 40 ms of it is 0.16 s
    assert plan.crossfade_duration == pytest.approx(0.16)
    assert plan.fadein_trim_start == pytest.approx(3.0 - 0.16)


def test_cut_ends_the_outgoing_on_a_downbeat_and_lands_the_incoming_one_there() -> None:
    """A cut stops A on its downbeat; B plays from just before its first downbeat."""
    planner, plan = _plan(_analysis(120.0), _shifted(_analysis(120.0), 7.87), "cut", exit_at=224.3)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadeout_trim is not None
    assert plan.fadeout_trim.end_pos == plan.fade_out_window == pytest.approx(29.0)
    assert plan.fadein_trim_start == pytest.approx(7.87 - CUT_SECONDS)
    assert plan.fadeout_curve == "qsin"
    assert not plan.tempo_plan


def test_cut_lands_on_a_vocal_that_starts_on_the_one() -> None:
    """A vocal starting on B's one, as the vocal timeline rounds it, is no lead-in."""
    inc = _with_vocal_activity(_shifted(_analysis(120.0), 1.55), [(1.45, 30.0)])
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0)

    assert planner.outcome == "applied"
    assert plan.fadein_trim_start == pytest.approx(1.55 - CUT_SECONDS)


@pytest.mark.parametrize(
    ("incoming", "fade_in_seconds", "reason"),
    [
        (_with_vocal_activity(_shifted(_analysis(120.0), 3.0), [(0.5, 9.0)]), 45.0, "vocal"),
        (_shifted(_analysis(120.0), 7.87), 7.0, "no_room"),
        (_with_vocal_activity(_silent_after_its_one(), [(0.5, 9.0)]), 45.0, "vocal"),
        (_silent_after_its_one(), 2.5, "no_room"),
        (_soft_intro(8), 10.0, "no_room"),
    ],
    ids=[
        "sung-lead-in-longer-than-a-bar",
        "one-beyond-the-received-head",
        "sung-lead-in-to-the-downbeat-after-a-silent-bar-longer-than-a-bar",
        "downbeat-after-a-silent-bar-beyond-the-received-head",
        "downbeat-after-a-quiet-intro-beyond-the-received-head",
    ],
)
def test_a_cut_that_cannot_land_on_the_one_ships_the_default_plan(
    incoming: AudioAnalysisData, fade_in_seconds: float, reason: str
) -> None:
    """A cut would pre-roll more than a bar of B's lead-in, or play B's silence: it falls back."""
    planner, _ = _plan(
        _analysis(120.0), incoming, "cut", exit_at=225.0, fade_in_seconds=fade_in_seconds
    )

    assert (planner.outcome, planner.reason) == ("fallback", reason)


def _sung_pickup(silent_bins: int = 0) -> AudioAnalysisData:
    """Return a track sung from 0.32 s, its one at 1.1 s, silent for its first bins (Pepas)."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[:silent_bins] = 0.0
    return _with_vocal_activity(_shifted(_analysis(120.0, rms_energy=rms), 1.1), [(0.32, 30.0)])


# the 240 s test tracks' analysis bins: the timeline has B sung from its third, at 0.27 s
BIN = 240.0 / 1800


@pytest.mark.parametrize(("silent_bins", "entry"), [(0, 2 * BIN - 0.25), (1, BIN)])
def test_a_cut_into_a_sung_pickup_pre_rolls_it_under_the_outgoing_track(
    silent_bins: int, entry: float
) -> None:
    """B's pickup plays under A's last beats from a detector lag before it is sung, never silent."""
    planner, plan = _plan(_analysis(120.0), _sung_pickup(silent_bins), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert TAIL_START + plan.fade_out_window == pytest.approx(224.0)
    assert plan.fadein_trim_start == pytest.approx(entry, abs=1e-6)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(1.1)
    # A at full to its exit, B at full from the start of the overlap
    assert plan.fade_seconds == CUT_SECONDS


def test_a_cut_pre_rolls_a_pickup_to_the_downbeat_after_a_silent_bar() -> None:
    """B's one is followed by silence and B sings into its next one: the pickup to that one rolls."""
    inc = _with_vocal_activity(_silent_after_its_one(), [(2.2, 9.0)])
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=225.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(16 * BIN - 0.25)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("sung", "outcome", "reason"),
    [((221.0, 222.0), "applied", None), ((223.6, 224.0), "fallback", "vocal")],
    ids=["phrase-ends-before-the-pickup", "last-words-over-the-pickup"],
)
def test_a_pre_roll_never_sings_over_the_outgoing_vocal(
    sung: tuple[float, float], outcome: str, reason: str | None
) -> None:
    """Both decks play at full over a pre-roll: A's last words over B's pickup fall back."""
    out = _with_vocal_activity(_analysis(120.0), [sung])

    planner, plan = _plan(out, _sung_pickup(), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == (outcome, reason)
    if outcome == "applied":
        assert plan.metrics.collision_seconds == 0.0


def test_a_quick_fade_keeps_the_one_of_a_sung_pickup_on_the_downbeat() -> None:
    """The fade starts as B's pickup does, so B's one lands on A's downbeat four bars from the exit."""
    planner, plan = _plan(_analysis(120.0), _sung_pickup(), "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(2 * BIN - 0.25)
    assert plan.crossfade_duration == pytest.approx(8.0 + 1.1 - 2 * BIN + 0.25)
    assert plan.fade_seconds is None


@pytest.mark.parametrize(
    "incoming",
    [
        _with_vocal_activity(_shifted(_analysis(120.0), 3.0), [(0.5, 9.0)]),
        _with_vocal_activity(_with_pickup(_shifted(_analysis(120.0), 0.2), 5), [(0.27, 30.0)]),
    ],
    ids=["lead-in-before-the-grid", "lead-in-over-five-beats"],
)
def test_a_quick_fade_never_plays_a_lead_in_longer_than_a_bar_from_the_head(
    incoming: AudioAnalysisData,
) -> None:
    """From its head B's one would land the lead-in after A's downbeat, off A's beats: it falls back."""
    planner, plan = _plan(_analysis(120.0), incoming, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("fallback", "vocal")
    assert plan == SmartCrossFadePlanner(LOGGER).plan(_analysis(120.0), incoming, 45.0)


def test_a_quick_fade_under_a_bar_fades_over_a_sung_pickup_onto_the_one() -> None:
    """Between far tempos B's sung lead-in fades in under A's last beats, its one on A's exit."""
    inc = _with_vocal_activity(_shifted(_analysis(128.0), 1.1), [(0.32, 30.0)])
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0, audible_from=0.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert TAIL_START + plan.fade_out_window == pytest.approx(223.2)
    assert plan.fadein_trim_start == pytest.approx(2 * BIN - 0.25)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(1.1)
    # one equal-power fade over the whole pickup, as any quick fade
    assert plan.fade_seconds is None
    assert plan.fadeout_curve == "qsin"
    assert not plan.tempo_plan


@pytest.mark.parametrize(
    "head",
    [_head(0.0, (0.95, 1.1)), _head(0.0, (1.04, 1.1)), _head(0.0, (1.07, 1.1)), None],
    ids=["150-ms-breath", "60-ms-breath", "30-ms-breath", "no-head"],
)
def test_a_quick_fade_under_a_bar_holds_a_over_a_breath_only_the_head_shows(
    head: npt.NDArray[np.bool_] | None,
) -> None:
    """
    B breathes before its one in less than an analysis bin: only B's PCM head shows it.

    A fading out over it would leave the room near silence (-37 dB): A holds at full, as for
    a cut. Without B's head the breath cannot be ruled out, and A holds too.
    """
    inc = _with_vocal_activity(_shifted(_analysis(128.0), 1.1), [(0.32, 30.0)])
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0, head=head)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(1.1)
    assert plan.fade_seconds == CUT_SECONDS


def test_a_quick_fade_under_a_bar_onto_a_breath_in_its_lead_in_becomes_a_cut() -> None:
    """B's unsung beats before its one break for 60 ms just before it: A's fade would die there."""
    planner, plan = _plan(
        _analysis(100.0),
        _with_pickup(_shifted(_analysis(128.0), 0.1), 2),
        "quick_fade",
        exit_at=224.0,
        head=_head(0.1, (0.1 + 2 * 60.0 / 128.0 - 0.06, 0.1 + 2 * 60.0 / 128.0)),
    )

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadein_trim_start == pytest.approx(0.1 + 2 * 60.0 / 128.0 - CUT_SECONDS)


def test_a_quick_fade_under_a_bar_holds_a_over_a_break_in_a_sung_pickup() -> None:
    """B sings to 1.44 s and breaks from 1.47 s to near its one at 1.94 s: A plays on at full."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int(1.47 / 240.0 * 1800) : int(1.9 / 240.0 * 1800)] = 0.001
    inc = _with_vocal_activity(
        _shifted(_analysis(95.0, rms_energy=rms), 1.94), [(0.0, 1.44), (1.94, 30.0)]
    )
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert plan.fadein_trim_start is None
    assert plan.crossfade_duration == pytest.approx(1.94)
    # both decks at full between 20 ms fades, as a cut's pre-roll
    assert plan.fade_seconds == CUT_SECONDS


@pytest.mark.parametrize("style", ["cut", "quick_fade"])
@pytest.mark.parametrize(("incoming_bpm", "pickup"), [(128.0, 2), (104.0, 3)])
def test_a_sung_pickup_whose_beats_would_drift_lands_its_first_beat_on_the_exit(
    style: str, incoming_bpm: float, pickup: int
) -> None:
    """B's lead-in beats would drift over 40 ms off A's: B plays under A only to its first beat."""
    inc = _with_vocal_activity(
        _with_pickup(_shifted(_analysis(incoming_bpm), 0.1), pickup), [(0.13, 30.0)]
    )
    planner, plan = _plan(_analysis(100.0), inc, style, exit_at=224.0)

    assert (planner.outcome, planner.reason) == (
        "applied",
        None if style == "cut" else "shortened",
    )
    assert TAIL_START + plan.fade_out_window == pytest.approx(223.2)
    # from its start, its first beat on A's exit, both at full until then
    assert plan.fadein_trim_start is None
    assert plan.crossfade_duration == pytest.approx(0.1)
    assert plan.fade_seconds == CUT_SECONDS


def test_a_sung_pickup_whose_beats_keep_to_a_s_pre_rolls_onto_the_one() -> None:
    """At 120 vs 122 BPM B's two lead-in beats drift 8 and 16 ms: its one lands on A's exit."""
    inc = _with_vocal_activity(_with_pickup(_shifted(_analysis(122.0), 0.1), 2), [(0.13, 30.0)])
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start is None
    assert plan.crossfade_duration == pytest.approx(0.1 + 2 * 60.0 / 122.0)


@pytest.mark.parametrize("style", ["cut", "quick_fade"])
def test_a_sung_pickup_whose_beats_would_drift_never_plays_a_break_alone(style: str) -> None:
    """After its first beat B's lead-in breaks before its one: alone after A, the room would go quiet."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int(0.6 / 240.0 * 1800) : int(1.0 / 240.0 * 1800)] = 0.001
    inc = _with_vocal_activity(
        _with_pickup(_shifted(_analysis(128.0, rms_energy=rms), 0.1), 2), [(0.13, 30.0)]
    )
    planner, _ = _plan(_analysis(100.0), inc, style, exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("fallback", "vocal")


def _pepas(bin_one_db: float = -25.0) -> AudioAnalysisData:
    """
    Return Pepas' head: silent, sung from 0.32 s, its one at 1.1 s at 130 bpm.

    The beat grid runs back over the silence to 0.18 s and 0.64 s; the analysis bin after the
    silent first one reads ``bin_one_db`` under the track's level (the onset's smear).
    """
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[0] = 0.0
    rms[1] = 0.5 * 10 ** (bin_one_db / 20)
    inc = _with_pickup(_shifted(_analysis(130.0, rms_energy=rms), 0.18), 2)
    return _with_vocal_activity(inc, [(0.32, 30.0)])


@pytest.mark.parametrize("style", ["cut", "quick_fade"])
@pytest.mark.parametrize(
    ("bin_one_db", "audible_from", "entry"),
    [(-25.0, None, BIN), (0.0, 0.38, 0.38)],
    ids=["heard-in-the-analysis", "heard-in-the-pcm-head"],
)
def test_a_lead_in_lands_its_first_beat_b_is_heard_on(
    style: str, bin_one_db: float, audible_from: float | None, entry: float
) -> None:
    """
    Pepas' grid beat at 0.18 s is before B is heard: A's exit takes its next beat, 0.64 s.

    Landed on 0.18 s, the room heard 0.2 s of near silence between PROVENZA and Pepas.
    """
    planner, plan = _plan(
        _analysis(100.0), _pepas(bin_one_db), style, exit_at=224.0, audible_from=audible_from
    )

    assert planner.outcome == "applied"
    assert TAIL_START + plan.fade_out_window == pytest.approx(223.2)
    assert plan.fadein_trim_start == pytest.approx(entry)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(0.18 + 60.0 / 130.0)
    # both decks at full over the pickup until B's beat lands on A's exit
    assert plan.fade_seconds == CUT_SECONDS


@pytest.mark.parametrize(
    ("audible_from", "one"),
    [(None, 0.1), (0.11, 0.1), (0.3, 2.1)],
    ids=["no-head", "heard-on-the-one", "heard-after-the-one"],
)
def test_a_cut_never_lands_before_the_pcm_head_says_b_starts(
    audible_from: float | None, one: float
) -> None:
    """B's grid starts on 0.1 s and every bin reads loud; its audio starts 0.3 s in: B's next one."""
    inc = _shifted(_analysis(120.0), 0.1)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0, audible_from=audible_from)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(one - CUT_SECONDS)
    assert plan.crossfade_duration == CUT_SECONDS


@pytest.mark.parametrize("style", ["cut", "quick_fade"])
def test_a_quick_fade_under_a_bar_fades_only_over_b_heard(style: str) -> None:
    """At 100 vs 128 BPM the fade before B's one starts no earlier than B is heard (0.25 s)."""
    inc = _shifted(_analysis(128.0), 0.1)
    planner, plan = _plan(_analysis(100.0), inc, style, exit_at=224.0, audible_from=0.27)

    assert planner.outcome == "applied"
    # B's one at 0.1 s is before it is heard: A's exit lands B's next one, 1.975 s
    assert (plan.fadein_trim_start or 0.0) + plan.crossfade_duration == pytest.approx(
        0.1 + 4 * 60 / 128
    )
    assert plan.fadein_trim_start >= 0.25


def _quiet_first_bar() -> AudioAnalysisData:
    """Return q128's head: its first bar 11 dB under its level, its last beat near silent."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    # the one 1 s in at 120 bpm; the bar to 3 s, its last beat from 2.5 s
    rms[int(1.0 / BIN + 0.5) : int(3.0 / BIN + 0.5)] = 0.5 * 10 ** (-11 / 20)
    rms[int(2.5 / BIN + 0.5) : int(3.0 / BIN + 0.5)] = 0.5 * 10 ** (-44 / 20)
    return _shifted(_analysis(120.0, rms_energy=rms), 1.0)


@pytest.mark.parametrize("style", ["cut", "quick_fade"])
def test_a_switch_never_lands_on_a_quiet_bar_that_falls_silent(style: str) -> None:
    """B's quiet first bar goes near silent three beats in: B never started, so it is a gap."""
    planner, plan = _plan(_analysis(100.0), _quiet_first_bar(), style, exit_at=224.0)

    assert planner.outcome == "applied"
    assert (plan.fadein_trim_start or 0.0) + plan.crossfade_duration == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("level", "one"),
    [(0.001, 3.0), (0.02, 1.0), (0.5, 1.0)],
    ids=["silent", "quiet", "loud"],
)
def test_a_cut_lands_on_the_first_downbeat_whose_bar_sounds(level: float, one: float) -> None:
    """B falling silent in the bar after its one would leave the room in silence: B's next one."""
    planner, plan = _plan(_analysis(120.0), _silent_after_its_one(level), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadein_trim_start == pytest.approx(one - CUT_SECONDS)


@pytest.mark.parametrize(
    ("level", "one"),
    [(0.015, 17.0), (0.035, 17.0), (0.05, 1.0)],
    ids=["quiet", "23-db-under", "soft"],
)
def test_a_cut_lands_past_an_intro_too_quiet_to_hear_alone(level: float, one: float) -> None:
    """Eight bars of B 30 dB under the rest, never silent, are near silence once A stops."""
    planner, plan = _plan(_analysis(120.0), _soft_intro(8, level), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(one - CUT_SECONDS)


@pytest.mark.parametrize(
    ("bars", "outcome", "reason"),
    [
        (3, "applied", None),
        (4, "fallback", "no_room"),
        (8, "fallback", "no_room"),
    ],
    ids=["ends-under-the-fade", "ends-on-the-fade-s-end", "plays-on-after-the-fade"],
)
def test_a_quick_fade_never_leaves_a_quiet_intro_playing_alone(
    bars: int, outcome: str, reason: str | None
) -> None:
    """
    B's quiet intro may play under A's four-bar fade, not where A has faded out, nor after it.

    The room would hear A's fade die into B 30 dB under its level: the default plan ships.
    """
    planner, _ = _plan(_analysis(120.0), _soft_intro(bars), "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == (outcome, reason)


@pytest.mark.parametrize(
    ("head", "outcome"),
    [(None, "applied"), (_head(1.0), "applied"), (_head(1.0, (2.94, 3.0)), "fallback")],
    ids=["no-head", "sounds-through", "breath-before-its-next-one"],
)
def test_a_whole_bar_quick_fade_never_dies_into_a_breath_only_the_head_shows(
    head: npt.NDArray[np.bool_] | None, outcome: str
) -> None:
    """A one-bar fade ends on B's next one, 3 s in; a 60 ms breath before it is A's fade dying out."""
    planner, plan = _plan(
        _analysis(120.0), _shifted(_analysis(120.0), 1.0), "quick_fade", exit_at=198.0, head=head
    )

    assert planner.outcome == outcome
    if outcome == "applied":
        assert plan.fadein_trim_start == pytest.approx(1.0)
        assert plan.crossfade_duration == pytest.approx(2.0)


def _stops_late_in_a_bar(bar: float = 1.0) -> AudioAnalysisData:
    """Return a track whose one is 1 s in, silent for beats 2.5 to 3 of its bar from ``bar``."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int((bar + 1.25) / 240.0 * 1800) : int((bar + 1.5) / 240.0 * 1800)] = 0.001
    return _shifted(_analysis(120.0, rms_energy=rms), 1.0)


def test_a_cut_keeps_the_one_when_b_stops_later_in_the_bar() -> None:
    """B's own stop two and a half beats in is heard after B has started: the cut stays on the one."""
    planner, plan = _plan(_analysis(120.0), _stops_late_in_a_bar(), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(1.0 - CUT_SECONDS)


@pytest.mark.parametrize(
    ("silent", "head"),
    [((0.85, 1.25), None), ((0.94, 1.34), None), ((0.85, 1.25), _head(0.1, (0.85, 1.25)))],
    ids=["in-the-bins", "late-in-a-bin", "only-in-the-head"],
)
def test_a_cut_hears_a_stop_inside_the_landing_s_beats_as_a_gap(
    silent: tuple[float, float], head: npt.NDArray[np.bool_] | None
) -> None:
    """
    B stops for 0.4 s from 1.5 (or 1.68) beats after its one, inside the landing's 1.75: B's next one.

    The 133 ms bins blur it: the first fully silent one may start past 1.75 beats, or, with
    B's PCM head read, none at all.
    """
    rms = np.full(1800, 0.5, dtype=np.float32)
    if head is None:
        for i in range(int(silent[0] / BIN), int(silent[1] / BIN) + 1):
            hidden = min(silent[1], (i + 1) * BIN) - max(silent[0], i * BIN)
            rms[i] = 0.5 * np.sqrt(max(0.0, 1.0 - hidden / BIN))
    inc = _shifted(_analysis(120.0, rms_energy=rms), 0.1)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0, head=head)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(2.1 - CUT_SECONDS)


def test_a_quick_fade_plays_on_into_b_s_own_stop() -> None:
    """B's own stop late in the bar it plays alone once A has faded out: the quick fade ships."""
    planner, plan = _plan(_analysis(120.0), _stops_late_in_a_bar(9.0), "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert (plan.fadein_trim_start or 0.0) + plan.crossfade_duration == pytest.approx(9.0)


def test_a_cut_reads_silence_against_b_s_own_level() -> None:
    """A loud master's dip in B's first beat, 35.6 dB under its level but 36.5 dB under its peak bin, is a gap."""
    rms = np.full(1800, 0.9, dtype=np.float32)
    rms[0] = 1.0
    rms[int(1.2 / 240.0 * 1800) : int(1.4 / 240.0 * 1800)] = 0.015
    inc = _shifted(_analysis(120.0, rms_energy=rms), 1.0)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(3.0 - CUT_SECONDS)


def test_a_cut_reads_the_silence_before_the_one_as_before_it() -> None:
    """A silent bin centred before B's one, as B starts late in it, keeps the cut on the one."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[: int(1.1 / 240.0 * 1800)] = 0.001
    inc = _shifted(_analysis(120.0, rms_energy=rms), 1.05)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0, audible_from=1.06)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(1.05 - CUT_SECONDS)


@pytest.mark.parametrize("onset", [0.2, 0.38, 0.58, 1.0])
def test_without_its_head_a_cut_keeps_clear_of_where_a_silent_b_may_start(onset: float) -> None:
    """
    B is silent to ``onset`` and its grid sits 40 ms early; its PCM head was not read.

    B may start anywhere in its first loud analysis bin, or in the 100 ms energy window past it
    (0.58 s reads loud in the bin ending 0.533 s): the cut lands no earlier, on B's next one,
    never 40 ms of silence before B.
    """
    # the analysis' energy: 100 ms windows (their share of B's audio), averaged into the bins
    windows = np.clip((np.arange(2400) + 1) * 0.1 - onset, 0.0, 0.1) / 0.1 * 0.25
    power = np.concatenate(([0.0], np.cumsum(windows)))
    edges = np.interp(np.linspace(0, 2400, 1801), np.arange(2401), power)
    rms = np.sqrt(np.diff(edges) / (2400 / 1800)).astype(np.float32)
    inc = _shifted(_analysis(120.0, rms_energy=rms), onset - 0.04)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(onset - 0.04 + 2.0)


@pytest.mark.parametrize(
    ("style", "incoming_bpm"),
    [("cut", 120.0), ("quick_fade", 150.0)],
    ids=["cut", "quick-fade-under-a-bar"],
)
def test_cut_inside_a_mastered_fade_keeps_its_fade_curve(style: str, incoming_bpm: float) -> None:
    """A cut inside the record's own fade-out still fades its few milliseconds, never stops dead."""
    out, _ = _mastered_fade_pair()
    planner, plan = _plan(out, _analysis(incoming_bpm), style, exit_at=239.0)

    assert planner.outcome == "applied"
    assert plan.crossfade_duration == CUT_SECONDS
    assert plan.fadeout_curve == "qsin"


def test_a_request_that_cuts_the_outgoing_vocal_ships_the_default_plan() -> None:
    """A cut inside A's sung phrase falls back; after the phrase it is applied."""
    out = _with_vocal_activity(_analysis(120.0), [(220.0, 230.0)])
    inc = _analysis(120.0)

    planner, plan = _plan(out, inc, "cut", exit_at=225.0)
    assert (planner.outcome, planner.reason) == ("fallback", "vocal")
    assert plan == SmartCrossFadePlanner(LOGGER).plan(out, inc, 45.0)

    planner, plan = _plan(out, inc, "cut", exit_at=231.0)
    assert planner.outcome == "applied"
    assert TAIL_START + plan.fade_out_window == pytest.approx(230.0)


def _outro_from(seconds: float) -> np.ndarray:
    """Return a 240 s track's energy, dropping at ``seconds``: Smart Fades' own exit is there."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[0] = 1.0
    rms[int(seconds / 240.0 * 1800) :] = 0.1
    return rms


@pytest.mark.parametrize(("style", "bars"), [("cut", 0), ("blend", 4)])
def test_without_an_exit_a_sung_phrase_moves_the_exit_past_it(style: str, bars: int) -> None:
    """Smart Fades' exit (228 s) is inside A's last phrase: the first downbeat after it is used."""
    out = _with_vocal_activity(_analysis(120.0, rms_energy=_outro_from(228.0)), [(215.0, 231.5)])

    planner, plan = _plan(out, _analysis(120.0), style, bars=bars)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert TAIL_START + plan.fade_out_window == pytest.approx(232.0)


def test_without_an_exit_a_phrase_sung_to_the_end_still_falls_back() -> None:
    """Sung past its last downbeat (239 s), A has no exit that keeps the phrase whole."""
    out = _with_vocal_activity(
        _shifted(_analysis(120.0, rms_energy=_outro_from(228.0)), 1.0), [(215.0, 239.9)]
    )

    planner, _ = _plan(out, _analysis(120.0), "cut")

    assert (planner.outcome, planner.reason) == ("fallback", "vocal")


@pytest.mark.parametrize("adlib", [(226.0, 230.0), (234.0, 237.0)])
@pytest.mark.parametrize("style", ["cut", "quick_fade"])
def test_without_an_exit_a_late_ad_lib_does_not_hold_the_outro(
    style: str, adlib: tuple[float, float]
) -> None:
    """Smart Fades leaves at 214 s; keeping an ad-lib 6+ bars into the quiet outro would not."""
    out = _with_vocal_activity(
        _analysis(120.0, rms_energy=_outro_from(215.0)), [(200.0, 213.0), adlib]
    )

    planner, plan = _plan(out, _analysis(122.0), style)

    assert (planner.outcome, planner.reason) == ("fallback", "vocal")
    assert plan == SmartCrossFadePlanner(LOGGER).plan(out, _analysis(122.0), 45.0)


def test_without_an_exit_a_blend_never_walks_back_from_smart_fades_exit() -> None:
    """Past the end of A's beat grid (226 s) no blend fits; an exit at 224 s would leave A early."""
    out, inc = _grid_ending_at(_analysis(120.0), 226.0), _analysis(124.0)

    planner, plan = _plan(out, inc, "blend", bars=4)

    assert (planner.outcome, planner.reason) == ("fallback", "no_room")
    assert plan == SmartCrossFadePlanner(LOGGER).plan(out, inc, 45.0)


@pytest.mark.parametrize(("style", "bars"), [("cut", 0), ("blend", 4)])
def test_a_named_exit_between_phrases_leaves_the_later_phrase_out(style: str, bars: int) -> None:
    """An exit the client names in a gap is applied; A's phrase after it is not played."""
    out = _with_vocal_activity(_analysis(120.0), [(200.0, 213.5), (222.0, 230.0)])

    planner, plan = _plan(out, _analysis(120.0), style, bars=bars, exit_at=216.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert TAIL_START + plan.fade_out_window == pytest.approx(216.0)
    if style == "blend":
        assert plan.tier is not TransitionTier.QUICK_FADE


@pytest.mark.parametrize(
    ("style", "bars", "exit_at", "incoming_bpm"),
    [
        ("blend", 4, 230.0, 122.0),
        ("quick_fade", 0, 224.0, 122.0),
        ("cut", 0, 224.0, 122.0),
        ("filter_sweep", 4, 230.0, 122.0),
        ("echo_out", 0, 224.0, 122.0),
        ("blend", 8, 0.0, 150.0),
    ],
    ids=["blend", "quick_fade", "cut", "filter_sweep", "echo_out", "fallback"],
)
def test_every_path_masks_the_outgoing_grid_to_its_exit(
    style: str, bars: int, exit_at: float, incoming_bpm: float
) -> None:
    """Like the default planner, each path leaves the outgoing grid masked to the plan's exit."""
    planner, plan = _plan(_analysis(120.0), _analysis(incoming_bpm), style, bars, exit_at)

    assert planner.outgoing.beats.max() <= plan.fade_out_window


def test_request_round_trips_through_extra_attributes() -> None:
    """A request survives the trip through a queue item's extra attributes."""
    request = TransitionRequest("blend", "next", 8, 230.0)
    attributes = dict(request.to_attributes())
    attributes["playback_speed"] = 1.0
    assert {key for key in attributes if key.startswith(REQUEST_PREFIX)} == {
        f"{REQUEST_PREFIX}{name}" for name in ("style", "next_item_id", "bars", "exit_at")
    }

    assert TransitionRequest.read(attributes) == request
    assert TransitionRequest.read(attributes, "next") == request
    assert TransitionRequest.read(attributes, "other") is None
    # taken for planning, the boundary names its next item from then on
    assert TransitionRequest.take(attributes, "next") == request
    assert attributes["transition_next_item_id"] == "next"
    assert TransitionRequest.drop(attributes) == request
    assert attributes == {"playback_speed": 1.0, "transition_next_item_id": "next"}
    assert TransitionRequest.drop(attributes) is None
    assert TransitionRequest.read({f"{REQUEST_PREFIX}style": "bogus"}) is None


@pytest.mark.parametrize(("bars", "exit_at"), [(8, 230.0), (16, 0.0)])
def test_filter_sweep_is_the_blend_handed_over_by_filters(bars: int, exit_at: float) -> None:
    """A sweep keeps the blend's exit, length and ramp; filters replace its EQ."""
    out, inc = _analysis(120.0), _analysis(122.0)
    blend_planner, blend = _plan(out, inc, "blend", bars=bars, exit_at=exit_at)
    planner, plan = _plan(out, inc, "filter_sweep", bars=bars, exit_at=exit_at)

    # sixteen bars are held to Smart Fades' cap of eight, as for the blend
    expected = ("applied", None if bars == 8 else "shortened")
    assert (planner.outcome, planner.reason) == (blend_planner.outcome, blend_planner.reason)
    assert (planner.outcome, planner.reason) == expected
    assert (plan.fade_out_window, plan.crossfade_duration, plan.tempo_plan) == (
        blend.fade_out_window,
        blend.crossfade_duration,
        blend.tempo_plan,
    )
    assert plan.eq_plan.low_out is None
    assert plan.eq_plan.low_in is None
    assert plan.eq_plan.mid_out is None
    assert plan.sweep_out is not None
    assert plan.sweep_in is not None
    # A: 10 Hz from the start, rising over exactly the overlap (in A-input time) to 8 kHz
    ratio = plan.tempo_plan.steps[-1][1]
    start = plan.fade_out_window - plan.crossfade_duration * ratio
    assert plan.sweep_out.steps[0] == (0.0, 10.0)
    assert plan.sweep_out.steps[1] == pytest.approx((start, 10.0))
    assert plan.sweep_out.steps[-1] == pytest.approx((plan.fade_out_window, 8000.0))
    # ...but dry until the overlap, its high-pass fading in over the overlap's first second
    assert plan.sweep_out.mix_steps[0] == (0.0, 0.0)
    assert plan.sweep_out.mix_steps[1] == pytest.approx((start, 0.0))
    assert plan.sweep_out.mix_steps[-1] == pytest.approx((start + 1.0, 1.0))
    # B: opens from 250 Hz by 3/4 of the overlap, dry by 9/10 of it
    assert plan.sweep_in.steps[0] == (0.0, 250.0)
    assert plan.sweep_in.steps[-1][1] == pytest.approx(8000.0)
    assert plan.sweep_in.mix_steps[0][1] == 1.0
    assert plan.sweep_in.mix_steps[-1] == pytest.approx((0.9 * plan.crossfade_duration, 0.0))
    assert _bars_of_outgoing(plan, 120.0) == pytest.approx(8.0, abs=0.01)


def test_filter_sweep_where_smart_fades_will_not_beatmatch_ships_the_default_plan() -> None:
    """A sweep falls back like the blend it is built on."""
    out, inc = _analysis(120.0), _analysis(150.0)
    planner, plan = _plan(out, inc, "filter_sweep", bars=8)

    assert (planner.outcome, planner.reason) == ("fallback", "not_blendable")
    assert plan == SmartCrossFadePlanner(LOGGER).plan(out, inc, 45.0)


@pytest.mark.parametrize("incoming_bpm", [120.0, 150.0], ids=["same-tempo", "tempo-jump"])
def test_echo_out_is_the_cut_with_its_last_beat_ringing_on(incoming_bpm: float) -> None:
    """
    An echo out keeps the cut's exit and entry; A's beat repeats for two of B's bars.

    The repeats ring under B, so they keep to B's beat: across a tempo jump all of them ring.
    """
    out, inc = _analysis(120.0), _shifted(_analysis(incoming_bpm), 7.87)
    _, cut = _plan(out, inc, "cut", exit_at=224.3)
    planner, plan = _plan(out, inc, "echo_out", exit_at=224.3)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.echo_out == EchoOut(beat=0.5, period=60.0 / incoming_bpm, repeats=8)
    assert plan.sweep_out is None
    assert (
        plan.fade_out_window,
        plan.crossfade_duration,
        plan.fadein_trim_start,
        plan.fadeout_curve,
    ) == (cut.fade_out_window, CUT_SECONDS, cut.fadein_trim_start, "qsin")


@pytest.mark.parametrize(
    ("fade_in_seconds", "outcome"),
    [(12.0, ("applied", None)), (9.0, ("fallback", "no_room"))],
    ids=["two-bars-fit", "under-a-bar-fits"],
)
def test_echo_out_needs_a_bar_of_the_incoming_head_for_its_tail(
    fade_in_seconds: float, outcome: tuple[str, str | None]
) -> None:
    """The echo must fit the incoming audio the mix receives, at least for a bar."""
    out, inc = _analysis(120.0), _shifted(_analysis(120.0), 7.87)
    planner, plan = _plan(out, inc, "echo_out", exit_at=224.3, fade_in_seconds=fade_in_seconds)

    assert (planner.outcome, planner.reason) == outcome
    if outcome[0] == "fallback":
        assert plan == SmartCrossFadePlanner(LOGGER).plan(out, inc, 45.0)


@pytest.mark.parametrize(
    ("outgoing_vocal", "incoming_vocal", "outcome"),
    [
        ((225.0, 229.9), (0.5, 12.0), ("fallback", "vocal")),
        ((225.0, 229.9), (30.0, 40.0), ("applied", None)),
        ((215.0, 222.0), (0.5, 12.0), ("applied", None)),
    ],
    ids=["sung-beat-over-sung-head", "incoming-sings-later", "last-beat-unsung"],
)
def test_echo_out_never_echoes_a_sung_beat_over_the_incoming_vocal(
    outgoing_vocal: tuple[float, float],
    incoming_vocal: tuple[float, float],
    outcome: tuple[str, str | None],
) -> None:
    """A cut there is fine; its echo would repeat A's last word over B's vocal."""
    out = _with_vocal_activity(_analysis(120.0), [outgoing_vocal])
    inc = _with_vocal_activity(_analysis(120.0), [incoming_vocal])
    cut_planner, _ = _plan(out, inc, "cut", exit_at=230.0)
    planner, _ = _plan(out, inc, "echo_out", exit_at=230.0)

    assert cut_planner.outcome == "applied"
    assert (planner.outcome, planner.reason) == outcome


@pytest.mark.parametrize(
    ("grid", "bpm"), [(0.52, 120.0), (0.5, 123.0)], ids=["slower-than-its-bpm", "bpm-off-the-grid"]
)
def test_echo_out_repeats_the_outgoing_tracks_real_last_beat(grid: float, bpm: float) -> None:
    """
    A grid running at another tempo than its bpm says echoes its own beat, not the nominal one.

    Only the first repeat is that beat, landed on B's one: the rest keep to B's beat, all
    two bars of them, however far A's beat is from it.
    """
    out = _analysis(bpm)
    assert out.beats is not None
    out.beats = [index * grid for index in range(len(out.beats))]
    out.downbeats = out.beats[::4]
    planner, plan = _plan(out, _analysis(120.0), "echo_out", exit_at=224.0)

    assert planner.outcome == "applied"
    assert plan.echo_out is not None
    assert plan.echo_out.beat == pytest.approx(grid, abs=1e-3)
    assert (plan.echo_out.period, plan.echo_out.repeats) == (0.5, 8)


@pytest.mark.parametrize("bpm", [95.0, 128.0, 174.0])
def test_echo_out_reads_a_steady_tempo_through_a_grid_on_analysis_frames(bpm: float) -> None:
    """
    A grid on the analysis' 20 ms frames reads the bpm's tempo, wherever the exit falls.

    Its beats are a frame long or short; the echo keeps the bpm's beat over them, so its
    repeats never drift off B's.
    """
    out, inc = _analysis(bpm), _shifted(_analysis(bpm), 7.87)
    for analysis in (out, inc):
        assert analysis.beats is not None
        analysis.beats = [round(beat * 50) / 50 for beat in analysis.beats]
        analysis.downbeats = analysis.beats[::4]
    for exit_at in np.arange(224.0, 232.0, 0.5):
        planner, plan = _plan(out, inc, "echo_out", exit_at=float(exit_at))

        assert planner.outcome == "applied"
        assert plan.echo_out == EchoOut(beat=60.0 / bpm, period=60.0 / bpm, repeats=8)


@pytest.mark.parametrize(
    ("one", "lead"),
    [(0.0, CUT_SECONDS), (0.01, 0.01), (CUT_SECONDS, 0.0), (0.3, 0.0)],
    ids=["one-on-the-first-sample", "one-inside-the-cut", "one-at-the-cut", "one-trimmed-to"],
)
def test_echo_out_starts_on_an_incoming_one_inside_the_cut(one: float, lead: float) -> None:
    """
    B cannot start before its first sample, so a one in its first 20 ms plays before A ends.

    The echo then starts on B's one too, that much before A's end.
    """
    planner, plan = _plan(
        _analysis(120.0), _shifted(_analysis(120.0), one), "echo_out", exit_at=224.0
    )

    assert planner.outcome == "applied"
    assert plan.echo_out is not None
    assert plan.echo_out.lead == pytest.approx(lead, abs=1e-6)


def test_a_one_bar_filter_sweep_steps_finely_enough_not_to_zipper() -> None:
    """The sweep's commands come about 150 to an overlap: 10-20 ms apart on a one-bar sweep."""
    planner, plan = _plan(_analysis(120.0), _analysis(122.0), "filter_sweep", bars=1)

    assert planner.outcome == "applied"
    assert plan.sweep_in is not None
    times = [t for t, _ in plan.sweep_in.steps]
    assert max(b - a for a, b in itertools.pairwise(times)) <= 0.02
