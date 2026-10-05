"""Tests for RequestedTransitionPlanner: the transition an API client asked for."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from music_assistant.controllers.streams.smart_fades.models import TransitionPlan, TransitionTier
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
) -> tuple[RequestedTransitionPlanner, TransitionPlan]:
    """Plan a requested transition over a held tail of ``buffer`` seconds."""
    planner = RequestedTransitionPlanner(
        LOGGER, TransitionRequest(style, "next", bars, exit_at), fade_in_seconds
    )
    return planner, planner.plan(out, inc, buffer)


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


@pytest.mark.parametrize(
    ("incoming", "entry", "lead"),
    [
        (_sung_pickup(), 2 * BIN - 0.25, 1.1 - 2 * BIN + 0.25),
        (_with_vocal_activity(_shifted(_analysis(120.0), 3.0), [(0.5, 9.0)]), None, 0.0),
    ],
    ids=["pickup-under-the-fade", "lead-in-longer-than-a-bar-plays-from-the-head"],
)
def test_a_quick_fade_keeps_the_one_of_a_sung_pickup_on_the_downbeat(
    incoming: AudioAnalysisData, entry: float | None, lead: float
) -> None:
    """The fade starts as B's pickup does, so B's one lands on A's downbeat four bars from the exit."""
    planner, plan = _plan(_analysis(120.0), incoming, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == (pytest.approx(entry) if entry else None)
    assert plan.crossfade_duration == pytest.approx(8.0 + lead)
    assert plan.fade_seconds is None


def test_a_quick_fade_under_a_bar_fades_over_a_sung_pickup_onto_the_one() -> None:
    """Between far tempos B's sung lead-in fades in under A's last beats, its one on A's exit."""
    inc = _with_vocal_activity(_shifted(_analysis(128.0), 1.1), [(0.32, 30.0)])
    planner, plan = _plan(_analysis(100.0), inc, "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", "shortened")
    assert TAIL_START + plan.fade_out_window == pytest.approx(223.2)
    assert plan.fadein_trim_start == pytest.approx(2 * BIN - 0.25)
    assert plan.fadein_trim_start + plan.crossfade_duration == pytest.approx(1.1)
    # one equal-power fade over the whole pickup, as any quick fade
    assert plan.fade_seconds is None
    assert plan.fadeout_curve == "qsin"
    assert not plan.tempo_plan


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


@pytest.mark.parametrize(("level", "one"), [(0.015, 17.0), (0.05, 1.0)], ids=["quiet", "soft"])
def test_a_cut_lands_past_an_intro_too_quiet_to_hear_alone(level: float, one: float) -> None:
    """Eight bars of B 30 dB under the rest, never silent, are near silence once A stops."""
    planner, plan = _plan(_analysis(120.0), _soft_intro(8, level), "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(one - CUT_SECONDS)


@pytest.mark.parametrize(
    ("bars", "outcome", "reason"),
    [(4, "applied", None), (8, "fallback", "no_room")],
    ids=["ends-under-the-fade", "plays-on-after-the-fade"],
)
def test_a_quick_fade_never_leaves_a_quiet_intro_playing_alone(
    bars: int, outcome: str, reason: str | None
) -> None:
    """B's quiet intro may play under A's four-bar fade, not on after it: the default plan ships."""
    planner, _ = _plan(_analysis(120.0), _soft_intro(bars), "quick_fade", exit_at=224.0)

    assert (planner.outcome, planner.reason) == (outcome, reason)


def test_a_cut_reads_the_silence_before_the_one_as_before_it() -> None:
    """A silent bin centred before B's one, as B starts late in it, keeps the cut on the one."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[: int(1.1 / 240.0 * 1800)] = 0.001
    inc = _shifted(_analysis(120.0, rms_energy=rms), 1.05)
    planner, plan = _plan(_analysis(120.0), inc, "cut", exit_at=224.0)

    assert (planner.outcome, planner.reason) == ("applied", None)
    assert plan.fadein_trim_start == pytest.approx(1.05 - CUT_SECONDS)


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
        ("blend", 8, 0.0, 150.0),
    ],
    ids=["blend", "quick_fade", "cut", "fallback"],
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
