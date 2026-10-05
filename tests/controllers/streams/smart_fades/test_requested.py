"""Tests for RequestedTransitionPlanner: the transition an API client asked for."""

from __future__ import annotations

import logging

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
    assert steps[-1][0] - steps[0][0] >= 8.0
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
        assert steps[-1][0] - steps[0][0] >= 8.0
    else:
        assert planner.reason == "no_room"


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
        (_with_vocal_activity(_shifted(_analysis(120.0), 3.0), [(1.2, 9.0)]), 45.0, "vocal"),
        (_shifted(_analysis(120.0), 7.87), 7.0, "no_room"),
    ],
    ids=["sung-lead-in-before-the-one", "one-beyond-the-received-head"],
)
def test_a_cut_that_cannot_land_on_the_one_ships_the_default_plan(
    incoming: AudioAnalysisData, fade_in_seconds: float, reason: str
) -> None:
    """A cut would behead B's lead-in, or play B's silence after A stops: it falls back."""
    planner, _ = _plan(
        _analysis(120.0), incoming, "cut", exit_at=225.0, fade_in_seconds=fade_in_seconds
    )

    assert (planner.outcome, planner.reason) == ("fallback", reason)


def test_cut_inside_a_mastered_fade_keeps_its_fade_curve() -> None:
    """A cut inside the record's own fade-out still fades its few milliseconds, never stops dead."""
    out, inc = _mastered_fade_pair()
    planner, plan = _plan(out, inc, "cut", exit_at=239.0)

    assert planner.outcome == "applied"
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
