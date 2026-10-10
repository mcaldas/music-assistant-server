"""Tests for the smart-fades transition plan value objects."""

from __future__ import annotations

import pytest

from music_assistant.controllers.streams.smart_fades.models import (
    EqPlan,
    TempoPlan,
    TransitionPlan,
    TransitionTier,
)


class TestTempoPlan:
    """Cover the TempoPlan stretch-savings integral and truthiness."""

    def test_empty_plan_is_falsy_and_saves_nothing(self) -> None:
        """An empty ramp stretches no time."""
        plan = TempoPlan()
        assert not plan
        assert plan.savings_until(45.0) == 0.0

    def test_non_empty_plan_is_truthy(self) -> None:
        """A plan with steps is truthy so renderers gate on it directly."""
        assert TempoPlan(steps=[(0.0, 1.05)])

    def test_speed_up_saves_positive_time(self) -> None:
        """A ratio > 1 (faster) removes time from the rendered stream."""
        plan = TempoPlan(steps=[(35.0, 1.0), (40.0, 1.05)])
        # ratio-1.0 segment [35,40] saves nothing; [40,45] runs at 1.05
        assert plan.savings_until(45.0) == pytest.approx(5.0 * (1.0 - 1.0 / 1.05))
        assert plan.savings_until(40.0) == 0.0

    def test_slow_down_lengthens_stream(self) -> None:
        """A ratio < 1 (slower) yields negative savings (stream lengthened)."""
        plan = TempoPlan(steps=[(35.0, 1.0), (40.0, 0.95)])
        assert plan.savings_until(45.0) == pytest.approx(5.0 * (1.0 - 1.0 / 0.95))

    def test_first_step_after_zero_stretches_from_start(self) -> None:
        """Rubberband starts at the first step's ratio, so the pre-step span is stretched."""
        plan = TempoPlan(steps=[(20.0, 1.004)])
        assert plan.savings_until(10.0) == pytest.approx(10.0 * (1.0 - 1.0 / 1.004))
        assert plan.savings_until(45.0) == pytest.approx(45.0 * (1.0 - 1.0 / 1.004))

    def test_ramp_points_start_where_the_stretch_starts(self) -> None:
        """The points run from the first stretched second to the rendered second asked for."""
        plan = TempoPlan(steps=[(2.0, 1.0), (4.0, 2.0)])
        # normal speed up to 4 s, then two input seconds for every rendered one
        assert plan.ramp_points(5.0) == [(4.0, 4.0), (6.0, 5.0)]

    def test_ramp_points_keep_every_step_ahead_of_the_end(self) -> None:
        """Each change of ratio ahead of the end is a point."""
        plan = TempoPlan(steps=[(2.0, 1.0), (4.0, 2.0), (6.0, 0.5)])
        # [4, 6] at 2.0 renders in one second; from 6 on, two rendered seconds a second
        assert plan.ramp_points(7.0) == [(4.0, 4.0), (6.0, 5.0), (7.0, 7.0)]

    def test_ramp_points_of_a_single_late_step_start_at_zero(self) -> None:
        """Rubberband starts at the first step's ratio, so the stretch starts with the tail."""
        plan = TempoPlan(steps=[(3.0, 0.5)])
        assert plan.ramp_points(4.0) == [(0.0, 0.0), (2.0, 4.0)]

    def test_ramp_points_agree_with_the_savings(self) -> None:
        """A point's rendered second is its input second less what the stretch saved by then."""
        plan = TempoPlan(steps=[(30.0, 1.0), (32.5, 1.007), (35.0, 1.014), (37.5, 1.0209)])
        points = plan.ramp_points(40.0)
        assert [p[0] for p in points[:-1]] == [32.5, 35.0, 37.5]
        assert points[-1][1] == 40.0
        for at, rendered in points:
            assert rendered == pytest.approx(at - plan.savings_until(at))

    def test_a_step_on_the_end_is_not_a_point_of_its_own(self) -> None:
        """A step a millisecond ahead of the end would round onto it: the end stands for both."""
        plan = TempoPlan(steps=[(2.0, 1.0), (4.0, 2.0), (6.0, 0.5)])
        points = plan.ramp_points(5.001)
        assert len(points) == 2
        assert points[0] == (4.0, 4.0)
        assert points[1] == pytest.approx((6.002, 5.001))

    def test_no_ramp_points_without_a_stretch_ahead_of_the_end(self) -> None:
        """Nothing is reported when the stream runs at normal speed up to the end asked for."""
        assert TempoPlan().ramp_points(10.0) == []
        assert TempoPlan(steps=[(2.0, 1.0), (4.0, 2.0)]).ramp_points(3.0) == []
        assert TempoPlan(steps=[(2.0, 1.0), (4.0, 2.0)]).ramp_points(0.0) == []


def test_transition_plan_defaults_to_neutral_eq() -> None:
    """TransitionPlan can be created without eq_plan and defaults to neutral."""
    plan = TransitionPlan(
        tier=TransitionTier.QUICK_FADE, fade_out_window=10.0, crossfade_duration=5.0
    )
    assert plan.eq_plan.low_out is None
    assert plan.eq_plan.mid_out is None


def test_eq_plan_neutral_factory() -> None:
    """EqPlan.neutral() factory creates a plan with all schedules None."""
    eq = EqPlan.neutral(swap_at=2.5)
    assert eq.swap_at == 2.5
    assert all(
        s is None for s in (eq.low_out, eq.low_in, eq.high_out, eq.high_in, eq.mid_out, eq.mid_in)
    )
