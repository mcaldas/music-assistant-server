"""Tests for TransitionRenderer — plan to filter-chain + timing."""

from __future__ import annotations

import logging

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.smart_fades.filters import (
    EchoOutFilter,
    FadeInTrimFilter,
    FadeOutTrimFilter,
    GradualTimeStretchFilter,
    PeakFilter,
    ShelfFilter,
    ShelfType,
    StreamingCrossfadeFilter,
    SweepFilter,
)
from music_assistant.controllers.streams.smart_fades.models import (
    EchoOut,
    EqPlan,
    FadeOutTrim,
    ShelfSchedule,
    SweepSchedule,
    TempoPlan,
    TransitionPlan,
    TransitionTier,
)
from music_assistant.controllers.streams.smart_fades.renderer import TransitionRenderer

LOGGER = logging.getLogger(__name__)
PCM = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, bit_depth=16, channels=2)


def _seconds(seconds: float) -> int:
    return int(seconds * PCM.pcm_sample_size)


def _eq_plan() -> EqPlan:
    return EqPlan(
        swap_at=6.0,
        low_out=ShelfSchedule(ShelfType.LOW, 100, [(0.0, 0.0), (36.0, -26.0)]),
        low_in=ShelfSchedule(ShelfType.LOW, 100, [(0.0, -26.0), (7.0, 0.0)]),
        high_out=ShelfSchedule(ShelfType.HIGH, 13000, [(0.0, 0.0), (38.0, -20.0)]),
        high_in=ShelfSchedule(ShelfType.HIGH, 13000, [(0.0, -20.0), (5.0, 0.0)]),
    )


def _plan(**overrides: object) -> TransitionPlan:
    defaults: dict[str, object] = {
        "tier": TransitionTier.FULL_BLEND,
        "fade_out_window": 40.0,
        "crossfade_duration": 10.0,
        "eq_plan": _eq_plan(),
    }
    defaults.update(overrides)
    return TransitionPlan(**defaults)  # type: ignore[arg-type]


class TestTransitionRenderer:
    """The renderer assembles the filter chain and timing from a plan."""

    def test_full_chain_order(self) -> None:
        """Every optional stage present renders in the fixed chain order."""
        plan = _plan(
            tempo_plan=TempoPlan(steps=[(30.0, 1.0), (35.0, 1.02)]),
            fadeout_trim=FadeOutTrim(end_pos=40.0, trimmed_seconds=5.0),
            fadein_trim_start=1.0,
        )
        filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
        assert [type(f) for f in filters] == [
            FadeOutTrimFilter,
            ShelfFilter,  # A low (pre-stretch: schedules stay in musical input time)
            ShelfFilter,  # A high
            GradualTimeStretchFilter,
            FadeInTrimFilter,
            ShelfFilter,  # B low
            ShelfFilter,  # B high
            StreamingCrossfadeFilter,
        ]

    def test_minimal_chain_is_shelves_then_crossfade(self) -> None:
        """With no trims and no stretch, only the four shelves and crossfade render."""
        filters, _ = TransitionRenderer(LOGGER).render(_plan(), PCM, _seconds(45))
        assert [type(f) for f in filters] == [
            ShelfFilter,
            ShelfFilter,
            ShelfFilter,
            ShelfFilter,
            StreamingCrossfadeFilter,
        ]

    def test_bypassed_low_shelves_are_skipped(self) -> None:
        """A bypassed (None) low shelf on either side emits no ShelfFilter for it."""
        eq_plan = _eq_plan()
        eq_plan.low_out = None
        eq_plan.low_in = None
        filters, _ = TransitionRenderer(LOGGER).render(_plan(eq_plan=eq_plan), PCM, _seconds(45))
        assert [type(f) for f in filters] == [
            ShelfFilter,  # A high
            ShelfFilter,  # B high
            StreamingCrossfadeFilter,
        ]

    def test_all_shelves_bypassed_leaves_only_crossfade(self) -> None:
        """When every shelf is bypassed the chain is a pure crossfade."""
        eq_plan = _eq_plan()
        eq_plan.low_out = None
        eq_plan.low_in = None
        eq_plan.high_out = None
        eq_plan.high_in = None
        filters, _ = TransitionRenderer(LOGGER).render(_plan(eq_plan=eq_plan), PCM, _seconds(45))
        assert [type(f) for f in filters] == [StreamingCrossfadeFilter]

    def test_timing_accounts_for_both_tracks(self) -> None:
        """PRE+CF spans the fade-out, TRIM+CF+POST spans the fade-in."""
        plan = _plan(fadein_trim_start=1.0)
        _, timing = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
        # no stretch -> fade_out_seconds == fade_out_window
        assert timing.pre_crossfade_duration + timing.crossfade_duration == pytest.approx(40.0)
        assert (
            timing.fadein_trimmed_duration
            + timing.crossfade_duration
            + timing.post_crossfade_duration
            == pytest.approx(45.0)
        )
        assert timing.fadein_trimmed_duration == pytest.approx(1.0)

    def test_timing_clamps_crossfade_to_short_fadein(self) -> None:
        """A short incoming buffer clamps the crossfade so POST never goes negative."""
        plan = _plan(crossfade_duration=10.0, fadein_trim_start=1.0)
        _, timing = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(5))
        assert timing.crossfade_duration == pytest.approx(4.0)
        assert timing.post_crossfade_duration == 0.0

    def test_short_fadein_clamps_acrossfade_overlap(self) -> None:
        """
        Regression: the acrossfade overlap must never exceed the audio it is fed.

        acrossfade silently emits nothing (a hard cut) when its requested overlap
        exceeds the incoming buffer, so the filter must use the same clamped,
        frame-exact overlap as the timing info.
        """
        plan = _plan(crossfade_duration=10.0, fadein_trim_start=1.0)
        filters, timing = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(5))
        crossfade = filters[-1]
        assert isinstance(crossfade, StreamingCrossfadeFilter)
        # 5s buffer minus the 1s trim leaves 4s of incoming audio
        assert crossfade.crossfade_samples == 4 * PCM.sample_rate
        assert timing.crossfade_duration == pytest.approx(4.0)

    def test_full_buffers_render_plan_duration_as_samples(self) -> None:
        """With full buffers the filter carries the plan duration, frame-exact."""
        filters, timing = TransitionRenderer(LOGGER).render(_plan(), PCM, _seconds(45))
        crossfade = filters[-1]
        assert isinstance(crossfade, StreamingCrossfadeFilter)
        assert crossfade.crossfade_samples == 10 * PCM.sample_rate
        assert timing.crossfade_duration == pytest.approx(10.0)

    def test_fadeout_curve_flows_into_the_crossfade_filter(self) -> None:
        """The plan's fadeout_curve becomes the outgoing stream's fade curve."""
        plan = _plan(fadeout_curve="nofade")
        filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
        crossfade = filters[-1]
        assert isinstance(crossfade, StreamingCrossfadeFilter)
        assert "curve=nofade" in crossfade.apply("[fadein]", "[fadeout]")[0]

    @pytest.mark.parametrize(
        ("fade_seconds", "samples"), [(None, 441000), (0.02, 882), (20.0, 441000)]
    )
    def test_fade_seconds_flow_into_the_crossfade_filter(
        self, fade_seconds: float | None, samples: int
    ) -> None:
        """The plan's fade length becomes both streams' fade, never longer than the overlap."""
        plan = _plan(fade_seconds=fade_seconds)
        filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
        crossfade = filters[-1]
        assert isinstance(crossfade, StreamingCrossfadeFilter)
        fadeout, fadein, _ = crossfade.apply("[fadein]", "[fadeout]")
        assert f"start_sample={30 * 44100 + 441000 - samples}:nb_samples={samples}:" in fadeout
        assert f"start_sample=0:nb_samples={samples}:" in fadein

    def test_stretch_savings_shorten_fadeout_accounting(self) -> None:
        """A speed-up ramp removes time from the rendered fade-out total."""
        plan = _plan(tempo_plan=TempoPlan(steps=[(30.0, 1.0), (35.0, 1.02)]))
        _, timing = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45))
        expected_fade_out = 40.0 - 5.0 * (1.0 - 1.0 / 1.02)
        assert timing.pre_crossfade_duration + timing.crossfade_duration == pytest.approx(
            expected_fade_out
        )


class TestMidSwapRendering:
    """A PEAK ShelfSchedule (mid swap) renders as a PeakFilter; None is skipped."""

    def test_peak_schedule_renders_as_peak_filter(self) -> None:
        """Populated mid_out/mid_in schedules render as PeakFilter instances."""
        eq_plan = _eq_plan()
        eq_plan.mid_out = ShelfSchedule(ShelfType.PEAK, 1200, [(0.0, 0.0), (36.0, -8.0)])
        eq_plan.mid_in = ShelfSchedule(ShelfType.PEAK, 1200, [(0.0, -8.0), (7.0, 0.0)])
        filters, _ = TransitionRenderer(LOGGER).render(_plan(eq_plan=eq_plan), PCM, _seconds(45))
        assert [type(f) for f in filters] == [
            ShelfFilter,  # A low
            ShelfFilter,  # A high
            PeakFilter,  # A mid
            ShelfFilter,  # B low
            ShelfFilter,  # B high
            PeakFilter,  # B mid
            StreamingCrossfadeFilter,
        ]

    def test_mid_none_is_skipped(self) -> None:
        """A bypassed (None) mid schedule emits no PeakFilter."""
        filters, _ = TransitionRenderer(LOGGER).render(_plan(), PCM, _seconds(45))
        assert not any(isinstance(f, PeakFilter) for f in filters)


def test_sweeps_wrap_the_stretch_like_the_shelves() -> None:
    """A's sweep runs before the stretch (input time), B's after B's trim and shelves."""
    plan = _plan(
        eq_plan=EqPlan.neutral(),
        tempo_plan=TempoPlan(steps=[(0.0, 1.0), (20.0, 1.02)]),
        fadeout_trim=FadeOutTrim(40.0, 5.0),
        fadein_trim_start=1.5,
        sweep_out=SweepSchedule([(0.0, 20.0), (29.8, 20.0), (40.0, 8000.0)]),
        sweep_in=SweepSchedule([(0.0, 250.0), (7.5, 8000.0)], [(7.5, 1.0), (9.0, 0.0)]),
    )
    filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45.0))

    assert [type(f) for f in filters] == [
        FadeOutTrimFilter,
        SweepFilter,
        GradualTimeStretchFilter,
        FadeInTrimFilter,
        SweepFilter,
        StreamingCrossfadeFilter,
    ]
    sweep_out, sweep_in = filters[1], filters[4]
    assert isinstance(sweep_out, SweepFilter)
    assert isinstance(sweep_in, SweepFilter)
    assert (sweep_out.kind, sweep_out.stream_type) == ("highpass", "fadeout")
    assert (sweep_in.kind, sweep_in.stream_type, sweep_in.mix_steps) == (
        "lowpass",
        "fadein",
        [(7.5, 1.0), (9.0, 0.0)],
    )


def test_echo_out_renders_on_the_crossfade_timeline() -> None:
    """The echo sits right before the crossfade, sized by its pre-point and overlap, in samples."""
    plan = _plan(
        tier=TransitionTier.QUICK_FADE,
        eq_plan=EqPlan.neutral(),
        fade_out_window=29.0,
        crossfade_duration=0.02,
        fadeout_trim=FadeOutTrim(29.0, 16.0),
        echo_out=EchoOut(beat=0.5, period=0.4, repeats=8, lead=0.01),
    )
    filters, _ = TransitionRenderer(LOGGER).render(plan, PCM, _seconds(45.0))

    assert [type(f) for f in filters] == [
        FadeOutTrimFilter,
        EchoOutFilter,
        StreamingCrossfadeFilter,
    ]
    echo, crossfade = filters[1], filters[2]
    assert isinstance(echo, EchoOutFilter)
    assert isinstance(crossfade, StreamingCrossfadeFilter)
    assert (echo.pre_crossfade_samples, echo.crossfade_samples) == (
        crossfade.pre_crossfade_samples,
        crossfade.crossfade_samples,
    )
    assert (
        echo.beat_samples,
        echo.period_samples,
        echo.lead_samples,
        echo.repeats,
        echo.sample_rate,
    ) == (22050, 17640, 441, 8, 44100)
