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
    RequestedTransitionPlanner,
    TransitionRequest,
)
from music_assistant.controllers.streams.smart_fades.vocal import parse_vocal_probabilities
from tests.controllers.streams.smart_fades.conftest import _analysis_with_bands
from tests.controllers.streams.smart_fades.test_planner import _analysis, _vocal_probabilities

LOGGER = logging.getLogger(__name__)


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
