"""Tests for planning a transition into a track that starts later (analysis_from)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pytest
from music_assistant_models.enums import CrossfadeMode

from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.controllers.streams.smart_fades.fades import SmartCrossFade
from music_assistant.controllers.streams.smart_fades.helpers import (
    analysis_from,
    audible_start,
)
from music_assistant.controllers.streams.smart_fades.planner.requested import (
    RequestedTransitionPlanner,
    TransitionRequest,
)
from music_assistant.controllers.streams.smart_fades.vocal import parse_vocal_probabilities
from tests.controllers.streams.smart_fades.conftest import _analysis_with_bands
from tests.controllers.streams.smart_fades.test_planner import _analysis, _vocal_probabilities
from tests.controllers.streams.test_smartfade_transition_timings import (
    PCM,
    _make_mixer,
    _seconds,
    _streamdetails,
)

LOGGER = logging.getLogger(__name__)
# a part of a 240 s, 120 BPM track (2 s bars, a downbeat every 2 s) from its 60 s downbeat
START = 60.0


def test_analysis_from_shifts_the_grids_and_rebins_the_envelopes() -> None:
    """Grids move by the start; band and vocal envelopes stay 1800 bins on the same seconds."""
    full = _analysis_with_bands(0.5, 0.4, 0.3, 0.2)
    assert full.extra_data is not None
    full.extra_data["vocal_activity"] = _vocal_probabilities(240.0, [(100.0, 110.0)])
    full.rms_energy = [i / 1800 for i in range(1800)]
    part = analysis_from(full, START)

    assert part.duration == 180.0
    assert part.downbeats is not None
    assert part.beats is not None
    assert part.downbeats[0] == 0.0
    assert part.downbeats[1] == pytest.approx(2.0)
    assert len(part.beats) == len([b for b in full.beats or [] if b >= START])
    assert part.rms_energy is not None
    assert len(part.rms_energy) == 1800
    # each new bin takes the source bin under its centre: 60 s plus a bin of 0.1 s, halved
    assert part.rms_energy[0] == pytest.approx(int((START + 0.05) / 240 * 1800) / 1800)
    assert all(len(env) == 1800 for env in part.extra_data["band_rms"].values())  # type: ignore[index]
    timeline = parse_vocal_probabilities(part)
    assert timeline is not None
    sung = next(i for i, p in enumerate(timeline.probabilities) if p >= 0.5)
    # the vocal at 100 s now sits 40 s into the part
    assert sung * timeline.frame_duration == pytest.approx(40.0, abs=0.2)
    assert full.duration == 240.0  # the stored row is untouched


@pytest.mark.parametrize("offset", [0.01, 0.0])
def test_a_downbeat_a_frame_before_the_start_stays_on_it(offset: float) -> None:
    """A start a frame after a downbeat (the grid's rounding) keeps that downbeat at 0."""
    part = analysis_from(_analysis(120.0), START + offset)
    assert part.downbeats is not None
    assert part.downbeats[0] == 0.0
    assert part.downbeats[1] == pytest.approx(2.0 - offset)


def test_a_start_mid_bar_drops_the_bar_it_left() -> None:
    """Half a bar in, the first downbeat is the next bar's."""
    part = analysis_from(_analysis(120.0), START + 1.0)
    assert part.downbeats is not None
    assert part.downbeats[0] == pytest.approx(1.0)


@pytest.mark.parametrize("start", [0.0, -1.0, 240.0, 300.0])
def test_analysis_from_outside_the_track_changes_nothing(start: float) -> None:
    """At or before 0, or at or past the end, the analysis is the stored one."""
    full = _analysis(120.0)
    assert analysis_from(full, start) is full


def _build_kwargs() -> dict[str, Any]:
    return {
        "fade_in_streamdetails": _streamdetails("in"),
        "fade_out_streamdetails": _streamdetails("out"),
        "pcm_format": PCM,
        "standard_crossfade_duration": 10,
        "mode": CrossfadeMode.SMART_CROSSFADE,
        "fade_out_data": b"\x00" * _seconds(45),
        "fade_in_bytes_len": _seconds(45),
    }


async def test_the_mixer_starts_the_incoming_analysis_only_when_told() -> None:
    """Without a start the planner reads the stored row; with one, the part's copy."""
    out, inc = _analysis(120.0), _analysis(122.0)
    mixer = _make_mixer({"out": out, "in": inc})
    plain = await mixer.build(**_build_kwargs())
    started = await mixer.build(**_build_kwargs(), fade_in_start=START)
    assert isinstance(plain, SmartCrossFade)
    assert isinstance(started, SmartCrossFade)
    assert plain.fade_in_analysis is inc
    assert started.fade_in_analysis.duration == 240.0 - START
    assert inc.duration == 240.0


async def test_a_request_into_a_started_item_has_its_entry_fixed() -> None:
    """The requested planner is told the client chose the entry only when there is a start."""
    mixer = _make_mixer({"out": _analysis(120.0), "in": _analysis(122.0)})
    request = TransitionRequest("blend", "in", 8)
    plain = await mixer.build(**_build_kwargs(), request=request)
    started = await mixer.build(**_build_kwargs(), request=request, fade_in_start=START)
    assert isinstance(plain, SmartCrossFade)
    assert isinstance(started, SmartCrossFade)
    assert isinstance(plain.planner, RequestedTransitionPlanner)
    assert isinstance(started.planner, RequestedTransitionPlanner)
    assert not plain.planner.fixed_entry
    assert started.planner.fixed_entry


def _part_with_a_quiet_lead_in() -> Any:
    """Return a part from 60 s whose kick comes in 4 bars (8 s) in, after a quiet lead-in."""
    rms = np.full(1800, 0.5, dtype=np.float32)
    rms[int(START / 240 * 1800) : int((START + 8.0) / 240 * 1800)] = 0.12
    return analysis_from(_analysis(120.0, rms_energy=rms), START)


@pytest.mark.parametrize(("fixed", "entry"), [(False, 4.0), (True, 0.0)])
def test_a_fixed_entry_keeps_the_lead_in(fixed: bool, entry: float) -> None:
    """A 2-bar blend would enter the part 2 bars in, toward its kick; fixed, at its start."""
    planner = RequestedTransitionPlanner(
        LOGGER, TransitionRequest("blend", "n", 2), 45.0, fixed_entry=fixed
    )
    plan = planner.plan(_analysis(122.0), _part_with_a_quiet_lead_in(), 45.0)
    assert planner.outcome == "applied"
    assert (plan.fadein_trim_start or 0.0) == pytest.approx(entry)


def test_a_fixed_entry_is_the_first_downbeat_at_or_after_the_start() -> None:
    """A start half a bar before a downbeat enters on that downbeat."""
    part = analysis_from(_analysis(120.0), START - 1.0)
    planner = RequestedTransitionPlanner(
        LOGGER, TransitionRequest("blend", "n", 4), 45.0, fixed_entry=True
    )
    plan = planner.plan(_analysis(122.0), part, 45.0)
    assert planner.outcome == "applied"
    assert plan.fadein_trim_start == pytest.approx(1.0)


async def test_the_incoming_head_is_read_from_the_start() -> None:
    """The audible head the planner gets begins at the start, not at the track's beginning."""
    mixer = _make_mixer()
    frame = PCM.bit_depth // 8 * PCM.channels
    loud = (8000).to_bytes(2, "little", signed=True) * PCM.channels

    async def _source() -> Any:
        # silent up to 12.5 s, then sounding
        for sec in range(20):
            sounding = 0 if sec < 12 else PCM.sample_rate // (2 if sec == 12 else 1)
            yield b"\x00" * (PCM.sample_rate - sounding) * frame + loud * sounding
            await asyncio.sleep(0)

    audio_buffer = AudioBuffer(PCM)
    audio_buffer.fill(_source(), source_name="in")
    while not audio_buffer.eof:
        await asyncio.sleep(0.01)
    details = _streamdetails("in")
    details.buffer = audio_buffer
    from_start = await mixer._audible_head(details)
    from_part = await mixer._audible_head(details, 12.25)
    assert from_start is not None
    assert from_part is not None
    assert audible_start(from_start) == pytest.approx(3.0)  # nothing in its first 3 s
    assert audible_start(from_part) == pytest.approx(0.25)
    # 3 s from the start's second, less the quarter before the start
    assert len(from_part) == 275
    await audio_buffer.clear()
