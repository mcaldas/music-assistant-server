"""
Smart Fades - the candidate/policy transition planner.

``SmartCrossFadePlanner.plan()`` is a thin orchestration of the pipeline:
build the immutable ``TransitionContext``, let the generators propose
candidate specs, build each into a timed candidate, score them all with the
rejection/penalty policies, finalize the winner's EQ - or, when every
candidate is rejected, retry with late-anchored rescue candidates (the
ungated audible-end ladder plus a modest rescue rung), then ship a plain
equal-power fallback crossfade - or, when even that collides too severely,
the click-free emergency handoff as a last resort. Alternative strategies
slot in as sibling ``TransitionPlanner`` subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.controllers.streams.smart_fades.models import SmartFadeNotApplicable

from .assembly import EmergencyHandoffFactory, FallbackCrossfadeFactory, PlanAssembler
from .candidates import (
    CandidateFactory,
    CandidateSpec,
    RescueAnchorGenerator,
    TrimClosingAnchorGenerator,
    default_generators,
)
from .context import TransitionContext, build_transition_context
from .policies import default_policies
from .selection import CandidateSelector

if TYPE_CHECKING:
    import logging

    from music_assistant.controllers.streams.smart_fades.models import Deck, TransitionPlan
    from music_assistant.models.audio_analysis import AudioAnalysisData


class TransitionPlanner(ABC):
    """Abstract base class for transition planners."""

    # set by plan(): the outgoing grid masked to the plan's exit
    outgoing: Deck

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the planner."""
        self.logger = logger

    @abstractmethod
    def plan(
        self,
        fade_out_analysis: AudioAnalysisData,
        fade_in_analysis: AudioAnalysisData,
        buffer_duration: float,
    ) -> TransitionPlan:
        """
        Build a ``TransitionPlan`` from the two tracks' analysis data.

        Pure over the analysis rows and the available holdback window — touches
        no audio bytes.  Raises ``SmartFadeNotApplicable`` when the tracks cannot
        yield this transition and the caller should fall back.

        :param fade_out_analysis: Analysis data for the outgoing track.
        :param fade_in_analysis: Analysis data for the incoming track.
        :param buffer_duration: Length in seconds of the available fade-out holdback.
        """


class SmartCrossFadePlanner(TransitionPlanner):
    """Plans a defensive, musically-aligned crossfade that never edits the music."""

    def __init__(self, logger: logging.Logger, fixed_entry: bool = False) -> None:
        """
        Initialize the planner.

        :param logger: Logger for debug output.
        :param fixed_entry: Whether the client chose where the incoming audio starts
            (``set_start_position``): every candidate then enters it at its first downbeat,
            never deeper in.
        """
        super().__init__(logger)
        self.fixed_entry = fixed_entry

    def plan(
        self,
        fade_out_analysis: AudioAnalysisData,
        fade_in_analysis: AudioAnalysisData,
        buffer_duration: float,
    ) -> TransitionPlan:
        """
        Build a smart-crossfade ``TransitionPlan`` from the two tracks' analysis.

        Vocal-aware protections engage per deck: each track with a validated
        FireRed vocal-activity timeline gets its vocals protected, while a
        track without one is planned on energy facts alone.

        :param fade_out_analysis: Analysis data for the outgoing track.
        :param fade_in_analysis: Analysis data for the incoming track.
        :param buffer_duration: Length in seconds of the available fade-out holdback.
        """
        ctx = build_transition_context(
            fade_out_analysis, fade_in_analysis, buffer_duration, self.logger
        )
        factory = CandidateFactory(ctx, self.logger)
        specs = self._entered(
            ctx, [spec for generator in default_generators() for spec in generator.generate(ctx)]
        )
        candidates = [candidate for spec in specs if (candidate := factory.build(spec)) is not None]
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            self.logger.log(
                VERBOSE_LOG_LEVEL,
                "generated %d specs (%s), %d built",
                len(specs),
                dict(Counter(spec.source for spec in specs)),
                len(candidates),
            )
        if not candidates:
            raise SmartFadeNotApplicable("no feasible transition candidate")
        selector = CandidateSelector(default_policies(), self.logger)
        winner = selector.select(candidates, ctx)
        if winner is None:
            # every phrased candidate breached a hard rejection: retry with the
            # ungated audible-end ladder plus a modest late-anchored rescue rung
            # before falling back to the handoff
            rescue_specs = self._entered(
                ctx,
                [
                    *TrimClosingAnchorGenerator(min_gap=0.0).generate(ctx),
                    *RescueAnchorGenerator().generate(ctx),
                ],
            )
            rescue_candidates = [
                candidate for spec in rescue_specs if (candidate := factory.build(spec)) is not None
            ]
            winner = selector.select(rescue_candidates, ctx) if rescue_candidates else None
            if winner is not None:
                self.logger.debug(
                    "shipping a rescue-pass candidate (source=%s) instead of the emergency handoff",
                    winner.candidate.spec.source,
                )
        if winner is None:
            # a plain volume crossfade reads far less abrupt than the click-free
            # handoff, so it ships unless its vocal collision is too severe
            fallback = FallbackCrossfadeFactory(ctx, factory, self.logger).build()
            if fallback is not None:
                self.logger.debug("shipping plain fallback crossfade")
                plan = fallback
            else:
                self.logger.debug("shipping click-free emergency handoff")
                plan = EmergencyHandoffFactory(ctx, factory, self.logger).build()
        else:
            plan = PlanAssembler(ctx, self.logger).finalize(winner.candidate)
        # the caller reads the outgoing grid off the planner after a successful
        # plan and expects it masked to the plan's own anchor
        self.outgoing = replace(
            ctx.outgoing,
            beats=ctx.outgoing.beats[ctx.outgoing.beats <= plan.fade_out_window],
            downbeats=ctx.outgoing.downbeats[ctx.outgoing.downbeats <= plan.fade_out_window],
        )
        return plan

    def _entered(self, ctx: TransitionContext, specs: list[CandidateSpec]) -> list[CandidateSpec]:
        """Pin every spec's entry to the incoming first downbeat when the client chose it."""
        if not self.fixed_entry or not len(ctx.incoming.downbeats):
            return specs
        entry = float(ctx.incoming.downbeats[0])
        # pinned, specs that differed only in their entry are one
        return list(dict.fromkeys(replace(spec, entry_s=entry) for spec in specs))
