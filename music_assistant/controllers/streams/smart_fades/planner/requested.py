"""
Smart Fades - the planner for a transition an API client asked for.

``player_queues/set_transition`` stores a request on the outgoing queue item; the stream
engine hands it to the mixer, which then plans the boundary with ``RequestedTransitionPlanner``
instead of ``SmartCrossFadePlanner``. It builds the requested shape from the default planner's
own pieces and ships the default plan, with a reason, when the request cannot be honoured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from music_assistant.controllers.streams.smart_fades.helpers import (
    SMART_CROSSFADE_DURATION,
    sustained_energy_floor,
)
from music_assistant.controllers.streams.smart_fades.models import (
    FadeOutTrim,
    TempoPlan,
    TransitionPlan,
    TransitionTier,
)

from .assembly import PlanAssembler
from .candidates import (
    _MAX_UNHEARD_INTRO_S,
    RUNG_LADDER,
    Candidate,
    CandidateFactory,
    CandidateSpec,
    _entry_options,
    bars_ladder,
)
from .context import TIME_STRETCH_BPM_PERCENTAGE_THRESHOLD, build_transition_context, choose_tier
from .planner import SmartCrossFadePlanner, TransitionPlanner
from .policies import (
    AudibleTrimPolicy,
    OverlapPreferencePolicy,
    Verdict,
    VocalTruncationPolicy,
    default_policies,
)
from .selection import CandidateSelector

if TYPE_CHECKING:
    import logging

    from music_assistant_models.constants import EXTRA_ATTRIBUTES_TYPES

    from music_assistant.controllers.streams.smart_fades.models import Deck
    from music_assistant.models.audio_analysis import AudioAnalysisData

    from .context import TransitionContext

REQUEST_PREFIX = "requested_transition_"
REQUEST_STYLES = ("blend", "quick_fade", "cut")
# a cut's overlap: just long enough to keep the switch click-free
CUT_SECONDS = 0.02
# the shortest tempo ramp a requested blend ships; the factory fits one into the (up to)
# 10s before the overlap, and clips it at the tail's start (a 16-bar blend's spans ~7.7s)
_MIN_RAMP_SECONDS = 6.0
# how far apart two unramped decks' beats may drift over a requested quick fade; it never
# gets shorter than a bar, so far-apart tempos drift more, as in the default quick fade
_QUICK_FADE_DRIFT_S = 0.04
# a vocal window starting this little before B's one is the 1800-bin timeline's
# rounding, not a pickup
_VOCAL_ONSET_SLACK_S = 0.25
# how many bars past Smart Fades' own exit a phrase may move the exit when no exit is named;
# further on, A would play its outro alone to keep a late ad-lib
_MAX_EXIT_DELAY_BARS = 4
# a bin of the stored, peak-normalized energy (~0.1-0.2s) at or under this is silent, as
# Smart Fades' sustained energy floor counts it (-40 dB)
_SILENT_RMS = 0.01
# a bar whose stored energy sits this far under the track's sustained level (-24 dB) for
# most of it is a soft intro or the start of a fade-in: heard alone after A, near silence
_QUIET_BAR_FRACTION = 0.063


@dataclass(frozen=True, slots=True)
class TransitionRequest:
    """How an API client asked the boundary after one queue item to play."""

    style: str  # one of REQUEST_STYLES
    next_item_id: str  # the queue item the request was made for
    bars: int = 0  # blend only: one of RUNG_LADDER
    exit_at: float = 0.0  # outgoing-song seconds where its audio ends; 0 = Smart Fades' exit

    def to_attributes(self) -> dict[str, EXTRA_ATTRIBUTES_TYPES]:
        """Return the request as the queue item's ``requested_transition_*`` extra attributes."""
        return {f"{REQUEST_PREFIX}{key}": value for key, value in asdict(self).items()}

    @classmethod
    def read(
        cls, attributes: dict[str, EXTRA_ATTRIBUTES_TYPES], next_item_id: str | None = None
    ) -> TransitionRequest | None:
        """
        Return the request stored in a queue item's extra attributes, if any.

        :param attributes: Extra attributes of the outgoing queue item.
        :param next_item_id: Return the request only when it was made for this next item.
        """
        style = attributes.get(f"{REQUEST_PREFIX}style")
        if style not in REQUEST_STYLES:
            return None
        request = cls(
            style=str(style),
            next_item_id=str(attributes.get(f"{REQUEST_PREFIX}next_item_id")),
            bars=int(attributes.get(f"{REQUEST_PREFIX}bars") or 0),
            exit_at=float(attributes.get(f"{REQUEST_PREFIX}exit_at") or 0.0),
        )
        if next_item_id is not None and request.next_item_id != next_item_id:
            return None
        return request

    @classmethod
    def take(
        cls, attributes: dict[str, EXTRA_ATTRIBUTES_TYPES], next_item_id: str
    ) -> TransitionRequest | None:
        """
        Return the request for the boundary into ``next_item_id`` as it is planned.

        The item names its next one from here on, as its transition report will, so
        ``player_queues/set_transition`` refuses changes the plan could no longer take.

        :param attributes: Extra attributes of the outgoing queue item.
        :param next_item_id: The queue item the boundary leads into.
        """
        attributes["transition_next_item_id"] = next_item_id
        return cls.read(attributes, next_item_id)

    @classmethod
    def drop(cls, attributes: dict[str, EXTRA_ATTRIBUTES_TYPES]) -> TransitionRequest | None:
        """
        Remove the request from a queue item's extra attributes and return it.

        :param attributes: Extra attributes of the outgoing queue item.
        """
        request = cls.read(attributes)
        for key in [key for key in attributes if key.startswith(REQUEST_PREFIX)]:
            del attributes[key]
        return request


class _PhraseCutPolicy(VocalTruncationPolicy):
    """Reject an exit inside an outgoing vocal phrase; later phrases are left out, not cut."""

    def evaluate(self, candidate: Candidate, ctx: TransitionContext) -> Verdict:
        """Judge one candidate against the shared per-transition context."""
        if ctx.vocal_out_scoring is None:
            return Verdict.ok()
        exit_s = candidate.plan.fade_out_window
        if any(
            left < exit_s and min(right, ctx.audio_end) - exit_s > self.max_truncated_vocal
            for left, right in ctx.vocal_out_scoring.windows
        ):
            return Verdict.reject("cuts into an audible outgoing vocal phrase")
        return Verdict.ok()


class RequestedTransitionPlanner(TransitionPlanner):
    """
    Plans the transition an API client asked for, from Smart Fades' own pieces.

    Unlike the default planner it may leave the outgoing track early, at the requested
    downbeat inside the held tail; a sung phrase after that exit is left out, one the exit
    cuts into is refused. Without a requested exit it leaves at the default planner's, moved
    to the nearest downbeat no sung phrase runs past. It adds no effects. A request it
    cannot honour ships the default planner's plan; ``outcome`` and ``reason`` say what
    happened.
    """

    def __init__(
        self,
        logger: logging.Logger,
        request: TransitionRequest,
        fade_in_seconds: float = float(SMART_CROSSFADE_DURATION),
    ) -> None:
        """
        Initialize the planner for one requested transition.

        :param logger: Logger for debug output.
        :param request: The client's request for this boundary.
        :param fade_in_seconds: Length of the incoming track's head the mix will receive.
        """
        super().__init__(logger)
        self.request = request
        self.fade_in_seconds = fade_in_seconds
        # "applied" | "fallback", read by the stream's transition report
        self.outcome = "applied"
        self.reason: str | None = None

    def plan(
        self,
        fade_out_analysis: AudioAnalysisData,
        fade_in_analysis: AudioAnalysisData,
        buffer_duration: float,
    ) -> TransitionPlan:
        """
        Build the requested ``TransitionPlan``, or the default one when it cannot be honoured.

        :param fade_out_analysis: Analysis data for the outgoing track.
        :param fade_in_analysis: Analysis data for the incoming track.
        :param buffer_duration: Length in seconds of the available fade-out holdback.
        """
        ctx = build_transition_context(
            fade_out_analysis, fade_in_analysis, buffer_duration, self.logger
        )
        plan = self._plan_request(ctx)
        self.logger.debug(
            "transition request %s bars=%s exit_at=%s: %s %s",
            self.request.style,
            self.request.bars,
            self.request.exit_at,
            self.outcome,
            self.reason or "",
        )
        if plan is None:
            default = SmartCrossFadePlanner(self.logger)
            plan = default.plan(fade_out_analysis, fade_in_analysis, buffer_duration)
            self.outgoing = default.outgoing
            return plan
        # the caller expects the outgoing grid masked to the plan's own exit, as the default planner does
        self.outgoing = replace(
            ctx.outgoing,
            beats=ctx.outgoing.beats[ctx.outgoing.beats <= plan.fade_out_window],
            downbeats=ctx.outgoing.downbeats[ctx.outgoing.downbeats <= plan.fade_out_window],
        )
        return plan

    def _plan_request(self, ctx: TransitionContext) -> TransitionPlan | None:
        """Build the requested plan on the first exit that takes it, or None (fallback)."""
        bar_out = ctx.outgoing.beats_per_bar * 60.0 / ctx.outgoing.bpm
        exits = [
            downbeat
            for downbeat in ctx.protective_downbeats
            if CUT_SECONDS < downbeat <= ctx.audio_end
        ]
        if not exits:
            self._fallback("no_room")
            return None
        if self.request.exit_at:
            # buffer-local; the client's exit is kept: only the downbeat nearest it is tried
            target = self.request.exit_at - ctx.buffer_offset
            exit_s = min(exits, key=lambda downbeat: abs(downbeat - target))
            if abs(exit_s - target) > bar_out / 2:
                # no downbeat of A near the asked exit in the tail it holds
                self._fallback("no_room")
                return None
            tries = [exit_s]
        else:
            # A's phrases are kept whole: the exit moves from Smart Fades' own to the nearest
            # downbeat no sung phrase of A runs past (the later one on a tie), a few bars at most
            sung_end = ctx.vocal_out_scoring.last_end() if ctx.vocal_out_scoring else 0.0
            latest = ctx.default_anchor + _MAX_EXIT_DELAY_BARS * bar_out
            tries = sorted(
                (
                    d
                    for d in exits
                    if sung_end - VocalTruncationPolicy.max_truncated_vocal <= d <= latest
                ),
                key=lambda downbeat: (abs(downbeat - ctx.default_anchor), -downbeat),
            ) or [min(exits, key=lambda downbeat: abs(downbeat - ctx.default_anchor))]
            # when that one does not plan only later ones are tried: an earlier exit keeps no
            # phrase whole that it does not, it would only leave A sooner (a blend past the end
            # of A's beat grid fails on every later downbeat)
            tries = [downbeat for downbeat in tries if downbeat >= tries[0]]
        first_reason = None
        for exit_s in tries:
            self.outcome, self.reason = "applied", None
            if (plan := self._plan_exit(ctx, exits, exit_s, bar_out)) is not None:
                return plan
            # the nearest exit's reason is the one reported when none works
            first_reason = first_reason or self.reason
        self._fallback(first_reason or "no_room")
        return None

    def _plan_exit(
        self, ctx: TransitionContext, exits: list[float], exit_s: float, bar_out: float
    ) -> TransitionPlan | None:
        """Build the requested plan ending on ``exit_s``, or None with the reason recorded."""
        # the request sets the exit and the length: the policies judging those stand down;
        # an exit the client named leaves A's later phrases out, so only a cut into one counts
        selector = CandidateSelector(
            [
                _PhraseCutPolicy()
                if self.request.exit_at and isinstance(policy, VocalTruncationPolicy)
                else policy
                for policy in default_policies()
                if not isinstance(policy, (AudibleTrimPolicy, OverlapPreferencePolicy))
            ],
            self.logger,
        )
        factory = CandidateFactory(ctx, self.logger)
        if self.request.style == "blend":
            if choose_tier(ctx.outgoing, ctx.incoming, exit_s)[1] is TransitionTier.QUICK_FADE:
                # the pair blends, but not this early in the tail (too few bars of A before it)
                blendable = ctx.tier is not TransitionTier.QUICK_FADE
                self._fallback("no_room" if blendable else "not_blendable")
                return None
            candidates = self._blend_candidates(ctx, factory, exit_s, bar_out)
        else:
            candidates = self._short_candidates(ctx, factory, exits, exit_s)
        if not candidates:
            if self.outcome != "fallback":
                self._fallback("no_room")
            return None
        winner = None
        for rung in candidates:
            if (scored := selector.select(rung, ctx)) is not None:
                winner = scored.candidate
                break
        if winner is None:
            # only the vocal policies reject in this set
            self._fallback("vocal")
            return None
        if self.request.style == "blend" and winner.spec.bars < self.request.bars:
            self.reason = "shortened"
        plan = PlanAssembler(ctx, self.logger).finalize(winner)
        if self.request.style == "cut":
            # inside a mastered fade the assembler picks "nofade", which would stop A dead
            plan = replace(plan, fadeout_curve="qsin")
        return plan

    def _blend_candidates(
        self, ctx: TransitionContext, factory: CandidateFactory, exit_s: float, bar_out: float
    ) -> list[list[Candidate]]:
        """Build the blend's rungs ending on ``exit_s``, longest first, each its full length."""
        ramped = 0.1 < ctx.bpm_diff_percent <= TIME_STRETCH_BPM_PERCENTAGE_THRESHOLD
        rungs: list[list[Candidate]] = []
        for bars in (bars for bars in RUNG_LADDER if bars <= self.request.bars):
            built = [
                candidate
                for entry in (None, *_entry_options(ctx, bars))
                if (
                    candidate := factory.build(
                        CandidateSpec(ctx.tier, bars, exit_s, entry, source="requested")
                    )
                )
                is not None
            ]
            rungs.append([c for c in built if self._fits(c, bars, bar_out, ramped)])
        return [rung for rung in rungs if rung]

    def _fits(self, candidate: Candidate, bars: int, bar_out: float, ramped: bool) -> bool:
        """Whether a built blend lasts its bars of A in the head of B it gets, ramped in time."""
        plan = candidate.plan
        steps = plan.tempo_plan.steps
        ratio = steps[-1][1] if steps else 1.0
        if candidate.spec.bars != bars or plan.crossfade_duration * ratio < (bars - 0.5) * bar_out:
            # capped, or snapped short of its bars (an exit past the end of A's beat grid)
            return False
        if (plan.fadein_trim_start or 0.0) + plan.crossfade_duration > min(
            float(SMART_CROSSFADE_DURATION), self.fade_in_seconds
        ):
            # past the head of B the mix receives the overlap is cut short, and B's own
            # stream takes over mid-fade
            return False
        # a squeezed ramp, or none, would blend the decks out of time
        return not ramped or (bool(steps) and steps[-1][0] - steps[0][0] >= _MIN_RAMP_SECONDS)

    def _short_candidates(
        self,
        ctx: TransitionContext,
        factory: CandidateFactory,
        exits: list[float],
        exit_s: float,
    ) -> list[list[Candidate]]:
        """Build the unramped quick fade or cut ending on ``exit_s``, B entering on its one."""
        cut = self.request.style == "cut"
        window = min(float(SMART_CROSSFADE_DURATION), self.fade_in_seconds)
        b_one = float(ctx.incoming.downbeats[0]) if len(ctx.incoming.downbeats) else 0.0
        if cut:
            # B's one lands where A ends; when B falls silent in the bar after it (a pickup
            # dying away before the song starts) or stays quiet through it (a soft intro, a
            # fade-in), the room would hear near silence once A stops, so the cut lands on
            # B's first later downbeat whose bar carries level
            overlap = CUT_SECONDS
            for one in [float(d) for d in ctx.incoming.downbeats] or [0.0]:
                entry = max(0.0, one - CUT_SECONDS)
                if self._sung_before(ctx, entry):
                    # a lead-in B sings before its one: cut off, or B's silence heard after A stops
                    self._fallback("vocal")
                    return []
                if entry + overlap > window:
                    return []
                if not _falls_quiet(ctx.incoming, one):
                    break
            else:
                return []
        else:
            # Smart Fades' quick-fade length, cut to the whole bars before the exit and to
            # what keeps the two unramped grids from audibly drifting apart
            drift = ctx.outgoing.beats_per_bar * abs(
                60.0 / ctx.outgoing.bpm - 60.0 / ctx.incoming.bpm
            )
            bars = min(bars_ladder(ctx, TransitionTier.QUICK_FADE)[0], exits.index(exit_s))
            if drift > 0.0:
                bars = min(bars, max(1, int(_QUICK_FADE_DRIFT_S / drift)))
            if bars == 0:
                return []
            overlap = exit_s - exits[exits.index(exit_s) - bars]
            # both decks on the one; B from its head, as today's quick fade, when that would
            # skip more than the fade plays, skip sung material, or pass the received head
            entry = b_one
            if (
                entry > overlap + _MAX_UNHEARD_INTRO_S
                or self._sung_before(ctx, entry)
                or entry + overlap > window
            ):
                entry = 0.0
            if overlap > window or _falls_quiet(ctx.incoming, entry + overlap):
                # past the head of B the mix receives; or B's bar once A has faded out is
                # near silence (a soft intro, a fade-in): the default plan ships
                return []
        spec = CandidateSpec(TransitionTier.QUICK_FADE, 1, exit_s, None, source="requested")
        plan = TransitionPlan(
            tier=TransitionTier.QUICK_FADE,
            fade_out_window=exit_s,
            crossfade_duration=overlap,
            tempo_plan=TempoPlan(),
            fadeout_trim=(
                FadeOutTrim(exit_s, ctx.buffer_duration - exit_s)
                if exit_s < ctx.buffer_duration
                else None
            ),
            fadein_trim_start=entry if entry > 0.0 else None,
        )
        return [[Candidate(spec, plan, factory.score(spec, plan), 1)]]

    @staticmethod
    def _sung_before(ctx: TransitionContext, entry: float) -> bool:
        """Whether B sings before ``entry``, beyond the rounding of its vocal timeline."""
        return ctx.vocal_in_scoring is not None and any(
            left < entry - _VOCAL_ONSET_SLACK_S for left, _ in ctx.vocal_in_scoring.windows
        )

    def _fallback(self, reason: str) -> None:
        """Record why the default plan ships instead of the requested one."""
        self.outcome, self.reason = "fallback", reason


def _falls_quiet(deck: Deck, start: float) -> bool:
    """Whether the deck's bar from ``start`` has a silent bin, or is quiet for most of it."""
    import numpy as np  # noqa: PLC0415

    rms = deck.analysis.rms_energy
    duration = deck.analysis.duration
    if rms is None or len(rms) == 0 or not duration:
        return False
    bins = np.asarray(rms, dtype=np.float32)
    bin_seconds = duration / len(bins)
    low = int(start / bin_seconds + 0.5)
    high = int((start + deck.beats_per_bar * 60.0 / deck.bpm) / bin_seconds + 0.5)
    bar = bins[low:high]
    # the bins centred in the bar; their median is deaf to a loud bin at its edge (the next
    # bar's onset, as the bins round it)
    return len(bar) > 0 and (
        float(bar.min()) <= _SILENT_RMS
        or float(np.median(bar)) < _QUIET_BAR_FRACTION * sustained_energy_floor(bins)
    )
