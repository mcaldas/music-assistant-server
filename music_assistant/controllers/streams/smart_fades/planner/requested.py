"""
Smart Fades - the planner for a transition an API client asked for.

``player_queues/set_transition`` stores a request on the outgoing queue item; the stream
engine hands it to the mixer, which then plans the boundary with ``RequestedTransitionPlanner``
instead of ``SmartCrossFadePlanner``. It builds the requested shape from the default planner's
own pieces and ships the default plan, with a reason, when the request cannot be honoured.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from music_assistant.controllers.streams.smart_fades.helpers import (
    AUDIBLE_WINDOW_S,
    SMART_CROSSFADE_DURATION,
    audible_start,
    db_ramp,
    sustained_energy_floor,
)
from music_assistant.controllers.streams.smart_fades.models import (
    EchoOut,
    EqPlan,
    FadeOutTrim,
    PlanMetrics,
    SmartFadeNotApplicable,
    SweepSchedule,
    TempoPlan,
    TransitionPlan,
    TransitionTier,
)
from music_assistant.controllers.streams.smart_fades.vocal import VOCAL_LEFT_PADDING

from .assembly import PlanAssembler
from .candidates import (
    _GRID_FRAME_S,
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

    import numpy as np
    import numpy.typing as npt
    from music_assistant_models.constants import EXTRA_ATTRIBUTES_TYPES

    from music_assistant.controllers.streams.smart_fades.models import Deck
    from music_assistant.models.audio_analysis import AudioAnalysisData

    from .context import TransitionContext

REQUEST_PREFIX = "requested_transition_"
REQUEST_STYLES = ("blend", "quick_fade", "cut", "filter_sweep", "echo_out")
# the styles built on a blend's overlap; the others are built on a cut's or a quick fade's
BLEND_STYLES = ("blend", "filter_sweep")
# a cut's overlap: just long enough to keep the switch click-free
CUT_SECONDS = 0.02
# the shortest tempo ramp a requested blend ships; the factory fits one into the (up to)
# 10s before the overlap, and clips it at the tail's start (a 16-bar blend's spans ~7.7s)
_MIN_RAMP_SECONDS = 6.0
# how far apart two unramped decks' beats may drift over a requested quick fade; when a
# bar would drift further, the fade is that much shorter and lands B's one on A's exit
_QUICK_FADE_DRIFT_S = 0.04
# a vocal window starting this little before B's one is the 1800-bin timeline's
# rounding, not a pickup
_VOCAL_ONSET_SLACK_S = 0.25
# how many bars past Smart Fades' own exit a phrase may move the exit when no exit is named;
# further on, A would play its outro alone to keep a late ad-lib
_MAX_EXIT_DELAY_BARS = 4
# a bin of the stored energy (~0.1-0.2s) this far under the track's sustained level
# (-33.5 dB) is near silence once A stops
_SILENT_FRACTION = 0.021
# how far into a landing bar such a bin is a gap at the switch; later on it is B's own
# stop, heard after B has started (48 analysed songs: gaps began by 1.41 beats, stops at 1.96+)
_LANDING_BEATS = 1.75
# ... but only once B has reached its level there: a bar whose first beats sit this far under
# the sustained level (-6 dB) has not started, and a silent bin anywhere in it is a gap
_STARTED_FRACTION = 0.5
# a bar whose stored energy sits this far under the track's sustained level (-22 dB) for
# most of it is a soft intro or the start of a fade-in: heard alone after A, near silence.
# B carries level from the first bin at this line; no beat before it lands on A's exit
_QUIET_BAR_FRACTION = 0.079
# how far before the incoming audio's start (read from its PCM head) a landing may sit: under
# a 30 ms gap, and about a frame of the analysis' beat grid
_HEARD_SLACK_S = 0.02
# the analysis averages energy over 100 ms windows into its bins: a bin can read loud from audio
# up to a window past its end
_ENERGY_WINDOW_S = 0.1
# the end of a whole-bar quick fade where A has mostly faded out (under -16 dB): B must carry
# level there, or the room hears the fade die into B's silence
_FADED_FRACTION = 0.1
# a gap in the incoming PCM head: this long under its audible line, as a room hears one; the
# head shows gaps the analysis' ~0.1-0.2 s bins blur
_GAP_S = 0.03
# filter_sweep: the outgoing high-pass rises from 10 Hz over the whole overlap; the incoming
# low-pass opens over its first 3/4 and fades to dry by 9/10 of it, ahead of asendcmd's frame
# granularity (~0.1s), so the incoming track plays on untouched
_SWEEP_OUT_HZ = (10.0, 8000.0)
# the outgoing track plays dry until the overlap: even at 10 Hz a high-pass turns the phase of
# its bass, which can rebuild a limited master's peaks past full scale. The filter's wet share
# then rises over the first half of the overlap, at most this long, on asendcmd's 10 ms frames
_SWEEP_OUT_WET_S = 1.0
_SWEEP_IN_HZ = (250.0, 8000.0)
_SWEEP_IN_OPEN = 0.75
_SWEEP_IN_DRY = 0.9
# echo_out: how long the outgoing track's last beat rings on, in bars of the incoming track
# (it repeats on that track's beat); each tempo is read over this many bars of its grid
_ECHO_BARS = 2
# two bars of a grid on the analysis' 20 ms frames read a steady tempo within about 1%: a
# grid this far off the bpm is a tempo the track really plays at there
_GRID_TEMPO_SLACK = 0.02


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
        exit_s = candidate.plan.fade_out_window
        if any(
            left < exit_s and min(right, ctx.audio_end) - exit_s > self.max_truncated_vocal
            for left, right in self.phrases(ctx)
        ):
            return Verdict.reject("cuts into an audible outgoing vocal phrase")
        return Verdict.ok()


class RequestedTransitionPlanner(TransitionPlanner):
    """
    Plans the transition an API client asked for, from Smart Fades' own pieces.

    Unlike the default planner it may leave the outgoing track early, at the requested
    downbeat inside the held tail; a sung phrase after that exit is left out, one the exit
    cuts into is refused. Without a requested exit it leaves at the default planner's, moved
    to the nearest downbeat no sung phrase runs past. A filter sweep is a blend handed over
    by filters instead of EQ; an echo out is a cut whose last outgoing beat rings on. A
    request it cannot honour ships the default planner's plan, except that an exit asked on
    an outgoing track with an end position is kept: A is cut on that exit's downbeat.
    ``outcome`` and ``reason`` say what happened.
    """

    def __init__(
        self,
        logger: logging.Logger,
        request: TransitionRequest,
        fade_in_seconds: float = float(SMART_CROSSFADE_DURATION),
        cut_at_end: bool = False,
        incoming_head: npt.NDArray[np.bool_] | None = None,
        fixed_entry: bool = False,
    ) -> None:
        """
        Initialize the planner for one requested transition.

        :param logger: Logger for debug output.
        :param request: The client's request for this boundary.
        :param fade_in_seconds: Length of the incoming track's head the mix will receive.
        :param cut_at_end: Whether the outgoing tail ends where the queue item's audio is cut
            at its end position (``set_end_position``) rather than where the song ends.
        :param incoming_head: Which 10 ms windows of the incoming track's head are audible,
            from its start, read from the PCM the mixer holds; None when unknown.
        :param fixed_entry: Whether the client chose where the incoming audio starts
            (``set_start_position``): a blend then enters it at its first downbeat, never
            deeper in.
        """
        super().__init__(logger)
        self.request = request
        self.fade_in_seconds = fade_in_seconds
        self.cut_at_end = cut_at_end
        self.incoming_head = incoming_head
        self.fixed_entry = fixed_entry
        # where the incoming track's audio starts to sound, in its seconds
        self.incoming_audible_from = None if incoming_head is None else audible_start(incoming_head)
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
        try:
            ctx = build_transition_context(
                fade_out_analysis, fade_in_analysis, buffer_duration, self.logger
            )
        except SmartFadeNotApplicable:
            if not (self.cut_at_end and self.request.exit_at):
                raise
            # Smart Fades finds no exit of its own in this tail (its energy drops early), but
            # the client named one on an end position: A's downbeats are all that takes
            ctx = build_transition_context(
                fade_out_analysis, fade_in_analysis, buffer_duration, self.logger, own_exit=False
            )
        plan = self._plan_request(ctx)
        if plan is None and self.cut_at_end and self.request.exit_at:
            # the client fixed where A ends: a cut there, never Smart Fades' own exit, which can
            # land anywhere in the held tail (half of what the item plays)
            plan = self._cut_at_exit(ctx)
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
        exits = _exits(ctx)
        if not exits:
            self._fallback("no_room")
            return None
        if self.request.exit_at:
            # the client's exit is kept: only the downbeat nearest it is tried
            if (exit_s := self._requested_exit(ctx, exits, bar_out)) is None:
                # no downbeat of A near the asked exit in the tail it holds
                self._fallback("no_room")
                return None
            tries = [exit_s]
        else:
            # A's phrases are kept whole: the exit moves from Smart Fades' own to the nearest
            # downbeat no sung phrase of A runs past (the later one on a tie), a few bars at most;
            # a phrase sung through an end position is cut there whatever the exit
            truncation = VocalTruncationPolicy(self.cut_at_end)
            sung_end = max((right for _, right in truncation.phrases(ctx)), default=0.0)
            latest = ctx.default_anchor + _MAX_EXIT_DELAY_BARS * bar_out
            tries = sorted(
                (d for d in exits if sung_end - truncation.max_truncated_vocal <= d <= latest),
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

    def _requested_exit(
        self, ctx: TransitionContext, exits: list[float], bar_out: float
    ) -> float | None:
        """Return the downbeat (buffer-local) nearest the client's exit, None when none is near."""
        target = self.request.exit_at - ctx.buffer_offset
        exit_s = min(exits, key=lambda downbeat: abs(downbeat - target))
        # an exit asked past A's last downbeat, with no bar left after it before A's audio
        # ends (an end position mid-bar): the next downbeat is past the end, so take that one
        past_last = target > exits[-1] and exits[-1] + bar_out + _GRID_FRAME_S > ctx.audio_end
        if not past_last and abs(exit_s - target) > bar_out / 2:
            return None
        return exit_s

    def _cut_at_exit(self, ctx: TransitionContext) -> TransitionPlan | None:
        """
        Cut A on the downbeat at the client's exit, the request having failed, or None.

        The client fixed both ends (A's end position and exit, B's entry), so the vocal
        policies stand down: what the client cut, it cut. The outcome stays "fallback" with
        the request's own reason. None, for the default plan, when no downbeat lies near the
        exit or the cut itself cannot be built.
        """
        bar_out = ctx.outgoing.beats_per_bar * 60.0 / ctx.outgoing.bpm
        exits = _exits(ctx)
        exit_s = self._requested_exit(ctx, exits, bar_out) if exits else None
        if exit_s is None:
            return None
        reason = self.reason
        candidates = self._short_candidates(
            ctx, CandidateFactory(ctx, self.logger), exits, exit_s, bar_out, quick=False
        )
        self.outcome, self.reason = "fallback", reason
        if not candidates:
            return None
        # one candidate, and none of the policies left would reject it
        plan = PlanAssembler(ctx, self.logger).finalize(candidates[0][0])
        # inside a mastered fade the assembler picks "nofade", which would stop A dead
        return replace(plan, fadeout_curve="qsin")

    def _plan_exit(
        self, ctx: TransitionContext, exits: list[float], exit_s: float, bar_out: float
    ) -> TransitionPlan | None:
        """Build the requested plan ending on ``exit_s``, or None with the reason recorded."""
        # the request sets the exit and the length: the policies judging those stand down;
        # an exit the client named leaves A's later phrases out, so only a cut into one
        # counts; a phrase sung through an end position is cut there whatever the exit
        selector = CandidateSelector(
            [
                (_PhraseCutPolicy if self.request.exit_at else VocalTruncationPolicy)(
                    self.cut_at_end
                )
                if isinstance(policy, VocalTruncationPolicy)
                else policy
                for policy in default_policies()
                if not isinstance(policy, (AudibleTrimPolicy, OverlapPreferencePolicy))
            ],
            self.logger,
        )
        factory = CandidateFactory(ctx, self.logger)
        if self.request.style in BLEND_STYLES:
            if choose_tier(ctx.outgoing, ctx.incoming, exit_s)[1] is TransitionTier.QUICK_FADE:
                # the pair blends, but not this early in the tail (too few bars of A before it)
                blendable = ctx.tier is not TransitionTier.QUICK_FADE
                self._fallback("no_room" if blendable else "not_blendable")
                return None
            candidates = self._blend_candidates(ctx, factory, exit_s, bar_out)
        else:
            candidates = self._short_candidates(
                ctx, factory, exits, exit_s, bar_out, quick=self.request.style == "quick_fade"
            )
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
        if self.request.style in BLEND_STYLES and winner.spec.bars < self.request.bars:
            self.reason = "shortened"
        plan = PlanAssembler(ctx, self.logger).finalize(winner)
        if not winner.spec.bars:
            # a cut, or a quick fade under a bar: inside a mastered fade the assembler picks
            # "nofade", which would stop A dead
            plan = replace(plan, fadeout_curve="qsin")
            if self.request.style == "quick_fade":
                # Smart Fades' own quick fade lasts a bar at least
                self.reason = "shortened"
        if self.request.style == "filter_sweep":
            return _with_sweeps(plan)
        if self.request.style == "echo_out":
            return self._with_echo(ctx, plan)
        return plan

    def _blend_candidates(
        self, ctx: TransitionContext, factory: CandidateFactory, exit_s: float, bar_out: float
    ) -> list[list[Candidate]]:
        """Build the blend's rungs ending on ``exit_s``, longest first, each its full length."""
        ramped = 0.1 < ctx.bpm_diff_percent <= TIME_STRETCH_BPM_PERCENTAGE_THRESHOLD
        rungs: list[list[Candidate]] = []
        # where the client started B is its entry: pinned to B's first downbeat, so neither the
        # natural entry nor the rolling-intro alignment skips the lead-in it chose
        fixed = [float(ctx.incoming.downbeats[0])] if len(ctx.incoming.downbeats) else [None]
        for bars in (bars for bars in RUNG_LADDER if bars <= self.request.bars):
            built = [
                candidate
                for entry in (fixed if self.fixed_entry else (None, *_entry_options(ctx, bars)))
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
        bar_out: float,
        quick: bool,
    ) -> list[list[Candidate]]:
        """Build the unramped quick fade (or cut) ending on ``exit_s``, B's one on A's downbeat."""
        window = min(float(SMART_CROSSFADE_DURATION), self.fade_in_seconds)
        b_one = float(ctx.incoming.downbeats[0]) if len(ctx.incoming.downbeats) else 0.0
        # a cut's overlap, or a quick fade's whole bars
        bars, fade = 0, CUT_SECONDS
        if quick:
            # Smart Fades' quick-fade length, cut to the whole bars before the exit and to
            # what keeps the two unramped grids from audibly drifting apart
            bars = min(bars_ladder(ctx, TransitionTier.QUICK_FADE)[0], exits.index(exit_s))
            if bars == 0:
                return []
            drift = ctx.outgoing.beats_per_bar * abs(
                60.0 / ctx.outgoing.bpm - 60.0 / ctx.incoming.bpm
            )
            if drift > 0.0:
                bars = min(bars, int(_QUICK_FADE_DRIFT_S / drift))
                # how long the decks stay together when even a bar drifts too far
                fade = _QUICK_FADE_DRIFT_S / drift * bar_out
        if bars == 0:
            # B's one lands where A ends; when B falls silent in the bar after it (a pickup
            # dying away before the song starts) or stays quiet through it (a soft intro, a
            # fade-in), the room would hear near silence once A stops, so the cut lands on
            # B's first later downbeat whose bar carries level. A quick fade under a bar
            # fades in what B plays before that one under A's last beats (they meet on the
            # one and drift apart going back), only from B's first beat and only where it
            # carries level: A fading out over B's silence leaves the room a dip. Nothing of B
            # before it is heard plays alone after A: a grid beat or downbeat there (extrapolated
            # back over a silent head) would leave the room B's silence
            heard = self._heard_from(ctx)
            first_beat = max(
                float(ctx.incoming.beats[0]) if len(ctx.incoming.beats) else b_one, heard
            )
            for one in [float(d) for d in ctx.incoming.downbeats if d >= heard] or [heard]:
                overlap = max(CUT_SECONDS, min(fade, one - first_beat))
                if overlap > CUT_SECONDS and _falls_quiet(
                    ctx.incoming, one - overlap, overlap, self.incoming_head
                ):
                    overlap = CUT_SECONDS
                entry = max(0.0, one - overlap)
                hold = False
                if self._sung_before(ctx, entry):
                    # B sings a lead-in before its one: it comes in under A's last beats, so
                    # B's one still lands on A's exit (a pre-roll). Longer than a bar of A it
                    # is no cut any more: the default plan ships
                    entry = min(self._pickup_start(ctx), one - CUT_SECONDS)
                    if one - entry > min(bar_out, exit_s):
                        self._fallback("vocal")
                        return []
                    # a lead-in whose beats would drift off A's comes in under A only up to
                    # its first beat, which lands on A's exit; the rest plays alone after A,
                    # so it must carry level
                    land = self._pickup_landing(ctx, max(entry, heard), one)
                    if land < one and _falls_quiet(
                        ctx.incoming, land, one - land, self.incoming_head
                    ):
                        self._fallback("vocal")
                        return []
                    overlap = max(CUT_SECONDS, land - entry)
                    # a lead-in that breaks or turns quiet where A has mostly faded out would
                    # leave the room a dip: a quick fade then holds both decks as the cut does.
                    # A breath shorter than an analysis bin shows only in B's head; without
                    # it, A holds
                    hold = (
                        land < one
                        or _falls_quiet(ctx.incoming, one - overlap / 2, overlap / 2)
                        or _gap_in(self.incoming_head, one - overlap / 2, one) is not False
                    )
                if entry + overlap > window:
                    return []
                if not _falls_quiet(ctx.incoming, one, head=self.incoming_head):
                    break
            else:
                return []
        else:
            overlap, hold = exit_s - exits[exits.index(exit_s) - bars], False
            # both decks on the one; a lead-in B sings before it fades in ahead of the bars
            # (a pre-roll of at most a bar of A), so the one stays on A's downbeat. B from its
            # head, as today's quick fade, when that would skip more than the fade plays, or
            # pass the received head
            entry, lead = b_one, 0.0
            if self._sung_before(ctx, entry):
                entry = self._pickup_start(ctx)
                lead = b_one - entry
                if lead > min(bar_out, exit_s - overlap):
                    # from its head B's one would miss A's downbeat by the lead-in, its beats
                    # off A's: as for a cut, the default plan ships
                    self._fallback("vocal")
                    return []
            if entry > overlap + _MAX_UNHEARD_INTRO_S or b_one + overlap > window:
                entry, lead = 0.0, 0.0
            overlap += lead
            faded = overlap * _FADED_FRACTION
            if (
                overlap > window
                or _falls_quiet(ctx.incoming, entry + overlap, head=self.incoming_head)
                or _falls_quiet(ctx.incoming, entry + overlap - faded, faded, self.incoming_head)
            ):
                # past the head of B the mix receives; or B's bar once A has faded out, or the
                # end of the fade itself, is near silence (a soft intro, a fade-in, a stop):
                # the default plan ships
                return []
        spec = CandidateSpec(TransitionTier.QUICK_FADE, bars, exit_s, None, source="requested")
        # a cut's pre-roll holds both decks at full between its fades; a quick fade's stays
        # one equal-power fade over the whole overlap unless B's lead-in would leave a dip
        held = overlap > CUT_SECONDS and (not quick or hold)
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
            fade_seconds=CUT_SECONDS if held else None,
        )
        metrics = factory.score(spec, plan)
        if held:
            # both decks play at full over a pre-roll: every second both sing weighs one
            metrics = _at_full(metrics)
        return [[Candidate(spec, plan, metrics, 1)]]

    def _pickup_start(self, ctx: TransitionContext) -> float:
        """Where B's sung lead-in starts, its detector lag padded, never before B's audio."""
        import numpy as np  # noqa: PLC0415

        assert ctx.vocal_in_scoring is not None  # narrowed by _sung_before
        sung = ctx.vocal_in_scoring.windows[0][0]
        rms = ctx.incoming.analysis.rms_energy
        duration = ctx.incoming.analysis.duration
        audible = sung
        if rms is not None and len(rms) and duration:
            bins = np.asarray(rms, dtype=np.float32)
            loud = np.flatnonzero(bins > _SILENT_FRACTION * sustained_energy_floor(bins))
            if len(loud):
                audible = min(sung, float(loud[0]) * duration / len(rms))
        return max(0.0, sung - VOCAL_LEFT_PADDING, audible, self.incoming_audible_from or 0.0)

    def _heard_from(self, ctx: TransitionContext) -> float:
        """
        Return where B first carries level: no landing before it, where B would play alone.

        The first bin at the quiet-bar line, as a landing rounds to its nearest bin, and no
        earlier than the PCM head says B's audio starts (bins of ~0.1-0.2s blur that onset).
        """
        import numpy as np  # noqa: PLC0415

        heard = 0.0
        rms = ctx.incoming.analysis.rms_energy
        duration = ctx.incoming.analysis.duration
        if rms is not None and len(rms) and duration:
            bins = np.asarray(rms, dtype=np.float32)
            floor = sustained_energy_floor(bins)
            loud = np.flatnonzero(bins >= _QUIET_BAR_FRACTION * floor)
            if len(loud):
                first = int(loud[0])
                heard = max(0.0, (first - 0.5) * duration / len(bins))
                if (
                    self.incoming_head is None
                    and first
                    and bins[first - 1] <= _SILENT_FRACTION * floor
                ):
                    # after a silent head B may start anywhere in that bin or the energy window
                    # past it, and its grid sit a few frames early: without the PCM head no
                    # landing before then
                    heard = (first + 1) * duration / len(bins) + _ENERGY_WINDOW_S
        if self.incoming_audible_from is not None:
            heard = max(heard, self.incoming_audible_from - _HEARD_SLACK_S)
        return heard

    @staticmethod
    def _pickup_landing(ctx: TransitionContext, entry: float, one: float) -> float:
        """
        Return where in B a lead-in heard from ``entry`` meets A's exit downbeat.

        B's one, when every beat B plays before it then falls within the quick fade's drift
        budget of one of A's beats; otherwise B's first beat heard, so that none of B's beats
        plays under A's and B carries level once A has gone.
        """
        beat = 60.0 / ctx.outgoing.bpm
        lead = [float(b) for b in ctx.incoming.beats if entry <= b < one]
        if all(abs(one - b - round((one - b) / beat) * beat) <= _QUICK_FADE_DRIFT_S for b in lead):
            return one
        return lead[0]

    @staticmethod
    def _sung_before(ctx: TransitionContext, entry: float) -> bool:
        """Whether B sings before ``entry``, beyond the rounding of its vocal timeline."""
        return ctx.vocal_in_scoring is not None and any(
            left < entry - _VOCAL_ONSET_SLACK_S for left, _ in ctx.vocal_in_scoring.windows
        )

    def _with_echo(self, ctx: TransitionContext, plan: TransitionPlan) -> TransitionPlan | None:
        """Ring the cut's last outgoing beat on under the incoming track, or None to fall back."""
        exit_s, entry = plan.fade_out_window, plan.fadein_trim_start or 0.0
        # B's beat on A's exit: the one the cut lands on (B's first, or a later one when B falls
        # silent after it), or the first beat of a lead-in whose beats would drift off A's
        at = entry + plan.crossfade_duration
        landing = min((float(b) for b in ctx.incoming.beats), key=lambda b: abs(b - at), default=at)
        bar = ctx.incoming.beats_per_bar
        out_beat, in_beat = 60.0 / ctx.outgoing.bpm, 60.0 / ctx.incoming.bpm
        # A's last beat before the exit, from the bars of its grid before it (its tempo can
        # drift from the bpm): the first repeat lands that beat on B's
        last = [float(b) for b in ctx.outgoing.beats if b < exit_s - 0.25 * out_beat]
        beat = _grid_beat([*last[-_ECHO_BARS * ctx.outgoing.beats_per_bar :], exit_s], out_beat)
        # the repeats ring under B, so the rest land on B's beat, from its grid after that one
        head = [float(b) for b in ctx.incoming.beats if b > landing - 0.25 * in_beat]
        period = _grid_beat(head[: _ECHO_BARS * bar + 1], in_beat)
        # a beat inside the cut's overlap plays before A's end: the echo starts on it there too
        lead = max(0.0, at - landing)
        window = min(float(SMART_CROSSFADE_DURATION), self.fade_in_seconds)
        repeats = min(_ECHO_BARS * bar, int((window - landing) / period))
        if exit_s < beat or repeats < bar:
            # the last beat, or a bar of its echo, is not in the audio the mix receives
            self._fallback("no_room")
            return None
        if (
            ctx.vocal_collision_reliable
            and ctx.vocal_out_scoring is not None
            and ctx.vocal_in_scoring is not None
            and any(
                left < exit_s and right > exit_s - beat
                for left, right in ctx.vocal_out_scoring.windows
            )
            and any(
                left < landing + repeats * period and right > entry
                for left, right in ctx.vocal_in_scoring.windows
            )
        ):
            # a sung beat echoing over the incoming vocal is the collision the checks guard
            self._fallback("vocal")
            return None
        return replace(plan, echo_out=EchoOut(beat, period, repeats, lead))

    def _fallback(self, reason: str) -> None:
        """Record why the default plan ships instead of the requested one."""
        self.outcome, self.reason = "fallback", reason


def _exits(ctx: TransitionContext) -> list[float]:
    """Return A's downbeats in the held tail where it can leave, buffer-local."""
    # a downbeat less than a frame past A's end is on it (an end position on a bar line)
    return [
        min(downbeat, ctx.audio_end)
        for downbeat in ctx.protective_downbeats
        if CUT_SECONDS < downbeat <= ctx.audio_end + _GRID_FRAME_S
    ]


def _at_full(metrics: PlanMetrics) -> PlanMetrics:
    """Weigh every second of vocal collision as one, as for two decks both at full."""
    return replace(metrics, weighted_collision_seconds=metrics.collision_seconds)


def _gap_in(head: npt.NDArray[np.bool_] | None, start: float, end: float) -> bool | None:
    """
    Whether the incoming PCM head holds a gap between ``start`` and ``end``, in its seconds.

    None when no head was read, or it ends before ``end``.

    :param head: The incoming track's audible 10 ms windows, from its start.
    """
    import numpy as np  # noqa: PLC0415

    last = round(end / AUDIBLE_WINDOW_S)
    if head is None or last > len(head):
        return None
    quiet = ~head[max(0, round(start / AUDIBLE_WINDOW_S)) : last]
    run = round(_GAP_S / AUDIBLE_WINDOW_S)
    held = np.convolve(quiet.astype(np.int32), np.ones(run, dtype=np.int32), mode="valid")
    return len(quiet) >= run and bool((held == run).any())


def _falls_quiet(
    deck: Deck,
    start: float,
    seconds: float = 0.0,
    head: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Whether the deck falls silent from ``start`` before it has started, or stays quiet.

    :param deck: The incoming deck.
    :param start: Where its bar starts, in its own seconds.
    :param seconds: A span to judge instead of the bar, silent anywhere in it (one B fades
        in over).
    :param head: The incoming track's audible 10 ms windows from its start: a gap in the
        span, or in the landing's beats, is silence wherever they reach.
    """
    import numpy as np  # noqa: PLC0415

    beat = 60.0 / deck.bpm
    gap = _gap_in(head, start, start + (seconds or _LANDING_BEATS * beat))
    if gap:
        return True
    rms = deck.analysis.rms_energy
    duration = deck.analysis.duration
    if rms is None or len(rms) == 0 or not duration:
        return False
    bins = np.asarray(rms, dtype=np.float32)
    bin_seconds = duration / len(bins)
    low = int(start / bin_seconds + 0.5)
    high = int((start + (seconds or deck.beats_per_bar * beat)) / bin_seconds + 0.5)
    # every bin that starts inside the landing's beats counts; where the head does not show
    # them, also the next, the first a stop starting late in them silences whole
    landing = (
        high
        if seconds
        else max(
            low + 1,
            math.ceil((start + _LANDING_BEATS * beat) / bin_seconds) + (1 if gap is None else 0),
        )
    )
    bar = bins[low:high]
    if len(bar) == 0:
        return False
    floor = sustained_energy_floor(bins)
    # a silent bin late in the bar is B's own stop only once B has started at its level
    landed = bins[low:landing]
    if float(np.median(landed)) < _STARTED_FRACTION * floor:
        landed = bar
    # the bins centred in the bar; their median is deaf to a loud bin at its edge (the next
    # bar's onset, as the bins round it)
    return (
        float(landed.min()) <= _SILENT_FRACTION * floor
        or float(np.median(bar)) < _QUIET_BAR_FRACTION * floor
    )


def _with_sweeps(plan: TransitionPlan) -> TransitionPlan:
    """Hand a blend over with filters instead of EQ: A high-passed away, B opened from a low-pass."""
    ratio = plan.tempo_plan.steps[-1][1] if plan.tempo_plan else 1.0
    overlap = plan.crossfade_duration
    # A-side in A-input time (its ramp ends before the overlap), B-side in post-trim time
    start_out = plan.fade_out_window - overlap * ratio
    opened, dry = _SWEEP_IN_OPEN * overlap, _SWEEP_IN_DRY * overlap
    # about 150 steps over the overlap, 10-100 ms apart: a one-bar sweep must not zipper
    step = min(0.1, max(0.01, overlap / 150))
    return replace(
        plan,
        eq_plan=EqPlan.neutral(swap_at=plan.eq_plan.swap_at),
        sweep_out=SweepSchedule(
            [
                (0.0, _SWEEP_OUT_HZ[0]),
                *_hz_ramp(start_out, overlap * ratio, *_SWEEP_OUT_HZ, step),
            ],
            [
                (0.0, 0.0),
                *db_ramp(start_out, min(_SWEEP_OUT_WET_S, overlap * ratio / 2), 0.0, 1.0, 0.01),
            ],
        ),
        sweep_in=SweepSchedule(
            _hz_ramp(0.0, opened, *_SWEEP_IN_HZ, step),
            db_ramp(opened, dry - opened, 1.0, 0.0, step),
        ),
    )


def _grid_beat(times: list[float], nominal: float) -> float:
    """
    Return the beat a run of consecutive grid times plays at, or ``nominal`` (the bpm's).

    The run's mean beat is returned only where it departs from ``nominal`` by more than
    ``_GRID_TEMPO_SLACK``: nearer, it can be the rounding of the analysis' 20 ms frames.

    :param times: Consecutive beat times of one track's grid, in order.
    :param nominal: The beat its bpm gives; also the answer for a run of fewer than two
        times, or with a step more than half a beat off ``nominal`` (a gap or an extra beat).
    """
    gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
    if not gaps or any(abs(gap / nominal - 1.0) > 0.5 for gap in gaps):
        return nominal
    mean = (times[-1] - times[0]) / len(gaps)
    return mean if abs(mean / nominal - 1.0) > _GRID_TEMPO_SLACK else nominal


def _hz_ramp(
    start: float, duration: float, from_hz: float, to_hz: float, step: float
) -> list[tuple[float, float]]:
    """Return a cutoff schedule that moves evenly in octaves, on ``db_ramp``'s step grid."""
    return [
        (t, 2.0**octave)
        for t, octave in db_ramp(start, duration, math.log2(from_hz), math.log2(to_hz), step)
    ]
