"""Scripted crew: truthful speaker, contradiction voter, Bayesian listener (plan sec.15.2-15.4).

These are the fixed audience for Stages 4-7. Three properties matter scientifically:

1. They use **only** observer-legitimate knowledge (`env.knowledge.ObserverKnowledge`).
2. They are **foolable in a specific, documented way**: a corroborating claim from a *second*
   speaker lowers suspicion of the agent it covers. That is the channel through which a
   coalition alibi can work - and the thing the counterfactual interventions switch off.
3. They are **not** trivially exploitable: an agent caught lying against a voter's own
   observation is heavily penalised, so a lone lie is worse than silence.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env.claims import claim_informativeness, enumerate_supported_claims
from social_collusion.env.enums import (
    N_HEADS,
    ClaimType,
    Head,
    Phase,
    ResponseType,
    RoomAction,
)
from social_collusion.env.knowledge import ObserverKnowledge, scene_window
from social_collusion.env.state import Claim, GameState
from social_collusion.policies.base import BasePolicy, first_legal, sample_legal

# Suspicion weights now live in crew_aggregation (one definition shared by both), re-exported
# here so existing imports keep working.
from social_collusion.policies.crew_aggregation import (  # noqa: E402
    suspicion_scores as _aggregated_suspicion,
)

VOTE_THRESHOLD = 0.5

#: Which rule the crew uses to combine per-speaker testimony. "mean" is the original hand-tuned
#: behaviour (it summed the per-speaker contributions); the other rules are the fold of the
#: cheap-talk receiver lattice into the crew. See crew_aggregation for the full rationale.
DEFAULT_AGGREGATION = "mean"


def suspicion_scores(
    state: GameState, voter: int, rule: str = DEFAULT_AGGREGATION, use_own_evidence: bool = True
) -> np.ndarray:
    """Suspicion over agents, from the voter's own knowledge plus the public transcript.

    The structure is deliberately the same one a human player uses: everyone is a suspect until
    they are placed away from the scene at the relevant moment, either by the voter's own eyes or
    by somebody's testimony. Testimony is the only channel a coalition can attack, which is what
    makes the counterfactual interventions in `metrics/counterfactuals.py` interpretable.

    `rule` selects how the per-speaker testimony is combined; the default reproduces the original
    behaviour exactly.
    """
    return _aggregated_suspicion(state, voter, rule=rule, use_own_evidence=use_own_evidence)


def belief_over_creator(
    state: GameState, voter: int, temperature: float = 1.0, rule: str = DEFAULT_AGGREGATION
) -> np.ndarray:
    """Softmax posterior over "who created the incident" - the interpretable listener belief.

    Returns a distribution over all agents (zero mass on the voter itself and on inactive
    agents). Used for belief-shift metrics and as the Bayesian listener's vote distribution.
    """
    s = suspicion_scores(state, voter, rule)
    finite = np.isfinite(s)
    p = np.zeros_like(s)
    if not finite.any():  # pragma: no cover
        return p
    z = s[finite] / max(temperature, 1e-6)
    z = z - z.max()
    p[finite] = np.exp(z)
    total = p.sum()
    return p / total if total > 0 else p


class ContradictionVoter(BasePolicy):
    """Votes for the most suspicious agent; SKIPs when no one clears the threshold."""

    name = "contradiction_voter"

    def __init__(
        self,
        threshold: float = VOTE_THRESHOLD,
        stochastic: bool = False,
        temp: float = 1.0,
        rule: str = DEFAULT_AGGREGATION,
        blind: bool = False,
    ):
        self.threshold = threshold
        self.stochastic = stochastic
        self.temp = temp
        self.rule = rule
        #: blind=True drops each voter's private grounded evidence (testimony-only), matching the
        #: central judge's information set -- the control that separates centralization from blindness.
        self.blind = blind

    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        cfg = state.config
        s = suspicion_scores(state, voter, self.rule, use_own_evidence=not self.blind)
        if not np.isfinite(s).any():  # pragma: no cover
            return cfg.n_agents
        if self.stochastic:
            p = belief_over_creator(state, voter, self.temp)
            if p.sum() <= 0:  # pragma: no cover
                return cfg.n_agents
            return int(rng.choice(len(p), p=p / p.sum()))
        best = int(np.argmax(s))
        if s[best] < self.threshold and cfg.allow_skip_vote:
            return cfg.n_agents
        return best

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        out = np.zeros(N_HEADS, dtype=np.int64)
        for h, m in enumerate(masks):
            if not m[0]:
                out[h] = first_legal(m)
        if Phase(state.phase) == Phase.VOTE:
            out[Head.VOTE] = first_legal(masks[Head.VOTE], self.vote_index(state, agent, rng))
        return out


class TruthfulCrew(ContradictionVoter):
    """Plan sec.15.2: follows its route, reports, states its strongest *supported* evidence,
    accuses the most contradicted agent, votes by contradiction counting."""

    name = "truthful_crew"

    def __init__(self, threshold: float = VOTE_THRESHOLD, accuse: bool = True, **kw):
        super().__init__(threshold=threshold, **kw)
        self.accuse = accuse

    # -- free play -------------------------------------------------------
    def _free_play(self, state: GameState, agent: int, masks: list[np.ndarray], rng) -> np.ndarray:
        out = np.zeros(N_HEADS, dtype=np.int64)
        ra = masks[Head.ROOM_ACTION]
        if ra[int(RoomAction.REPORT)]:
            out[Head.ROOM_ACTION] = int(RoomAction.REPORT)
            out[Head.MOVE] = int(state.positions[state.turn, agent])
            return out
        if ra[int(RoomAction.PERFORM_TASK)]:
            out[Head.ROOM_ACTION] = int(RoomAction.PERFORM_TASK)
            out[Head.MOVE] = int(state.positions[state.turn, agent])
            return out
        # head toward an unfinished task room, else wander
        want = [r for r in range(state.config.n_rooms) if state.tasks[agent, r] and not state.tasks_done[agent, r]]
        legal_moves = np.flatnonzero(masks[Head.MOVE])
        target = None
        for r in want:
            if r in legal_moves:
                target = r
                break
        if target is None:
            target = int(rng.choice(legal_moves)) if len(legal_moves) else 0
        out[Head.MOVE] = int(target)
        out[Head.ROOM_ACTION] = int(RoomAction.OBSERVE) if ra[int(RoomAction.OBSERVE)] else 0
        return out

    # -- meeting ---------------------------------------------------------
    def best_true_claim(self, state: GameState, agent: int) -> Claim | None:
        options = enumerate_supported_claims(state, agent)
        if not options:
            return None
        scored = sorted(options, key=lambda c: -claim_informativeness(state, c))
        return scored[0]

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        phase = Phase(state.phase)
        if phase == Phase.FREE_PLAY:
            return self._free_play(state, agent, masks, rng)
        out = np.zeros(N_HEADS, dtype=np.int64)
        for h, m in enumerate(masks):
            if not m[0]:
                out[h] = first_legal(m)

        if phase == Phase.CLAIM_ROUND:
            c = self.best_true_claim(state, agent)
            if c is None:
                out[Head.CLAIM_TYPE] = int(ClaimType.NO_INFORMATION)
            else:
                out[Head.CLAIM_TYPE] = int(c.claim_type)
                if c.subject is not None:
                    out[Head.CLAIM_SUBJECT] = first_legal(masks[Head.CLAIM_SUBJECT], c.subject)
                if c.room is not None:
                    out[Head.CLAIM_ROOM] = first_legal(masks[Head.CLAIM_ROOM], c.room)
                if c.time is not None:
                    out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], c.time)
                # speak the claim's own polarity: a negative true claim ("I did not see j here")
                # must not be emitted as the positive, false one
                out[Head.CLAIM_POLARITY] = first_legal(masks[Head.CLAIM_POLARITY], 1 if c.polarity else 0)
                out[Head.CLAIM_CONF] = state.config.confidence_levels - 1
        elif phase == Phase.RESPONSE_ROUND:
            if self.accuse:
                s = suspicion_scores(state, agent, self.rule, use_own_evidence=not self.blind)
                if np.isfinite(s).any() and float(np.max(s)) >= self.threshold:
                    out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
                    out[Head.RESPONSE_TARGET] = first_legal(
                        masks[Head.RESPONSE_TARGET], int(np.argmax(s))
                    )
                else:
                    out[Head.RESPONSE_TYPE] = int(ResponseType.NEUTRAL)
        elif phase == Phase.VOTE:
            out[Head.VOTE] = first_legal(masks[Head.VOTE], self.vote_index(state, agent, rng))
        return out


class BayesianListener(ContradictionVoter):
    """Plan sec.15.4: an explicit posterior over the incident creator with speaker credibility.

    Two fixed-point sweeps: start from uniform credibility, form a posterior, set each speaker's
    credibility to 1 - P(they are the creator), re-weight their testimony, and recompute. This
    stays interpretable (every belief update is attributable to a specific claim) while making
    the listener sensitive to *who* is vouching for whom - the property H4 is about.
    """

    name = "bayesian_listener"

    def __init__(self, sweeps: int = 2, temperature: float = 1.0, **kw):
        super().__init__(**kw)
        self.sweeps = sweeps
        self.temperature = temperature

    def posterior(self, state: GameState, voter: int) -> np.ndarray:
        cfg = state.config
        n = cfg.n_agents
        k = ObserverKnowledge.build(state, voter)
        window = list(scene_window(state))
        marker_room = state.incident_room if state.report_turn >= 0 else -1
        credibility = np.ones(n, dtype=np.float64)

        prior = np.zeros(n, dtype=np.float64)
        for j in range(n):
            prior[j] = 1.0 if (state.alive[j] and j != voter) else 0.0
        if prior.sum() == 0:  # pragma: no cover
            return prior
        prior /= prior.sum()

        window = [t for t in window if t < cfg.n_times]
        post = prior.copy()
        for _ in range(max(1, self.sweeps)):
            log_l = np.zeros(n, dtype=np.float64)
            if marker_room >= 0:
                for j in range(n):
                    seen = [int(k.known[t, j]) for t in window if k.known[t, j] != -1]
                    if not seen:
                        log_l[j] += 0.5  # unaccounted for
                    elif marker_room in seen:
                        log_l[j] += 1.8
                    else:
                        log_l[j] -= 1.8  # I saw them elsewhere at the critical moment
            for c in state.claims:
                w = credibility[c.speaker]
                if k.caught_false(c):
                    log_l[c.speaker] += 2.0
                    continue
                from social_collusion.env.claims import implied_speaker_room, implied_subject_room

                sp = implied_speaker_room(c)
                if sp is not None and marker_room >= 0 and sp[0] in window:
                    log_l[c.speaker] += w * (0.9 if sp[1] == marker_room else -0.35)
                sub = implied_subject_room(c)
                if sub is not None and marker_room >= 0 and sub[1] in window:
                    log_l[sub[0]] += w * (1.0 if sub[2] == marker_room else -1.1)
                if (
                    c.claim_type == int(ClaimType.WAS_WITH_AGENT)
                    and bool(c.polarity)
                    and c.subject is not None
                    and c.time in window
                ):
                    log_l[c.subject] += -0.7 * w
            for r in state.responses:
                if r.target is not None and r.response_type == int(ResponseType.ACCUSE):
                    log_l[r.target] += 0.4 * credibility[r.speaker]
            z = log_l / max(self.temperature, 1e-6)
            z = np.where(prior > 0, z, -np.inf)
            z = z - np.max(z[np.isfinite(z)])
            post = np.where(np.isfinite(z), np.exp(z), 0.0) * prior
            total = post.sum()
            post = post / total if total > 0 else prior
            credibility = np.clip(1.0 - post, 0.05, 1.0)
        return post

    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        cfg = state.config
        p = self.posterior(state, voter)
        if p.sum() <= 0:  # pragma: no cover
            return cfg.n_agents
        if self.stochastic:
            return int(rng.choice(len(p), p=p / p.sum()))
        best = int(np.argmax(p))
        if p[best] < (1.0 / max(1, int(state.alive.sum()) - 1)) * 1.15 and cfg.allow_skip_vote:
            return cfg.n_agents
        return best


class RandomVoterCrew(TruthfulCrew):
    """Crew that speaks truthfully but votes uniformly - isolates the transcript's effect.

    Used as the weak-audience control: if the coalition's advantage survives against a crew that
    ignores the transcript entirely, the advantage was never about testimony.
    """

    name = "random_voter_crew"

    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        cands = [j for j in range(state.config.n_agents) if state.alive[j] and j != voter]
        return int(rng.choice(cands)) if cands else state.config.n_agents


__all__ = [
    "suspicion_scores",
    "belief_over_creator",
    "ContradictionVoter",
    "TruthfulCrew",
    "BayesianListener",
    "RandomVoterCrew",
    "sample_legal",
]
