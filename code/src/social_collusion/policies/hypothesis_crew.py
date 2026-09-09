"""Candidate-coalition feasibility scoring with hard observer-legitimate constraints.

This is an explicitly heuristic score, not a calibrated Bayesian posterior. Candidate coalitions
have equal initial weight. For each speaking coalition member a Bernoulli `honesty` gate chooses
whether its claims join the mandatory honest record. A branch scores one if its constraints are
feasible without an incident, or the fraction of (coalition member, public scene time) creator
assignments that are feasible after an incident. The score sums these gate-weighted indicators.

Publicly known crew (dead agents who were not publicly ejected, and any revealed crew) cannot be
coalition members. Publicly revealed coalition members must belong to each candidate coalition.
Private positive and negative observations, accepted claims, and public marker location are joint
hard constraints. No pseudocount restores an impossible candidate. If all candidates fail, the
rule returns zero scores and abstains when skipping is allowed. Manufactured corroboration has
no general immunity guarantee: claims can change the feasibility of the optional honest gates.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from social_collusion.env.claims import claim_constraints, satisfiable
from social_collusion.env.enums import N_HEADS, ClaimType, EventType, Head, Phase, ResponseType
from social_collusion.env.knowledge import UNKNOWN, ObserverKnowledge, observed_horizon, scene_window
from social_collusion.env.state import DEAD, GameState
from social_collusion.policies.base import first_legal
from social_collusion.policies.crew_aggregation import own_evidence
from social_collusion.policies.scripted_crew import TruthfulCrew

# Compatibility name only: impossible hypotheses always have zero weight.
EPS = 0.0
DEFAULT_HONESTY = 0.5
DEFAULT_THRESHOLD = 0.5


def _public_roles(state):
    """Infer roles only from public alive status and the public ejection record."""
    ejected, revealed_crew, revealed_coalition = set(), set(), set()
    for event in state.events:
        if event.type != EventType.AGENT_EJECTED:
            continue
        agent = event.payload.get("agent")
        if agent is None:
            continue
        ejected.add(int(agent))
        role = event.payload.get("role")
        if role == "CREW":
            revealed_crew.add(int(agent))
        elif role == "COALITION":
            revealed_coalition.add(int(agent))
    murdered = {j for j, alive in enumerate(state.alive) if not alive and j not in ejected}
    return murdered | revealed_crew, revealed_coalition


def _public_inactive_since(state):
    """Earliest guaranteed inactive times inferable from public events and alive status.

    Round-start survivors distinguish earlier deaths from this round's murder. A forensic
    report fixes the current victim's final active instant; otherwise only absence by the
    report is asserted. Ejection at the end of this meeting does not change earlier positions.
    """
    start_alive = set(range(state.config.n_agents))
    ejected = set()
    for event in state.events:
        if event.type == EventType.ROUND_STARTED:
            start_alive = set(map(int, event.payload["alive"]))
            ejected.clear()
        elif event.type == EventType.AGENT_EJECTED and event.payload.get("agent") is not None:
            ejected.add(int(event.payload["agent"]))
    inactive = {j: 0 for j in range(state.config.n_agents) if j not in start_alive}
    if state.report_turn >= 0:
        since = (int(state.incident_time) + 1
                 if state.config.incident_time_public and state.incident_time >= 0
                 else int(state.report_turn))
        for j in start_alive:
            if not state.alive[j] and j not in ejected:
                inactive[j] = since
    return tuple(sorted(inactive.items()))


def _active_constraints(constraints, inactive):
    """Eliminate publicly absent subjects before the finite-room feasibility check."""
    out = []
    for c in constraints:
        kind = c[0]
        if kind in ("eq_const", "neq_const"):
            _, agent, time, _ = c
            absent = agent in inactive and time >= inactive[agent]
        elif kind in ("eq_pair", "neq_pair"):
            _, a, b, time = c
            absent = any(agent in inactive and time >= inactive[agent] for agent in (a, b))
        else:
            absent = False
        if absent:
            if kind in ("eq_const", "eq_pair"):
                return None
            continue  # An inactive agent is absent from every room and every living partner.
        out.append(c)
    return out


def coalition_posterior(
    state: GameState,
    voter: int,
    k: ObserverKnowledge | None = None,
    use_private: bool = True,
    honesty: float = DEFAULT_HONESTY,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized membership and creator *scores* (legacy function name).

    Positive total evidence gives membership sum n_coalition and creator sum one for a reported
    incident (zero without one). An inconsistent view gives two all-zero arrays, not a fallback
    distribution over impossible worlds. `honesty` is a declared gate weight, not an estimated
    behavioral frequency. Nonzero legacy smoothing is rejected.
    """
    if not 0.0 <= honesty <= 1.0:
        raise ValueError("honesty must lie in [0, 1]")
    if eps != 0.0:
        raise ValueError("impossible hypotheses cannot receive a smoothing pseudocount")
    cfg = state.config
    if not 0 <= voter < cfg.n_agents:
        raise ValueError("voter must be a valid agent")
    known_crew, known_coalition = _public_roles(state)
    known_crew.add(voter)  # This policy is the observer's reasoning conditional on being crew.
    marker = int(state.incident_room) if state.report_turn >= 0 else -1
    window = tuple(t for t in scene_window(state) if 0 <= t < cfg.n_times) if marker >= 0 else ()
    private = []
    if use_private:
        k = k or ObserverKnowledge.build(state, voter)
        upto = observed_horizon(state)
        for t in range(upto + 1):
            here = int(k.present[t])
            if here == DEAD:
                continue
            for j in range(cfg.n_agents):
                room = int(k.known[t, j])
                private.append(("eq_const", j, t, room) if room != UNKNOWN
                               else ("neq_const", j, t, here))
        if bool(state.marker_seen[voter]) and state.incident_room >= 0:
            private.append(("marker", int(state.incident_room)))
    if marker >= 0:
        private.append(("marker", marker))
    by_speaker = {}
    for claim in state.claims:
        if claim.claim_type != int(ClaimType.NO_INFORMATION):
            by_speaker.setdefault(int(claim.speaker), []).extend(claim_constraints(claim))
    records = tuple((speaker, tuple(sorted(set(cons), key=repr))) for speaker, cons in sorted(by_speaker.items()))
    # At terminal-state diagnostics a player ejected by this meeting was still a possible
    # creator. Current ballots identify that meeting's participants using public information.
    living = tuple(j for j in range(cfg.n_agents) if state.alive[j] or state.votes[j] >= 0)
    member, creator = _score_view(cfg, tuple(sorted(known_crew)), tuple(sorted(known_coalition)),
                                  living, marker, window, records,
                                  tuple(sorted(set(private), key=repr)), _public_inactive_since(state),
                                  float(honesty))
    # Cache results are immutable; callers receive independent arrays.
    return np.array(member, dtype=np.float64), np.array(creator, dtype=np.float64)


@lru_cache(maxsize=128)
def _score_view(cfg, known_crew, known_coalition, living, marker, window, records, private, inactive, honesty):
    n, m = cfg.n_agents, cfg.n_coalition
    candidates = [j for j in range(n) if j not in known_crew]
    by_speaker = dict(records)
    member = np.zeros(n, dtype=np.float64)
    creator = np.zeros(n, dtype=np.float64)
    total = 0.0
    memo = {}
    inactive = dict(inactive)

    def feasible(excluded, j=None, t=None):
        key = excluded, j, t
        if key not in memo:
            cons = list(private)
            cons.extend(c for speaker, cs in records if speaker not in excluded for c in cs)
            if j is not None:
                cons.append(("eq_const", j, t, marker))
            cons = _active_constraints(cons, inactive)
            memo[key] = cons is not None and satisfiable(cons, cfg)
        return memo[key]

    for hypothesis in itertools.combinations(candidates, m):
        if not set(known_coalition).issubset(hypothesis):
            continue
        spoke = [j for j in hypothesis if j in by_speaker]
        # The branch dropping every optional claim is a superset of all other branches.
        if not feasible(tuple(spoke)):
            continue
        weight = 0.0
        creator_weight = np.zeros(n, dtype=np.float64)
        for bits in itertools.product((False, True), repeat=len(spoke)):
            branch = np.prod([honesty if bit else 1.0-honesty for bit in bits])
            if branch == 0.0:
                continue
            excluded = tuple(j for j, bit in zip(spoke, bits) if not bit)
            if not feasible(excluded):
                continue
            if marker >= 0 and window:
                for j in hypothesis:
                    if j not in living:
                        continue
                    contribution = branch * sum(feasible(excluded, j, t) for t in window) / (m * len(window))
                    creator_weight[j] += contribution
                    weight += contribution
            else:
                weight += branch
        if weight == 0.0:
            continue
        total += weight
        for j in hypothesis:
            member[j] += weight
        creator += creator_weight
    if total > 0.0:
        member /= total
        creator /= total
    return tuple(member), tuple(creator)


class HypothesisCrew(TruthfulCrew):
    """Truthful speaker that accuses and votes from normalized feasibility scores.

    Ejects the agent with the highest membership score when that score exceeds
    `threshold` (a declared decision threshold, not calibrated probability); otherwise skips. Ties in membership
    are broken by the creator score, then by the voter's own grounded evidence.
    """

    name = "hypothesis_crew"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, honesty: float = DEFAULT_HONESTY, **kw):
        kw.setdefault("rule", "mean")
        super().__init__(threshold=threshold, **kw)
        self.honesty = honesty

    def _target(self, state: GameState, voter: int) -> int | None:
        cfg = state.config
        member, creator = coalition_posterior(state, voter, use_private=not self.blind, honesty=self.honesty)
        alive = [j for j in range(cfg.n_agents) if state.alive[j] and j != voter]
        if not alive:
            return None
        own = own_evidence(state, voter) if not self.blind else np.zeros(cfg.n_agents)
        best = max(alive, key=lambda j: (member[j], creator[j], own[j], -j))
        if member[best] <= self.threshold:
            return None
        return best

    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        cfg = state.config
        tgt = self._target(state, voter)
        if tgt is None:
            return cfg.n_agents if cfg.allow_skip_vote else first_legal_candidate(state, voter)
        return tgt

    def act_single(self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
        phase = Phase(state.phase)
        if phase != Phase.RESPONSE_ROUND:
            return super().act_single(state, agent, masks, rng)
        out = np.zeros(N_HEADS, dtype=np.int64)
        for h, mk in enumerate(masks):
            if not mk[0]:
                out[h] = first_legal(mk)
        if self.accuse:
            tgt = self._target(state, agent)
            if tgt is not None:
                out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
                out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], tgt)
            else:
                out[Head.RESPONSE_TYPE] = int(ResponseType.NEUTRAL)
        return out


def first_legal_candidate(state: GameState, voter: int) -> int:
    for j in range(state.config.n_agents):
        if state.alive[j] and j != voter:
            return j
    return state.config.n_agents


#: pseudo-rule name used by the F2 sweep and the static evaluations
HYPOTHESIS_RULE = "hypothesis"


def crew_for_rule(rule: str, **kw) -> TruthfulCrew:
    """A truthful crew that aggregates testimony with `rule`; `hypothesis` selects this module."""
    if rule == HYPOTHESIS_RULE:
        return HypothesisCrew(**kw)
    return TruthfulCrew(rule=rule, **kw)


__all__ = ["HypothesisCrew", "coalition_posterior", "crew_for_rule", "HYPOTHESIS_RULE", "DEFAULT_HONESTY", "DEFAULT_THRESHOLD"]
