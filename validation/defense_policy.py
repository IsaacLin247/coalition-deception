"""Supplementary decision rules over the unchanged hypothesis feasibility scores.

This module deliberately imports the frozen benchmark implementation. It does not patch
the solver, information set, action masks, training collector, or ballot aggregation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from social_collusion.env.knowledge import ObserverKnowledge, observed_horizon
from social_collusion.env.state import GameState
from social_collusion.policies.crew_aggregation import own_evidence
from social_collusion.policies.hypothesis_crew import (
    HypothesisCrew,
    _public_inactive_since,
    _public_roles,
    coalition_posterior,
    first_legal_candidate,
)

TieBreak = Literal["index", "random", "skip"]
RANDOMIZATION_VERSION = "observer-view-sha256-pcg64-v1"


@dataclass(frozen=True)
class DefenseSpec:
    """A prospectively specified setting; the scores are not calibrated probabilities."""

    name: str
    tie_break: TieBreak = "index"
    threshold: float = 0.5
    honesty: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if self.tie_break not in ("index", "random", "skip"):
            raise ValueError("tie_break must be index, random, or skip")
        for name in ("threshold", "honesty"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DefenseDecision:
    """Inspect a decision without advancing the policy or environment RNG."""

    target: int | None
    tied_candidates: tuple[int, ...]
    top_membership: float | None
    above_threshold: bool


def _public_claim(claim) -> tuple:
    # Claim.to_dict() also exports privileged correctness labels. Never use it here.
    return (
        int(claim.speaker), int(claim.claim_type),
        None if claim.subject is None else int(claim.subject),
        None if claim.room is None else int(claim.room),
        None if claim.time is None else int(claim.time),
        bool(claim.polarity), int(claim.confidence), int(claim.order),
    )


def observer_view_key(state: GameState, voter: int, *, blind: bool = False) -> dict:
    """Only the permitted evidence shared by response and vote decisions.

    Responses, cast ballots, phase, and sequential speaking position are omitted: the
    hypothesis rule does not use them to rank living candidates, and including them
    would spuriously redraw the same observer's tie between accusation and vote.
    The episode's hidden simulator seed is deliberately absent.
    """
    cfg = state.config
    known_crew, known_coalition = _public_roles(state)
    marker_public = state.report_turn >= 0
    private = None
    if not blind:
        knowledge = ObserverKnowledge.build(state, voter)
        upto = observed_horizon(state) + 1
        marker_seen = bool(state.marker_seen[voter])
        private = {
            "present": knowledge.present[:upto].tolist(),
            "known": knowledge.known[:upto].tolist(),
            "marker_room": int(state.incident_room) if marker_seen else None,
        }
    return {
        "n_agents": int(cfg.n_agents),
        "n_coalition": int(cfg.n_coalition),
        "voter": int(voter),
        "round": int(state.round),
        "alive": state.alive.tolist(),
        "public_roles": [sorted(known_crew), sorted(known_coalition)],
        "public_inactive_since": _public_inactive_since(state),
        "report_turn": int(state.report_turn),
        "marker_room": int(state.incident_room) if marker_public else None,
        "incident_time": int(state.incident_time)
        if marker_public and cfg.incident_time_public else None,
        "claims": [_public_claim(claim) for claim in state.claims],
        "private": private,
    }


class ValidatedHypothesisCrew(HypothesisCrew):
    """Apply an index, random, or abstaining final tie rule to frozen scores.

    Random ties use a separately assigned replicate seed and observer-view key. This
    is a fixed pseudorandom function per replicate, not a new draw on every call.
    Identical views get identical targets, independent of batch or diagnostic order.
    """

    def __init__(self, spec: DefenseSpec, *, tie_seed: int = 0,
                 record_decisions: bool = False, **kwargs) -> None:
        if not isinstance(tie_seed, (int, np.integer)) or isinstance(tie_seed, bool):
            raise ValueError("tie_seed must be an integer independent of hidden game state")
        if "threshold" in kwargs or "honesty" in kwargs:
            raise ValueError("threshold and honesty must be set in DefenseSpec")
        super().__init__(threshold=spec.threshold, honesty=spec.honesty, **kwargs)
        self.spec = spec
        self.tie_seed = int(tie_seed)
        self.name = spec.name
        self.record_decisions = bool(record_decisions)
        self.audit_records: list[dict] = []

    def _random_target(self, state: GameState, voter: int, tied: tuple[int, ...]) -> int:
        payload = {
            "version": RANDOMIZATION_VERSION,
            "tie_seed": self.tie_seed,
            "view": observer_view_key(state, voter, blind=self.blind),
            "tied_candidates": [int(j) for j in tied],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(blob.encode("utf-8")).digest()
        # An explicit generator avoids sharing or advancing the engine's RNG stream.
        local_rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
        return tied[int(local_rng.integers(len(tied)))]

    def decision(self, state: GameState, voter: int) -> DefenseDecision:
        member, creator = coalition_posterior(
            state, voter, use_private=not self.blind, honesty=self.honesty,
        )
        alive = tuple(j for j in range(state.config.n_agents) if state.alive[j] and j != voter)
        if not alive:
            return DefenseDecision(None, (), None, False)
        own = own_evidence(state, voter) if not self.blind else np.zeros(state.config.n_agents)
        ranks = {j: (member[j], creator[j], own[j]) for j in alive}
        best_rank = max(ranks.values())
        # Deliberately exact tuple equality: near-equal scores are not new ties.
        tied = tuple(j for j in alive if ranks[j] == best_rank)
        top_membership = float(best_rank[0])
        above = top_membership > self.threshold
        if not above or (len(tied) > 1 and self.spec.tie_break == "skip"):
            return DefenseDecision(None, tied, top_membership, above)
        if len(tied) > 1 and self.spec.tie_break == "random":
            target = self._random_target(state, voter, tied)
        else:
            target = tied[0]
        return DefenseDecision(target, tied, top_membership, above)

    def _target(self, state: GameState, voter: int) -> int | None:
        return self.decision(state, voter).target


    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        decision = self.decision(state, voter)
        cfg = state.config
        if decision.target is None:
            vote = cfg.n_agents if cfg.allow_skip_vote else first_legal_candidate(state, voter)
        else:
            vote = decision.target
        if self.record_decisions:
            self.audit_records.append({
                # Diagnostic join only: VecEnv advances retained states in place.
                # The runner strips this transient token before persistence.
                "_state_token": id(state),
                "round": int(state.round),
                "voter": int(voter),
                "n_agents": int(cfg.n_agents),
                "top_membership": decision.top_membership,
                "tie_count": len(decision.tied_candidates),
                "tied_candidates": list(decision.tied_candidates),
                "threshold": float(self.threshold),
                "honesty": float(self.honesty),
                "tie_break": self.spec.tie_break,
                "above_threshold": decision.above_threshold,
                "proposed_target": decision.target,
                "target": int(vote) if vote < cfg.n_agents else None,
                "skip": bool(vote == cfg.n_agents),
            })
        return int(vote)


def make_defense(spec: DefenseSpec | dict, seed: int | None = None, *,
                 tie_seed: int | None = None, **kwargs) -> ValidatedHypothesisCrew:
    """Construct a policy directly from a protocol JSON object or DefenseSpec."""
    if isinstance(spec, dict):
        spec = DefenseSpec(**spec)
    if seed is not None and tie_seed is not None:
        raise ValueError("provide seed or tie_seed, not both")
    resolved_seed = tie_seed if tie_seed is not None else (0 if seed is None else seed)
    return ValidatedHypothesisCrew(spec, tie_seed=resolved_seed, **kwargs)


__all__ = [
    "DefenseSpec", "DefenseDecision", "ValidatedHypothesisCrew", "make_defense",
    "observer_view_key", "RANDOMIZATION_VERSION",
]
