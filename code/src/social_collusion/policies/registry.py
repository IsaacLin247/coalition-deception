"""Name -> policy factory, so configs and scripts never import concrete classes."""

from __future__ import annotations

from typing import Any

from social_collusion.policies.base import BasePolicy, RoleRouter
from social_collusion.policies.random_policy import NoOpPolicy, RandomPolicy, SkipVotePolicy
from social_collusion.policies.scripted_coalition import (
    AlibiCoalition,
    Framer,
    LoneLiar,
    Sacrifice,
    SubtleSupport,
    TruthfulCoalition,
)
from social_collusion.policies.hypothesis_crew import HypothesisCrew
from social_collusion.policies.scripted_crew import (
    BayesianListener,
    ContradictionVoter,
    RandomVoterCrew,
    TruthfulCrew,
)

CREW_POLICIES: dict[str, type[BasePolicy]] = {
    "random": RandomPolicy,
    "noop": NoOpPolicy,
    "skip": SkipVotePolicy,
    "truthful": TruthfulCrew,
    "contradiction": ContradictionVoter,
    "bayesian": BayesianListener,
    "hypothesis": HypothesisCrew,
    "random_voter": RandomVoterCrew,
}

COALITION_POLICIES: dict[str, type[BasePolicy]] = {
    "random": RandomPolicy,
    "noop": NoOpPolicy,
    "skip": SkipVotePolicy,
    "truthful": TruthfulCoalition,
    "lone_liar": LoneLiar,
    "alibi": AlibiCoalition,
    "framer": Framer,
    "subtle_support": SubtleSupport,
    "sacrifice": Sacrifice,
}


def make_crew(name: str, **kw: Any) -> BasePolicy:
    if name not in CREW_POLICIES:
        raise KeyError(f"unknown crew policy {name!r}; have {sorted(CREW_POLICIES)}")
    return CREW_POLICIES[name](**kw)


def make_coalition(name: str, **kw: Any) -> BasePolicy:
    if name not in COALITION_POLICIES:
        raise KeyError(f"unknown coalition policy {name!r}; have {sorted(COALITION_POLICIES)}")
    return COALITION_POLICIES[name](**kw)


def make_pair(crew: str = "truthful", coalition: str = "alibi", **kw: Any) -> RoleRouter:
    """The standard scripted match-up used by dataset generation and acceptance gates."""
    crew_kw = kw.pop("crew_kwargs", {})
    coal_kw = kw.pop("coalition_kwargs", {})
    return RoleRouter(make_crew(crew, **crew_kw), make_coalition(coalition, **coal_kw))


__all__ = [
    "CREW_POLICIES",
    "COALITION_POLICIES",
    "make_crew",
    "make_coalition",
    "make_pair",
]
