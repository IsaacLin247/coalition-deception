"""Scripted baselines and the supervised belief listener."""

from social_collusion.policies.base import AgentRouter, BasePolicy, RoleRouter
from social_collusion.policies.random_policy import NoOpPolicy, RandomPolicy, SkipVotePolicy
from social_collusion.policies.registry import (
    COALITION_POLICIES,
    CREW_POLICIES,
    make_coalition,
    make_crew,
    make_pair,
)
from social_collusion.policies.scripted_coalition import (
    SCRIPTED_COALITIONS,
    AlibiCoalition,
    Framer,
    LoneLiar,
    Sacrifice,
    SubtleSupport,
    TruthfulCoalition,
)
from social_collusion.policies.scripted_crew import (
    BayesianListener,
    ContradictionVoter,
    RandomVoterCrew,
    TruthfulCrew,
    belief_over_creator,
    suspicion_scores,
)

__all__ = [
    "BasePolicy",
    "RoleRouter",
    "AgentRouter",
    "RandomPolicy",
    "NoOpPolicy",
    "SkipVotePolicy",
    "TruthfulCrew",
    "ContradictionVoter",
    "BayesianListener",
    "RandomVoterCrew",
    "suspicion_scores",
    "belief_over_creator",
    "TruthfulCoalition",
    "LoneLiar",
    "AlibiCoalition",
    "Framer",
    "SubtleSupport",
    "Sacrifice",
    "SCRIPTED_COALITIONS",
    "CREW_POLICIES",
    "COALITION_POLICIES",
    "make_crew",
    "make_coalition",
    "make_pair",
]
