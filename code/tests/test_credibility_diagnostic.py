"""Realized credibility diagnostics must describe the people who could vote."""

import importlib.util
from pathlib import Path

import numpy as np

from social_collusion.config import load_env_config
from social_collusion.env.enums import ClaimType, Role
from social_collusion.env.state import Claim
from social_collusion.env.transition import reset


def _module():
    path = Path(__file__).resolve().parents[1] / "analysis/credibility_weight_diagnostic.py"
    spec = importlib.util.spec_from_file_location("credibility_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ejected_honest_voter_keeps_the_same_pre_vote_diagnostic():
    module = _module()
    cfg = load_env_config("meeting_only").with_(n_agents=7)
    state = reset(cfg, np.random.default_rng(7))
    voters = np.flatnonzero(state.alive)
    state.votes[voters] = cfg.n_agents
    state.claims = [Claim(speaker=int(i), claim_type=int(ClaimType.NO_INFORMATION)) for i in voters]
    expected = module.voter_records(state, module.ca.TAU_SOFT)
    honest = np.flatnonzero(state.alive & (state.roles == int(Role.CREW)))
    state.ejected = int(honest[0])
    state.alive[state.ejected] = False
    actual = module.voter_records(state, module.ca.TAU_SOFT)
    assert len(actual) == len(honest)
    assert actual == expected
    assert not state.alive[state.ejected]
    assert not state.alive[state.incident_victim]


def test_top_pick_excludes_agents_dead_before_the_meeting(monkeypatch):
    module = _module()
    cfg = load_env_config("meeting_only").with_(n_agents=7)
    state = reset(cfg, np.random.default_rng(9))
    voters = np.flatnonzero(state.alive)
    state.votes[voters] = cfg.n_agents
    state.claims = [Claim(speaker=int(i), claim_type=int(ClaimType.NO_INFORMATION)) for i in voters]
    coalition_target = int(state.coalition[0])
    victim = state.incident_victim
    scores = np.zeros(cfg.n_agents)
    scores[coalition_target], scores[victim] = 1.0, 100.0
    monkeypatch.setattr(module.ca, "own_evidence", lambda *args: scores.copy())
    records = module.voter_records(state, module.ca.TAU_SOFT)
    assert records
    assert all(r["mean_top_is_coalition"] == 1.0 and r["soft_top_is_coalition"] == 1.0 for r in records)
