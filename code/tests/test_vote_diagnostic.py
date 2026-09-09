"""Ballot diagnostics must include players removed after casting their ballot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env.enums import Role
from social_collusion.env.transition import reset


def _diagnostic():
    path = Path(__file__).resolve().parents[1] / "experiments/dependence/hypothesis_vote_diagnostic.py"
    spec = importlib.util.spec_from_file_location("vote_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    cfg = load_env_config("meeting_only").with_(n_agents=7, n_coalition=2)
    state = reset(cfg, np.random.default_rng(42), seed=42)
    state.roles[:] = int(Role.CREW)
    state.roles[:2] = int(Role.COALITION)
    state.alive[:] = True
    state.votes[:] = cfg.n_agents
    state.incident_creator = -1
    state.incident_room = -1
    state.claims = []
    return state


def test_ejected_crew_ballot_can_change_counterfactual_plurality(monkeypatch):
    module = _diagnostic()
    state = _state()
    # Coalition elects 2. Crew alone would tie 2 and 4; dropping 2's vote
    # incorrectly makes 2 the crew's unique plurality winner.
    state.votes[:] = [2, 2, 4, 2, 7, 7, 7]
    state.ejected = 2
    state.alive[2] = False
    seen = []

    def posterior(s, voter):
        seen.append((voter, bool(s.alive[2])))
        return np.zeros(7), np.zeros(7)

    monkeypatch.setattr(module, "coalition_posterior", posterior)
    result = module.decompose([state])
    assert result["coalition_decided_false_ejections"] == 1.0
    assert result["crew_skip_rate"] == pytest.approx(3 / 5)
    assert (2, True) in seen
    assert not state.alive[2]  # diagnostics leave the stored outcome unchanged


def test_same_target_denominator_includes_coalition_ejection(monkeypatch):
    module = _diagnostic()
    state = _state()
    state.votes[:] = [2, 2, 0, 0, 0, 0, 0]
    state.ejected = 0
    state.alive[0] = False
    monkeypatch.setattr(module, "coalition_posterior", lambda s, v: (np.zeros(7), np.zeros(7)))
    result = module.decompose([state])
    assert result["coalition_same_target_rate"] == 1.0


@pytest.mark.parametrize("private_history", [False, True])
def test_diagnostic_replays_saved_legacy_and_full_history_layouts(tmp_path, private_history):
    import torch

    from social_collusion.env import VecEnv
    from social_collusion.policies.base import RoleRouter
    from social_collusion.rl.torch_policy import TorchPolicy, evaluation_copy

    module = _diagnostic()
    cfg = load_env_config("full_short_game").with_(
        n_agents=5, max_rounds=1, actor_private_history=private_history,
    )
    checkpoint = TorchPolicy(cfg, hidden_dim=16).save(tmp_path / "checkpoint.pt")
    if not private_history:
        # Archived checkpoints have no actor_private_history configuration key.
        blob = torch.load(checkpoint, weights_only=False)
        del blob["env_config"]["actor_private_history"]
        torch.save(blob, checkpoint)
    coal = module.load_role_checkpoint(checkpoint, Role.COALITION)
    replay_cfg = module.checkpoint_env_config(coal, 5)
    assert replay_cfg.actor_private_history == private_history
    policy = evaluation_copy(RoleRouter(module.crew_for_rule("mean"), coal), 782)
    states = VecEnv(replay_cfg, 2, seed=782).run_episodes(policy, 2)
    assert len(states) == 2
    assert all(state.config == cfg for state in states)
    with pytest.raises(ValueError, match="n_agents=7"):
        module.checkpoint_env_config(coal, 7)
