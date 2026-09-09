"""PettingZoo conformance (plan Task 7 / sec.9.3) - part of CI, not a one-off check."""

from __future__ import annotations

import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

from social_collusion.env.pettingzoo_env import env as aec_env  # noqa: E402
from social_collusion.env.pettingzoo_env import parallel_env  # noqa: E402


def test_parallel_api(cfg):
    parallel_api_test(parallel_env(cfg), num_cycles=50)


def test_parallel_api_spatial(cfg_spatial):
    parallel_api_test(parallel_env(cfg_spatial), num_cycles=50)


def test_aec_conversion_runs(cfg):
    e = aec_env(cfg)
    e.reset(seed=0)
    steps = 0
    for agent in e.agent_iter(max_iter=200):
        _, _, term, trunc, _ = e.last()
        e.step(None if (term or trunc) else e.action_space(agent).sample())
        steps += 1
    assert steps > 0


def test_infos_carry_action_masks(cfg):
    e = parallel_env(cfg)
    _, infos = e.reset(seed=1)
    for a in e.agents:
        masks = infos[a]["action_mask"]
        assert len(masks) == len(e.action_space(a).nvec)
        for m, size in zip(masks, e.action_space(a).nvec):
            assert m.shape == (size,)
            assert m.sum() >= 1


def test_illegal_actions_are_clamped_and_counted(cfg):
    e = parallel_env(cfg)
    e.reset(seed=2)
    total_invalid = 0
    while e.agents:
        actions = {a: e.action_space(a).nvec - 1 for a in e.agents}  # deliberately extreme
        _, _, term, trunc, infos = e.step(actions)
        total_invalid += max(i.get("invalid_action_attempts", 0) for i in infos.values())
    assert total_invalid > 0, "extreme actions should have needed clamping"


def test_episode_terminates_and_empties_the_agent_set(cfg):
    e = parallel_env(cfg)
    e.reset(seed=3)
    n = 0
    while e.agents and n < 100:
        obs, rew, term, trunc, info = e.step(
            {a: e.action_space(a).sample() for a in e.agents}
        )
        n += 1
    assert not e.agents


def test_render_returns_public_text_only(cfg):
    e = parallel_env(cfg, render_mode="ansi")
    e.reset(seed=4)
    text = e.render()
    assert "PRIVILEGED" not in text


def test_state_is_the_privileged_critic_view(cfg):
    from social_collusion.env.observations import critic_dim

    e = parallel_env(cfg)
    e.reset(seed=5)
    s = e.state()
    assert s.shape == (critic_dim(cfg),)
    assert np.isfinite(s).all()
