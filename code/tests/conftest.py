from __future__ import annotations

import numpy as np
import pytest

from social_collusion.config import EnvConfig, load_env_config
from social_collusion.env import Phase, compute_masks, reset, sample_masked, transition
from social_collusion.seeding import episode_rng


@pytest.fixture(scope="session")
def cfg() -> EnvConfig:
    return load_env_config("meeting_only")


@pytest.fixture(scope="session")
def cfg_spatial() -> EnvConfig:
    return load_env_config("one_incident")


def play_random(cfg: EnvConfig, seed: int, episode: int = 0):
    """Random legal play to TERMINAL; returns (terminal state, recorded joint actions)."""
    rng = episode_rng(seed, episode)
    st = reset(cfg, rng, seed=seed)
    actions = []
    while st.phase != int(Phase.TERMINAL):
        a = sample_masked(compute_masks(st), rng)
        actions.append(np.array(a, copy=True))
        st = transition(st, a, rng, inplace=True, validate=True).state
    return st, actions


@pytest.fixture(scope="session")
def scripted_states(cfg):
    from social_collusion.env import VecEnv
    from social_collusion.policies import make_pair

    return VecEnv(cfg, 32, seed=1234).run_episodes(make_pair("truthful", "alibi"), 200)
