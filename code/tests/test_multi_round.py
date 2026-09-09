"""Multi-round game loop: gated behind max_rounds, bit-identical at max_rounds=1."""
from __future__ import annotations

import numpy as np

from social_collusion.config import load_env_config
from social_collusion.env import VecEnv
from social_collusion.env.transition import living_counts
from social_collusion.policies import make_pair


def _run(na, mr, seed=0, n=200):
    cfg = load_env_config("full_short_game").with_(n_agents=na, n_coalition=2, max_rounds=mr)
    return VecEnv(cfg, 64, seed=seed).run_episodes(make_pair("truthful", "alibi"), n)


def test_single_round_default_stays_round_zero():
    # max_rounds=1 must never loop: every game ends in its first round.
    for s in _run(9, 1):
        assert s.round == 0
        assert s.done


def test_multi_round_loops_and_terminates():
    states = _run(9, 6)
    rounds = np.array([s.round for s in states])
    assert rounds.max() >= 2, "a bigger game should reach at least a third round sometimes"
    for s in states:
        assert s.done
        coal, crew = living_counts(s)
        # terminal iff the coalition is wiped, at parity, or the round cap was hit
        assert coal == 0 or coal >= crew or s.round + 1 >= s.config.max_rounds


def test_reputation_and_round_persist_shape():
    for s in _run(9, 4, n=50):
        assert s.reputation.shape == (s.config.n_agents,)
        assert 0 <= s.round < s.config.max_rounds


def test_max_rounds_one_is_bit_identical():
    # the new machinery must not perturb the original single-round outcomes
    base = load_env_config("full_short_game").with_(n_agents=9, n_coalition=2)
    a = VecEnv(base, 64, seed=3).run_episodes(make_pair("truthful", "alibi"), 200)
    b = VecEnv(base.with_(max_rounds=1), 64, seed=3).run_episodes(make_pair("truthful", "alibi"), 200)
    assert [s.ejected for s in a] == [s.ejected for s in b]
    assert [int(s.incident_creator) for s in a] == [int(s.incident_creator) for s in b]
