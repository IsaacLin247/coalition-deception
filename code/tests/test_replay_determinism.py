"""Determinism guarantee of plan sec.12.3.

Given (config, seed, ordered actions) the episode must reproduce with identical per-step state
hashes. This is the property that makes every counterfactual comparison meaningful: if replay
drifted, an intervention's effect could not be separated from simulator noise.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env import Phase, compute_masks, reset, transition
from social_collusion.policies import make_pair
from social_collusion.replay import (
    read,
    read_many,
    rebuild_state,
    record_episode,
    resimulate,
    validate_record,
    write,
    write_many,
)
from social_collusion.seeding import episode_rng
from tests.conftest import play_random


def test_same_seed_same_actions_same_hashes(cfg):
    for e in range(10):
        st1, acts1 = play_random(cfg, seed=3001, episode=e)
        st2, acts2 = play_random(cfg, seed=3001, episode=e)
        assert st1.state_hash() == st2.state_hash()
        assert st1.event_log_hash() == st2.event_log_hash()
        assert all(np.array_equal(a, b) for a, b in zip(acts1, acts2))


def test_different_seeds_diverge(cfg):
    a, _ = play_random(cfg, seed=3003, episode=0)
    b, _ = play_random(cfg, seed=3005, episode=0)
    assert a.state_hash() != b.state_hash()


def test_replay_reproduces_every_step_hash(cfg):
    pol = make_pair("truthful", "alibi")
    for e in range(8):
        _, rec = record_episode(cfg, pol, seed=3007, episode_index=e)
        state, hashes = resimulate(rec)
        assert state.state_hash() == rec.final_state_hash
        assert [s.state_hash for s in rec.steps] == hashes


def test_validate_record_accepts_and_rejects(cfg):
    pol = make_pair("truthful", "framer")
    _, rec = record_episode(cfg, pol, seed=3009, episode_index=1)
    assert validate_record(rec).ok
    rec.final_state_hash = "0" * 32
    assert not validate_record(rec).ok


def test_replay_roundtrips_through_disk(cfg, tmp_path):
    pol = make_pair("truthful", "alibi")
    _, rec = record_episode(cfg, pol, seed=3011, episode_index=2)
    p = write(rec, tmp_path / "ep.json")
    assert validate_record(read(p)).ok
    pgz = write(rec, tmp_path / "ep.json.gz")
    assert validate_record(read(pgz)).ok
    bundle = write_many([rec, rec], tmp_path / "bundle.jsonl")
    assert len(read_many(bundle)) == 2


def test_replay_is_portable_without_the_policy(cfg, tmp_path):
    """Acceptance gate of plan Stage 14: a replay must render from the file alone."""
    from social_collusion.env.render_text import render_state

    pol = make_pair("truthful", "alibi")
    _, rec = record_episode(cfg, pol, seed=3013, episode_index=3)
    p = write(rec, tmp_path / "portable.json")
    del pol
    state = rebuild_state(read(p))
    text = render_state(state, privileged=True)
    assert "outcome" in text and state.done


def test_inplace_and_pure_transitions_agree(cfg):
    rng_a = episode_rng(3015, 0)
    rng_b = episode_rng(3015, 0)
    st_a = reset(cfg, rng_a, seed=3015)
    st_b = reset(cfg, rng_b, seed=3015)
    from social_collusion.env import sample_masked

    while st_a.phase != int(Phase.TERMINAL):
        a = sample_masked(compute_masks(st_a), rng_a)
        st_a = transition(st_a, a, rng_a, inplace=True).state
        st_b = transition(st_b, a, rng_b, inplace=False).state
    assert st_a.state_hash() == st_b.state_hash()


def test_state_hash_is_sensitive(cfg):
    st, _ = play_random(cfg, seed=3017)
    h = st.state_hash()
    st.votes[0] = (int(st.votes[0]) + 1) % cfg.n_agents
    assert st.state_hash() != h


def test_episode_rng_indices_are_independent(cfg):
    hashes = {play_random(cfg, seed=3019, episode=e)[0].state_hash() for e in range(40)}
    assert len(hashes) > 30, "episode indices should give near-independent worlds"
