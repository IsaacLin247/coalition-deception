"""Actor observations must never contain what the actor cannot know (plan sec.8.2).

The strategy is adversarial: scramble every hidden field and assert the observation vector is
bit-identical. A leak of a single float would flip a bit here.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env.enums import Role
from social_collusion.env.observations import (
    build_actor_obs,
    build_critic_state,
    critic_dim,
    get_spec,
    last_seen,
    obs_dim,
    scramble_hidden_state,
)
from tests.conftest import play_random


def test_hidden_state_does_not_leak(cfg):
    rng = np.random.default_rng(0)
    for e in range(30):
        st, _ = play_random(cfg, seed=1001, episode=e)
        for i in range(cfg.n_agents):
            if st.roles[i] != int(Role.CREW):
                continue
            o1 = build_actor_obs(st, i).copy()
            o2 = build_actor_obs(scramble_hidden_state(st, i, rng), i)
            assert np.array_equal(o1, o2), f"observation leak for crew agent {i}"


def test_coalition_sees_only_its_own_partner(cfg):
    """A coalition member may see its partner (that is the condition), nothing more.

    With `partner_known` and a 2-member coalition, roles are not hidden from a coalition member -
    knowing your partner fixes everyone else's role by elimination - so roles are held fixed and
    the genuinely hidden fields (who did it, others' symbols and sightings, claim truth labels)
    are scrambled.
    """
    rng = np.random.default_rng(1)
    for e in range(20):
        st, _ = play_random(cfg, seed=1003, episode=e)
        for i in st.coalition:
            i = int(i)
            scrambled = scramble_hidden_state(st, i, rng, scramble_roles=False)
            o1 = build_actor_obs(st, i).copy()
            o2 = build_actor_obs(scrambled, i)
            assert np.array_equal(o1, o2)


def test_partner_mask_is_empty_when_partner_is_hidden(cfg):
    hidden = cfg.with_(partner_known=False)
    spec = get_spec(hidden)
    for e in range(10):
        st, _ = play_random(hidden, seed=1005, episode=e)
        for i in st.coalition:
            o = build_actor_obs(st, int(i))
            assert o[spec.slices["partner_mask"]].sum() == 0.0


def test_partner_mask_is_set_when_partner_is_known(cfg):
    spec = get_spec(cfg)
    for e in range(10):
        st, _ = play_random(cfg, seed=1007, episode=e)
        for i in st.coalition:
            o = build_actor_obs(st, int(i))
            block = o[spec.slices["partner_mask"]]
            assert block.sum() == 1.0
            assert int(np.argmax(block)) == st.partner_of(int(i))
        for i in st.crew:
            o = build_actor_obs(st, int(i))
            assert o[spec.slices["partner_mask"]].sum() == 0.0


def test_last_seen_only_reports_colocation(cfg):
    for e in range(20):
        st, _ = play_random(cfg, seed=1009, episode=e)
        for i in range(cfg.n_agents):
            rooms, times = last_seen(st, i)
            for j in range(cfg.n_agents):
                if j == i or times[j] < 0:
                    continue
                t = int(times[j])
                assert int(st.positions[t, i]) == int(st.positions[t, j]) == int(rooms[j])


def test_observation_shape_and_range(cfg):
    spec = get_spec(cfg)
    for e in range(10):
        st, _ = play_random(cfg, seed=1011, episode=e)
        for i in range(cfg.n_agents):
            o = build_actor_obs(st, i)
            assert o.shape == (obs_dim(cfg),) == (spec.size,)
            assert np.isfinite(o).all()
            assert o.min() >= 0.0 and o.max() <= 1.0


def test_ablation_zeroes_only_the_named_block(cfg):
    spec = get_spec(cfg)
    st, _ = play_random(cfg, seed=1013)
    o = build_actor_obs(st, 0)
    a = spec.ablate(o, ["transcript"])
    assert a[spec.slices["transcript"]].sum() == 0.0
    for name, sl in spec.slices.items():
        if name != "transcript":
            assert np.array_equal(a[sl], o[sl])


def test_critic_state_does_contain_privileged_information(cfg):
    """The mirror image: the centralised critic *should* see what the actor cannot."""
    rng = np.random.default_rng(2)
    st, _ = play_random(cfg, seed=1015)
    c1 = build_critic_state(st)
    c2 = build_critic_state(scramble_hidden_state(st, 0, rng))
    assert c1.shape == (critic_dim(cfg),)
    assert not np.array_equal(c1, c2), "critic state should depend on hidden roles/creator"


def test_dead_agents_get_a_well_formed_observation(cfg):
    for e in range(10):
        st, _ = play_random(cfg, seed=1017, episode=e)
        o = build_actor_obs(st, int(st.incident_victim))
        assert np.isfinite(o).all()
