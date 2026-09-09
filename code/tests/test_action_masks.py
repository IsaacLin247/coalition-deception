"""Action masks: never empty, never cross-head, and semantically correct."""

from __future__ import annotations

import numpy as np

from social_collusion.env import Phase, compute_masks, reset, sample_masked, transition
from social_collusion.env.action_masks import active_agents, head_sizes, is_legal
from social_collusion.env.enums import N_HEADS, Head, RoomAction
from social_collusion.env.state import DEAD, empty_joint_action
from social_collusion.seeding import episode_rng


def _walk(cfg, seed=0, episode=0):
    rng = episode_rng(seed, episode)
    st = reset(cfg, rng, seed=seed)
    while st.phase != int(Phase.TERMINAL):
        yield st
        st = transition(st, sample_masked(compute_masks(st), rng), rng, inplace=True).state
    yield st


def test_no_mask_is_ever_empty(cfg, cfg_spatial):
    for c in (cfg, cfg_spatial):
        for e in range(15):
            for st in _walk(c, seed=2001, episode=e):
                for h, m in enumerate(compute_masks(st)):
                    assert m.any(axis=1).all(), f"empty mask on head {Head(h).name}"


def test_index_zero_is_always_legal_for_inactive_agents(cfg):
    for e in range(10):
        for st in _walk(cfg, seed=2003, episode=e):
            masks = compute_masks(st)
            act = active_agents(st)
            for i in np.flatnonzero(~act):
                for h in range(N_HEADS):
                    assert masks[h][i, 0]
                    assert masks[h][i].sum() == 1, "inactive agents must have a single option"


def test_every_legal_combination_is_accepted(cfg):
    """No cross-head constraints: any product of per-head legal choices must be legal."""
    rng = np.random.default_rng(0)
    for e in range(10):
        for st in _walk(cfg, seed=2005, episode=e):
            if st.phase == int(Phase.TERMINAL):
                continue
            masks = compute_masks(st)
            for _ in range(8):
                a = empty_joint_action(cfg)
                for i in range(cfg.n_agents):
                    for h in range(N_HEADS):
                        legal = np.flatnonzero(masks[h][i])
                        a[i, h] = int(rng.choice(legal))
                ok, why = is_legal(st, a)
                assert ok, why


def test_head_sizes_match_masks(cfg):
    sizes = head_sizes(cfg)
    st = next(_walk(cfg, seed=2007))
    for h, m in enumerate(compute_masks(st)):
        assert m.shape == (cfg.n_agents, sizes[h])


def test_claim_masks_exclude_self_and_time_zero(cfg):
    for e in range(10):
        for st in _walk(cfg, seed=2009, episode=e):
            if st.phase != int(Phase.CLAIM_ROUND):
                continue
            masks = compute_masks(st)
            speaker = int(st.speaking_order[st.sub_step])
            assert not masks[Head.CLAIM_SUBJECT][speaker, speaker]
            assert not masks[Head.CLAIM_TIME][speaker, 0], "t=0 is excluded so ENTER/LEAVE have t-1"
            assert masks[Head.CLAIM_TIME][speaker, 1 : st.horizon + 1].all()
            if st.horizon + 1 < cfg.n_times:
                assert not masks[Head.CLAIM_TIME][speaker, st.horizon + 1 :].any()


def test_vote_masks(cfg):
    for e in range(10):
        for st in _walk(cfg, seed=2011, episode=e):
            if st.phase != int(Phase.VOTE):
                continue
            masks = compute_masks(st)
            for i in np.flatnonzero(active_agents(st)):
                v = masks[Head.VOTE][i]
                assert not v[i], "self-vote must be masked out"
                assert v[cfg.n_agents], "SKIP must be available"
                for j in range(cfg.n_agents):
                    if not st.alive[j]:
                        assert not v[j], "cannot vote for an inactive agent"


def test_response_targets_are_alive_and_not_self(cfg):
    for e in range(10):
        for st in _walk(cfg, seed=2013, episode=e):
            if st.phase != int(Phase.RESPONSE_ROUND):
                continue
            masks = compute_masks(st)
            i = int(st.speaking_order[st.sub_step])
            t = masks[Head.RESPONSE_TARGET][i]
            assert not t[i]
            for j in range(cfg.n_agents):
                if not st.alive[j]:
                    assert not t[j]


def test_movement_masks_are_adjacency(cfg_spatial):
    for e in range(10):
        for st in _walk(cfg_spatial, seed=2015, episode=e):
            if st.phase != int(Phase.FREE_PLAY):
                continue
            masks = compute_masks(st)
            for i in np.flatnonzero(active_agents(st)):
                here = int(st.positions[st.turn, i])
                allowed = set(np.flatnonzero(masks[Head.MOVE][i]).tolist())
                assert allowed == {here} | set(cfg_spatial.neighbors(here))


def test_crew_can_never_create_an_incident(cfg_spatial):
    from social_collusion.env.enums import Role

    for e in range(15):
        for st in _walk(cfg_spatial, seed=2017, episode=e):
            if st.phase != int(Phase.FREE_PLAY):
                continue
            masks = compute_masks(st)
            for i in range(cfg_spatial.n_agents):
                if st.roles[i] == int(Role.CREW):
                    assert not masks[Head.ROOM_ACTION][i, int(RoomAction.CREATE_INCIDENT)]


def test_report_requires_a_visible_marker(cfg_spatial):
    for e in range(15):
        for st in _walk(cfg_spatial, seed=2019, episode=e):
            if st.phase != int(Phase.FREE_PLAY):
                continue
            masks = compute_masks(st)
            for i in np.flatnonzero(active_agents(st)):
                if masks[Head.ROOM_ACTION][i, int(RoomAction.REPORT)]:
                    assert st.incident_time >= 0
                    assert int(st.positions[st.turn, i]) == st.incident_room
                    assert st.turn > st.incident_time


def test_dead_agents_are_never_active(cfg):
    for e in range(10):
        for st in _walk(cfg, seed=2021, episode=e):
            act = active_agents(st)
            for i in range(cfg.n_agents):
                if not st.alive[i]:
                    assert not act[i]


def test_symbol_head_is_coalition_only(cfg):
    from social_collusion.env.enums import Role

    chan = cfg.with_(enable_symbol_channel=True, symbol_mode="learned", n_symbols=4)
    for e in range(10):
        for st in _walk(chan, seed=2023, episode=e):
            if st.phase != int(Phase.SYMBOL):
                continue
            masks = compute_masks(st)
            for i in range(chan.n_agents):
                n_legal = int(masks[Head.SYMBOL][i].sum())
                if st.roles[i] == int(Role.COALITION) and st.alive[i]:
                    assert n_legal == chan.n_symbols
                else:
                    assert n_legal == 1


def test_positions_of_dead_agents_are_dead(cfg):
    for e in range(10):
        st = list(_walk(cfg, seed=2025, episode=e))[-1]
        assert (st.positions[st.incident_time + 1 :, st.incident_victim] == DEAD).all()
