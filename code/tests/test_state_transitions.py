"""Phase machine, incident rules, voting rules, and terminal accounting."""

from __future__ import annotations

import numpy as np
import pytest

from social_collusion.env import Phase, Role, compute_masks, reset, transition
from social_collusion.env.enums import Head, RoomAction
from social_collusion.env.rewards import coalition_favorable, terminal_rewards
from social_collusion.env.state import DEAD, empty_joint_action
from social_collusion.env.transition import IllegalActionError
from social_collusion.seeding import episode_rng
from tests.conftest import play_random


def test_meeting_only_phase_sequence(cfg):
    rng = episode_rng(0, 0)
    st = reset(cfg, rng, seed=0)
    assert st.phase == int(Phase.REPORT)
    seen = [Phase(st.phase).name]
    from social_collusion.env import sample_masked

    while st.phase != int(Phase.TERMINAL):
        st = transition(st, sample_masked(compute_masks(st), rng), rng, inplace=True).state
        seen.append(Phase(st.phase).name)
    assert seen[0] == "REPORT"
    assert "CLAIM_ROUND" in seen and "RESPONSE_ROUND" in seen and "VOTE" in seen
    assert seen[-1] == "TERMINAL"


def test_episode_length_is_fixed_for_meeting_only(cfg):
    lengths = set()
    for e in range(25):
        rng = episode_rng(5, e)
        st = reset(cfg, rng, seed=5)
        from social_collusion.env import sample_masked

        n = 0
        while st.phase != int(Phase.TERMINAL):
            st = transition(st, sample_masked(compute_masks(st), rng), rng, inplace=True).state
            n += 1
        lengths.add(n)
    # exactly one victim, so the speaking order is always n_agents-1 long
    assert lengths == {1 + 2 * (cfg.n_agents - 1) + 1}


def test_exactly_one_incident_with_a_crew_victim(cfg):
    for e in range(50):
        st, _ = play_random(cfg, seed=3, episode=e)
        assert st.incident_creator >= 0 and st.incident_victim >= 0
        assert st.roles[st.incident_creator] == int(Role.COALITION)
        assert st.roles[st.incident_victim] == int(Role.CREW)
        assert not st.alive[st.incident_victim] or st.ejected == st.incident_victim
        assert st.positions[st.incident_time, st.incident_victim] == st.incident_room
        assert (st.positions[st.incident_time + 1 :, st.incident_victim] == DEAD).all()


def test_incident_isolation_rule(cfg):
    """No second crew member may be present when the incident happens."""
    for e in range(60):
        st, _ = play_random(cfg, seed=9, episode=e)
        t, room = st.incident_time, st.incident_room
        crew_present = [
            j
            for j in range(cfg.n_agents)
            if st.roles[j] == int(Role.CREW) and int(st.positions[t, j]) == room
        ]
        assert crew_present == [st.incident_victim]


def test_victim_never_speaks_or_votes(cfg):
    for e in range(30):
        st, _ = play_random(cfg, seed=11, episode=e)
        v = st.incident_victim
        assert v not in list(st.speaking_order)
        assert st.votes[v] == -1
        assert all(c.speaker != v for c in st.claims)


def test_plurality_and_tie_rules(cfg):
    ties = 0
    for e in range(120):
        st, _ = play_random(cfg, seed=17, episode=e)
        tally = np.zeros(cfg.n_agents, dtype=int)
        for i in range(cfg.n_agents):
            v = int(st.votes[i])
            if 0 <= v < cfg.n_agents:
                tally[v] += 1
        top = tally.max()
        winners = np.flatnonzero(tally == top)
        if top == 0 or (len(winners) > 1 and not cfg.tie_ejects):
            assert st.ejected == -1
            ties += 1
        else:
            assert st.ejected == int(winners[0])
            assert not st.alive[st.ejected]
    assert ties > 0, "random play should produce at least one tie/no-ejection"


def test_no_self_vote(cfg):
    for e in range(40):
        st, _ = play_random(cfg, seed=23, episode=e)
        for i in range(cfg.n_agents):
            assert int(st.votes[i]) != i


def test_terminal_rewards_are_zero_sum_in_sign(cfg):
    for e in range(40):
        st, _ = play_random(cfg, seed=29, episode=e)
        r = terminal_rewards(st)
        fav = coalition_favorable(st)
        for i in range(cfg.n_agents):
            if st.roles[i] == int(Role.CREW):
                assert r[i] == (-1.0 if fav else 1.0)
        assert set(np.abs(r).tolist()) == {1.0}


def test_shared_vs_individual_reward(cfg):
    shared = cfg.with_(shared_reward=True)
    indiv = cfg.with_(shared_reward=False)
    for e in range(40):
        st_s, _ = play_random(shared, seed=31, episode=e)
        st_i, _ = play_random(indiv, seed=31, episode=e)
        rs, ri = terminal_rewards(st_s), terminal_rewards(st_i)
        # the two engines see identical worlds; only the coalition's reward rule differs
        assert st_s.incident_creator == st_i.incident_creator
        for i in st_s.coalition:
            if st_s.ejected >= 0 and st_s.roles[st_s.ejected] == int(Role.COALITION):
                if int(i) != st_s.ejected:
                    # shared: both lose. individual: the survivor still wins.
                    assert rs[i] == -1.0 and ri[i] == 1.0


def test_illegal_action_raises_in_strict_mode(cfg):
    rng = episode_rng(0, 0)
    st = reset(cfg, rng, seed=0)
    st = transition(st, None, rng, inplace=True).state  # REPORT -> CLAIM_ROUND
    a = empty_joint_action(cfg)
    speaker = int(st.speaking_order[0])
    a[speaker, Head.CLAIM_SUBJECT] = speaker  # claiming about yourself is masked out
    with pytest.raises(IllegalActionError):
        transition(st, a, rng, validate=True)


def test_transition_is_pure_by_default(cfg):
    rng = episode_rng(0, 0)
    st = reset(cfg, rng, seed=0)
    before = st.state_hash()
    transition(st, None, rng, inplace=False, validate=True)
    assert st.state_hash() == before, "pure transition must not mutate its input"


def test_spatial_variant_runs_and_may_produce_no_incident(cfg_spatial):
    outcomes = []
    for e in range(40):
        st, _ = play_random(cfg_spatial, seed=41, episode=e)
        assert st.phase == int(Phase.TERMINAL)
        outcomes.append(st.incident_creator >= 0)
        if st.incident_creator >= 0:
            assert st.roles[st.incident_creator] == int(Role.COALITION)
    assert len(outcomes) == 40


def test_spatial_movement_respects_adjacency(cfg_spatial):
    for e in range(30):
        st, _ = play_random(cfg_spatial, seed=43, episode=e)
        for t in range(1, st.horizon + 1):
            for i in range(cfg_spatial.n_agents):
                a, b = int(st.positions[t - 1, i]), int(st.positions[t, i])
                if a == DEAD or b == DEAD:
                    continue
                assert a == b or b in cfg_spatial.neighbors(a)


def test_create_incident_requires_a_legal_target(cfg_spatial):
    """CREATE_INCIDENT is masked off exactly when no legal victim exists (no cross-head trap)."""
    from social_collusion.env.action_masks import valid_incident_targets

    rng = episode_rng(0, 0)
    st = reset(cfg_spatial, rng, seed=0)
    checked = 0
    from social_collusion.env import sample_masked

    while st.phase == int(Phase.FREE_PLAY):
        masks = compute_masks(st)
        for i in range(cfg_spatial.n_agents):
            legal_here = masks[Head.ROOM_ACTION][i, int(RoomAction.CREATE_INCIDENT)]
            has_target = valid_incident_targets(st, i, st.turn).any()
            assert legal_here == has_target
            checked += 1
        st = transition(st, sample_masked(masks, rng), rng, inplace=True).state
    assert checked > 0
