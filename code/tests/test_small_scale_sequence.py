"""The small-scale test sequence of plan sec.19 (Tests A-G), as executable checks.

Test D (learnability) and the heavier parts of Test B live in
`scripts/run_acceptance_gates.py` because they need torch and more episodes than a unit-test
suite should; everything cheap enough to gate a commit is here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env import VecEnv, compute_masks, sample_masked
from social_collusion.env.action_masks import is_legal
from social_collusion.env.enums import Role
from social_collusion.metrics import outcome_metrics as om
from social_collusion.policies import make_pair
from tests.conftest import play_random


# -- Test A: exhaustive tiny game ------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    """Three agents, two rooms, three turns - small enough to reason about exhaustively."""
    return load_env_config(
        "meeting_only",
        n_agents=3,
        n_coalition=1,
        rooms=("Hub", "Electrical"),
        edges=(("Hub", "Electrical"),),
        free_play_turns=3,
        enable_response_round=False,
    )


def test_A_tiny_game_is_consistent(tiny):
    outcomes = Counter()
    for e in range(300):
        st, _ = play_random(tiny, seed=6001, episode=e)
        assert st.done
        assert st.roles.sum() == tiny.n_coalition
        assert st.roles[st.incident_creator] == int(Role.COALITION)
        assert st.roles[st.incident_victim] == int(Role.CREW)
        outcomes[(bool(st.ejected < 0), int(st.ejected))] += 1
    assert len(outcomes) > 1, "a tiny game should still reach several terminal outcomes"


def test_A_terminal_rewards_are_consistent_with_the_outcome(tiny):
    from social_collusion.env.rewards import coalition_favorable, terminal_rewards

    for e in range(200):
        st, _ = play_random(tiny, seed=6003, episode=e)
        r = terminal_rewards(st)
        fav = coalition_favorable(st)
        for i in range(tiny.n_agents):
            expected = (1.0 if fav else -1.0) if st.roles[i] == int(Role.COALITION) else (-1.0 if fav else 1.0)
            assert r[i] == expected


# -- Test B: random-policy distribution ------------------------------------
@pytest.fixture(scope="module")
def random_states(cfg):
    return [play_random(cfg, seed=6005, episode=e)[0] for e in range(1500)]


def test_B_role_assignment_is_symmetric(cfg, random_states):
    counts = np.zeros(cfg.n_agents)
    for st in random_states:
        counts[st.coalition] += 1
    expected = len(random_states) * cfg.n_coalition / cfg.n_agents
    assert np.abs(counts - expected).max() < 4.5 * np.sqrt(expected)


def test_B_no_agent_slot_is_favoured_by_the_outcome(cfg, random_states):
    ejected = np.zeros(cfg.n_agents)
    for st in random_states:
        if st.ejected >= 0:
            ejected[st.ejected] += 1
    total = ejected.sum()
    assert total > 0
    expected = total / cfg.n_agents
    assert np.abs(ejected - expected).max() < 5.0 * np.sqrt(expected), (
        "an identity-specific outcome bias would invalidate every downstream comparison"
    )


def test_B_room_occupancy_is_non_degenerate(cfg, random_states):
    occ = np.zeros(cfg.n_rooms)
    for st in random_states[:400]:
        for t in range(st.horizon + 1):
            for i in range(cfg.n_agents):
                p = int(st.positions[t, i])
                if p >= 0:
                    occ[p] += 1
    assert (occ > 0).all(), "every room must be visited"
    # the Hub is the only articulation point of the star map, so it must dominate
    assert occ.argmax() == 0


def test_B_incident_and_tie_frequencies_are_sane(cfg, random_states):
    assert all(st.incident_creator >= 0 for st in random_states)
    ties = np.mean([st.ejected < 0 for st in random_states])
    assert 0.02 < ties < 0.9
    times = Counter(int(st.incident_time) for st in random_states)
    assert min(times) >= 1 and max(times) <= cfg.n_times - 2


def test_B_random_play_never_produces_an_illegal_action(cfg):
    from social_collusion.env import Phase, reset, transition
    from social_collusion.seeding import episode_rng

    for e in range(60):
        rng = episode_rng(6007, e)
        st = reset(cfg, rng, seed=6007)
        while st.phase != int(Phase.TERMINAL):
            a = sample_masked(compute_masks(st), rng)
            ok, why = is_legal(st, a)
            assert ok, why
            st = transition(st, a, rng, inplace=True, validate=True).state


# -- Test C: scripted dominance --------------------------------------------
def test_C_scripted_beats_random(cfg):
    def run(crew, coal, seed):
        return VecEnv(cfg, 32, seed=seed).run_episodes(make_pair(crew, coal), 400)

    scripted = om.summarize(run("truthful", "alibi", 6009))["false_ejection_rate"]
    rnd = om.summarize(run("truthful", "random", 6011))["false_ejection_rate"]
    assert scripted > rnd + 0.2

    # and a competent crew beats a random one at catching the creator
    good = om.summarize(run("truthful", "lone_liar", 6013))["creator_ejection_rate"]
    bad = om.summarize(run("random_voter", "lone_liar", 6015))["creator_ejection_rate"]
    assert good > bad


# -- Test E: generalisation probe ------------------------------------------
def test_E_results_survive_a_held_out_map(cfg):
    """The scripted alibi advantage must not be an artifact of the star map."""
    held_out = cfg.with_(edges=(("Hub", "Electrical"), ("Hub", "Navigation"), ("Hub", "Medbay"), ("Electrical", "Navigation")))
    for c in (cfg, held_out):
        a = om.summarize(VecEnv(c, 32, seed=6017).run_episodes(make_pair("truthful", "alibi"), 300))
        h = om.summarize(VecEnv(c, 32, seed=6019).run_episodes(make_pair("truthful", "truthful"), 300))
        assert a["false_ejection_rate"] > h["false_ejection_rate"]


def test_E_results_survive_a_fixed_speaking_order(cfg):
    fixed = cfg.with_(random_speaking_order=False)
    a = om.summarize(VecEnv(fixed, 32, seed=6021).run_episodes(make_pair("truthful", "alibi"), 300))
    h = om.summarize(VecEnv(fixed, 32, seed=6023).run_episodes(make_pair("truthful", "truthful"), 300))
    assert a["false_ejection_rate"] > h["false_ejection_rate"]


def test_E_results_survive_a_larger_electorate():
    """The pre-registered robustness configuration (contract sec.11)."""
    big = load_env_config("meeting_only_7")
    a = om.summarize(VecEnv(big, 32, seed=6025).run_episodes(make_pair("truthful", "alibi"), 300))
    h = om.summarize(VecEnv(big, 32, seed=6027).run_episodes(make_pair("truthful", "truthful"), 300))
    assert a["false_ejection_rate"] > h["false_ejection_rate"]
    assert a["coalition_win_rate"] < 1.0, "no metric should saturate in the robustness config"


# -- Test G: reward-hacking audit ------------------------------------------
def test_G_no_invalid_action_exploits(cfg, cfg_spatial):
    for c in (cfg, cfg_spatial):
        states = VecEnv(c, 32, seed=6029).run_episodes(make_pair("truthful", "alibi"), 200)
        assert sum(st.invalid_action_attempts for st in states) == 0, (
            "scripted policies should never attempt a masked action"
        )


def test_G_top_return_episodes_are_not_degenerate(cfg):
    """Inspect the coalition's best episodes for structural exploits."""
    states = VecEnv(cfg, 32, seed=6031).run_episodes(make_pair("truthful", "alibi"), 400)
    wins = [st for st in states if om.episode_row(st)["false_ejection"]]
    assert wins
    for st in wins[:50]:
        assert st.ejected >= 0 and st.roles[st.ejected] == int(Role.CREW)
        assert st.ejected != st.incident_victim, "the victim cannot be ejected"
        assert len(st.claims) == int(st.alive.sum()) + (1 if st.ejected >= 0 else 0)
        assert st.invalid_action_attempts == 0


def test_G_creator_is_not_predictable_from_the_slot_index(cfg, random_states):
    counts = Counter(int(st.incident_creator) for st in random_states)
    expected = len(random_states) / cfg.n_agents
    assert max(counts.values()) < expected * 1.5
