"""The dependence-aware credibility mechanism (env/dependence.py) and its three integration points."""

from __future__ import annotations

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env import VecEnv, dependence
from social_collusion.env.enums import Role
from social_collusion.metrics import outcome_metrics as om
from social_collusion.policies import crew_aggregation as ca
from social_collusion.policies import make_pair
from social_collusion.policies.scripted_crew import TruthfulCrew


@pytest.fixture(scope="module")
def alibi_states(cfg):
    return VecEnv(cfg, 32, seed=7171).run_episodes(make_pair("truthful", "alibi"), 120)


@pytest.fixture(scope="module")
def honest_states(cfg):
    return VecEnv(cfg, 32, seed=7172).run_episodes(make_pair("truthful", "truthful"), 120)


def test_agreement_matrices_are_symmetric_and_bounded(alibi_states):
    for st in alibi_states[:40]:
        agree, opp = dependence.meeting_agreement(st)
        assert np.allclose(agree, agree.T) and np.allclose(opp, opp.T)
        assert (agree <= opp + 1e-12).all() and (agree >= 0).all()
        D = dependence.windowed_dependence(st)
        assert D.min() >= 0.0 and D.max() <= 1.0 + 1e-12
        assert np.allclose(np.diag(D), 0.0)


def test_penalties_lie_in_unit_interval_and_strength_zero_is_neutral(alibi_states):
    for st in alibi_states[:40]:
        pen = dependence.agent_penalties(st)
        assert (pen > 0).all() and (pen <= 1.0 + 1e-12).all()
        assert np.allclose(dependence.agent_penalties(st, strength=0.0), 1.0)
        stronger = dependence.agent_penalties(st, strength=8.0)
        assert (stronger <= pen + 1e-12).all()


def test_concentration_flags_a_closed_dyad():
    """Two agents that agree only with each other must have higher excess than agents whose
    agreement is spread across the room."""
    D = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0.6, 0.6],
            [0, 0, 0.6, 0, 0.6],
            [0, 0, 0.6, 0.6, 0],
        ],
        dtype=float,
    )
    alive = np.ones(5, dtype=bool)
    ex = dependence.concentration(D, alive, top_j=1)
    assert ex[0] == ex[1] > ex[2] == ex[3] == ex[4]
    assert ex[2] > 0.0  # some concentration is inevitable with one non-agreeing pair


def test_coalition_pair_is_more_dependent_than_honest_pairs_under_alibi(alibi_states):
    s = dependence.summarize_dependence(alibi_states)
    assert s["coalition_pair_dependence"] > s["crew_pair_dependence"]
    assert s["coalition_penalty"] < s["crew_penalty"]


def test_dependence_rule_reduces_to_soft_credibility_at_zero_strength(cfg, alibi_states):
    zero = cfg.with_(dependence_penalty=0.0)
    for st in alibi_states[:30]:
        st0 = st.copy()
        st0.config = zero
        for v in range(cfg.n_agents):
            if not st.alive[v]:
                continue
            a = ca.suspicion_scores(st0, v, "soft_credibility")
            b = ca.suspicion_scores(st0, v, "dependence_aware")
            fin = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[fin], b[fin], atol=1e-10)


def test_dependence_rule_is_listed_and_masks_like_the_others(cfg, alibi_states):
    assert "dependence_aware" in ca.RULES
    for st in alibi_states[:20]:
        for v in range(cfg.n_agents):
            if st.alive[v]:
                s = ca.suspicion_scores(st, v, "dependence_aware")
                assert s[v] == -np.inf
                assert np.isfinite(s[np.isfinite(s)]).all()


def test_dependence_rule_is_no_worse_than_soft_credibility_against_scripted_alibi(cfg):
    """The mechanism must at least not amplify the manufactured-corroboration attack."""
    fe = {}
    for rule in ("soft_credibility", "dependence_aware"):
        c = cfg.with_(dependence_penalty=3.0)
        pol = make_pair("truthful", "alibi", crew_kwargs={"rule": rule})
        states = VecEnv(c, 32, seed=7173).run_episodes(pol, 300)
        fe[rule] = om.summarize(states)["false_ejection_rate"]
    assert fe["dependence_aware"] <= fe["soft_credibility"] + 0.02


def test_dependence_weighted_vote_matches_majority_when_strength_is_zero():
    base = load_env_config("full_short_game").with_(n_agents=7, n_coalition=2, max_rounds=1)
    a = VecEnv(base.with_(vote_aggregation="majority"), 32, seed=11).run_episodes(
        make_pair("truthful", "alibi"), 150
    )
    b = VecEnv(
        base.with_(vote_aggregation="dependence_weighted", dependence_penalty=0.0), 32, seed=11
    ).run_episodes(make_pair("truthful", "alibi"), 150)
    assert [s.ejected for s in a] == [s.ejected for s in b]


def test_dependence_weighted_vote_runs_in_multi_round_and_logs_window():
    cfg = load_env_config("full_short_game").with_(
        n_agents=7,
        n_coalition=2,
        max_rounds=6,
        vote_aggregation="dependence_weighted",
        dependence_window=3,
    )
    states = VecEnv(cfg, 32, seed=12).run_episodes(make_pair("truthful", "alibi"), 100)
    for s in states:
        assert s.done
        assert len(s.dependence_log) == len(s.round_log)
        for agree, opp in s.dependence_log:
            assert agree.shape == (7, 7) and (agree <= opp + 1e-12).all()
    multi = [s for s in states if len(s.round_log) >= 2]
    assert multi, "some games should last more than one round"


def test_judge_dependence_aware_runs(cfg):
    c = cfg.with_(vote_aggregation="judge_dependence_aware")
    states = VecEnv(c, 32, seed=13).run_episodes(make_pair("truthful", "alibi"), 60)
    assert all(s.done for s in states)
    assert any(s.ejected >= 0 for s in states)


def test_default_config_never_touches_the_log(cfg, honest_states):
    for s in honest_states[:30]:
        assert s.dependence_log == []


def test_truthful_crew_accepts_the_new_rule_in_spatial_game():
    cfg = load_env_config("full_short_game").with_(n_agents=5, n_coalition=2, max_rounds=1)
    pol = make_pair("truthful", "alibi", crew_kwargs={"rule": "dependence_aware"})
    states = VecEnv(cfg, 16, seed=14).run_episodes(pol, 40)
    assert all(s.done for s in states)
    assert all(int(Role.COALITION) in set(s.roles.tolist()) for s in states)
    _ = TruthfulCrew(rule="dependence_aware")
