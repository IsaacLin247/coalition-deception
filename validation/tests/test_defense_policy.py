"""Synthetic observer views only; no training or Monte Carlo game simulations."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env.action_masks import head_sizes
from social_collusion.env.enums import Head, Phase, ResponseType, Role, TruthLabel
from social_collusion.env.state import Claim, GameState, Response, empty_joint_action
from social_collusion.policies.hypothesis_crew import HypothesisCrew
from validation import defense_policy as defense


def _view(n=5):
    cfg = load_env_config("meeting_only").with_(n_agents=n, n_coalition=2)
    st = GameState(config=cfg, seed=23, phase=int(Phase.VOTE), horizon=2, turn=2)
    st.roles = np.array([0] * (n - 2) + [1, 1], dtype=np.int8)
    st.alive = np.ones(n, dtype=bool)
    st.positions = np.full((cfg.n_times, n), 2, dtype=np.int8)
    st.positions[:, 0] = 0
    st.marker_seen = np.zeros(n, dtype=bool)
    st.marker_seen_time = np.full(n, -1, dtype=np.int8)
    st.votes = np.full(n, -1, dtype=np.int8)
    return st


def _mock_scores(monkeypatch, member, creator=None, own=None):
    member = np.asarray(member, dtype=float)
    creator = np.zeros_like(member) if creator is None else np.asarray(creator, dtype=float)
    own = np.zeros_like(member) if own is None else np.asarray(own, dtype=float)
    monkeypatch.setattr(defense, "coalition_posterior", lambda *a, **kw: (member, creator))
    monkeypatch.setattr(defense, "own_evidence", lambda *a, **kw: own)


@pytest.mark.parametrize("tie_break", ["index", "random", "skip"])
def test_strict_threshold_and_unique_target(monkeypatch, tie_break):
    _mock_scores(monkeypatch, [0, 0.7, 0.2, 0.6, 0.5])
    policy = defense.make_defense({"name": "test", "tie_break": tie_break, "threshold": 0.7})
    assert policy.decision(_view(), 0).target is None
    policy = defense.make_defense(defense.DefenseSpec("test", tie_break, threshold=0.69))
    assert policy.decision(_view(), 0).target == 1


def test_ranks_use_creator_then_own_evidence_before_final_tie(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .7], [0, .2, .3, .3, .9], [0, 9, 0, 1, 20])
    policy = defense.make_defense(defense.DefenseSpec("skip", "skip"))
    result = policy.decision(_view(), 0)
    assert result.tied_candidates == (3,)
    assert result.target == 3


def test_near_equal_values_are_not_ties(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, np.nextafter(.8, 1), .2, .1])
    result = defense.make_defense(defense.DefenseSpec("skip", "skip")).decision(_view(), 0)
    assert result.tied_candidates == (2,)
    assert result.target == 2


def test_exact_final_tie_and_inactive_self_exclusion(monkeypatch):
    _mock_scores(monkeypatch, [1, .8, .8, 1, .1])
    st = _view()
    st.alive[3] = False
    baseline = defense.make_defense(defense.DefenseSpec("index"))
    skipping = defense.make_defense(defense.DefenseSpec("skip", "skip"))
    assert baseline.decision(st, 0).tied_candidates == (1, 2)
    assert baseline._target(st, 0) == 1
    assert skipping._target(st, 0) is None


def test_index_baseline_matches_frozen_rule_on_bounded_real_score_views():
    old = HypothesisCrew(threshold=.49)
    new = defense.make_defense(defense.DefenseSpec("index", threshold=.49))
    st = _view()
    for voter in range(st.config.n_agents):
        assert new._target(st, voter) == old._target(st, voter)
    st.claims = [Claim(1, 1, room=1, time=1), Claim(1, 1, room=2, time=1)]
    for voter in range(st.config.n_agents):
        assert new._target(st, voter) == old._target(st, voter)


def test_all_zero_and_no_candidate_return_abstention():
    policy = defense.make_defense(defense.DefenseSpec("test"), blind=True)
    st = _view()
    st.claims = [Claim(j, 1, room=r, time=1) for j in [1, 2, 3] for r in [0, 1]]
    assert policy.decision(st, 0).top_membership == 0
    assert policy.vote_index(st, 0, np.random.default_rng(0)) == st.config.n_agents
    st.alive[1:] = False
    assert policy.decision(st, 0) == defense.DefenseDecision(None, (), None, False)


def test_honesty_gate_passed_to_unchanged_scoring_function(monkeypatch):
    seen = []
    original = defense.coalition_posterior

    def checked(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(defense, "coalition_posterior", checked)
    policy = defense.make_defense(defense.DefenseSpec("gate", honesty=.25), blind=True)
    policy.decision(_view(), 0)
    assert seen == [{"use_private": False, "honesty": .25}]


def test_response_vote_consistency_replay_and_engine_rng_isolation(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .8])
    policy = defense.make_defense(defense.DefenseSpec("random", "random"), tie_seed=7261)
    st = _view()
    st.phase = int(Phase.RESPONSE_ROUND)
    masks = [np.ones(n, dtype=bool) for n in head_sizes(st.config)]
    rng = np.random.default_rng(482)
    before = copy.deepcopy(rng.bit_generator.state)
    action = policy.act_single(st, 0, masks, rng)
    assert action[Head.RESPONSE_TYPE] == int(ResponseType.ACCUSE)
    target = int(action[Head.RESPONSE_TARGET])
    st.responses.append(Response(2, int(ResponseType.ACCUSE), target=4, order=2))
    st.sub_step += 1
    st.phase = int(Phase.VOTE)
    assert policy.act_single(st, 0, masks, rng)[Head.VOTE] == target
    assert before == rng.bit_generator.state
    rebuilt = defense.make_defense(policy.spec.to_dict(), tie_seed=7261)
    rebuilt.reset(batch_size=32)
    assert rebuilt.vote_index(st.copy(), 0, rng) == target
    assert before == rng.bit_generator.state


def test_hidden_state_and_annotations_cannot_change_random_target(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .8])
    st = _view()
    st.config = st.config.with_(incident_time_public=False)
    st.claims = [Claim(1, 1, room=2, time=1)]
    policy = defense.make_defense(defense.DefenseSpec("random", "random"), tie_seed=104)
    before_key = defense.observer_view_key(st, 0)
    before_target = policy._target(st, 0)
    st.seed = 9898
    st.roles[1:] = st.roles[1:][::-1]
    st.incident_creator, st.incident_victim, st.incident_time = 1, 2, 1
    st.symbols = np.array([4, 2, 0, 1, 0])
    st.claims[0].label = int(TruthLabel.FALSE)
    st.claims[0].content_true = False
    st.claims[0].supported = False
    st.positions[:, 1:] = 1  # Still outside observer 0's room; no new private fact.
    assert defense.observer_view_key(st, 0) == before_key
    assert policy._target(st, 0) == before_target


def test_blind_key_excludes_private_position_and_marker_information():
    st = _view()
    before = defense.observer_view_key(st, 0, blind=True)
    st.positions[:, 0] = 2
    st.marker_seen[0] = True
    st.incident_room = 1
    assert defense.observer_view_key(st, 0, blind=True) == before
    assert defense.observer_view_key(st, 0, blind=False)["private"] is not None


def test_random_ties_have_no_low_index_concentration_under_id_relabeling(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .8])
    st = _view()
    relabeled = st.copy()
    permutation = np.array([3, 4, 1, 0, 2])  # Old ID -> new ID.
    relabeled.positions[:, permutation] = st.positions
    relabeled.alive[permutation] = st.alive
    relabeled.roles[permutation] = st.roles
    counts = Counter()
    relabeled_counts = Counter()
    seeds = 2048
    for seed in range(seeds):
        policy = defense.make_defense(defense.DefenseSpec("random", "random"), tie_seed=seed)
        # Test the random selector separately: both lists represent the same tied roles.
        counts[policy._random_target(st, 0, (1, 2, 3, 4))] += 1
        target = policy._random_target(relabeled, int(permutation[0]), tuple(sorted(permutation[1:])))
        relabeled_counts[int(np.flatnonzero(permutation == target)[0])] += 1
    for count in [*counts.values(), *relabeled_counts.values()]:
        assert abs(count / seeds - .25) < .045
    assert set(counts) == set(relabeled_counts) == {1, 2, 3, 4}


def test_skip_on_tie_does_not_veto_two_coalition_plurality_ballots(monkeypatch):
    # A single synthetic tally transition, not a generated game or empirical study.
    from social_collusion.env import transition as transition_function
    import importlib

    transition_module = importlib.import_module(transition_function.__module__)
    monkeypatch.setattr(transition_module, "_end_round", lambda state, rng: None)
    _mock_scores(monkeypatch, [.8, .8, .8, .8, .8])
    st = _view()
    policy = defense.make_defense(defense.DefenseSpec("skip", "skip"))
    actions = empty_joint_action(st.config)
    rng = np.random.default_rng(51)
    for voter in range(3):
        actions[voter, Head.VOTE] = policy.vote_index(st, voter, rng)
        assert actions[voter, Head.VOTE] == st.config.n_agents
    actions[3:, Head.VOTE] = 0
    transition_module._step_vote(st, actions, rng)
    assert st.ejected == 0
    assert st.meeting_log[-1]["false_ejection"]
    assert st.roles[0] == int(Role.CREW)


@pytest.mark.parametrize("field,value", [
    ("tie_break", "epsilon"), ("threshold", float("nan")), ("threshold", -.1),
    ("honesty", float("inf")), ("honesty", 1.1), ("name", ""),
])
def test_invalid_spec_rejected(field, value):
    values = {"name": "test", field: value}
    with pytest.raises(ValueError):
        defense.DefenseSpec(**values)


def test_invalid_seed_and_conflicting_parameters_rejected():
    spec = defense.DefenseSpec("test")
    for seed in [1.5, "2", True]:
        with pytest.raises(ValueError, match="tie_seed"):
            defense.make_defense(spec, tie_seed=seed)
    with pytest.raises(ValueError, match="DefenseSpec"):
        defense.make_defense(spec, threshold=.7)
    with pytest.raises(ValueError, match="not both"):
        defense.make_defense(spec, seed=12, tie_seed=13)


def test_seed_alias_and_pre_ejection_recording(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .8])
    spec = defense.DefenseSpec("random", "random")
    policy = defense.make_defense(spec, 926, record_decisions=True)
    alias = defense.make_defense(spec, tie_seed=926)
    st = _view()
    rng = np.random.default_rng(8)
    vote = policy.vote_index(st, 0, rng)
    assert vote == alias.vote_index(st, 0, rng)
    assert len(policy.audit_records) == 1
    assert policy.audit_records[0]["target"] == vote
    assert policy.audit_records[0]["top_membership"] == .8
    assert policy.audit_records[0]["tie_count"] == 4
    assert not policy.audit_records[0]["skip"]
    assert not alias.audit_records
    # Descriptive inspection does not add records; one record per vote_index call.
    policy.decision(st, 0)
    assert len(policy.audit_records) == 1
    policy.audit_records.clear()
    assert not policy.audit_records


def test_skip_disabled_preserves_frozen_first_legal_fallback(monkeypatch):
    _mock_scores(monkeypatch, [0, .8, .8, .8, .8])
    st = _view()
    st.config = st.config.with_(allow_skip_vote=False)
    policy = defense.make_defense(defense.DefenseSpec("skip", "skip"), record_decisions=True)
    assert policy.vote_index(st, 0, np.random.default_rng(0)) == 1
    assert policy.audit_records[0]["proposed_target"] is None
    assert policy.audit_records[0]["target"] == 1
    assert not policy.audit_records[0]["skip"]
