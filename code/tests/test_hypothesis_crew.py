"""Hard-constrained hypothesis scores and observer-information regressions."""

from __future__ import annotations

import pytest

import numpy as np

from social_collusion.config import load_env_config
from social_collusion.env import VecEnv
from social_collusion.env.enums import ClaimType, EventType, Phase, Role, TruthLabel
from social_collusion.env.state import Claim, Event, GameState
from social_collusion.policies.base import RoleRouter
from social_collusion.policies.hypothesis_crew import HypothesisCrew, coalition_posterior, crew_for_rule
from social_collusion.policies.registry import make_coalition
from social_collusion.policies.scripted_crew import TruthfulCrew


def _states(condition: str, rule: str, episodes: int = 40, seed: int = 3):
    cfg = load_env_config("meeting_only").with_(n_agents=7, n_coalition=2, coalition_objective="false_ejection")
    return VecEnv(cfg, 8, seed=seed).run_episodes(RoleRouter(crew_for_rule(rule), make_coalition(condition)), episodes)


def test_posterior_is_a_distribution_over_pairs():
    for st in _states("alibi", "hypothesis", episodes=10):
        for voter in range(st.config.n_agents):
            if st.roles[voter] != int(Role.CREW):
                continue
            member, creator = coalition_posterior(st, voter)
            assert member[voter] == 0.0
            assert np.isclose(member.sum(), st.config.n_coalition)
            assert np.all(member >= 0) and np.all(member <= 1 + 1e-9)
            if st.report_turn >= 0:
                assert np.all(creator >= 0)
                assert np.isclose(creator.sum(), 1.0)
                assert member[st.incident_victim] == 0.0


def test_truthful_crew_never_lies():
    """The scripted crew speaks its own polarity: no crew claim is labelled FALSE."""
    for st in _states("truthful", "mean", episodes=30):
        for c in st.claims:
            if st.roles[c.speaker] == int(Role.CREW):
                assert int(c.label) != int(TruthLabel.FALSE)


def test_crew_for_rule_dispatch():
    assert isinstance(crew_for_rule("hypothesis"), HypothesisCrew)
    assert isinstance(crew_for_rule("mean"), TruthfulCrew)
    assert not isinstance(crew_for_rule("mean"), HypothesisCrew)


def _view(n=5):
    cfg = load_env_config("meeting_only").with_(n_agents=n, n_coalition=2)
    st = GameState(config=cfg, seed=0, phase=int(Phase.VOTE), horizon=2, turn=2)
    st.roles = np.array([0] * (n-2) + [1, 1], dtype=np.int8)
    st.alive = np.ones(n, dtype=bool)
    st.positions = np.full((cfg.n_times, n), 2, dtype=np.int8)
    st.positions[:, 0] = 0
    st.marker_seen = np.zeros(n, dtype=bool)
    st.marker_seen_time = np.full(n, -1, dtype=np.int8)
    st.votes = np.full(n, -1, dtype=np.int8)
    return st


def test_murdered_victim_is_known_crew_without_hidden_role_lookup():
    st = _view()
    st.alive[4] = False
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    # Deliberately misleading hidden metadata must not override public death information.
    st.incident_victim = 3
    st.roles[4] = int(Role.COALITION)
    member, creator = coalition_posterior(st, 0, use_private=False)
    assert member[4] == creator[4] == 0.0
    assert np.allclose(member[1:4], 2/3)
    assert np.allclose(creator[1:4], 1/3)


def test_publicly_ejected_player_is_not_assumed_murdered_or_honest():
    st = _view()
    st.alive[1] = False
    st.events.append(Event(EventType.AGENT_EJECTED, 0, int(Phase.VOTE), {"agent": 1, "role": None}))
    member, _ = coalition_posterior(st, 0, use_private=False)
    assert member[1] > 0.0
    st.events[-1] = Event(EventType.AGENT_EJECTED, 0, int(Phase.VOTE), {"agent": 1, "role": "CREW"})
    revealed, _ = coalition_posterior(st, 0, use_private=False)
    assert revealed[1] == 0.0


def test_revealed_coalition_member_is_in_every_surviving_hypothesis():
    st = _view()
    st.alive[1] = False
    st.events.append(Event(EventType.AGENT_EJECTED, 0, int(Phase.VOTE), {"agent": 1, "role": "COALITION"}))
    member, _ = coalition_posterior(st, 0, use_private=False)
    assert member[1] == 1.0


def test_impossible_views_return_zero_and_abstain_without_resurrection():
    st = _view()
    # Three incompatible speakers cannot all fit inside a coalition of size two.
    st.claims = [Claim(j, int(ClaimType.SELF_LOCATION), room=r, time=1)
                 for j in [1, 2, 3] for r in [0, 1]]
    member, creator = coalition_posterior(st, 0, use_private=False)
    assert np.array_equal(member, np.zeros(5))
    assert np.array_equal(creator, np.zeros(5))
    assert HypothesisCrew(blind=True).vote_index(st, 0, np.random.default_rng(0)) == 5
    with pytest.raises(ValueError, match="pseudocount"):
        coalition_posterior(st, 0, eps=1e-3)


def test_private_negative_evidence_eliminates_creator_assignments():
    st = _view()
    st.config = st.config.with_(incident_requires_isolation=False)
    st.positions[:, 0] = 1
    st.positions[:, 2] = 1
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    _, creator = coalition_posterior(st, 0)
    assert creator[1] == creator[3] == creator[4] == 0.0
    assert creator[2] == 1.0


def test_private_facts_and_pair_claims_are_solved_jointly():
    st = _view()
    st.positions[:, 3] = 0
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    # Neither player 1 nor 2 was directly seen by the voter. Their accepted pair equality
    # and player 2's self-placement jointly clear player 1 of the scene as well.
    st.claims = [Claim(1, int(ClaimType.WAS_WITH_AGENT), subject=2, time=1),
                 Claim(2, int(ClaimType.SELF_LOCATION), room=2, time=1)]
    _, creator = coalition_posterior(st, 0, honesty=1.0)
    assert creator[1] == creator[2] == creator[3] == 0.0
    assert creator[4] == pytest.approx(1.0)


def test_public_marker_conflict_is_a_hard_constraint():
    st = _view()
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    st.claims = [Claim(1, int(ClaimType.FOUND_MARKER), room=2, time=2)]
    member, _ = coalition_posterior(st, 0, use_private=False, honesty=0.0)
    assert member[1] == 1.0  # Every hypothesis treating speaker 1 as honest is impossible.


def test_hidden_labels_roles_and_creator_identity_do_not_change_scores():
    st = _view()
    st.claims = [Claim(1, int(ClaimType.SELF_LOCATION), room=2, time=1)]
    before = coalition_posterior(st, 0)
    st.roles[1:] = st.roles[1:][::-1]
    st.incident_creator, st.incident_victim = 1, 2
    st.claims[0].label = int(TruthLabel.CONTRADICTORY)
    st.claims[0].content_true = False
    after = coalition_posterior(st, 0)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_cached_scores_are_isolated_and_refresh_when_public_claims_change():
    st = _view()
    member, _ = coalition_posterior(st, 0, use_private=False)
    member[1] = 77.0
    assert coalition_posterior(st, 0, use_private=False)[0][1] == 0.5
    st.claims = [Claim(1, int(ClaimType.SELF_LOCATION), room=r, time=1) for r in [1, 2]]
    assert coalition_posterior(st, 0, use_private=False)[0][1] == 1.0


def test_honesty_gate_must_be_a_probability():
    with pytest.raises(ValueError, match="honesty"):
        coalition_posterior(_view(), 0, honesty=1.1)


def test_public_death_allows_absence_from_every_room():
    st = _view()
    st.alive[4] = False
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    # Different honest speakers can truthfully fail to see the dead victim in every room.
    # Use each speaker's own two-time movement so the speaker entailments remain feasible.
    st.claims = [Claim(1 + r % 2, int(ClaimType.SAW_AGENT_IN_ROOM), subject=4,
                      room=r, time=2, polarity=False) for r in range(2)]
    st.claims += [Claim(3, int(ClaimType.WAS_WITH_AGENT), subject=4, time=2, polarity=False)]
    # Room exclusions themselves must not require the dead subject to occupy another room.
    from social_collusion.policies.hypothesis_crew import _active_constraints
    from social_collusion.env.claims import satisfiable
    cons = [("neq_const", 4, 2, r) for r in range(st.config.n_rooms)]
    assert satisfiable(_active_constraints(cons, {4: 2}), st.config)
    member, _ = coalition_posterior(st, 0, use_private=False, honesty=1.0)
    assert member.sum() == pytest.approx(2.0)
    assert member[4] == 0.0


def test_public_post_death_sighting_cannot_be_honest():
    st = _view()
    st.alive[4] = False
    st.incident_room, st.incident_time, st.report_turn = 1, 1, 2
    st.claims = [Claim(1, int(ClaimType.SAW_AGENT_IN_ROOM), subject=4, room=2, time=2)]
    member, _ = coalition_posterior(st, 0, use_private=False, honesty=0.0)
    assert member[1] == 1.0
    # The victim occupied its last room at the incident instant, so this is not impossible.
    st.claims[0].time = 1
    earlier, _ = coalition_posterior(st, 0, use_private=False, honesty=0.0)
    assert earlier[1] < 1.0


def test_hidden_incident_time_does_not_determine_death_constraints():
    st = _view()
    st.config = st.config.with_(incident_time_public=False)
    st.alive[4] = False
    st.incident_room, st.incident_time, st.report_turn = 1, 0, 2
    st.claims = [Claim(1, int(ClaimType.SAW_AGENT_IN_ROOM), subject=4, room=2, time=1)]
    before = coalition_posterior(st, 0, use_private=False)
    st.incident_time = 1
    after = coalition_posterior(st, 0, use_private=False)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_previous_round_dead_agents_are_absent_at_current_round_start():
    st = _view()
    st.alive[4] = False
    st.events = [Event(EventType.AGENT_EJECTED, 2, int(Phase.VOTE), {"agent": 4, "role": None}),
                 Event(EventType.ROUND_STARTED, 0, int(Phase.FREE_PLAY), {"round": 1, "alive": [0, 1, 2, 3]})]
    st.claims = [Claim(1, int(ClaimType.WAS_WITH_AGENT), subject=4, time=0)]
    member, _ = coalition_posterior(st, 0, use_private=False, honesty=0.0)
    assert member[1] == 1.0
    assert member[4] > 0.0  # An earlier ejection of unknown role is not a known crew death.
