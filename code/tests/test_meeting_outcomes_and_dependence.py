"""Decision-time dependence and whole-game harm must survive later ejections/rounds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env import dependence
from social_collusion.env.enums import EventType, Head, Phase, Role
from social_collusion.env.rewards import terminal_rewards
from social_collusion.env.state import empty_joint_action
from social_collusion.env.transition import reset, transition
from social_collusion.metrics import outcome_metrics as om


def _vote(state, target=None):
    state.phase = int(Phase.VOTE)
    actions = empty_joint_action(state.config)
    for i in np.flatnonzero(state.alive):
        actions[i, Head.VOTE] = state.config.n_agents if target is None or i == target else target
    return actions


def test_whole_game_harm_includes_incident_free_and_earlier_ejections():
    cfg = load_env_config("full_short_game").with_(n_agents=9, max_rounds=3, coalition_objective="false_ejection")
    rng = np.random.default_rng(19)
    state = reset(cfg, rng)
    state.roles[:] = int(Role.CREW)
    state.roles[:2] = int(Role.COALITION)

    # First meeting: a false ejection despite no incident.
    state = transition(state, _vote(state, 2), rng).state
    assert state.round == 1
    assert state.crew_framed_count == 0  # preserve the original incident-gated reward

    # Second meeting: an incident-bearing false ejection.
    state.incident_creator, state.incident_victim = 0, 3
    state.incident_time, state.incident_room = 1, 0
    state.alive[3] = False
    state.add_event(EventType.INCIDENT_CREATED, creator=0, victim=3, room=0)
    state = transition(state, _vote(state, 4), rng).state
    assert state.round == 2
    assert state.crew_framed_count == 1

    # The last meeting skips. A final-meeting metric must not erase the earlier harm.
    state = transition(state, _vote(state), rng).state
    assert state.done
    assert terminal_rewards(state)[0] == 1.0
    row = om.episode_row(state)
    assert not row["false_ejection"]
    assert row["any_false_ejection"]
    assert row["total_false_ejections"] == 2
    assert row["total_incident_false_ejections"] == row["total_incident_free_false_ejections"] == 1
    assert row["total_meetings"] == 3
    assert row["total_incident_meetings"] == 1
    assert row["total_incident_free_meetings"] == 2
    assert row["false_ejections_per_meeting"] == pytest.approx(2 / 3)

    restored = state.copy()
    restored.meeting_log.clear()
    assert om.meeting_outcomes(restored) == om.meeting_outcomes(state)
    assert om.whole_game_row(restored) == om.whole_game_row(state)
    assert len(state.meeting_log) == 3  # copy owns its log

    summary = om.summarize([state])
    assert summary["terminal_meeting_false_ejection_rate"] == 0.0
    assert summary["any_false_ejection_rate"] == 1.0
    assert summary["mean_false_ejections"] == 2.0
    assert summary["false_ejections_per_meeting"] == pytest.approx(2 / 3)
    assert summary["false_ejections_per_incident_meeting"] == 1.0
    assert summary["false_ejections_per_incident_free_meeting"] == 0.5


def test_whole_game_rates_pool_meetings_and_keep_undefined_denominators():
    cfg = load_env_config("full_short_game")
    a, b = reset(cfg, np.random.default_rng(1)), reset(cfg, np.random.default_rng(2))
    a.meeting_log = [{"had_incident": False, "false_ejection": True}]
    b.meeting_log = [{"had_incident": False, "false_ejection": False} for _ in range(3)]
    summary = om.whole_game_summary([a, b])
    assert summary["any_false_ejection_rate"] == 0.5
    assert summary["mean_false_ejections"] == 0.5
    assert summary["false_ejections_per_meeting"] == 0.25
    assert summary["false_ejections_per_incident_free_meeting"] == 0.25
    assert summary["any_incident_false_ejection_rate"] == 0.0
    assert np.isnan(summary["false_ejections_per_incident_meeting"])


def test_dependence_records_full_electorate_and_does_not_duplicate_current_round():
    cfg = load_env_config("full_short_game").with_(n_agents=7, max_rounds=3, dependence_window=3)
    rng = np.random.default_rng(11)
    state = reset(cfg, rng)
    state.round = 2  # terminate at this vote, retaining its record for the diagnostic
    off_diagonal = np.ones((7, 7)) - np.eye(7)
    state.dependence_log = [(np.zeros((7, 7)), off_diagonal.copy()), (off_diagonal.copy(), off_diagonal.copy())]
    state.dependence_logged_round = 1
    target = int(state.coalition[0])
    actions = _vote(state, target)
    before = state.copy()
    before.votes[:] = actions[:, Head.VOTE]
    expected_agreement = dependence.meeting_agreement(before)
    expected_window = dependence.windowed_dependence(before)
    expected_penalty = dependence.agent_penalties(before)
    expected_realized = dependence.realized_dependence(before)
    expected_tally = dependence.weighted_vote_tally(before, expected_penalty)

    terminal = transition(state, actions, rng).state
    assert terminal.done and terminal.ejected == target and not terminal.alive[target]
    assert terminal.dependence_logged_round == terminal.round == 2
    assert len(terminal.dependence_log) == 3
    for actual, expected in zip(terminal.dependence_log[-1], expected_agreement):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(dependence.windowed_dependence(terminal), expected_window)
    np.testing.assert_allclose(dependence.agent_penalties(terminal), expected_penalty)
    np.testing.assert_allclose(dependence.weighted_vote_tally(terminal, expected_penalty), expected_tally)
    assert dependence.realized_dependence(terminal) == expected_realized
    assert dependence.voting_population(terminal)[target]

    # The per-channel analysis must use exactly the same electorate as the mechanism.
    path = Path(__file__).resolve().parents[1] / "experiments/dependence/channel_diagnostic.py"
    spec = importlib.util.spec_from_file_location("decision_channel_diagnostic", path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    for channel in diagnostic.CHANNELS:
        for a, b in zip(diagnostic.channel_matrices(before)[channel], diagnostic.channel_matrices(terminal)[channel]):
            np.testing.assert_array_equal(a, b)


def test_logged_current_votes_are_excluded_when_reconstructing_pre_vote_weights():
    cfg = load_env_config("full_short_game").with_(n_agents=7, max_rounds=2, dependence_window=2)
    state = reset(cfg, np.random.default_rng(5))
    state.round = 1
    state.phase = int(Phase.VOTE)
    state.votes[:] = cfg.n_agents
    state.votes[:2] = 4
    zero = np.zeros((7, 7))
    state.dependence_log = [(zero.copy(), np.ones((7, 7)) - np.eye(7))]
    expected = dependence.windowed_dependence(state, include_votes=False)
    state.dependence_log.append(dependence.meeting_agreement(state, include_votes=True))
    state.dependence_logged_round = state.round
    np.testing.assert_array_equal(dependence.windowed_dependence(state, include_votes=False), expected)
