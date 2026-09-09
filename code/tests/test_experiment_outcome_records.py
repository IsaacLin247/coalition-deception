"""Experiment exports must retain earlier/incident-free harm and auditable game rows."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from social_collusion.config import load_env_config
from social_collusion.env.enums import Phase, Role
from social_collusion.env.transition import reset
from social_collusion.metrics import outcome_metrics as om


def _load_scripts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import train_f3_matched_cycle as cycle

    path = root / "experiments/multigen/run_multigen.py"
    spec = importlib.util.spec_from_file_location("multigen_outcome_records", path)
    multigen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multigen)
    return cycle, multigen


def _terminal_game(rounds, seed=1):
    cfg = load_env_config("full_short_game").with_(max_rounds=rounds)
    state = reset(cfg, np.random.default_rng(seed))
    state.roles[:] = int(Role.CREW)
    state.roles[:2] = int(Role.COALITION)
    state.phase, state.done = int(Phase.TERMINAL), True
    state.ejected = -1
    state.meeting_log = [
        {"round": 0, "had_incident": False, "ejected": 2, "false_ejection": True,
         "creator_ejected": False, "any_ejection": True},
    ]
    if rounds > 1:
        state.meeting_log.append(
            {"round": 1, "had_incident": True, "ejected": -1, "false_ejection": False,
             "creator_ejected": False, "any_ejection": False},
        )
        state.round_log = [(0, 0, 0, 0, 1), (1, 1, 0, 0, 0)]
    return state


@pytest.mark.parametrize("rounds", [1, 2])
def test_round_export_retains_incident_free_harm_and_legacy_framing(monkeypatch, rounds):
    cycle, _ = _load_scripts(monkeypatch)
    state = _terminal_game(rounds)
    rows = cycle._round_rows([state], "C0_vs_D0", 123)
    assert len(rows) == rounds
    assert sum(r["false_ejection"] for r in rows) == 1
    assert rows[0]["incident_free_false_ejection"] == 1
    assert rows[0]["incident_false_ejection"] == 0
    assert rows[0]["ejected"] == 2
    # The old column was ungated in the single-round fallback and gated otherwise.
    assert rows[0]["crew_framed"] == int(rounds == 1)


def test_multigen_exports_allow_independent_recomputation(monkeypatch, tmp_path):
    _, multigen = _load_scripts(monkeypatch)
    states = [_terminal_game(2, 1), _terminal_game(2, 2)]
    states[1].meeting_log[0].update(ejected=-1, false_ejection=False, any_ejection=False)
    record = multigen.save_evaluation_records(states, "C0_vs_D0", 123, tmp_path,
                                              "C0_vs_D0", "crossplay")
    with (tmp_path / record["episode_metrics"]).open() as f:
        games = list(csv.DictReader(f))
    with (tmp_path / record["round_metrics"]).open() as f:
        meetings = list(csv.DictReader(f))
    assert record["n_episodes"] == len(games) == 2
    assert record["n_meetings"] == len(meetings) == 4
    assert [int(r["episode_index"]) for r in games] == [0, 1]
    summary = om.summarize(states)
    assert np.mean([r["any_false_ejection"] == "True" for r in games]) == summary["any_false_ejection_rate"]
    assert np.mean([int(r["total_false_ejections"]) for r in games]) == summary["mean_false_ejections"]
    assert np.mean([int(r["false_ejection"]) for r in meetings]) == summary["false_ejections_per_meeting"]


def test_replay_sampling_is_isolated_and_identified(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from social_collusion.policies.scripted_crew import TruthfulCrew
    from social_collusion.rl.torch_policy import TorchPolicy

    cycle, _ = _load_scripts(monkeypatch)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = load_env_config("meeting_only")
        actor = TorchPolicy(cfg, hidden_dim=16, seed=9)
        before = actor.generator.get_state().clone()
        _, _, _, replays = cycle.evaluate_pair(
            cfg, TruthfulCrew(), actor, "C0_vs_D0", 2, 123, tmp_path, 1, {},
        )
        assert torch.equal(before, actor.generator.get_state())
        assert replays[0]["valid"]
        assert replays[0]["trajectory_source"] == "independent_serial_replay"
        assert replays[0]["actor_seed"] != 123
    finally:
        torch.set_num_threads(old_threads)
