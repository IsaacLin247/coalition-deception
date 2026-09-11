"""Synthetic integrity checks only; no environment games or training are executed."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jobs
import runner


def static_job():
    return {"name": "dev_s100", "kind": "static", "phase": "development", "seed": 100,
            "eval_seed": 200, "n_agents": 7, "max_rounds": 1, "episodes": 2,
            "eval_batch_size": 2, "defenses": [{"name": "original", "tie_break": "index", "threshold": .5}],
            "opponents": [{"name": "truthful", "kind": "scripted", "condition": "truthful"}]}


def design(job=None):
    return {"schema_version": 1, "study": "test", "jobs": [job or static_job()]}


def write_synthetic_result(path, job):
    path.mkdir(exist_ok=True)
    cell = "truthful__original"
    payload = {"summary": {"n_episodes": 2, "false_ejection_rate": .5},
               "episodes": [{"episode_index": 0, "eval_seed": 200, "false_ejection": True},
                            {"episode_index": 1, "eval_seed": 200, "false_ejection": False}]}
    runner.write_json(path / f"{cell}.json", payload)
    result = {"job": job, "status": "complete", "evaluations": [{"cell": cell, "path": f"{cell}.json"}], "stages": []}
    runner.write_json(path / "result.json", result)
    return payload, result


def test_job_rejects_ambiguous_or_unsafe_names():
    j = static_job()
    j["name"] = "../outside"
    with pytest.raises(ValueError, match="unsafe"):
        jobs.validate_design(design(j))
    j = static_job()
    with pytest.raises(ValueError, match="duplicate job"):
        jobs.validate_design({**design(j), "jobs": [j, copy.deepcopy(j)]})


def test_cycles_and_fractional_episode_batches_are_rejected():
    j = static_job()
    j["dependencies"] = [j["name"]]
    with pytest.raises(ValueError, match="cyclic"):
        jobs.validate_design(design(j))
    j = {**static_job(), "kind": "adaptive", "updates": 400, "n_envs": 32,
         "rollout_episodes": 33, "checkpoint_every": 50, "monitor_episodes": 200, "device": "cpu"}
    with pytest.raises(ValueError, match="whole vector"):
        jobs.validate_design(design(j))


def test_selected_parameters_cannot_be_overridden_after_selection():
    j = static_job()
    j["defenses"] = [{"name": "selected", "selection": True}]
    with pytest.raises(ValueError, match="requires a frozen"):
        jobs.resolve_job(j, None)
    selection = {"selected_defense": {"name": "winning_candidate", "threshold": .65, "tie_break": "random"}}
    resolved = jobs.resolve_job(j, selection)
    assert resolved["defenses"] == [{"name": "selected", "threshold": .65, "tie_break": "random"}]
    assert j["defenses"][0] == {"name": "selected", "selection": True}
    j["defenses"][0]["threshold"] = .8
    with pytest.raises(ValueError, match="cannot override"):
        jobs.resolve_job(j, selection)


def test_exact_episode_indices_and_raw_outcomes_are_required(tmp_path):
    job = static_job()
    payload, _ = write_synthetic_result(tmp_path, job)
    runner.validate_artifacts(job, tmp_path)
    payload["episodes"][1]["episode_index"] = 2
    runner.write_json(tmp_path / "truthful__original.json", payload)
    with pytest.raises(ValueError, match="complete planned set"):
        runner.validate_artifacts(job, tmp_path)
    payload["episodes"][1]["episode_index"] = 1
    payload["summary"]["false_ejection_rate"] = .75
    runner.write_json(tmp_path / "truthful__original.json", payload)
    with pytest.raises(ValueError, match="raw primary"):
        runner.validate_artifacts(job, tmp_path)


def test_completion_rejects_tampering_even_when_summary_unchanged(tmp_path):
    job = static_job()
    write_synthetic_result(tmp_path, job)
    marker = {"job_sha256": jobs.digest(job), "protocol_sha256": "abc", "selection_sha256": None,
              "returncode": 0, "artifacts": runner.artifact_manifest(tmp_path)}
    runner.write_json(tmp_path / "complete.json", marker)
    assert runner.completed(job, tmp_path, "abc", None)
    path = tmp_path / "truthful__original.json"
    path.write_text(path.read_text() + "\n")
    assert not runner.completed(job, tmp_path, "abc", None)
    assert not runner.completed(job, tmp_path, "different", None)


def test_final_launch_requires_frozen_selection_for_same_protocol_and_source(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    runner.write_json(protocol_path, {"any": "bytes"})
    protocol = {"source_sha256": "frozen_source"}
    with pytest.raises(ValueError, match="require --selection"):
        runner.load_selection(None, protocol_path, protocol, "final")
    selection_path = tmp_path / "selection.json"
    selection = {"status": "frozen", "protocol_sha256": runner.sha256(protocol_path), "source_sha256": "frozen_source"}
    runner.write_json(selection_path, selection)
    assert runner.load_selection(selection_path, protocol_path, protocol, "final")[0] == selection
    selection["source_sha256"] = "wrong_source"
    runner.write_json(selection_path, selection)
    with pytest.raises(ValueError, match="selection source"):
        runner.load_selection(selection_path, protocol_path, protocol, "final")


def test_preserve_undefined_metrics_and_every_record():
    value = [{"conditional": float("nan"), "outcome": False}, {"conditional": .5, "outcome": True}]
    assert runner.json_safe(value) == [{"conditional": None, "outcome": False}, {"conditional": .5, "outcome": True}]


def test_decision_records_omit_unretained_batch_suffix_and_preserve_episode_links():
    kept = [object(), object()]
    discarded = object()
    records = [{"_state_token": id(discarded), "voter": 0},
               {"_state_token": id(kept[1]), "voter": 2},
               {"_state_token": id(kept[0]), "voter": 1}]
    assert runner.retained_decisions(records, kept, 37) == [
        {"episode_index": 1, "eval_seed": 37, "voter": 2},
        {"episode_index": 0, "eval_seed": 37, "voter": 1}]
    assert "_state_token" in records[0]


def test_decision_audit_requires_exact_actual_honest_ballots(tmp_path):
    job = static_job()
    payload, _ = write_synthetic_result(tmp_path, job)
    for episode in payload["episodes"]:
        episode.update(roles=[0, 1], votes=[2, 0])
    payload["defense_decisions"] = [{"episode_index": i, "eval_seed": 200, "voter": 0,
                                    "n_agents": 2, "skip": True, "target": None} for i in (0, 1)]
    runner.write_json(tmp_path / "truthful__original.json", payload)
    runner.validate_artifacts(job, tmp_path)
    payload["defense_decisions"][1]["episode_index"] = 2
    runner.write_json(tmp_path / "truthful__original.json", payload)
    with pytest.raises(ValueError, match="retained honest ballots"):
        runner.validate_artifacts(job, tmp_path)


def test_exclusive_freeze_write_does_not_replace_existing_protocol(tmp_path):
    p = tmp_path / "protocol.json"
    runner.write_json(p, {"first": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.write_json(p, {"second": True}, exclusive=True)
    assert json.loads(p.read_text()) == {"first": True}


def test_reported_stage_cannot_shrink_frozen_budget(tmp_path):
    job = {**static_job(), "kind": "adaptive", "updates": 400, "rollout_episodes": 4,
           "checkpoint_every": 1}
    payload, result = write_synthetic_result(tmp_path, job)
    (tmp_path / "truthful__original.json").rename(tmp_path / "original__original.json")
    result["evaluations"][0].update(cell="original__original", path="original__original.json")
    stage_dir = tmp_path / "coalition_vs_original"
    stage_dir.mkdir()
    runner.write_json(stage_dir / "history.json", {"updates_completed": 2, "history": [
        {"updates_completed": 1, "episodes": 4}, {"updates_completed": 2, "episodes": 8}]})
    result["stages"] = [{"path": "coalition_vs_original", "updates": 2, "seed": job["seed"]}]
    runner.write_json(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="seed or budget differs"):
        runner.validate_artifacts(job, tmp_path)


def test_build_design_has_all_fresh_seeds_and_mixed_development_regimes():
    inputs = {"attacks": [{"development_seed": seed, "name": f"legacy_{attack}",
                           "path": f"validation/inputs/{attack}_{seed}.pt", "sha256": "a" * 64,
                           "environment": "full_short_game", "env_overrides": {"dependence_window": 3}}
                          for seed in range(100, 108) for attack in ("mean", "soft", "hypothesis")]}
    d = jobs.build_design(inputs)
    assert len(d["jobs"]) == 224
    dev = [j for j in d["jobs"] if j["phase"] == "development"]
    assert {j["seed"] for j in dev} == set(range(100, 108))
    assert len(dev[0]["defenses"]) == 14
    assert [o["environment"] for o in dev[0]["opponents"]] == ["meeting_only"] * 4 + ["full_short_game"] * 3
    for kind in ("static", "adaptive", "matched_cycle"):
        final = [j for j in d["jobs"] if j["phase"] == "final" and j["kind"] == kind]
        assert {j["seed"] for j in final} == set(range(1000, 1072))
        assert all(j["episodes"] == 1000 for j in final)
