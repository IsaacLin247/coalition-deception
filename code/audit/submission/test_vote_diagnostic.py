"""External diagnostic regressions; production source package stays frozen."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).with_name("replicate_vote_diagnostic.py")
    spec = importlib.util.spec_from_file_location("f8_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ballot_necessity_does_not_imply_coalition_alone_sufficiency(diagnostic):
    # Coalition splits its votes. The innocent wins 3–2 overall, but coalition-only is tied.
    roles = [1, 1, 0, 0, 0]
    votes = [2, 3, 3, 2, 2]
    row = diagnostic.ballot_metrics(roles, votes, 2)
    assert row["crew_only_ejected"] == 2  # This particular example is crew-sufficient.
    assert row["sufficient_alone_for_observed_false_ejection"] == 0
    # One crew vote for each target plus a skip: coalition tie cannot break a full tie either.
    # Instead use coalition disagreement with 3 different candidates and a 2–1–1 full result.
    votes = [2, 3, 4, 2, 5]
    row = diagnostic.ballot_metrics(roles, votes, 2)
    assert row["necessary_for_observed_false_ejection"] == 1
    assert row["sufficient_alone_for_observed_false_ejection"] == 0
    assert row["crew_votes_for_ejected"] == 1


def test_coalition_sufficiency_does_not_imply_necessity(diagnostic):
    roles, votes = [1, 1, 0, 0, 0], [2, 2, 5, 2, 2]
    row = diagnostic.ballot_metrics(roles, votes, 2)
    assert row["sufficient_alone_for_observed_false_ejection"] == 1
    assert row["necessary_for_observed_false_ejection"] == 0


def test_ballot_removal_can_eject_a_different_innocent(diagnostic):
    roles, votes = [1, 1, 0, 0, 0], [2, 2, 3, 2, 3]
    row = diagnostic.ballot_metrics(roles, votes, 2)
    assert row["necessary_for_observed_false_ejection"] == 1
    assert row["crew_only_ejected"] == 3
    assert row["crew_only_any_false_ejection"] == 1


def test_complete_tally_detects_wrong_ejection_and_skips(diagnostic):
    assert diagnostic.plurality([5]*5, 5)[0] == -1
    assert diagnostic.plurality([2, 3, 5, 5, 5], 5)[0] == -1
    with pytest.raises(ValueError, match="complete plurality"):
        diagnostic.ballot_metrics([1,1,0,0,0], [2,2,5,5,5], 3)


def test_zero_denominators_and_single_seed_sd_are_undefined(diagnostic):
    metrics = diagnostic.summarize_records([])
    assert metrics["false_ejection_rate"] is None
    assert metrics["coalition_ballots_necessary_share_of_fe"] is None
    result = diagnostic.summarize_seeds([{"n_agents":7,"seed":0,"trained":"soft","against":"mean","metrics":metrics}])
    cell = result["n7|soft|mean"]["metrics"]["false_ejection_rate"]
    assert cell == {"mean":None,"sd":None,"defined_seeds":0,"total_seeds":1}


def test_fixed_jobs_reject_missing_and_duplicate_seeds(diagnostic):
    path = (Path(__file__).resolve().parents[2] / "submission_protocol.json")
    study = json.loads(path.read_text())
    jobs = diagnostic.required_jobs(study)
    assert len(jobs) == 15
    assert len([j for j in jobs if "_n7_" in j["name"]]) == 10
    study["jobs"] = [j for j in study["jobs"] if j["name"] != jobs[0]["name"]]
    with pytest.raises(ValueError, match="omits prescribed"):
        diagnostic.required_jobs(study)
    study["jobs"] = jobs + [jobs[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.required_jobs(study)


def test_seed_draws_are_cell_independent_and_separate(diagnostic):
    assert diagnostic.episode_seeds(7, 0, 3) == diagnostic.episode_seeds(7, 0, 3)
    assert diagnostic.episode_seeds(7, 0, 3) != diagnostic.episode_seeds(7, 0, 4)
    assert diagnostic.episode_seeds(7, 0, 3)[0] not in (987654321,1987654321)
    assert diagnostic.episode_seeds(7, 0, 3)[0] != diagnostic.episode_seeds(9, 0, 3)[0]


def test_source_manifest_tampering_rejected(diagnostic, tmp_path):
    (tmp_path/"pyproject.toml").write_text("original")
    files = {"pyproject.toml":diagnostic.sha_file(tmp_path/"pyproject.toml")}
    study = {"sources":files,"source_sha256":diagnostic.digest_json(files)}
    diagnostic.verify_source(tmp_path, study)
    (tmp_path/"pyproject.toml").write_text("changed")
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        diagnostic.verify_source(tmp_path, study)


def test_completion_gate_never_accepts_partial_job_set(diagnostic, monkeypatch):
    study = json.loads((Path(__file__).resolve().parents[2] / "submission_protocol.json").read_text())
    called = []
    def unavailable(run, n, seed, study, job=None):
        called.append(job["name"])
        raise FileNotFoundError("not completed")
    monkeypatch.setattr(diagnostic, "validate_run", unavailable)
    with pytest.raises(ValueError, match="all 15 jobs"):
        diagnostic.gate_all(Path("unused"), study)
    assert len(called) == 15


def test_completion_source_and_job_hashes_are_both_required(diagnostic, tmp_path):
    import csv
    study = json.loads((Path(__file__).resolve().parents[2] / "submission_protocol.json").read_text())
    job = diagnostic.required_jobs(study)[0]
    run = tmp_path/job["name"]
    run.mkdir()
    args = {"smoke":False,"n_agents":7,"seed":0,"defenses":"soft,mean,hypothesis","name":job["name"]}
    meta = {"args":args,"env_config":{},"algo_config":{}}
    diagnostic.write_json(run/"experiment_config.json",meta)
    diagnostic.write_json(run/"result.json",{**meta,"status":"complete"})
    diagnostic.write_json(run/"runmeta.json",{**meta,"source_sha256":study["source_sha256"]})
    with (run/"crossplay.csv").open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','coalition_trained_vs','defense','n_episodes','false_ejection_rate'])
        writer.writeheader()
        for trained in diagnostic.RULES:
            for against in diagnostic.RULES:
                writer.writerow(dict(seed=0,coalition_trained_vs=trained,defense=against,n_episodes=1000,false_ejection_rate=0.0))
    marker={"name":job["name"],"returncode":0,"artifacts_valid":True,
            "source_sha256":study["source_sha256"],"job_sha256":diagnostic.job_hash(job)}
    path=tmp_path/'_control'/f"{job['name']}.complete.json"
    for key in ['source_sha256','job_sha256']:
        diagnostic.write_json(path,{**marker,key:'incorrect'})
        with pytest.raises(ValueError,match='fingerprint mismatch'):
            diagnostic.validate_run(run,7,0,study,job=job)


def test_intermediate_checkpoint_integrity_checks(diagnostic):
    import copy
    import torch
    meta = {"env_config":{"n_agents":7},"algo_config":{"total_updates":400}}
    final = {**meta,"extra":{"seed":0,"updates":400,"algo":meta["algo_config"]},
             "state_dict":{"weight":torch.ones(2,3)}}
    intermediate = copy.deepcopy(final)
    intermediate["extra"]["updates"] = 50
    diagnostic.validate_checkpoint_blob(intermediate,meta,0,50,"checkpoint_50.pt",reference=final)
    with pytest.raises(ValueError,match='metadata mismatch'):
        diagnostic.validate_checkpoint_blob(intermediate,meta,0,100,"checkpoint_100.pt",reference=final)
    intermediate["state_dict"]["weight"][0,0] = float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        diagnostic.validate_checkpoint_blob(intermediate,meta,0,50,"checkpoint_50.pt",reference=final)
    intermediate["state_dict"]["weight"] = torch.ones(3,2)
    with pytest.raises(ValueError,match='keys/shapes/dtypes'):
        diagnostic.validate_checkpoint_blob(intermediate,meta,0,50,"checkpoint_50.pt",reference=final)


def test_wait_marker_gate_requires_every_matching_completion(diagnostic, tmp_path):
    study = json.loads((Path(__file__).resolve().parents[2] / "submission_protocol.json").read_text())
    jobs = diagnostic.required_jobs(study)
    assert len(diagnostic.pending_markers(tmp_path, study)) == 15
    for job in jobs:
        path=tmp_path/'_control'/f"{job['name']}.complete.json"
        diagnostic.write_json(path,{"name":job['name'],"returncode":0,"artifacts_valid":True,
                                   "source_sha256":study['source_sha256'],"job_sha256":diagnostic.job_hash(job)})
    assert diagnostic.pending_markers(tmp_path, study) == {}
    path=tmp_path/'_control'/f"{jobs[-1]['name']}.complete.json"
    marker=diagnostic.read_json(path);marker['job_sha256']='wrong';diagnostic.write_json(path,marker)
    assert list(diagnostic.pending_markers(tmp_path, study)) == [jobs[-1]['name']]


def test_waiting_sidecar_does_not_create_output_directory(diagnostic, tmp_path):
    from types import SimpleNamespace
    args=SimpleNamespace(out=tmp_path/'result',smoke=True)
    diagnostic.write_status(args,'waiting',missing_jobs=['pending'])
    assert not args.out.exists()
    assert diagnostic.read_json(tmp_path/'result.status.json')['status']=='waiting'
