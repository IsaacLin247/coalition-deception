"""Regression tests for seed pairing, multiplicity and the fresh-result gate."""

import json
import hashlib

import numpy as np
import pytest

from analysis import analyze_submission as analysis
from experiments.submission.run_study import study_jobs


def test_exact_mean_flips_and_holm_keep_magnitudes_and_missing_family_members():
    assert analysis.exact_mean_sign_flip([3, 2, -1]) == .5
    assert analysis.exact_mean_sign_flip([1] * 5) == .0625
    assert analysis.exact_mean_sign_flip([1] * 10) == .001953125
    assert analysis.holm([.01, .04, None]) == [.03, .08, None]
    assert analysis.holm([.001953125] * 26) == [.05078125] * 26
    with pytest.raises(ValueError):
        analysis.exact_mean_sign_flip([1, np.nan])


def test_contrasts_pair_seed_ids_not_input_order_and_interim_suppresses_inference():
    data = analysis.Dataset()
    a, b = ("g", "t", "a", "m"), ("g", "t", "b", "m")
    data.cells[a] = {1: .4, 0: .9}
    data.cells[b] = {0: .1, 1: .3}
    comparison = analysis.Comparison("f", "contrast", [(1., a), (-1., b)], [0, 1])
    result = analysis.comparison_results([comparison], data, infer=True)[0]
    assert result["per_seed_differences"] == pytest.approx([.8, .1])
    assert result["mean"] == pytest.approx(.45)
    assert result["ci95"] == pytest.approx([.1, .8])
    interim = analysis.comparison_results([comparison], data, infer=False)[0]
    assert interim["p_exact"] is interim["p_holm"] is interim["ci95"] is None


def test_reference_summaries_use_planned_common_seeds():
    jobs = study_jobs()
    comparisons, _ = analysis.planned_comparisons(jobs)
    comparison = next(c for c in comparisons if c.family == "reward_controls")
    assert comparison.seeds == list(range(5))
    data = analysis.Dataset()
    for _, key in comparison.terms:
        data.cells[key] = {s: .1 if s < 5 else .9 for s in range(10)}
    summaries = analysis.matched_reference_summaries([comparison], data)
    assert all(row["included_seeds"] == list(range(5)) for row in summaries)
    assert all(row["mean"] == pytest.approx(.1) for row in summaries)


def test_response_gap_uses_correct_matrix_orientation_and_reports_mc_uncertainty():
    rows = analysis.restricted_response_gaps([[.2, .8], [.4, .6]], 500)
    assert [r["gap"] for r in rows] == pytest.approx([.2, .4])
    assert [r["best_observed_coalition"] for r in rows] == [1, 0]
    assert [r["best_observed_defender"] for r in rows] == [0, 0]
    assert all(r["mc_low"] <= r["gap"] <= r["mc_high"] for r in rows)
    with pytest.raises(ValueError):
        analysis.restricted_response_gaps([[.2, np.nan], [.4, .6]], 500)


def test_raw_harm_reconstruction_separates_terminal_whole_game_and_pooled_rates():
    rows = [dict(false_ejection="True", coalition_favorable="True", coalition_game_won="False",
                 total_false_ejections="1", total_meetings="1"),
            dict(false_ejection="False", coalition_favorable="False", coalition_game_won="True",
                 total_false_ejections="2", total_meetings="4")]
    summary = analysis.episode_summary(rows)
    assert summary["false_ejection_rate"] == .5
    assert summary["any_false_ejection_rate"] == 1.
    assert summary["mean_false_ejections"] == 1.5
    assert summary["false_ejections_per_meeting"] == .6
    wrong = dict(summary, false_ejections_per_meeting=.75)  # mean per-game rates
    with pytest.raises(ValueError, match="false_ejections_per_meeting"):
        analysis.check_episode_summary(wrong, rows, 8)


def fake_f1(tmp_path, seed, fingerprint="fresh", holdout_base=1987654321):
    job = dict(name=f"f1_tenseed_crew5_s{seed}", family="f1", artifacts=["result.json", "history.json", "checkpoint_final.pt"])
    directory = tmp_path / "runs" / job["name"]
    directory.mkdir(parents=True)
    metrics = {"false_ejection_rate": .2, "coalition_favorable_rate": .4}
    analysis.write_json(directory / "result.json", dict(holdout_seed=holdout_base + seed,
                                                       initial_holdout=metrics, final_holdout=metrics))
    analysis.write_json(directory / "history.json", dict(history=[{}], updates_completed=1,
                                                        final_metrics=metrics, eval_curve=[]))
    analysis.write_json(directory / "runmeta.json", dict(source_sha256=fingerprint))
    (directory / "checkpoint_final.pt").write_bytes(b"fixture only")
    analysis.write_json(tmp_path / "runs/_control" / f"{job['name']}.complete.json",
                        dict(source_sha256=fingerprint, returncode=0,
                             job_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()))
    return job


def test_final_requires_every_seed_and_fresh_provenance(tmp_path):
    first = fake_f1(tmp_path, 0)
    second = dict(first, name="f1_tenseed_crew5_s1")
    protocol = tmp_path / "protocol.json"
    analysis.write_json(protocol, dict(study="post_audit_replication_20260909", source_sha256="fresh", jobs=[first, second]))
    args = ["--protocol", str(protocol), "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]
    assert analysis.main(args) == 2
    assert not (tmp_path / "out/paired_effects.csv").exists()
    assert analysis.main(args + ["--interim"]) == 0
    rows = analysis.read_json(tmp_path / "out/paired_effects.json")["rows"]
    assert all(r["expected_seeds"] == [0, 1] and r["included_seeds"] == [0] for r in rows)
    assert all(r["p_exact"] is r["ci95"] is None for r in rows)
    fake_f1(tmp_path, 1, fingerprint="archived")
    assert analysis.main(args) == 2
    assert not (tmp_path / "out/paired_effects.csv").exists()  # stale interim moved aside
    assert (tmp_path / "out/superseded").exists()


def test_f1_rejects_monitored_evidence_as_final_holdout(tmp_path):
    job = fake_f1(tmp_path, 0, holdout_base=987654321)
    with pytest.raises(ValueError, match="independent final evidence"):
        analysis.ingest(job, tmp_path / "runs" / job["name"], analysis.Dataset())


def test_complete_protocol_emits_final_paired_inference(tmp_path):
    jobs = [fake_f1(tmp_path, seed) for seed in range(2)]
    protocol = tmp_path / "protocol.json"
    analysis.write_json(protocol, dict(study="post_audit_replication_20260909", source_sha256="fresh", jobs=jobs))
    assert analysis.main(["--protocol", str(protocol), "--runs", str(tmp_path / "runs"),
                          "--out", str(tmp_path / "out")]) == 0
    assert analysis.read_json(tmp_path / "out/status.json")["inference_enabled"]
    contrasts = analysis.read_json(tmp_path / "out/paired_effects.json")["rows"]
    assert all(c["included_seeds"] == [0, 1] and c["p_exact"] == c["p_holm"] == 1. for c in contrasts)
