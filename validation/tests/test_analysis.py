"""Independent synthetic data regressions for the publication inference gate."""
from __future__ import annotations

import copy
import json

import pytest

from validation import analyze


def _specs():
    return [
        {"name": "hypothesis_index_050", "tie_break": "index", "threshold": .5, "honesty": .5},
        {"name": "selected", "tie_break": "index", "threshold": .5, "honesty": .5},
        {"name": "soft", "rule": "soft_credibility"},
    ]


def _job(kind="static", seed=1000):
    return dict(name=f"{kind}_{seed}", kind=kind, phase="final", seed=seed,
                eval_seed=3200000000 + seed, n_agents=7,
                max_rounds=8 if kind == "matched_cycle" else 1,
                episodes=1000, defenses=_specs(), opponents=[{"name": "truthful"}])


def _payload(fe_episodes=37):
    job = _job()
    cell = "truthful__selected"
    episodes, meetings, decisions = [], [], []
    for idx in range(1000):
        false_ejection = idx < fe_episodes
        votes = [7, 7, 7, 7, -1, 0 if false_ejection else 7, 0 if false_ejection else 7]
        alive = [not false_ejection, True, True, True, False, True, True]
        ejected = 0 if false_ejection else -1
        episodes.append(dict(
            episode_index=idx, eval_seed=job["eval_seed"], pair=cell,
            roles=[0, 0, 0, 0, 0, 1, 1], votes=votes, alive=alive,
            ejected=ejected, incident_creator=5,
            false_ejection=false_ejection, any_false_ejection=false_ejection,
            total_false_ejections=int(false_ejection), total_meetings=1,
            creator_ejected=False, no_ejection=not false_ejection, coalition_game_won=False,
        ))
        meetings.append(dict(episode_index=idx, eval_seed=job["eval_seed"], pair=cell,
                             round=0, ejected=ejected, false_ejection=int(false_ejection),
                             any_ejection=int(false_ejection)))
        for voter in range(4):
            tied = [j for j in range(7) if j != voter and j != 4]
            decisions.append(dict(episode_index=idx, eval_seed=job["eval_seed"], voter=voter,
                                  round=0, target=None, skip=True, threshold=.5, honesty=.5,
                                  tie_break="index", tied_candidates=tied, tie_count=len(tied)))
    summary = dict(n_episodes=1000, false_ejection_rate=fe_episodes / 1000,
                   any_false_ejection_rate=fe_episodes / 1000, mean_false_ejections=fe_episodes / 1000,
                   creator_ejection_rate=0., no_ejection_rate=1 - fe_episodes / 1000,
                   coalition_game_win_rate=None)
    payload = dict(cell=cell, episodes=episodes, meetings=meetings, summary=summary,
                   defense_decisions=decisions,
                   environment=analyze.resolved_environment(job, cell))
    return payload, job, cell


def _contrast_rows():
    needed = {(kind, cell) for _, _, terms in analyze.PRIMARY for kind, cell, _, _ in terms}
    rows = []
    for seed in analyze.FINAL_SEEDS:
        for kind, cell in sorted(needed):
            counts = dict(false_ejection=100, any_false_ejection=200,
                          coalition_game_won=300, total_false_ejections=250)
            if cell == "C1__D1":
                counts.update(coalition_game_won=400, any_false_ejection=210, total_false_ejections=180)
            if cell == "selected__selected":
                counts["false_ejection"] = 180
            rows.append(dict(kind=kind, seed=seed, cell=cell, denominator=1000, counts=counts))
    return rows


def test_independent_ballot_and_role_reconstruction_recovers_integer_counts():
    payload, job, cell = _payload()
    verified = analyze.verify_cell(payload, job, cell, payload["summary"])
    assert verified["counts"]["false_ejection"] == 37
    assert verified["counts"]["total_false_ejections"] == 37
    assert verified["terminal_honest_votes"] == 4000  # Includes the innocent just ejected.
    assert verified["decision_records"] == 4000
    assert verified["rates"]["false_ejection"] == .037


def test_changed_role_with_unchanged_labels_is_detected_independently():
    payload, job, cell = _payload()
    payload["episodes"][0]["roles"][0] = 1
    payload["episodes"][0]["roles"][5] = 0
    with pytest.raises(ValueError, match="creator|outcome|role/ejection"):
        analyze.verify_cell(payload, job, cell)


def test_changed_plurality_ballots_rejected_even_when_all_summary_labels_agree():
    payload, job, cell = _payload()
    payload["episodes"][0]["votes"][5:] = [2, 2]
    with pytest.raises(ValueError, match="plurality tally"):
        analyze.verify_cell(payload, job, cell)


@pytest.mark.parametrize("field,value", [("any_false_ejection", False), ("total_false_ejections", 2)])
def test_meeting_logs_independently_detect_changed_whole_game_outcomes(field, value):
    payload, job, cell = _payload()
    payload["episodes"][0][field] = value
    with pytest.raises(ValueError, match="outcome|meeting rows|innocent-ejection count"):
        analyze.verify_cell(payload, job, cell)


def test_summary_rounding_is_not_used_to_reconstruct_counts():
    payload, job, cell = _payload(fe_episodes=37)
    payload["summary"]["false_ejection_rate"] = .04
    with pytest.raises(ValueError, match="summary does not match raw integer"):
        analyze.verify_cell(payload, job, cell)


@pytest.mark.parametrize("corrupt", ["duplicate_index", "missing_row", "extra_row", "missing_meeting"])
def test_complete_fixed_episode_and_meeting_sets_required(corrupt):
    payload, job, cell = _payload()
    if corrupt == "duplicate_index":
        payload["episodes"][1]["episode_index"] = 0
    elif corrupt == "missing_row":
        payload["episodes"].pop()
    elif corrupt == "extra_row":
        payload["episodes"].append(copy.deepcopy(payload["episodes"][-1]))
    else:
        payload["meetings"].pop()
    with pytest.raises(ValueError, match="episode|meeting"):
        analyze.verify_cell(payload, job, cell)


@pytest.mark.parametrize("corrupt", ["extra_game", "post_ejection_omission", "wrong_target", "duplicate_voter", "leaked_token"])
def test_diagnostics_must_match_retained_pre_ejection_electorate(corrupt):
    payload, job, cell = _payload()
    records = payload["defense_decisions"]
    if corrupt == "extra_game":
        records.append({**records[-1], "episode_index": 1000})
    elif corrupt == "post_ejection_omission":
        records.pop(0)
    elif corrupt == "wrong_target":
        records[0]["target"], records[0]["skip"] = 1, False
    elif corrupt == "duplicate_voter":
        records[1]["voter"] = records[0]["voter"]
    else:
        records[0]["_state_token"] = 123456789
    with pytest.raises(ValueError, match="audit|state identity"):
        analyze.verify_cell(payload, job, cell)


def test_primary_six_use_integer_counts_and_direct_p6_difference():
    results = analyze.primary_results(_contrast_rows())
    assert [r["contrast"] for r in results] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert results[3]["integer_differences"] == [100] * 72
    assert results[4]["integer_differences"] == [-70] * 72
    assert results[5]["integer_differences"] == [90] * 72
    assert results[4]["mean"] == pytest.approx(-.07)
    assert results[5]["mean"] == pytest.approx(.09)
    assert all(r["family_size"] == 6 for r in results)
    assert all(r["family"] == "post_review_validation" for r in results)


def test_alias_selected_original_retains_zero_contrasts_and_p_one():
    rows = _contrast_rows()
    for row in rows:
        if row["cell"] == "selected__selected":
            row["counts"]["false_ejection"] = 100
    results = analyze.primary_results(rows)
    for index in (0, 2):
        assert results[index]["mean"] == 0
        assert results[index]["p_exact"] == results[index]["p_holm"] == 1
        assert not results[index]["reject_holm_005"]


@pytest.mark.parametrize("corrupt", ["missing_seed", "duplicate_row", "extra_seed", "floating_count", "wrong_denominator"])
def test_primary_gate_rejects_cohort_and_numerator_changes(corrupt):
    rows = _contrast_rows()
    if corrupt == "missing_seed":
        rows = [row for row in rows if row["seed"] != 1071]
    elif corrupt == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif corrupt == "extra_seed":
        rows.append({**rows[0], "seed": 1080})
    elif corrupt == "floating_count":
        rows[0]["counts"]["false_ejection"] = 100.0
    else:
        rows[0]["denominator"] = 2000
    with pytest.raises(ValueError):
        analyze.primary_results(rows)


def test_final_grid_requires_all_three_complete_72_seed_cohorts():
    jobs = [_job(kind, seed) for kind in analyze.KINDS for seed in analyze.FINAL_SEEDS]
    analyze.verify_final_grid(jobs)
    jobs[-1]["seed"] = 1000
    with pytest.raises(ValueError, match="missing, duplicate, or substituted"):
        analyze.verify_final_grid(jobs)


def test_incomplete_run_only_writes_counts_and_removes_stale_inference(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    jobs = [_job(kind, seed) for kind in analyze.KINDS for seed in analyze.FINAL_SEEDS]
    development = [dict(_job(seed=seed), phase="development") for seed in range(100, 108)]
    protocol = dict(design={"jobs": development + jobs}, source_sha256="source")
    monkeypatch.setattr(analyze.runner, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(analyze.runner, "verify_source", lambda value: None)
    monkeypatch.setattr(analyze.runner, "verify_inputs", lambda value: None)
    monkeypatch.setattr(analyze.runner, "load_selection", lambda *args: ({"status": "frozen"}, "selection"))
    monkeypatch.setattr(analyze.runner, "completed", lambda job, *args: job["phase"] == "development")
    out = tmp_path / "analysis"
    out.mkdir()
    (out / "primary.json").write_text('[{"p_holm": 0.01}]')
    status = analyze.analyze(protocol_path, tmp_path / "results", None, out)
    assert not status["inference_enabled"]
    assert status["complete_development_jobs"] == 8
    assert status["complete_final_jobs"] == 0
    assert len(status["incomplete_jobs"]) == 216
    assert sorted(path.name for path in out.iterdir()) == ["status.json"]
    assert "p_exact" not in (out / "status.json").read_text()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_literals(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            analyze.read_json(path)


def test_smoke_settings_require_two_flags_and_explicit_complete_budget():
    design = dict(smoke_only=True, analysis=dict(smoke_only=True, final_seed_count=2,
                  episodes_per_cell=16), selection=dict(development_seeds=[100, 101]),
                  primary_inference=dict(episode_denominator=16, lattice_scale=16, family_size=6),
                  jobs=[_job(kind, seed) for kind in analyze.KINDS for seed in (3000, 3001)])
    for job in design["jobs"]:
        job["episodes"] = 16
    settings = analyze.analysis_settings(design)
    assert settings == ([3000, 3001], 16, [100, 101], True)
    analyze.verify_final_grid(design["jobs"], settings[0], settings[1])
    design["analysis"]["smoke_only"] = False
    with pytest.raises(ValueError, match="explicit flags"):
        analyze.analysis_settings(design)


def test_metadata_cannot_change_primary_algebra_or_production_denominator():
    metadata = json.loads((analyze.HERE / "design_metadata.json").read_text())
    assert analyze.analysis_settings(metadata)[0] == analyze.FINAL_SEEDS
    metadata["primary_inference"]["primary_contrasts"][5]["terms"][-1][0] = -1
    with pytest.raises(ValueError, match="algebra"):
        analyze.analysis_settings(metadata)
    with pytest.raises(ValueError, match="production denominator"):
        analyze.analysis_settings({"analysis": {"episodes_per_cell": 16}})


def test_paired_t_sensitivity_keeps_constant_cases_and_primary_unchanged():
    rows = _contrast_rows()
    baseline = analyze.primary_results(rows)
    sensitivity = analyze.primary_results(rows, t_sensitivity=True)
    for first, second in zip(baseline, sensitivity):
        assert all(second[key] == value for key, value in first.items())
    assert sensitivity[0]["t_sensitivity_p"] == 1
    assert sensitivity[3]["t_sensitivity_p"] == 0
    # A varying constructed 72-seed contrast must produce a finite sensitivity p.
    for row in rows:
        if row["cell"] == "selected__selected":
            row["counts"]["false_ejection"] = row["seed"] - 900
    varied = analyze.primary_results(rows, t_sensitivity=True)
    assert 0 < varied[1]["t_sensitivity_p"] < 1


def test_small_smoke_primary_cohort_uses_its_integer_denominator():
    rows = [r for r in _contrast_rows() if r["seed"] in (1000, 1001)]
    for row in rows:
        row["denominator"] = 16
        row["counts"] = {key: value // 50 for key, value in row["counts"].items()}
    result = analyze.primary_results(rows, [1000, 1001], 16)
    assert result[3]["integer_differences"] == [2, 2]
    assert result[3]["denominator"] == 16
    assert result[3]["mean"] == .125
    assert result[3]["p_exact"] == .5


def test_multi_round_counts_and_terminal_parity_reconstructed_from_raw_records():
    payload, job, cell = _payload(fe_episodes=1000)
    job.update(kind="matched_cycle", max_rounds=8)
    cell = "C1__D1"
    payload["cell"] = cell
    payload["environment"] = analyze.resolved_environment(job, cell)
    payload["defense_decisions"] = []
    payload["meetings"] = []
    for row in payload["episodes"]:
        row.update(pair=cell, ejected=1, votes=[-1, 7, 7, 7, -1, 1, 1],
                   alive=[False, False, True, True, False, True, True],
                   total_meetings=2, total_false_ejections=2, coalition_game_won=True)
        for round_index in range(2):
            payload["meetings"].append(dict(pair=cell, eval_seed=job["eval_seed"],
                episode_index=row["episode_index"], round=round_index, ejected=round_index,
                false_ejection=1, any_ejection=1))
    payload["summary"].update(mean_false_ejections=2., coalition_game_win_rate=1.)
    result = analyze.verify_cell(payload, job, cell)
    assert result["counts"]["total_false_ejections"] == 2000
    assert result["counts"]["coalition_game_won"] == 1000
    assert result["rates"]["total_false_ejections"] == 2


@pytest.mark.parametrize("field,value", [
    ("actor_private_history", False), ("incident_time_public", False),
    ("coalition_objective", "false_ejection"), ("synthetic_evidence", True),
])
def test_full_environment_guard_rejects_regime_changes_with_unchanged_population(field, value):
    payload, job, cell = _payload()
    assert payload["environment"][field] != value
    payload["environment"][field] = value
    with pytest.raises(ValueError, match="full cell environment"):
        analyze.verify_cell(payload, job, cell)


def test_static_opponent_environment_and_override_precedence_are_verified():
    payload, job, cell = _payload()
    job["env_overrides"] = {"incident_time_public": False, "actor_private_history": False}
    job["opponents"][0].update(environment="meeting_only", env_overrides={"incident_time_public": True})
    expected = analyze.resolved_environment(job, cell)
    assert expected["synthetic_evidence"] is True
    assert expected["incident_time_public"] is True
    assert expected["actor_private_history"] is False
    with pytest.raises(ValueError, match="full cell environment"):
        analyze.verify_cell(payload, job, cell)
    payload["environment"] = json.loads(json.dumps(expected))
    assert analyze.verify_cell(payload, job, cell)["counts"]["false_ejection"] == 37
