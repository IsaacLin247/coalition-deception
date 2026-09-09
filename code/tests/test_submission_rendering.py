"""Rendering must preserve estimands, cohort denominators and inference gates."""
import json

import numpy as np
import pytest

from analysis import render_submission as render


def row(group, table, cell, metric, mean, seeds=(0, 1)):
    return dict(group=group, table=table, cell=cell, metric=metric, mean=mean, sd=.02,
                expected_seeds=list(seeds), included_seeds=list(seeds), n_seeds=len(seeds), complete=True)


def test_cycle_matrix_keeps_whole_game_terminal_and_count_outcomes_separate():
    metrics = {"false_ejection_rate": .1, "any_false_ejection_rate": .7,
               "mean_false_ejections": 1.5, "coalition_game_win_rate": .4}
    rows = [row("f4_tenseed_crew5", "crossplay", cell, metric, value + .01 * index)
            for metric, value in metrics.items() for index, cell in enumerate(render.PAIR_COORDINATES)]
    for metric, base in metrics.items():
        matrix, _ = render.cycle_matrix(rows, metric)
        assert matrix.tolist() == [[base, base + .01], [base + .03, base + .02]]
    assert "Terminal-meeting" in render.metric_label("false_ejection_rate", 8)
    assert "At least one" in render.metric_label("any_false_ejection_rate", 8)
    assert "Mean" in render.metric_label("mean_false_ejections", 8)


def test_matched_reference_cannot_be_replaced_by_full_ten_seed_average():
    group = "multigen_none_n7_r1_k10"
    rows = [row(group, "loop_h4", "c_stage_mean", "false_ejection_rate", .2, range(5)),
            row(group, "loop_h4", "c_stage_mean", "false_ejection_rate", .8, range(10))]
    chosen = render.select_matched(rows, {group}, rounds=1, table="loop_h4")
    assert len(chosen) == 1 and chosen[0]["mean"] == .2
    assert chosen[0]["included_seeds"] == list(range(5))
    mismatched = rows + [row("multigen_rwbal_n7_r1_k4", "loop_h4", "c_stage_mean", "false_ejection_rate", .3, range(5, 10))]
    with pytest.raises(ValueError, match="same seed IDs"):
        render.select_matched(mismatched, {group, "multigen_rwbal_n7_r1_k4"}, rounds=1, table="loop_h4")


def test_interim_groups_require_all_planned_job_ids():
    first, second = "f1_tenseed_crew3", "f1_tenseed_crew5"
    status = {"completed": [f"{first}_s0", f"{first}_s1", f"{second}_s0"]}
    per_seed = [{"job": name} for name in status["completed"]]
    summaries = [row(group, "holdout", "final", "false_ejection_rate", .2) for group in (first, second)]
    assert render.complete_groups(status, summaries, per_seed) == {first}
    with pytest.raises(ValueError, match="different completed jobs"):
        render.complete_groups(status, summaries, per_seed[:2])


def test_stale_summary_mean_is_detected_from_checked_seed_values():
    summary = row("f1_tenseed_crew5", "holdout", "final", "false_ejection_rate", .9)
    values = {render.key(summary): {0: .1, 1: .3}}
    with pytest.raises(ValueError, match="mean mismatch"):
        render.checked_summaries([summary], values)


def test_f1_table_uses_holdout_endpoints_and_interim_drops_inference(tmp_path):
    group = "f1_tenseed_crew5"
    cells = []
    for metric in ("false_ejection_rate", "coalition_favorable_rate"):
        cells += [row(group, "holdout", "initial", metric, .1), row(group, "holdout", "final", metric, .2),
                  row(group, "monitored_curve", "0", metric, .8), row(group, "monitored_curve", "400", metric, .9)]
    status = dict(completed_jobs=2, expected_jobs=240)
    renderer = render.Renderer(tmp_path, status,
                               dict(cell_summary=cells, matched_reference_summary=[],
                                    paired_effects=[{"p_exact": .01, "ci95": [.1, .2]}]), {group}, interim=True)
    renderer.render()
    displayed = json.loads((tmp_path / "table_data.json").read_text())["rows"]
    assert {record["table"] for record in displayed} == {"holdout"}
    assert {record["mean"] for record in displayed} == {.1, .2}
    assert renderer.effects == []
    text = (tmp_path / "tables.md").read_text()
    assert "Suppressed in interim" in text and "Exact p" not in text
    assert (tmp_path / "f1_learning_and_holdout.pdf").exists()


def test_stage_order_places_each_defender_before_its_adapted_coalition():
    unordered = ["C10", "D2", "C0", "C1", "D1", "C2", "D10"]
    assert sorted(unordered, key=lambda cell: render.cell_order(cell, "stages")) == ["C0", "D1", "C1", "D2", "C2", "D10", "C10"]


def test_final_mode_refuses_incomplete_status_even_without_partial_tables(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps(dict(status="INTERIM / FINAL INFERENCE BLOCKED", inference_enabled=False)))
    assert render.main(["--analysis", str(tmp_path)]) == 2
    assert not (tmp_path / "presentation").exists()


def test_all_non_f1_templates_accept_complete_schematic_analyzer_cells(tmp_path, monkeypatch):
    """Exercise future-family layouts without pretending fixture values are research results."""
    import matplotlib.pyplot as plt

    cells, matched, groups = [], [], set()

    def add(group, table, condition, metrics):
        groups.add(group)
        for metric in metrics:
            value = 1.2 if metric == "mean_false_ejections" else .2
            cells.append(row(group, table, condition, metric, value, range(10)))

    for group in ("f2_tenseed_crew5", "depstatic_n7_r1"):
        for condition in render.CONDITIONS:
            for rule in render.RULES:
                add(group, "rules", f"{condition}|{rule}", ["false_ejection_rate"])
    for family in ("f3", "f4"):
        metrics = (["false_ejection_rate", "coalition_favorable_rate"] if family == "f3" else
                   ["any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"])
        for cell in render.PAIR_COORDINATES:
            add(f"{family}_tenseed_crew5", "crossplay", cell, metrics)
    for group, defenses in (("counterattack_hyp_n7_r1", ("soft", "mean", "hypothesis")),
                            ("counterattack_n7_r1", ("soft", "rule", "both"))):
        for first in defenses:
            for second in defenses:
                add(group, "crossplay", f"{first}|{second}", ["false_ejection_rate", "coalition_favorable_rate"])
    for group in ("multigen_none_n7_r8_k4", "multigen_none_n7_r1_k10", "multigen_rwbal_n7_r1_k4"):
        info = render.scope(group)
        metrics = (["false_ejection_rate", "coalition_favorable_rate"] if info["rounds"] == 1 else
                   ["any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"])
        add(group, "stages", "C0", metrics)
        for generation in range(1, info["generations"] + 1):
            for side in ("C", "D"):
                add(group, "stages", f"{side}{generation}", metrics)
        for i in range(info["generations"] + 1):
            for j in range(info["generations"] + 1):
                add(group, "crossplay", f"C{i}|D{j}", metrics)
            add(group, "restricted_pool_gap", str(i), [metrics[0]])
        for descriptor in ("c_stage_mean", "d_stage_mean", "matrix_adaptation_gain"):
            add(group, f"loop_h{info['generations']}", descriptor, metrics)
    for rounds in (1, 8):
        metrics = (["false_ejection_rate"] if rounds == 1 else
                   ["any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"])
        metrics += ["same_target_vote_rate", "coalition_false_claim_rate", "creator_survival"]
        for condition in render.ABLATIONS:
            add(f"ablation_n7_r{rounds}", "ablation", condition, metrics)
    groups.add("f3_budget_d800_crew5")
    for group in ("f3_budget_d800_crew5", "f3_tenseed_crew5"):
        for cell in ("coalition0_vs_learned_crew1", "coalition1_vs_learned_crew1"):
            matched.append(row(group, "crossplay", cell, "false_ejection_rate", .12, range(5)))
    for group in ("multigen_none_n7_r1_k10", "multigen_rwbal_n7_r1_k4"):
        for descriptor in ("c_stage_mean", "d_stage_mean", "matrix_adaptation_gain"):
            matched.append(row(group, "loop_h4", descriptor, "false_ejection_rate", .13, range(5)))
    renderer = render.Renderer(tmp_path, dict(completed_jobs=100, expected_jobs=240),
                               dict(cell_summary=cells, matched_reference_summary=matched, paired_effects=[]), groups, interim=True)
    def close_figure(figure, identifier, caption):
        renderer.figures.append(dict(file=identifier, caption=caption))
        plt.close(figure)
    monkeypatch.setattr(renderer, "savefig", close_figure)
    renderer.render()
    text = (tmp_path / "tables.md").read_text()
    assert "Terminal-meeting false ejection" in text and "Mean false ejections / game" in text
    assert "matched five-seed cohort" in text
    assert "0.120 ± 0.020" in text and "0.130 ± 0.020" in text
    assert any("response_gaps" in record["file"] for record in renderer.figures)
    assert renderer.effects == []
