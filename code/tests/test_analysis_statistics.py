"""Regression checks for the statistical estimands reported in the paper."""

import csv
import json
import math

import pytest

from analysis import analyze_ablation_budget as ablation
from analysis import analyze_revision as revision


@pytest.mark.parametrize("sign_flip", [ablation.exact_sign_p, revision.exact_sign_p])
def test_sign_flip_uses_difference_magnitudes(sign_flip):
    # The binomial sign test returns 1 here. Enumerating the eight signed means
    # gives four outcomes at least as extreme as the observed mean 4/3.
    assert sign_flip([3.0, 2.0, -1.0]) == 0.5
    assert sign_flip([1.0] * 5) == 0.0625
    assert sign_flip([0.0, 0.0]) == 1.0
    assert math.isnan(sign_flip([float("nan")]))


def test_reward_reference_uses_the_control_seed_intersection(monkeypatch, tmp_path):
    def run(level):
        rows = [{"generation": 0, "side": "C", "coalition_favorable_rate": level,
                 "false_ejection_rate": level}]
        for generation in range(1, 5):
            rows.extend({"generation": generation, "side": side,
                         "coalition_favorable_rate": level,
                         "false_ejection_rate": level} for side in ("D", "C"))
        return {"rows": rows, "crossplay": None}

    def fake_load(_root, pattern):
        runs = {0: run(0.1), 1: run(0.3)}
        if pattern.startswith("multigen_none"):
            runs[2] = run(0.9)  # available in the reference only
        return runs

    monkeypatch.setattr(revision, "load_loop", fake_load)
    monkeypatch.setattr(revision, "MG_FIG", tmp_path)
    monkeypatch.setattr(revision, "save_fig", lambda figure, path: None)
    result = revision.analyze_reward(tmp_path, 1, [])
    for variant in result["variants"].values():
        assert variant["summary"]["seeds"] == [0, 1]
        assert variant["summary"]["n_seeds"] == 2
        assert variant["summary"]["generation0_mean"] == pytest.approx(0.2)


def test_ablation_uses_incident_gated_survival_not_buggy_complement(monkeypatch, tmp_path):
    for rounds in (1, 8):
        directory = tmp_path / f"ablation_n7_r{rounds}_s0"
        directory.mkdir()
        rows = [dict(max_rounds=rounds, condition=condition, false_ejection_rate=0.2,
                     same_target_vote_rate=0.3, coalition_false_claim_rate=0.4,
                     coalition_game_win_rate=0.5, creator_survival=0.1,
                     creator_survival_rate=0.9) for condition in ablation.CONDITIONS]
        (directory / "result.json").write_text(json.dumps({"rows": rows}))
    monkeypatch.setattr(ablation.plt.Figure, "savefig", lambda *args, **kwargs: None)
    output = tmp_path / "summary.csv"
    ablation.analyze_ablation(tmp_path, tmp_path / "plot.png", output, [])
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    survival = [r for r in rows if r["metric"] == "creator_survival"]
    assert len(survival) == 4
    assert all(float(r["mean"]) == 0.1 for r in survival)
    assert not any(r["metric"] == "creator_survival_rate" for r in rows)
