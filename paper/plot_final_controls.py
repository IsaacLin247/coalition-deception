#!/usr/bin/env python3
"""Regenerate the five custom paper figures from the published completed data.

Example, from the repository root:
    python paper/plot_final_controls.py --out results/paper_figures

The output manifest records exact row selections and input hashes. Error bars
are sample SD across seeds, not confidence intervals. No training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(data, out, manifest_path):
    names = ("final_analysis/status.json", "final_analysis/cell_summary.json",
             "final_analysis/matched_reference_summary.json", "vote_diagnostic/summary.json")
    inputs = {name: json.loads((data / name).read_text()) for name in names}
    status = inputs[names[0]]
    if (status["status"] != "complete" or not status["inference_enabled"]
            or status["completed_jobs"] != status["expected_jobs"] or status["completed_jobs"] != 240
            or status["problems"] or status["missing_comparisons"]):
        raise ValueError("Custom paper figures require the complete 240-job analysis")

    def index(rows):
        result = {(r["group"], r["table"], r["cell"], r["metric"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate summary identities")
        return result

    full = index(inputs[names[1]]["rows"])
    matched = index(inputs[names[2]]["rows"])
    votes = inputs[names[3]]["cells"]
    selected, selected_votes, figures = {}, {}, {}

    def row(group, table, cell, metric, match=False):
        value = (matched if match else full)[group, table, cell, metric]
        if not value["complete"] or value["n_seeds"] < 5:
            raise ValueError(f"Incomplete figure source: {(group, table, cell, metric)}")
        key = "|".join((group, table, cell, metric, "matched" if match else "full"))
        selected[key] = value
        return value

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    out.mkdir(parents=True, exist_ok=True)

    def bars(name, panels):
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.5), squeeze=False)
        records = []
        for ax, (title, labels, values) in zip(axes[0], panels):
            means, sds = [r["mean"] for r in values], [r["sd"] for r in values]
            if not np.isfinite(means).all() or not np.isfinite(sds).all():
                raise ValueError(f"Undefined figure values: {name}, {title}")
            ax.bar(range(len(means)), means, yerr=sds, capsize=3, color="#386b8a", alpha=.85)
            ax.set_xticks(range(len(means)), labels, rotation=25, ha="right")
            ax.set_title(title)
            ax.set_ylabel("Mean ± seed SD")
            ax.grid(axis="y", alpha=.2)
            records.append(dict(title=title, labels=labels, values=values))
        fig.tight_layout()
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
        figures[name] = dict(kind="bars", panels=records)

    ablations = ["default", "no_channel", "no_partner", "no_channel_no_partner"]
    labels = ["Default", "No channel", "Hidden partner", "Both removed"]
    bars("coordination_controls.pdf", [
        (title, labels, [row(group, "ablation", condition, metric) for condition in ablations])
        for title, group, metric in [
            ("Single-meeting false ejection", "ablation_n7_r1", "false_ejection_rate"),
            ("Multi-round coalition victory", "ablation_n7_r8", "coalition_game_win_rate")]])

    panels = []
    for family, metric, title, budgets in [
        ("f3", "false_ejection_rate", "Single FE", [400, 800, 1600]),
        ("f4", "coalition_game_win_rate", "Multi win", [400, 800])]:
        values = [row(f"{family}_tenseed_crew5" if budget == 400 else f"{family}_budget_d{budget}_crew5",
                      "crossplay", "coalition1_vs_learned_crew1", metric, True) for budget in budgets]
        panels.append((title + " after coalition adaptation", [str(b) for b in budgets], values))
    bars("defender_budgets.pdf", panels)

    panels = []
    for setting, metric, groups, labels, horizon in [
        ("Single FE", "false_ejection_rate", ["multigen_none_n7_r1_k10", "multigen_rwbal_n7_r1_k4",
         "multigen_rwmeet_n7_r1_k4"], ["Default", "Balanced", "Meeting"], 4),
        ("Multi any-FE", "any_false_ejection_rate", ["multigen_none_n7_r8_k10", "multigen_rwfe_n7_r8_k2"],
         ["Survival", "Framing count"], 2),
        ("Multi win", "coalition_game_win_rate", ["multigen_none_n7_r8_k10", "multigen_rwfe_n7_r8_k2"],
         ["Survival", "Framing count"], 2)]:
        panels.append((setting + " adaptation gain", labels,
                       [row(group, f"loop_h{horizon}", "matrix_adaptation_gain", metric, True)
                        for group in groups]))
    bars("reward_controls.pdf", panels)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    series = []
    for metric, label in [("coalition_js", "Coalition"), ("crew_js", "Crew")]:
        generations = list(range(1, 11) if metric == "coalition_js" else range(2, 11))
        values = [row("multigen_none_n7_r8_k10", "policy_distance", f"{i}|{i-1}", metric)
                  for i in generations]
        ax.errorbar(generations, [r["mean"] for r in values], yerr=[r["sd"] for r in values],
                    marker="o", capsize=3, label=label)
        series.append(dict(label=label, metric=metric, generations=generations, values=values))
    ax.set(xlabel="New policy generation", ylabel="Probe-set JS divergence",
           title="Successive same-side policies, 5+2 multi-round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "policy_distance.pdf", bbox_inches="tight")
    plt.close(fig)
    figures["policy_distance.pdf"] = dict(kind="successive_policy_distances", series=series)

    panels = []
    for metric, title in [("false_ejection_rate", "False ejection"), ("crew_skip_rate", "Honest abstention"),
                          ("coalition_ballots_necessary_share_of_fe", "Coalition ballots necessary")]:
        values = []
        for size in (7, 9):
            key = f"n{size}|hypothesis|hypothesis"
            value = votes[key]["metrics"][metric]
            expected = 10 if size == 7 else 5
            if value["defined_seeds"] != value["total_seeds"] or value["total_seeds"] != expected:
                raise ValueError(f"Incomplete ballot figure source: {key}, {metric}")
            selected_votes[key + "|" + metric] = value
            values.append(value)
        panels.append((title, ["5+2", "7+2"], values))
    bars("ballot_diagnostic.pdf", panels)

    manifest = dict(kind="Exact sources for the five custom completed-study paper figures",
        original_source_sha256=status["source_sha256"],
        input_sha256={name: digest(data / name) for name in names},
        plotter_sha256=digest(__file__),
        libraries=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
        selected_rows=selected, selected_ballot_metrics=selected_votes, figures=figures,
        note="All numerical inputs are copied without rounding. Error bars are sample SD across seeds. Budget/reward rows use matched cohorts and horizons. Policy distances compare same-side learned policies and omit scripted D0. Ballot necessity is conditional on false ejection under fixed recorded votes. PDF metadata timestamps can change across regeneration; this manifest binds source values and plotting code, not PDF byte identity.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the five regenerated PDFs")
    parser.add_argument("--manifest", type=Path, help="Default: OUT/custom_figure_sources.json")
    args = parser.parse_args(argv)
    result = generate(args.data, args.out, args.manifest or args.out / "custom_figure_sources.json")
    print(f"Generated {len(result['figures'])} figures from {len(result['selected_rows'])} main-study rows "
          f"and {len(result['selected_ballot_metrics'])} ballot metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
