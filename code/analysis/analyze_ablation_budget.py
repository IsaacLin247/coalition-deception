#!/usr/bin/env python3
"""Aggregate the coordination ablation and the defender-budget runs.

    python analysis/analyze_ablation_budget.py --runs "$SOCIAL_COLLUSION_RUNS" --out figures

Reads
  ablation_n7_r{1,8}_s*/result.json            (experiments/ablation/train_coalition_ablation.py)
  f3_budget_d{800,1600}_crew5_s*/crossplay_summary.csv and f4_budget_d800_crew5_s*/...
  f3_tenseed_crew5_s*/ and f4_tenseed_crew5_s*/ (the 400-update reference, same seeds)
and writes
  analysis/ablation_summary.csv, analysis/budget_summary.csv, figures/fig_ablation.png,
  figures/fig_budget.png and a text report.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONDITIONS = ("default", "no_channel", "no_partner", "no_channel_no_partner")
COND_LABELS = {"default": "channel + partner id", "no_channel": "no channel", "no_partner": "no partner id", "no_channel_no_partner": "neither"}
# Archived creator_survival_rate was computed as 1 - creator_ejection_rate,
# including incident-free episodes and the old -1 == -1 ejection bug. The
# separately recorded creator_survival requires a real incident and is valid.
METRICS_R1 = ("false_ejection_rate", "same_target_vote_rate", "creator_survival", "coalition_false_claim_rate")
METRICS_R8 = ("coalition_game_win_rate", "false_ejection_rate", "same_target_vote_rate", "coalition_false_claim_rate")
METRIC_LABELS = {"false_ejection_rate": "false ejection", "same_target_vote_rate": "same-target voting", "creator_survival": "incident creator survives (per episode)",
                 "coalition_false_claim_rate": "coalition false claims", "coalition_game_win_rate": "coalition game wins"}
PAIRS = (("C0 vs scripted", "coalition0_vs_soft_credibility"), ("C0 vs D1", "coalition0_vs_learned_crew1"), ("C1 vs D1", "coalition1_vs_learned_crew1"))


def exact_sign_p(diffs) -> float:
    """Two-sided exact sign-flip test of the mean paired seed difference."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return float("nan")
    observed = abs(float(d.mean()))
    extreme = sum(
        abs(float(np.mean(np.asarray(signs) * d))) >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=d.size)
    )
    return extreme / 2**d.size


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def analyze_ablation(runs: Path, out_fig: Path, out_csv: Path, report: list[str]) -> None:
    rows = []
    for r in (1, 8):
        for d in sorted(runs.glob(f"ablation_n7_r{r}_s*")):
            f = d / "result.json"
            if f.exists():
                rows.extend(json.loads(f.read_text())["rows"])
    if not rows:
        report.append("ablation: no runs found")
        return
    by = defaultdict(list)
    for row in rows:
        for m in set(METRICS_R1) | set(METRICS_R8):
            v = row.get(m)
            if v is not None and v == v:
                by[(int(row["max_rounds"]), row["condition"], m)].append(float(v))
    summary = []
    for r in (1, 8):
        metrics = METRICS_R1 if r == 1 else METRICS_R8
        for cond in CONDITIONS:
            for m in metrics:
                mu, sd = mean_sd(by[(r, cond, m)]) if by[(r, cond, m)] else (float("nan"), float("nan"))
                summary.append({"max_rounds": r, "condition": cond, "metric": m, "mean": mu, "sd": sd, "n_seeds": len(by[(r, cond, m)]),
                                "seed_values": json.dumps([round(x, 4) for x in by[(r, cond, m)]])})
        report.append(f"ablation, {'single meeting' if r == 1 else 'multi-round'}:")
        for m in metrics:
            base = by[(r, "default", m)]
            line = f"  {m}: default {np.mean(base):.3f}" if base else f"  {m}: (no default)"
            for cond in CONDITIONS[1:]:
                v = by[(r, cond, m)]
                if v and base and len(v) == len(base):
                    diffs = [a - b for a, b in zip(v, base)]
                    line += f" | {cond} {np.mean(v):.3f} (Δ {np.mean(diffs):+.3f}, p={exact_sign_p(diffs):.3f})"
            report.append(line)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.4))
    colors = {"default": "#c62828", "no_channel": "#ef6c00", "no_partner": "#6a1b9a", "no_channel_no_partner": "#546e7a"}
    for i, r in enumerate((1, 8)):
        metrics = METRICS_R1 if r == 1 else METRICS_R8
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            means = [np.mean(by[(r, c, m)]) if by[(r, c, m)] else np.nan for c in CONDITIONS]
            sds = [np.std(by[(r, c, m)], ddof=1) if len(by[(r, c, m)]) > 1 else 0 for c in CONDITIONS]
            ax.bar(range(len(CONDITIONS)), means, yerr=sds, capsize=3, color=[colors[c] for c in CONDITIONS], edgecolor="white")
            ax.set_xticks(range(len(CONDITIONS)), [COND_LABELS[c] for c in CONDITIONS], rotation=20, ha="right", fontsize=8)
            ax.set_title(f"{'single meeting' if r == 1 else 'multi-round'}: {METRIC_LABELS[m]}", fontsize=10, weight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=.25)
    fig.suptitle("What the learned coalition's coordination depends on (5+2 players, C0 stage, mean ± SD across 5 seeds)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".pdf"), bbox_inches="tight")


def _summary_rows(run: Path) -> dict[str, dict[str, float]]:
    f = run / "crossplay_summary.csv"
    if not f.exists():
        return {}
    return {row["pair"]: {k: float(v) for k, v in row.items() if v not in ("", None) and k not in ("pair", "vote_aggregation", "coalition_objective") and _isnum(v)} for row in read_csv(f)}


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def analyze_budget(runs: Path, out_fig: Path, out_csv: Path, report: list[str]) -> None:
    configs = [("single meeting", 1, "f3_tenseed_crew5", 400), ("single meeting", 1, "f3_budget_d800_crew5", 800), ("single meeting", 1, "f3_budget_d1600_crew5", 1600),
               ("multi-round", 8, "f4_tenseed_crew5", 400), ("multi-round", 8, "f4_budget_d800_crew5", 800)]
    metric_for = {1: "false_ejection_rate", 8: "coalition_game_win_rate"}
    table = defaultdict(dict)  # (regime, budget) -> pair -> list over seeds
    summary = []
    for regime, r, prefix, budget in configs:
        metric = metric_for[r]
        per_pair = defaultdict(list)
        for s in range(5):
            d = runs / f"{prefix}_s{s}"
            rows = _summary_rows(d)
            for label, pair in PAIRS:
                if pair in rows and metric in rows[pair]:
                    per_pair[label].append(rows[pair][metric])
        for label, _ in PAIRS:
            table[(regime, budget)][label] = per_pair[label]
            mu, sd = mean_sd(per_pair[label]) if per_pair[label] else (float("nan"), float("nan"))
            summary.append({"regime": regime, "defender_updates": budget, "pair": label, "metric": metric, "mean": mu, "sd": sd, "n_seeds": len(per_pair[label]),
                            "seed_values": json.dumps([round(x, 4) for x in per_pair[label]])})
    for regime in ("single meeting", "multi-round"):
        report.append(f"budget, {regime} ({metric_for[1 if regime == 'single meeting' else 8]}):")
        for (reg, budget), pairs in sorted(table.items()):
            if reg != regime:
                continue
            line = f"  D1 budget {budget:5d}: " + " | ".join(f"{label} {np.mean(v):.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f} (n={len(v)})" for label, v in pairs.items() if v)
            report.append(line)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, regime in zip(axes, ("single meeting", "multi-round")):
        budgets = sorted({b for (reg, b) in table if reg == regime})
        x = np.arange(len(budgets))
        width = 0.26
        colors = {"C0 vs scripted": "#90a4ae", "C0 vs D1": "#1565c0", "C1 vs D1": "#c62828"}
        for j, (label, _) in enumerate(PAIRS):
            means = [np.mean(table[(regime, b)][label]) if table[(regime, b)].get(label) else np.nan for b in budgets]
            sds = [np.std(table[(regime, b)][label], ddof=1) if len(table[(regime, b)].get(label, [])) > 1 else 0 for b in budgets]
            ax.bar(x + (j - 1) * width, means, width, yerr=sds, capsize=3, color=colors[label], label=label, edgecolor="white")
        ax.set_xticks(x, [f"D1: {b} updates" for b in budgets])
        ax.set_ylabel(METRIC_LABELS[metric_for[1 if regime == "single meeting" else 8]])
        ax.set_title(f"{regime} (5+2 players; C0 and C1 always 400 updates)", fontsize=10, weight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does a larger defender budget change the response cycle? (mean ± SD across 5 seeds)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".pdf"), bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("SOCIAL_COLLUSION_RUNS", "results")))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    analyze_ablation(args.runs, args.out / "fig_ablation.png", args.tables / "ablation_summary.csv", report)
    analyze_budget(args.runs, args.out / "fig_budget.png", args.tables / "budget_summary.csv", report)
    (args.tables / "ablation_budget_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
