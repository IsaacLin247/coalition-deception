#!/usr/bin/env python3
"""Neutral manuscript tables/plots from checked fresh-replication analyzer outputs.

No raw-run or historical-figure fallback. Default requires complete final analysis;
--interim renders descriptive results only for fully completed seed groups.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
INPUTS = ("status", "cell_summary", "paired_effects", "matched_reference_summary", "per_seed")
RULES = ("mean", "median", "trimmed", "sharp_credibility", "soft_credibility", "dependence_aware", "hypothesis")
CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
ABLATIONS = ("default", "no_channel", "no_partner", "no_channel_no_partner")
PAIR_COORDINATES = {
    "coalition0_vs_soft_credibility": (0, 0), "coalition0_vs_learned_crew1": (0, 1),
    "coalition1_vs_learned_crew1": (1, 1), "coalition1_vs_soft_credibility": (1, 0),
}
RULE_LABELS = {"mean": "Mean", "median": "Median", "trimmed": "Trimmed mean",
               "sharp_credibility": "Sharp credibility", "soft_credibility": "Soft credibility",
               "dependence_aware": "Dependence aware", "hypothesis": "Hypothesis elimination",
               "soft": "Soft credibility", "rule": "Dependence rule", "both": "Rule + tally"}


def scope(group):
    n = re.search(r"_n(\d+)", group)
    crew = re.search(r"_crew(\d+)", group)
    k = re.search(r"_k(\d+)", group)
    return dict(n=int(n[1]) if n else int(crew[1]) + 2,
                rounds=8 if "_r8" in group or group.startswith("f4_") else 1,
                generations=int(k[1]) if k else None)


def population(group):
    info = scope(group)
    return f"{info['n']-2}+2 players, " + ("single meeting" if info["rounds"] == 1 else "up to 8 rounds")


def metric_label(metric, rounds=1):
    return {
        "false_ejection_rate": "Terminal-meeting false ejection" if rounds > 1 else "Single-meeting false ejection",
        "any_false_ejection_rate": "At least one false ejection / game",
        "mean_false_ejections": "Mean false ejections / game",
        "coalition_game_win_rate": "Coalition game wins",
        "coalition_favorable_rate": "No coalition ejection at terminal meeting" if rounds > 1 else "No coalition member ejected",
        "false_ejections_per_meeting": "False ejections / completed meeting",
        "creator_survival": "Incident and creator survives (terminal meeting)" if rounds > 1 else "Incident and creator survives",
        "same_target_vote_rate": "Same-target votes (terminal meeting)" if rounds > 1 else "Same-target votes",
        "coalition_false_claim_rate": "Coalition false claims (terminal meeting)" if rounds > 1 else "Coalition false claims",
    }.get(metric, metric.replace("_", " "))


def key(row):
    return tuple(row[field] for field in ("group", "table", "cell", "metric"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_summaries(rows, seed_values):
    """Catch stale/mixed aggregate files using the analyzer's checked per-seed data."""
    for row in rows:
        values = seed_values.get(key(row), {})
        expected = list(row["expected_seeds"])
        included = sorted(s for s in expected if s in values)
        if included != row["included_seeds"] or row["n_seeds"] != len(included):
            raise ValueError(f"Summary/per-seed denominator mismatch: {key(row)}")
        if row["complete"] != (included == expected):
            raise ValueError(f"Incorrect completeness flag: {key(row)}")
        for field, computed in (("mean", np.mean([values[s] for s in included]) if included else None),
                                ("sd", np.std([values[s] for s in included], ddof=1) if len(included) > 1 else None)):
            if computed is None:
                if row[field] is not None:
                    raise ValueError(f"Undefined {field} rendered as a number: {key(row)}")
            elif row[field] is None or not np.isclose(float(row[field]), computed, rtol=0, atol=1e-10):
                raise ValueError(f"Summary/per-seed {field} mismatch: {key(row)}")


def complete_groups(status, summaries, per_seed):
    expected = {}
    completed = set(status["completed"])
    represented = {row["job"] for row in per_seed}
    if completed != represented:
        raise ValueError("Analyzer status and per_seed describe different completed jobs")
    for row in summaries:
        seeds = tuple(row["expected_seeds"])
        if row["group"] in expected and expected[row["group"]] != seeds:
            raise ValueError("Inconsistent full-group expected seed sets")
        expected[row["group"]] = seeds
    return {group for group, seeds in expected.items()
            if seeds and all(f"{group}_s{seed}" in completed for seed in seeds)}


def load_checked(directory):
    paths = {name: directory / f"{name}.json" for name in INPUTS}
    hashes = {name: sha(path) for name, path in paths.items()}
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    status = payloads["status"]
    if not re.fullmatch(r"[0-9a-f]{64}", status.get("source_sha256", "")):
        raise ValueError("Missing checked-analysis source identity")
    if status["completed_jobs"] != len(status["completed"]):
        raise ValueError("Status completion count disagrees with job identities")
    if any(payloads[name]["status"] != status["status"] for name in INPUTS if name != "status"):
        raise ValueError("Analyzer output files have inconsistent statuses")
    rows = {name: payloads[name]["rows"] for name in INPUTS if name != "status"}
    seed_values = defaultdict(dict)
    for row in rows["per_seed"]:
        if row["seed"] in seed_values[key(row)]:
            raise ValueError(f"Duplicate analyzer per-seed cell: {key(row)}")
        seed_values[key(row)][row["seed"]] = float(row["value"])
    checked_summaries(rows["cell_summary"], seed_values)
    checked_summaries(rows["matched_reference_summary"], seed_values)
    if not set(seed_values).issubset({key(row) for row in rows["cell_summary"]}):
        raise ValueError("Checked per-seed cells are absent from cell_summary")
    eligible = complete_groups(status, rows["cell_summary"], rows["per_seed"])
    for name, path in paths.items():
        if sha(path) != hashes[name]:
            raise ValueError("Analyzer output changed while being read; retry after analysis finishes")
    return status, rows, eligible, hashes


def select_matched(rows, eligible, *, rounds, table, count=5, variants=None):
    """Use the precomputed matched cohort; never substitute full-group reference means."""
    selected = [row for row in rows if row["complete"] and row["group"] in eligible
                and scope(row["group"])["rounds"] == rounds and row["table"] == table
                and len(row["expected_seeds"]) == count
                and (variants is None or row["group"] in variants)]
    if len({tuple(row["included_seeds"]) for row in selected}) > 1:
        raise ValueError("Matched-control rows do not share the same seed IDs")
    return selected


def cycle_matrix(rows, metric):
    selected = {row["cell"]: row for row in rows if row["metric"] == metric}
    if set(selected) != set(PAIR_COORDINATES):
        raise ValueError(f"Incomplete cycle crossplay cells for {metric}")
    means, sds = np.zeros((2, 2)), np.zeros((2, 2))
    for cell, (i, j) in PAIR_COORDINATES.items():
        means[i, j], sds[i, j] = selected[cell]["mean"], selected[cell]["sd"]
    return means, sds


def pretty(value):
    return "—" if value is None else f"{value:.3f}"


def cell_order(cell, table):
    if table == "holdout":
        return (0 if cell == "initial" else 1, "")
    if table == "stages":
        match = re.fullmatch(r"([CD])(\d+)", cell)
        if match:
            return (2 * int(match[2]) - int(match[1] == "D"), "")
    if cell in PAIR_COORDINATES:
        return (list(PAIR_COORDINATES).index(cell), "")
    return (0, tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", cell)))


def mean_sd(row):
    return f"{pretty(row['mean'])} ± {pretty(row['sd'])}"


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                          for field, value in row.items()} for row in rows)


class Renderer:
    def __init__(self, output, status, rows, eligible, interim):
        self.output, self.status, self.interim = output, status, interim
        self.eligible = eligible
        self.cells = [row for row in rows["cell_summary"] if row["group"] in eligible and row["complete"]]
        self.matched = rows["matched_reference_summary"]
        self.effects = [] if interim else rows["paired_effects"]
        self.table_rows, self.figures = [], []
        self.prefix = "INTERIM — " if interim else ""
        self.markdown = [f"# {self.prefix}Verified replication tables and figures", "",
                         f"{status['completed_jobs']}/{status['expected_jobs']} analyzed jobs; {len(eligible)} complete seed groups displayed. "
                         + ("Descriptive output only; inferential results are suppressed." if interim else "All planned jobs are complete."), "",
                         "Cells report mean ± sample SD across independent seeds. SD is between-seed variability, not a confidence interval. "
                         "Terminal-meeting outcomes and whole-game outcomes are labeled separately. No archived figures or captions are used.", ""]

    def rows(self, group, table, metrics=None):
        return [r for r in self.cells if r["group"] == group and r["table"] == table
                and (metrics is None or r["metric"] in metrics)]

    def table(self, title, rows, description=""):
        if not rows:
            return
        identifier = slug(title)
        for row in rows:
            self.table_rows.append(dict(presentation_table=identifier, **row))
        identities = sorted({(row["group"], row["cell"], tuple(row["included_seeds"])) for row in rows},
                            key=lambda identity: (identity[0], cell_order(identity[1], rows[0]["table"]), identity[2]))
        metrics = list(dict.fromkeys(row["metric"] for row in rows))
        mapping = {(r["group"], r["cell"], tuple(r["included_seeds"]), r["metric"]): r for r in rows}
        rounds = max(scope(r["group"])["rounds"] for r in rows)
        priority = (["any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"]
                    if rounds > 1 else ["false_ejection_rate", "coalition_favorable_rate"])
        metrics = [m for m in priority if m in metrics] + [m for m in metrics if m not in priority]
        self.markdown += [f"## {title}", "", description, "",
                          "| Setting / cell | Seeds | " + " | ".join(metric_label(m, rounds) for m in metrics) + " |",
                          "|---|---:|" + "---:|" * len(metrics)]
        for group, cell, seeds in identities:
            values = [mean_sd(mapping[group, cell, seeds, metric]) if (group, cell, seeds, metric) in mapping else "—" for metric in metrics]
            setting = f"`{group}` / {cell}".replace("|", " / ")
            self.markdown.append(f"| {setting} | {len(seeds)} | " + " | ".join(values) + " |")
        self.markdown.append("")

    def savefig(self, fig, identifier, caption):
        fig.tight_layout(rect=(0, .035, 1, .95))
        fig.text(.5, .007, "Mean ± seed SD; " + ("INTERIM descriptive complete groups" if self.interim else "verified final analysis"), ha="center", fontsize=8)
        for extension in ("pdf", "png"):
            fig.savefig(self.output / f"{identifier}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(dict(file=f"{identifier}.pdf", preview=f"{identifier}.png", caption=self.prefix + caption))
        self.markdown += [f"[{identifier}.pdf]({identifier}.pdf) — {self.prefix}{caption}", ""]

    def heatmaps(self, group, identifier, title, metrics, row_labels, column_labels, matrices):
        columns = min(2, len(metrics))
        fig, axes = plt.subplots(math.ceil(len(metrics) / columns), columns,
                                 figsize=(6.1 * columns, max(3.4, len(row_labels) * .38 + 1.3) * math.ceil(len(metrics) / columns)), squeeze=False)
        rounds = scope(group)["rounds"]
        for ax, metric in zip(axes.flat, metrics):
            means, sds = matrices[metric]
            upper = max(1., float(means.max()) * 1.1) if metric == "mean_false_ejections" else 1.
            image = ax.imshow(means, cmap="viridis", vmin=0, vmax=upper, aspect="auto")
            ax.set_xticks(range(len(column_labels)), column_labels, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
            ax.set_title(metric_label(metric, rounds), fontsize=10)
            for i, j in np.ndindex(means.shape):
                ax.text(j, i, f"{means[i,j]:.2f}\n±{sds[i,j]:.2f}", ha="center", va="center",
                        color="black" if means[i, j] > .55 * upper else "white",
                        fontsize=7 if means.shape[0] <= 7 else 5.5)
            fig.colorbar(image, ax=ax, shrink=.8)
        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")
        fig.suptitle(self.prefix + title + f" ({population(group)})", fontsize=12)
        self.savefig(fig, identifier, title + f" ({population(group)}). "
                     + "Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. "
                     + "Each cell is mean ± SD across the complete seed group.")

    def render_f1(self):
        groups = sorted(g for g in self.eligible if g.startswith("f1_"))
        if not groups:
            return
        metrics = ("false_ejection_rate", "coalition_favorable_rate")
        endpoints = [r for g in groups for r in self.rows(g, "holdout", metrics)]
        self.table("F1: untrained and trained holdout endpoints", endpoints,
                   "Both endpoints use the independent final evidence stream. Initial means the untrained actor; final means the fixed-budget checkpoint.")
        fig, axes = plt.subplots(2, len(groups), figsize=(5 * len(groups), 6.6), squeeze=False)
        for column, group in enumerate(groups):
            for index, metric in enumerate(metrics):
                curves = sorted(self.rows(group, "monitored_curve", (metric,)), key=lambda r: int(r["cell"]))
                if not curves:
                    raise ValueError(f"Missing monitored curve for completed F1 group {group}")
                x, y, sd = ([int(r["cell"]) for r in curves], np.array([r["mean"] for r in curves]), np.array([r["sd"] for r in curves]))
                axes[0, column].plot(x, y, label=metric_label(metric), color=f"C{index}")
                axes[0, column].fill_between(x, y - sd, y + sd, color=f"C{index}", alpha=.13)
                selected = {r["cell"]: r for r in endpoints if r["group"] == group and r["metric"] == metric}
                if set(selected) != {"initial", "final"}:
                    raise ValueError(f"Incomplete F1 holdout endpoints: {group}/{metric}")
                locations = np.array([0, 1]) + (index - .5) * .3
                axes[1, column].bar(locations, [selected[c]["mean"] for c in ("initial", "final")], width=.28,
                                    yerr=[selected[c]["sd"] for c in ("initial", "final")], capsize=3, color=f"C{index}")
            axes[0, column].set(title=population(group), xlabel="Completed PPO updates", ylabel="Monitored probability", ylim=(-.03, 1.03))
            axes[0, column].legend(fontsize=7)
            axes[1, column].set(xticks=[0, 1], xticklabels=["Untrained", "Final"], ylabel="Independent holdout probability", ylim=(-.03, 1.03))
        fig.suptitle(self.prefix + "F1: monitored learning curves and independent holdout endpoints")
        self.savefig(fig, "f1_learning_and_holdout", "F1 monitored curves and independent holdout endpoints. Curve axes use the analyzer's completed-update values; monitored evaluations are separate from both holdout endpoints.")

    def render_scripted(self):
        for group in sorted(g for g in self.eligible if g.startswith(("f2_", "depstatic_"))):
            rows = self.rows(group, "rules", ("false_ejection_rate",))
            mapping = {r["cell"]: r for r in rows}
            means, sds = np.zeros((len(RULES), len(CONDITIONS))), np.zeros((len(RULES), len(CONDITIONS)))
            for i, rule in enumerate(RULES):
                for j, condition in enumerate(CONDITIONS):
                    row = mapping[f"{condition}|{rule}"]
                    means[i, j], sds[i, j] = row["mean"], row["sd"]
            self.table(f"Scripted rule evaluation: {group}", rows)
            self.heatmaps(group, group + "_rules", "Scripted coalition and crew-rule evaluation",
                          ["false_ejection_rate"], [RULE_LABELS[r] for r in RULES], list(CONDITIONS),
                          {"false_ejection_rate": (means, sds)})
            if group.startswith("depstatic_"):
                self.table(f"Static learned-policy evaluation: {group}", self.rows(group, "learned", ("false_ejection_rate",)))
                self.table(f"Dependence strength sweep: {group}", self.rows(group, "strength_sweep", ("false_ejection_rate",)),
                           "Every recorded strength is shown; no best strength is selected.")

    def render_cycles(self):
        for group in sorted(g for g in self.eligible if g.startswith(("f3_tenseed_", "f4_tenseed_"))):
            rounds = scope(group)["rounds"]
            metrics = (("false_ejection_rate", "coalition_favorable_rate") if rounds == 1 else
                       ("any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"))
            rows = self.rows(group, "crossplay", metrics)
            self.table(f"Matched three-stage cycle: {group}", rows,
                       "C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. "
                       + ("Whole-game harm counts every completed meeting, including incident-free meetings. Terminal-meeting FE is a separate outcome." if rounds > 1 else "Each game contains one meeting."))
            self.heatmaps(group, group + "_crossplay", "Matched three-stage crossplay", metrics,
                          ["C0", "C1"], ["D0: scripted soft", "D1: learned"], {m: cycle_matrix(rows, m) for m in metrics})

    def render_counterattack(self):
        for group in sorted(g for g in self.eligible if g.startswith("counterattack_")):
            defenses = ("soft", "mean", "hypothesis") if "_hyp_" in group else ("soft", "rule", "both")
            metrics = ("false_ejection_rate", "coalition_favorable_rate")
            rows = self.rows(group, "crossplay", metrics)
            mapping = {(r["cell"], r["metric"]): r for r in rows}
            matrices = {}
            for metric in metrics:
                means, sds = np.zeros((3, 3)), np.zeros((3, 3))
                for i, trained in enumerate(defenses):
                    for j, evaluated in enumerate(defenses):
                        row = mapping[f"{trained}|{evaluated}", metric]
                        means[i, j], sds[i, j] = row["mean"], row["sd"]
                matrices[metric] = means, sds
            self.table(f"Adaptive-defense crossplay: {group}", rows,
                       "Cell names give training defense / evaluation defense; effects are defined separately with explicit signs.")
            self.heatmaps(group, group + "_crossplay", "Coalition training-defense × evaluation-defense crossplay", metrics,
                          ["Trained: " + RULE_LABELS[d] for d in defenses], [RULE_LABELS[d] for d in defenses], matrices)

    def render_loops(self):
        for group in sorted(g for g in self.eligible if g.startswith("multigen_")):
            info = scope(group)
            metrics = (("false_ejection_rate", "coalition_favorable_rate") if info["rounds"] == 1 else
                       ("any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"))
            stage_rows = self.rows(group, "stages", metrics)
            self.table(f"Generation stages: {group}", stage_rows,
                       "Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.")
            fig, axes = plt.subplots(math.ceil(len(metrics)/2), 2, figsize=(12, 3.7 * math.ceil(len(metrics)/2)), squeeze=False)
            sequence = ["C0"] + [f"{side}{g}" for g in range(1, info["generations"] + 1) for side in ("D", "C")]
            for ax, metric in zip(axes.flat, metrics):
                mapping = {r["cell"]: r for r in stage_rows if r["metric"] == metric}
                y = np.array([mapping[c]["mean"] for c in sequence])
                sd = np.array([mapping[c]["sd"] for c in sequence])
                x = np.arange(len(sequence))
                ax.plot(x, y, color="C0", marker="o", markersize=3)
                ax.fill_between(x, y-sd, y+sd, alpha=.15, color="C0")
                ax.set_xticks(x, sequence, rotation=60, fontsize=7)
                ax.set(title=metric_label(metric, info["rounds"]), xlabel="Completed generation stage", ylabel="Mean ± seed SD")
                ax.set_ylim(bottom=0, top=None if metric == "mean_false_ejections" else 1.03)
            fig.suptitle(self.prefix + "Multigeneration stage evaluations: " + group)
            self.savefig(fig, group + "_stages", f"Stage evaluations, {population(group)}. The band is between-seed SD; similar levels do not establish stationarity or convergence.")
            crossplay = self.rows(group, "crossplay", metrics)
            matrices = {}
            for metric in metrics:
                mapping = {r["cell"]: r for r in crossplay if r["metric"] == metric}
                dimension = info["generations"] + 1
                means, sds = np.zeros((dimension, dimension)), np.zeros((dimension, dimension))
                for i, j in np.ndindex(means.shape):
                    row = mapping[f"C{i}|D{j}"]
                    means[i, j], sds[i, j] = row["mean"], row["sd"]
                matrices[metric] = means, sds
            self.heatmaps(group, group + "_matrix", "Cross-generation evaluation matrix", metrics,
                          [f"C{i}" for i in range(dimension)], [f"D{j}" for j in range(dimension)], matrices)
            # Compact matrix tables retain every value without thousands of
            # repetitive long-form matchup rows in the manuscript document.
            self.markdown += [f"## Cross-generation matrices: {group}", ""]
            for metric in metrics:
                means, sds = matrices[metric]
                self.markdown += [f"**{metric_label(metric, info['rounds'])}**", "",
                                  "| Coalition / defender | " + " | ".join(f"D{j}" for j in range(dimension)) + " |",
                                  "|---|" + "---:|" * dimension]
                for i in range(dimension):
                    self.markdown.append(f"| C{i} | " + " | ".join(f"{means[i,j]:.3f} ± {sds[i,j]:.3f}" for j in range(dimension)) + " |")
                self.markdown.append("")
            self.table_rows.extend(dict(presentation_table=group + "_matrix", **r) for r in crossplay)
            self.table(f"Loop descriptors: {group}", self.rows(group, f"loop_h{info['generations']}", metrics))
            gaps = self.rows(group, "restricted_pool_gap")
            self.table(f"Finite observed-policy response gaps: {group}", gaps,
                       "Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. "
                       "This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.")
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            for metric in dict.fromkeys(r["metric"] for r in gaps):
                selected = sorted([r for r in gaps if r["metric"] == metric], key=lambda r: int(r["cell"]))
                x, y, sd = np.array([int(r["cell"]) for r in selected]), np.array([r["mean"] for r in selected]), np.array([r["sd"] for r in selected])
                ax.plot(x, y, marker="o", markersize=3, label=metric_label(metric, info["rounds"]))
                ax.fill_between(x, y-sd, y+sd, alpha=.13)
            ax.set(xlabel="Generation k", ylabel="Finite-pool response gap (mean ± SD)", ylim=(-.03, 1.03),
                   title=self.prefix + "Observed-policy response gaps: " + population(group))
            ax.legend(fontsize=7)
            self.savefig(fig, group + "_response_gaps", "Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.")

    def render_controls(self):
        for group in sorted(g for g in self.eligible if g.startswith("ablation_")):
            metrics = (("false_ejection_rate",) if scope(group)["rounds"] == 1 else
                       ("any_false_ejection_rate", "mean_false_ejections", "coalition_game_win_rate", "false_ejection_rate"))
            metrics += ("same_target_vote_rate", "coalition_false_claim_rate", "creator_survival")
            self.table(f"Coordination controls: {group}", self.rows(group, "ablation", metrics),
                       "Five complete training seeds per condition. Hidden-partner removes explicit identity and legality/message-availability oracles; ordinary content/outcome inference remains possible. Creator survival is incident-gated per episode.")
        for rounds in (1, 8):
            budget_groups = {g for g in self.eligible if g.startswith(f"f{3 if rounds == 1 else 4}_budget_")}
            if budget_groups:
                budget_groups.add(f"f{3 if rounds == 1 else 4}_tenseed_crew5")
                rows = select_matched(self.matched, self.eligible, rounds=rounds, table="crossplay", variants=budget_groups)
                self.table(f"Defender budgets: r={rounds}, matched five-seed cohort", rows,
                           "Every budget, including the 400-update reference, uses the same five seed IDs. Only the defender budget changes.")
            horizon = 4 if rounds == 1 else 2
            controls = {g for g in self.eligible if g.startswith("multigen_rw") and scope(g)["rounds"] == rounds}
            if controls:
                controls.add(f"multigen_none_n7_r{rounds}_k10")
                rows = select_matched(self.matched, self.eligible, rounds=rounds, table=f"loop_h{horizon}", variants=controls)
                self.table(f"Reward objectives: r={rounds}, horizon={horizon}, matched five-seed cohort", rows,
                           "Reference and reward controls use seeds 0–4 and the same fixed generation horizon. The full ten-seed reference mean is not substituted here.")
        controls = {"multigen_dependence_n7_r8_k4", "multigen_none_n7_r8_k10"}
        if controls.issubset(self.eligible):
            rows = select_matched(self.matched, self.eligible, rounds=8, table="loop_h4", count=10, variants=controls)
            self.table("Dependence-aware loop: matched ten-seed cohort, horizon=4", rows)

    def render_effects(self):
        if self.interim:
            self.markdown += ["## Paired inference", "", "Suppressed in interim presentations, including for completed groups.", ""]
            return
        families = defaultdict(list)
        for row in self.effects:
            if not row["complete"] or row["ci95"] is None or row["p_exact"] is None or row["p_holm"] is None:
                raise ValueError("Final analyzer output contains incomplete paired inference")
            families[row["family"]].append(row)
        self.markdown += ["## Paired effects", "", "Signed differences are defined by the analyzer's recorded contrasts. Intervals are pointwise 95% seed-bootstrap intervals. "
                          "Exact mean sign-flip p-values retain difference magnitudes; Holm correction uses the complete declared family. "
                          "Small seed counts limit attainable p-values, so nonsignificance does not establish absence or equivalence.", ""]
        for family, rows in sorted(families.items()):
            self.markdown += [f"### {family}", "", "| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |", "|---|---:|---:|---|---:|---:|"]
            for row in rows:
                contrast = row["contrast"].replace("|", " / ")
                interval = f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"
                self.markdown.append(f"| {contrast} | {row['n_seeds']} | {mean_sd(row)} | {interval} | {row['p_exact']:.4g} | {row['p_holm']:.4g} |")
            self.markdown.append("")

    def render(self):
        plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
        self.render_f1()
        self.render_scripted()
        self.render_cycles()
        self.render_counterattack()
        self.render_loops()
        self.render_controls()
        self.render_effects()
        (self.output / "tables.md").write_text("\n".join(self.markdown))
        write_csv(self.output / "table_data.csv", self.table_rows)
        (self.output / "table_data.json").write_text(json.dumps(dict(status=self.prefix + "descriptive cells", rows=self.table_rows), indent=2))
        write_csv(self.output / "figure_data.csv", self.cells)
        write_csv(self.output / "paired_effects.csv", self.effects)
        (self.output / "figure_captions.json").write_text(json.dumps(self.figures, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=REPO / "audit/submission/analysis")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--interim", action="store_true")
    args = parser.parse_args(argv)
    output = args.out or args.analysis / "presentation"
    preliminary = json.loads((args.analysis / "status.json").read_text())
    if not args.interim and (preliminary.get("status") != "complete" or not preliminary.get("inference_enabled")):
        print("Presentation refused: final analyzer status is incomplete. Use --interim for descriptive complete groups.")
        return 2
    status, rows, eligible, hashes = load_checked(args.analysis)
    complete = (status["status"] == "complete" and status["inference_enabled"]
                and status["completed_jobs"] == status["expected_jobs"] and not status["problems"]
                and not status["missing_comparisons"])
    if not complete and not args.interim:
        print("Presentation refused: final analyzer status is incomplete. Use --interim for descriptive complete groups.")
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".presentation_build_", dir=output.parent))
    renderer = Renderer(staged, status, rows, eligible, args.interim)
    renderer.render()
    metadata = dict(status="INTERIM descriptive complete groups" if args.interim else "complete",
                    included_groups=sorted(eligible), source_sha256=status["source_sha256"], input_hashes=hashes,
                    analysis_sha256=status["analysis_sha256"], renderer_sha256=sha(__file__),
                    created_utc=datetime.now(timezone.utc).isoformat(), figures=renderer.figures,
                    table_cells=len(renderer.table_rows), paired_contrasts=len(renderer.effects))
    (staged / "status.json").write_text(json.dumps(metadata, indent=2))
    if output.exists():
        archive = output.parent / "presentation_superseded"
        archive.mkdir(exist_ok=True)
        output.rename(archive / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    staged.rename(output)
    print(f"{metadata['status']}: {len(eligible)} groups, {len(renderer.figures)} figures, {len(renderer.table_rows)} table cells -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
