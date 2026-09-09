#!/usr/bin/env python3
"""Aggregation for the journal-revision experiments (reviewer points 1, 3 and 4).

1. Adaptive attack on hypothesis elimination (F8).
   Reads `counterattack_hyp_n{N}_r1_s*/{crossplay.csv, coalition_vs_*/history.json}` and writes
   `experiments/dependence/figures/hypothesis_counterattack_n{N}.{csv,json,pdf,png}`: the
   trained-against x evaluated-against matrix (mean +/- SD across seeds) for every outcome, the
   paired contrasts the paper quotes, and the training curves.
2. Reward ablation (control).
   Compares `multigen_rw{bal,meet}_n7_r1_k4_s*` and `multigen_rwfe_n7_r8_k2_s*` with the reference
   loops `multigen_none_n7_r{1,8}_k10_s*` on the same seeds, truncated to the ablation's horizon:
   stage levels, sawtooth amplitude, per-seed last-mover gain, and paired differences.
   Writes `experiments/multigen/figures/reward_ablation_r{1,8}.{csv,json,pdf,png}`.
3. Seed-level loop statistics with every available seed (extra seeds).
   For every loop configuration: per-seed last-mover gain, parity gap, stationary levels and
   late-minus-early change with SDs and exact sign-flip p-values; the defended-vs-undefended
   paired comparison; the counterattack table with all seeds.
   Writes `experiments/multigen/figures/loop_seed_stats.{csv,json}` and
   `experiments/dependence/figures/counterattack_seed_stats.json`.

    python analysis/analyze_revision.py --runs results

The independent unit is the training seed throughout; evaluation episodes give within-seed
precision only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from exp_common import COLORS, save_fig, use_paper_style  # noqa: E402

DEP_FIG = REPO / "experiments" / "dependence" / "figures"
MG_FIG = REPO / "experiments" / "multigen" / "figures"

#: outcomes read from crossplay.csv. The creator-ejection rate is deliberately left out: runs made
#: before the creator-ejection metric correction recorded it with the
#: incident-free convention; `hypothesis_vote_diagnostic.py` recomputes it per incident.
OUTCOMES = [
    ("false_ejection_rate", "false ejection"),
    ("coalition_favorable_rate", "coalition favorable"),
    ("no_ejection_rate", "no ejection"),
    ("same_target_vote_rate", "same-target voting"),
    ("coalition_false_claim_rate", "coalition false claims"),
]
DEF_LABEL = {"soft": "soft credibility", "mean": "mean", "hypothesis": "hypothesis elimination",
             "rule": "dependence-aware rule", "both": "rule + weighted vote"}


# ----------------------------------------------------------------------------- helpers
def exact_sign_p(diffs) -> float:
    """Two-sided exact sign-flip test of the mean paired seed-level difference.

    Retain magnitudes: a binomial test of the count of positive differences is
    a different test. NaNs mark undefined seed-level estimands and are omitted;
    an entirely undefined contrast has no p-value.
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return float("nan")
    observed = abs(float(d.mean()))
    extreme = sum(
        abs(float(np.mean(np.asarray(signs) * d))) >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=d.size)
    )
    return extreme / 2**d.size


def mean_sd(v) -> tuple[float, float]:
    x = np.asarray([t for t in v if t is not None], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), (float(x.std(ddof=1)) if x.size > 1 else 0.0)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v) -> float:
    if v is None or v == "" or v == "None":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def seed_of(d: Path) -> int:
    return int(str(d.name).rsplit("_s", 1)[1])


def contrast(name: str, a: list[float], b: list[float]) -> dict[str, Any]:
    """Paired per-seed difference a - b with SD, sign-flip p and the fraction of positive seeds."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, sd = mean_sd(d)
    return {"contrast": name, "n_seeds": int(np.isfinite(d).sum()), "mean": m, "sd": sd,
            "frac_positive": float(np.mean(d > 0)) if d.size else float("nan"), "sign_flip_p": exact_sign_p(d),
            "per_seed": [float(x) for x in d]}


# ----------------------------------------------------------------------------- 1. hypothesis
def analyze_hypothesis(root: Path, n: int, report: list[str]) -> dict[str, Any] | None:
    dirs = sorted((Path(d) for d in glob.glob(str(root / f"counterattack_hyp_n{n}_r1_s*"))), key=seed_of)
    dirs = [d for d in dirs if (d / "crossplay.csv").exists()]
    if not dirs:
        print(f"[hypothesis] no finished counterattack_hyp_n{n} runs; skipped")
        return None
    seeds = [seed_of(d) for d in dirs]
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    defenses: list[str] = []
    for d in dirs:
        for row in read_csv(d / "crossplay.csv"):
            key = (row["coalition_trained_vs"], row["defense"])
            if row["defense"] not in defenses:
                defenses.append(row["defense"])
            for m, _ in OUTCOMES:
                cells.setdefault(key, {}).setdefault(m, []).append(fnum(row.get(m)))
    table = []
    for tv in defenses:
        for ev in defenses:
            rec: dict[str, Any] = {"n_agents": n, "coalition_trained_vs": tv, "evaluated_vs": ev, "n_seeds": len(seeds)}
            for m, _ in OUTCOMES:
                mu, sd = mean_sd(cells[(tv, ev)][m])
                rec[f"{m}_mean"], rec[f"{m}_sd"] = mu, sd
            table.append(rec)
    save_csv(table, DEP_FIG / f"hypothesis_counterattack_n{n}.csv")

    fe = lambda tv, ev: cells[(tv, ev)]["false_ejection_rate"]  # noqa: E731
    contrasts = []
    if "hypothesis" in defenses and "soft" in defenses:
        contrasts.append(contrast("FE: C[hyp] vs hyp  minus  C[soft] vs hyp  (adaptive gain against hypothesis elimination)", fe("hypothesis", "hypothesis"), fe("soft", "hypothesis")))
        contrasts.append(contrast("FE: C[soft] vs soft  minus  C[hyp] vs hyp  (hypothesis advantage under adaptive attack, vs soft credibility)", fe("soft", "soft"), fe("hypothesis", "hypothesis")))
        contrasts.append(contrast("FE: C[hyp] vs soft  minus  C[soft] vs soft  (transfer of the hypothesis-trained coalition to soft credibility)", fe("hypothesis", "soft"), fe("soft", "soft")))
    if "hypothesis" in defenses and "mean" in defenses:
        contrasts.append(contrast("FE: C[mean] vs mean  minus  C[hyp] vs hyp  (hypothesis advantage under adaptive attack, vs mean)", fe("mean", "mean"), fe("hypothesis", "hypothesis")))
        contrasts.append(contrast("FE: C[hyp] vs mean  minus  C[mean] vs mean  (transfer of the hypothesis-trained coalition to the mean)", fe("hypothesis", "mean"), fe("mean", "mean")))
        contrasts.append(contrast("FE: C[mean] vs hyp  minus  C[hyp] vs hyp", fe("mean", "hypothesis"), fe("hypothesis", "hypothesis")))
    if "mean" in defenses and "soft" in defenses:
        contrasts.append(contrast("FE: C[soft] vs soft  minus  C[mean] vs mean", fe("soft", "soft"), fe("mean", "mean")))

    # training curves (rollout false-ejection rate per PPO update, mean +/- SD across seeds)
    curves: dict[str, np.ndarray] = {}
    for tv in defenses:
        hs = []
        for d in dirs:
            p = d / f"coalition_vs_{tv}" / "history.json"
            if p.exists():
                j = json.loads(p.read_text())
                h = j["history"] if isinstance(j, dict) else j
                hs.append([fnum(r.get("false_ejection_rate")) for r in h])
        if hs:
            L = min(len(h) for h in hs)
            curves[tv] = np.array([h[:L] for h in hs])

    summary = {"n_agents": n, "n_seeds": len(seeds), "seeds": seeds, "defenses": defenses,
               "table": table, "contrasts": contrasts,
               "training_endpoints": {tv: {"first_50_mean": float(np.nanmean(c[:, :50])), "last_50_mean": float(np.nanmean(c[:, -50:]))} for tv, c in curves.items()}}
    save_json(summary, DEP_FIG / f"hypothesis_counterattack_n{n}.json")

    # figure: training curves + trained-vs x evaluated-vs matrix
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    palette = {"soft": COLORS.get("soft_credibility", "#c44e52"), "mean": COLORS.get("mean", "#4c72b0"), "hypothesis": COLORS.get("hypothesis", "#2a9d8f")}
    for tv, c in curves.items():
        m = np.nanmean(c, axis=0)
        sd = np.nanstd(c, axis=0, ddof=1) if c.shape[0] > 1 else np.zeros_like(m)
        w = 10  # moving average for legibility
        ker = np.ones(w) / w
        ms, sds = np.convolve(m, ker, mode="valid"), np.convolve(sd, ker, mode="valid")
        x = np.arange(len(ms)) + w - 1
        ax.plot(x, ms, color=palette.get(tv, None), lw=1.6, label=f"trained vs {DEF_LABEL.get(tv, tv)}")
        ax.fill_between(x, ms - sds, ms + sds, color=palette.get(tv, None), alpha=0.15, lw=0)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("false-ejection rate (training rollouts)")
    ax.set_ylim(-0.02, 0.8)
    ax.set_title(f"a fresh coalition trained against each rule ({n - 2}+2, single meeting)")
    ax.legend()
    ax = axes[1]
    M = np.array([[cells[(tv, ev)]["false_ejection_rate"] for ev in defenses] for tv in defenses], dtype=float)
    Mm, Ms = np.nanmean(M, axis=2), (np.nanstd(M, axis=2, ddof=1) if M.shape[2] > 1 else np.zeros(M.shape[:2]))
    im = ax.imshow(Mm, cmap="Reds", vmin=0, vmax=max(0.6, float(np.nanmax(Mm))))
    for i in range(len(defenses)):
        for j in range(len(defenses)):
            ax.text(j, i, f"{Mm[i, j]:.2f}\n±{Ms[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black" if Mm[i, j] < 0.35 else "white")
    ax.set_xticks(range(len(defenses)))
    ax.set_xticklabels([DEF_LABEL.get(d, d) for d in defenses], rotation=20, ha="right")
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels([f"trained vs {DEF_LABEL.get(d, d)}" for d in defenses])
    ax.set_xlabel("evaluated against")
    ax.set_title(f"cross-evaluated false-ejection rate (mean ± SD, {len(seeds)} seeds)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, DEP_FIG / f"hypothesis_counterattack_n{n}")

    report.append(f"\n== Adaptive attack on hypothesis elimination, {n - 2}+2 ({len(seeds)} seeds: {seeds}) ==")
    for rec in table:
        report.append(f"  C[{rec['coalition_trained_vs']:10s}] vs {rec['evaluated_vs']:10s}: FE {rec['false_ejection_rate_mean']:.3f} ± {rec['false_ejection_rate_sd']:.3f} | favorable {rec['coalition_favorable_rate_mean']:.3f} | no ejection {rec['no_ejection_rate_mean']:.3f} | same-target {rec['same_target_vote_rate_mean']:.3f} | false claims {rec['coalition_false_claim_rate_mean']:.3f}")
    for c in contrasts:
        report.append(f"  {c['contrast']}: {c['mean']:+.3f} ± {c['sd']:.3f}, positive in {c['frac_positive']:.0%} of seeds, sign-flip p = {c['sign_flip_p']:.3f}")
    return summary


# ----------------------------------------------------------------------------- 2. reward ablation
def load_loop(root: Path, pattern: str) -> dict[int, dict[str, Any]]:
    runs = {}
    for d in glob.glob(str(root / pattern)):
        d = Path(d)
        if not (d / "generations.csv").exists():
            continue
        cp = d / "crossplay.json"
        runs[seed_of(d)] = {"dir": d, "rows": read_csv(d / "generations.csv"), "crossplay": json.loads(cp.read_text()) if cp.exists() else None}
    return runs


def loop_stats(run: dict[str, Any], key: str, n_stages: int | None = None) -> dict[str, float]:
    """Per-seed descriptors of one loop run, truncated to `n_stages` stages if given."""
    rows = run["rows"][:n_stages] if n_stages else run["rows"]
    v = np.array([fnum(r.get(key)) for r in rows])
    sides = [r["side"] for r in rows]
    c_idx = [i for i, s in enumerate(sides) if s == "C" and i > 0]
    d_idx = [i for i, s in enumerate(sides) if s == "D"]
    out = {"generation0": float(v[0]),
           "stage1": float(v[1]) if len(v) > 1 else float("nan"),
           "stage2": float(v[2]) if len(v) > 2 else float("nan"),
           "c_stage_mean": float(np.nanmean(v[c_idx])) if c_idx else float("nan"),
           "d_stage_mean": float(np.nanmean(v[d_idx])) if d_idx else float("nan"),
           "sawtooth": float(np.nanmean([abs(v[c] - v[d]) for d, c in zip(d_idx, c_idx)])) if c_idx else float("nan")}
    gens = np.array([int(r["generation"]) for r in rows])
    if len(c_idx) >= 2:
        out["c_stage_slope"] = float(np.polyfit(gens[c_idx], v[c_idx], 1)[0])
        out["d_stage_slope"] = float(np.polyfit(gens[d_idx], v[d_idx], 1)[0])
    if c_idx:
        pairs = list(zip(d_idx, c_idx))
        out["sawtooth_first"] = float(abs(v[pairs[0][1]] - v[pairs[0][0]]))
        out["sawtooth_last"] = float(abs(v[pairs[-1][1]] - v[pairs[-1][0]]))
    if len(c_idx) >= 3:
        out["c_stage_late_mean"] = float(np.nanmean(v[c_idx[-3:]]))
        out["d_stage_late_mean"] = float(np.nanmean(v[d_idx[-3:]]))
        out["late_minus_early"] = float(np.nanmean(v[c_idx[-3:]]) - np.nanmean(v[c_idx[:3]]))
        out["sawtooth_early"] = float(np.nanmean([abs(v[c] - v[d]) for d, c in list(zip(d_idx, c_idx))[:3]]))
        out["sawtooth_late"] = float(np.nanmean([abs(v[c] - v[d]) for d, c in list(zip(d_idx, c_idx))[-3:]]))
    cp = run["crossplay"]
    if cp and key in cp.get("matrices", {}):
        M = np.asarray(cp["matrices"][key], dtype=float)
        k1 = M.shape[0] if n_stages is None else min(M.shape[0], (n_stages + 1) // 2)
        M = M[:k1, :k1]
        out["last_mover_gain"] = float(np.mean([M[i, i] - M[i - 1, i] for i in range(1, k1)])) if k1 > 1 else float("nan")
        out["diag_mean"] = float(np.mean([M[i, i] for i in range(1, k1)])) if k1 > 1 else float("nan")
        same, opp = [], []
        for i in range(1, k1):
            for j in range(1, k1):
                if abs(i - j) < 2:
                    continue
                (same if (i - j) % 2 == 0 else opp).append(M[i, j])
        out["parity_gap"] = float(np.mean(same) - np.mean(opp)) if same and opp else float("nan")
        out["coalition_vs_scripted_D0_late"] = float(np.mean(M[1:, 0])) if k1 > 1 else float("nan")
    return out


def analyze_reward(root: Path, r: int, report: list[str]) -> dict[str, Any] | None:
    key = "coalition_favorable_rate" if r == 1 else "coalition_game_win_rate"
    variants = {1: [("rwbal", "multigen_rwbal_n7_r1_k4_s*", "both sides balanced_ejection"), ("rwmeet", "multigen_rwmeet_n7_r1_k4_s*", "both sides meeting_ejection")],
                8: [("rwfe", "multigen_rwfe_n7_r8_k2_s*", "both sides framing count (false_ejection)")]}[r]
    ref_label = {1: "side-specific objectives (paper: balanced_ejection / meeting_ejection)", 8: "survival objective (paper)"}[r]
    ref = load_loop(root, f"multigen_none_n7_r{r}_k10_s*")
    loops = {tag: load_loop(root, pat) for tag, pat, _ in variants}
    loops = {t: v for t, v in loops.items() if v}
    if not loops or not ref:
        print(f"[reward] no finished reward-ablation loops for r={r}; skipped")
        return None
    # Every displayed reference and control must use the same independent runs,
    # not just the paired contrasts. Extra reference seeds belong in the main
    # loop analysis, not in this matched control comparison.
    common_seeds = sorted(set(ref).intersection(*(set(v) for v in loops.values())))
    if not common_seeds:
        raise ValueError(f"No common reward-control seeds for max_rounds={r}")
    ref = {s: ref[s] for s in common_seeds}
    loops = {tag: {s: v[s] for s in common_seeds} for tag, v in loops.items()}
    n_stages = min(len(run["rows"]) for v in loops.values() for run in v.values())
    out: dict[str, Any] = {"max_rounds": r, "primary_metric": key, "n_stages": n_stages, "variants": {}, "reference_label": ref_label}
    rows_out = []
    per_seed_ref: dict[int, dict[str, float]] = {}
    labels = {tag: lab for tag, _, lab in variants}
    labels["ref"] = ref_label
    stage_curves: dict[str, np.ndarray] = {}
    fe_curves: dict[str, np.ndarray] = {}
    for tag, v in [("ref", ref)] + list(loops.items()):
        seeds = sorted(v)
        if tag != "ref":
            seeds = [s for s in seeds if s in ref]
        stats = {s: loop_stats(v[s], key, n_stages) for s in seeds}
        fe_stats = {s: loop_stats(v[s], "false_ejection_rate", n_stages) for s in seeds}
        if tag == "ref":
            per_seed_ref = stats
        rec: dict[str, Any] = {"variant": tag, "label": labels[tag], "n_seeds": len(seeds), "seeds": seeds}
        for name in ("generation0", "c_stage_mean", "d_stage_mean", "sawtooth", "last_mover_gain", "parity_gap", "diag_mean", "coalition_vs_scripted_D0_late"):
            m, sd = mean_sd([stats[s].get(name, float("nan")) for s in seeds])
            rec[f"{name}_mean"], rec[f"{name}_sd"] = m, sd
        for name in ("c_stage_mean", "d_stage_mean"):
            m, sd = mean_sd([fe_stats[s].get(name, float("nan")) for s in seeds])
            rec[f"fe_{name}_mean"], rec[f"fe_{name}_sd"] = m, sd
        lm = [stats[s].get("last_mover_gain", float("nan")) for s in seeds]
        rec["last_mover_frac_positive"] = float(np.mean(np.asarray(lm) > 0)) if lm else float("nan")
        rec["last_mover_sign_flip_p"] = exact_sign_p(lm)
        if tag != "ref":
            common = [s for s in seeds if s in per_seed_ref]
            for name in ("c_stage_mean", "d_stage_mean", "last_mover_gain", "sawtooth"):
                c = contrast(f"{name}: {tag} minus reference", [stats[s].get(name, float("nan")) for s in common], [per_seed_ref[s].get(name, float("nan")) for s in common])
                rec[f"delta_{name}_mean"], rec[f"delta_{name}_sd"], rec[f"delta_{name}_p"] = c["mean"], c["sd"], c["sign_flip_p"]
        rows_out.append(rec)
        out["variants"][tag] = {"summary": rec, "per_seed": stats}
        X = np.array([[fnum(row.get(key)) for row in v[s]["rows"][:n_stages]] for s in seeds])
        stage_curves[tag] = X
        fe_curves[tag] = np.array([[fnum(row.get("false_ejection_rate")) for row in v[s]["rows"][:n_stages]] for s in seeds])
    save_csv(rows_out, MG_FIG / f"reward_ablation_r{r}.csv")
    save_json(out, MG_FIG / f"reward_ablation_r{r}.json")

    import matplotlib.pyplot as plt

    stage_labels = [f"{row['side']}{int(row['generation'])}" for row in next(iter(ref.values()))["rows"][:n_stages]]
    x = np.arange(n_stages)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    palette = {"ref": "#333333", "rwbal": "#c44e52", "rwmeet": "#4c72b0", "rwfe": "#c44e52"}
    for ax, (curves, ylabel) in zip(axes, [(stage_curves, "coalition-favorable meeting rate" if r == 1 else "coalition game-win rate"), (fe_curves, "false-ejection rate")]):
        for tag, X in curves.items():
            m = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(m)
            ax.plot(x, m, color=palette.get(tag), lw=1.7, marker="o", ms=3.5, label=f"{labels[tag]} ({X.shape[0]} seeds)")
            ax.fill_between(x, m - sd, m + sd, color=palette.get(tag), alpha=0.13, lw=0)
        for i, lab in enumerate(stage_labels):
            if lab.startswith("D"):
                ax.axvspan(i - 0.5, i + 0.5, color=COLORS.get("crew", "#4c72b0"), alpha=0.06, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("stage (frozen match-up evaluated after it)")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"reward specification control, 5+2 {'single meeting' if r == 1 else 'multi-round game'} (mean ± SD across seeds)", fontsize=10.5, weight="bold")
    save_fig(fig, MG_FIG / f"reward_ablation_r{r}")

    report.append(f"\n== Reward ablation, 5+2 {'single meeting' if r == 1 else 'multi-round'} ({n_stages} stages) ==")
    for rec in rows_out:
        report.append(f"  {rec['variant']:6s} [{rec['label']}] n={rec['n_seeds']}: gen0 {rec['generation0_mean']:.3f} | C-stage {rec['c_stage_mean_mean']:.3f} ± {rec['c_stage_mean_sd']:.3f} | D-stage {rec['d_stage_mean_mean']:.3f} ± {rec['d_stage_mean_sd']:.3f} | sawtooth {rec['sawtooth_mean']:.3f} | last-mover {rec['last_mover_gain_mean']:+.3f} ± {rec['last_mover_gain_sd']:.3f} (positive {rec['last_mover_frac_positive']:.0%}, p={rec['last_mover_sign_flip_p']:.3f}) | FE after C {rec['fe_c_stage_mean_mean']:.3f} / after D {rec['fe_d_stage_mean_mean']:.3f}")
        if rec["variant"] != "ref":
            report.append(f"         minus reference (paired): C-stage {rec['delta_c_stage_mean_mean']:+.3f} ± {rec['delta_c_stage_mean_sd']:.3f} (p={rec['delta_c_stage_mean_p']:.3f}) | D-stage {rec['delta_d_stage_mean_mean']:+.3f} ± {rec['delta_d_stage_mean_sd']:.3f} (p={rec['delta_d_stage_mean_p']:.3f}) | last-mover {rec['delta_last_mover_gain_mean']:+.3f} ± {rec['delta_last_mover_gain_sd']:.3f} (p={rec['delta_last_mover_gain_p']:.3f})")
    return out


# ----------------------------------------------------------------------------- 3. seed stats
LOOPS = [
    ("none_n7_r8", "multigen_none_n7_r8_k10_s*", "coalition_game_win_rate", "5+2 multi-round, k=10"),
    ("none_n7_r1", "multigen_none_n7_r1_k10_s*", "coalition_favorable_rate", "5+2 single meeting, k=10"),
    ("none_n5_r8", "multigen_none_n5_r8_k10_s*", "coalition_game_win_rate", "3+2 multi-round, k=10"),
    ("none_n9_r8", "multigen_none_n9_r8_k3_s*", "coalition_game_win_rate", "7+2 multi-round, k=3"),
    ("dependence_n7_r8", "multigen_dependence_n7_r8_k4_s*", "coalition_game_win_rate", "5+2 multi-round, dependence-aware defense, k=4"),
]


def analyze_loops(root: Path, report: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows_out = []
    report.append("\n== Loop statistics with every available seed ==")
    runs_by_tag = {}
    for tag, pat, key, label in LOOPS:
        runs = load_loop(root, pat)
        if not runs:
            continue
        runs_by_tag[tag] = (runs, key)
        seeds = sorted(runs)
        stats = {s: loop_stats(runs[s], key) for s in seeds}
        rec: dict[str, Any] = {"tag": tag, "label": label, "primary_metric": key, "n_seeds": len(seeds), "seeds": seeds}
        for aux_key, aux_tag in (("false_ejection_rate", "fe"), ("mean_rounds_played", "rounds")):
            aux = {s: loop_stats(runs[s], aux_key) for s in seeds}
            for name in ("c_stage_mean", "d_stage_mean", "c_stage_late_mean", "d_stage_late_mean"):
                m, sd = mean_sd([aux[s].get(name, float("nan")) for s in seeds])
                rec[f"{aux_tag}_{name}_mean"], rec[f"{aux_tag}_{name}_sd"] = m, sd
        for name in ("generation0", "stage1", "stage2", "c_stage_mean", "d_stage_mean", "c_stage_slope", "d_stage_slope", "sawtooth_first", "sawtooth_last",
                     "c_stage_late_mean", "d_stage_late_mean", "sawtooth_early", "sawtooth_late", "late_minus_early", "last_mover_gain", "parity_gap", "diag_mean"):
            vals = [stats[s].get(name, float("nan")) for s in seeds]
            m, sd = mean_sd(vals)
            rec[f"{name}_mean"], rec[f"{name}_sd"] = m, sd
            if name in ("last_mover_gain", "parity_gap", "late_minus_early"):
                rec[f"{name}_frac_positive"] = float(np.mean(np.asarray(vals) > 0))
                rec[f"{name}_sign_flip_p"] = exact_sign_p(vals)
        rows_out.append(rec)
        out[tag] = {"summary": rec, "per_seed": stats}
        report.append(f"  {label} ({len(seeds)} seeds): C-stage late {rec['c_stage_late_mean_mean']:.3f} ± {rec['c_stage_late_mean_sd']:.3f} | D-stage late {rec['d_stage_late_mean_mean']:.3f} ± {rec['d_stage_late_mean_sd']:.3f} | sawtooth {rec['sawtooth_early_mean']:.2f} -> {rec['sawtooth_late_mean']:.2f} | late-early {rec['late_minus_early_mean']:+.3f} ± {rec['late_minus_early_sd']:.3f} | last-mover {rec['last_mover_gain_mean']:+.3f} ± {rec['last_mover_gain_sd']:.3f} (positive {rec['last_mover_gain_frac_positive']:.0%}, p={rec['last_mover_gain_sign_flip_p']:.3f}) | parity gap {rec['parity_gap_mean']:+.3f} ± {rec['parity_gap_sd']:.3f} (positive {rec['parity_gap_frac_positive']:.0%}, p={rec['parity_gap_sign_flip_p']:.3f})")
    # defended vs undefended, paired by seed over the defended horizon
    if "none_n7_r8" in runs_by_tag and "dependence_n7_r8" in runs_by_tag:
        und, key = runs_by_tag["none_n7_r8"]
        dfd, _ = runs_by_tag["dependence_n7_r8"]
        common = sorted(set(und) & set(dfd))
        n_stages = min(len(dfd[s]["rows"]) for s in common)
        comp = {}
        for metric in (key, "false_ejection_rate", "dep_coalition_pair_dependence", "coalition_false_claim_rate", "mean_rounds_played"):
            a = {s: loop_stats(dfd[s], metric, n_stages) for s in common}
            b = {s: loop_stats(und[s], metric, n_stages) for s in common}
            comp[metric] = {name: contrast(f"{metric} {name}: defended minus undefended", [a[s].get(name, float("nan")) for s in common], [b[s].get(name, float("nan")) for s in common]) | {"defended_mean": mean_sd([a[s].get(name, float("nan")) for s in common])[0], "undefended_mean": mean_sd([b[s].get(name, float("nan")) for s in common])[0]} for name in ("c_stage_mean", "c_stage_late_mean", "d_stage_mean", "last_mover_gain")}
        out["defended_vs_undefended_n7_r8"] = {"n_seeds": len(common), "seeds": common, "n_stages": n_stages, "contrasts": comp}
        report.append(f"  Defended vs undefended loop, 5+2 multi-round, first {n_stages} stages, {len(common)} paired seeds:")
        for metric, d in comp.items():
            for name, c in d.items():
                report.append(f"    {metric:32s} {name:16s}: defended {c['defended_mean']:.3f} vs undefended {c['undefended_mean']:.3f}, diff {c['mean']:+.3f} ± {c['sd']:.3f}, positive {c['frac_positive']:.0%}, p={c['sign_flip_p']:.3f}")
    save_csv(rows_out, MG_FIG / "loop_seed_stats.csv")
    save_json(out, MG_FIG / "loop_seed_stats.json")
    return out


def analyze_counterattack_seeds(root: Path, n: int, report: list[str]) -> dict[str, Any] | None:
    dirs = sorted((Path(d) for d in glob.glob(str(root / f"counterattack_n{n}_r1_s*"))), key=seed_of)
    dirs = [d for d in dirs if (d / "crossplay.csv").exists()]
    if not dirs:
        return None
    seeds = [seed_of(d) for d in dirs]
    cells: dict[tuple[str, str], list[float]] = {}
    for d in dirs:
        for row in read_csv(d / "crossplay.csv"):
            cells.setdefault((row["coalition_trained_vs"], row["defense"]), []).append(fnum(row["false_ejection_rate"]))
    table = {f"C[{tv}] vs {ev}": mean_sd(v) for (tv, ev), v in cells.items()}
    contrasts = [
        contrast("FE: C[soft] vs rule  minus  C[soft] vs soft (crew-rule placement, paper C0)", cells[("soft", "rule")], cells[("soft", "soft")]),
        contrast("FE: C[soft] vs both  minus  C[soft] vs soft (weighted vote, paper C0)", cells[("soft", "both")], cells[("soft", "soft")]),
        contrast("FE: C[rule] vs rule  minus  C[soft] vs soft", cells[("rule", "rule")], cells[("soft", "soft")]),
        contrast("FE: C[both] vs both  minus  C[soft] vs soft", cells[("both", "both")], cells[("soft", "soft")]),
        contrast("FE: C[rule] vs soft  minus  C[soft] vs soft", cells[("rule", "soft")], cells[("soft", "soft")]),
    ]
    out = {"n_agents": n, "n_seeds": len(seeds), "seeds": seeds, "table": {k: {"mean": m, "sd": sd} for k, (m, sd) in table.items()}, "contrasts": contrasts}
    save_json(out, DEP_FIG / f"counterattack_seed_stats_n{n}.json")
    report.append(f"\n== Dependence-aware counterattack, {n - 2}+2, all seeds ({len(seeds)}: {seeds}) ==")
    for k, (m, sd) in table.items():
        report.append(f"  {k:22s}: {m:.3f} ± {sd:.3f}")
    for c in contrasts:
        report.append(f"  {c['contrast']}: {c['mean']:+.3f} ± {c['sd']:.3f}, positive {c['frac_positive']:.0%}, p={c['sign_flip_p']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=REPO / "results")
    ap.add_argument("--report", type=Path, default=REPO / "analysis" / "revision_report.txt")
    args = ap.parse_args()
    use_paper_style()
    report: list[str] = ["Journal-revision experiments (every number: mean across training seeds; SD across seeds)"]
    for n in (7, 9):
        analyze_hypothesis(args.runs, n, report)
    for r in (1, 8):
        analyze_reward(args.runs, r, report)
    analyze_loops(args.runs, report)
    for n in (7, 5):
        analyze_counterattack_seeds(args.runs, n, report)
    args.report.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
