#!/usr/bin/env python3
"""Distributed sensor fusion under a coordinated compromised coalition (Task 4 demo).

A minimal engineering analogue of the social-deduction result. N sensors observe a slowly
drifting scalar signal s_t. Honest sensors report s_t + independent Gaussian noise. A small
compromised coalition reports s_t + b + eta_t with *coordinated* noise (pairwise correlation
rho), so its members corroborate each other every step. Four fusion rules run side by side:

    mean            plain average
    median          order statistic
    trust-weighted  weight_i ~ exp(gamma * #sensors within tolerance h of sensor i) -- the
                    corroboration-based credibility of the game (softmax with gain gamma)
    dependence-aware trust-weighted with weight_i *= exp(-lambda * excess_i), where excess_i is
                    how much more sensor i agreed with its closest partner than with the rest over
                    the last T steps (env/dependence.py transplanted to sensors)

Output: one figure (PDF + PNG) with the fused traces, the total weight the two trust rules hand to
the coalition over time, and the RMSE of every rule. Self-contained: numpy + matplotlib only.

    python experiments/sensor_fusion/sensor_fusion_demo.py [--rho 1.0 --bias 3.0 --steps 400]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def trust_weights(r: np.ndarray, gamma: float, h: float, penalty: np.ndarray | None = None) -> np.ndarray:
    agree = (np.abs(r[:, None] - r[None, :]) <= h).astype(float)
    np.fill_diagonal(agree, 0.0)
    c = agree.sum(1)
    w = np.exp(gamma * (c - c.max()))
    if penalty is not None:
        w = w * penalty
    return w / w.sum()


def dependence_penalty(window: np.ndarray, h: float, lam: float) -> np.ndarray:
    """window: (T, N) recent reports. Agreement rate per pair, concentration, penalty."""
    D = (np.abs(window[:, :, None] - window[:, None, :]) <= h).mean(0)
    np.fill_diagonal(D, np.nan)
    excess = np.clip(np.nanmax(D, 1) - np.nanmean(D, 1), 0.0, None)
    return np.exp(-lam * excess)


def run(n_honest=9, n_bad=3, steps=400, sigma=1.0, h=0.4, bias=3.0, rho=1.0, gamma=1.0, lam=4.0, T=8, seed=0):
    rng = np.random.default_rng(seed)
    # slowly drifting signal (integrated noise) so the fusion problem is not trivially constant
    s = np.cumsum(0.05 * rng.standard_normal(steps))
    honest = s[:, None] + sigma * rng.standard_normal((steps, n_honest))
    common = rng.standard_normal((steps, 1))
    idio = rng.standard_normal((steps, n_bad))
    bad = s[:, None] + bias + sigma * (math.sqrt(rho) * common + math.sqrt(1 - rho) * idio)
    R = np.concatenate([honest, bad], axis=1)
    est = {k: np.zeros(steps) for k in ("mean", "median", "trust", "dependence_aware")}
    w_bad = {"trust": np.zeros(steps), "dependence_aware": np.zeros(steps)}
    for t in range(steps):
        r = R[t]
        est["mean"][t] = r.mean()
        est["median"][t] = np.median(r)
        w = trust_weights(r, gamma, h)
        est["trust"][t] = w @ r
        w_bad["trust"][t] = w[n_honest:].sum()
        pen = dependence_penalty(R[max(0, t - T + 1) : t + 1], h, lam)
        wd = trust_weights(r, gamma, h, pen)
        est["dependence_aware"][t] = wd @ r
        w_bad["dependence_aware"][t] = wd[n_honest:].sum()
    rmse = {k: float(np.sqrt(np.mean((v - s) ** 2))) for k, v in est.items()}
    return s, R, est, w_bad, rmse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-honest", type=int, default=9)
    ap.add_argument("--n-bad", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bias", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--h", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20, help="independent runs for the error statistics (seed .. seed+seeds-1)")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    s, R, est, w_bad, rmse = run(args.n_honest, args.n_bad, args.steps, 1.0, args.h, args.bias, args.rho, args.gamma, args.lam, args.window, args.seed)
    # a decorrelated coalition for comparison (same bias, rho = 0)
    _, _, _, w_bad0, rmse0 = run(args.n_honest, args.n_bad, args.steps, 1.0, args.h, args.bias, 0.0, args.gamma, args.lam, args.window, args.seed)
    # error statistics over independent runs (the traces above are seed `args.seed`)
    keys_all = ("mean", "median", "trust", "dependence_aware")
    stats = {"coordinated": {k: [] for k in keys_all}, "independent": {k: [] for k in keys_all},
             "weight_coordinated": {k: [] for k in ("trust", "dependence_aware")}, "weight_independent": {k: [] for k in ("trust", "dependence_aware")}}
    for sd in range(args.seed, args.seed + args.seeds):
        _, _, _, wb, rm = run(args.n_honest, args.n_bad, args.steps, 1.0, args.h, args.bias, args.rho, args.gamma, args.lam, args.window, sd)
        _, _, _, wb0, rm0 = run(args.n_honest, args.n_bad, args.steps, 1.0, args.h, args.bias, 0.0, args.gamma, args.lam, args.window, sd)
        for k in keys_all:
            stats["coordinated"][k].append(rm[k]); stats["independent"][k].append(rm0[k])
        for k in ("trust", "dependence_aware"):
            stats["weight_coordinated"][k].append(float(wb[k].mean())); stats["weight_independent"][k].append(float(wb0[k].mean()))
    def ms(v):
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
    stats_summary = {grp: {k: ms(v) for k, v in d.items()} for grp, d in stats.items()}
    col = {"mean": "#8e44ad", "median": "#2e9d52", "trust": "#e0a900", "dependence_aware": "#00897b"}
    lab = {"mean": "mean", "median": "median", "trust": "trust-weighted (corroboration)", "dependence_aware": "dependence-aware trust"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"width_ratios": [2.2, 1.4, 1.0]})
    t = np.arange(args.steps)
    ax = axes[0]
    ax.plot(t, R[:, : args.n_honest], color="#b0bec5", lw=0.5, alpha=0.6)
    ax.plot(t, R[:, args.n_honest :], color="#ef9a9a", lw=0.6, alpha=0.8)
    ax.plot(t, s, color="k", lw=1.8, label="true signal")
    for k in ("trust", "dependence_aware", "median"):
        ax.plot(t, est[k], color=col[k], lw=1.3, label=lab[k])
    ax.set_xlabel("time step")
    ax.set_ylabel("signal")
    ax.set_title(f"{args.n_honest} honest sensors (grey) + {args.n_bad} coordinated compromised sensors (red, bias {args.bias:g}σ, ρ={args.rho:g})")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax = axes[1]
    for k in ("trust", "dependence_aware"):
        ax.plot(t, w_bad[k], color=col[k], lw=1.4, label=f"{lab[k]}, ρ={args.rho:g}")
        ax.plot(t, w_bad0[k], color=col[k], lw=1.0, ls="--", label=f"{lab[k]}, ρ=0")
    ax.axhline(args.n_bad / (args.n_honest + args.n_bad), color="#546e7a", ls=":", lw=1, label="uniform share f")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time step")
    ax.set_ylabel("total weight on compromised sensors")
    ax.set_title("how much trust the coalition captures")
    ax.legend(fontsize=7)
    ax = axes[2]
    keys = ("mean", "median", "trust", "dependence_aware")
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [stats_summary["coordinated"][k]["mean"] for k in keys], 0.4, yerr=[stats_summary["coordinated"][k]["sd"] for k in keys],
           capsize=2, color=[col[k] for k in keys], label=f"ρ={args.rho:g} (coordinated)")
    ax.bar(x + 0.2, [stats_summary["independent"][k]["mean"] for k in keys], 0.4, yerr=[stats_summary["independent"][k]["sd"] for k in keys],
           capsize=2, color=[col[k] for k in keys], alpha=0.45, hatch="//", label="ρ=0 (independent)")
    ax.set_xticks(x, ["mean", "median", "trust", "dep.-aware"], rotation=15)
    ax.set_ylabel("RMSE (units of σ)")
    ax.set_title(f"fusion error (mean ± SD over {args.seeds} runs)")
    ax.legend(fontsize=7)
    fig.suptitle(f"Trust-weighted sensor fusion is captured by a coordinated coalition; discounting concentrated agreement restores it "
                 f"(γ={args.gamma:g}, h={args.h:g}σ, λ={args.lam:g}, window {args.window})", fontsize=10.5, weight="bold")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"sensor_fusion_demo.{ext}", bbox_inches="tight", dpi=180)
    summary = {"args": vars(args) | {"out": str(args.out)}, "rmse_coordinated_seed0": rmse, "rmse_independent_seed0": rmse0,
               "mean_coalition_weight_coordinated_seed0": {k: float(v.mean()) for k, v in w_bad.items()},
               "mean_coalition_weight_independent_seed0": {k: float(v.mean()) for k, v in w_bad0.items()},
               "over_seeds": stats_summary}
    (args.out / "sensor_fusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2))
    print(f"figure -> {args.out / 'sensor_fusion_demo.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
