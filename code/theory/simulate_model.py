#!/usr/bin/env python3
"""Stylized model of credibility collapse under correlated testimony (theory/credibility_collapse_model.md).

    n honest agents      x_i = theta + eps_i,            eps_i ~ N(0, sigma^2) i.i.d.
    m coalition agents   y_j = theta + b + eta_j,        eta ~ N(0, sigma_c^2 R(rho)),  R = (1-rho) I + rho 11'
    corroboration        c_i = #{k != i : |r_i - r_k| <= h}          (indicator kernel <-> is_compatible)
    credibility weights  w_i = exp(gamma c_i) / sum_k exp(gamma c_k)  (softmax, gain gamma = 1 / tau)
    estimate             theta_hat = sum_i w_i r_i

Baselines: mean, median, symmetric trimmed mean. The dependence-aware variant multiplies each
weight by exp(-lambda * excess_i), where excess_i is the concentration of the agent's pairwise
agreement over a sliding window of T rounds. This is a single-channel analogue of env/dependence.py.
Closed-form coalition weights describe a deterministic mean-field surrogate, not the expectation
of the realized softmax weights; their amplification boundaries can differ even in sign.

This module is importable (all closed forms and the Monte Carlo are functions) and runnable:

    python theory/simulate_model.py            # full sweeps -> theory/figures/*.pdf, sweep CSVs
    python theory/simulate_model.py --quick    # coarse grids (a minute)

Only numpy / matplotlib are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

# The reporter count matches the 5+2 game, with one report trimmed per side. These noise and
# agreement parameters are illustrative: the game's honest claims are almost always jointly
# satisfiable, whereas h = 0.4 sigma deliberately gives p_HH = erf(0.2) = 0.22 here.
DEFAULTS = dict(n=5, m=2, sigma=1.0, sigma_c=1.0, h=0.4, b=4.0, gamma=1.0, lam=4.0, T=8, trim=0.25)


# --------------------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------------------
def Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def p_hh(sigma: float, h: float) -> float:
    """P(|eps_i - eps_k| <= h) for independent honest noise: eps_i - eps_k ~ N(0, 2 sigma^2)."""
    return float(math.erf(h / (2.0 * sigma)))


def p_cc(rho: float, sigma_c: float, h: float) -> float:
    """P(|eta_j - eta_k| <= h) for equicorrelated coalition noise: Var = 2 sigma_c^2 (1 - rho)."""
    if rho >= 1.0:
        return 1.0
    return float(math.erf(h / (2.0 * sigma_c * math.sqrt(1.0 - rho))))


def p_hc(b: float, sigma: float, sigma_c: float, h: float) -> float:
    """P(|x_i - y_j| <= h): x - y ~ N(-b, sigma^2 + sigma_c^2)."""
    s = math.sqrt(sigma**2 + sigma_c**2)
    return float(Phi((h - b) / s) - Phi((-h - b) / s))


def expected_corroboration(n, m, rho, b, sigma, sigma_c, h) -> tuple[float, float]:
    """(E c_H, E c_C): expected corroboration counts of an honest and of a coalition reporter."""
    phh, pcc, phc = p_hh(sigma, h), p_cc(rho, sigma_c, h), p_hc(b, sigma, sigma_c, h)
    return (n - 1) * phh + m * phc, (m - 1) * pcc + n * phc


def mean_field_coalition_weight(n, m, rho, b, gamma, sigma, sigma_c, h, penalty_gap: float = 0.0) -> float:
    """W_C = total credibility weight on the coalition when every reporter sits at its expected
    corroboration count (deterministic mean-field surrogate):

        W_C = f / (f + (1 - f) exp(-(gamma * dc - penalty_gap))),   dc = E c_C - E c_H.

    `penalty_gap` = lambda * (excess_C - excess_H) is the dependence-aware log-weight discount.
    """
    ch, cc = expected_corroboration(n, m, rho, b, sigma, sigma_c, h)
    f = m / (n + m)
    if m == 0:
        return 0.0
    z = gamma * (cc - ch) - penalty_gap + math.log(f / (1.0 - f))
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def collapse_condition(n, m, rho, gamma, sigma, sigma_c, h, b=None, penalty_gap: float = 0.0) -> float:
    """Signed margin of the collapse (majority-weight) condition; > 0 means W_C > 1/2:

        gamma * (E c_C - E c_H) - penalty_gap - ln((1 - f) / f).
    """
    b = 10.0 * max(sigma, sigma_c) + h if b is None else b  # b >> h: no cross corroboration
    ch, cc = expected_corroboration(n, m, rho, b, sigma, sigma_c, h)
    f = m / (n + m)
    return float(gamma * (cc - ch) - penalty_gap - math.log((1.0 - f) / f))


def critical_rho(n, m, gamma, sigma, sigma_c, h) -> float:
    """Infimum rho for mean-field majority capture with negligible cross agreement.

    For a minority coalition, zero gain or a singleton coalition cannot capture the weight.
    NaN means no strict majority is attainable on rho in [0, 1].
    """
    if m <= 1 or gamma <= 0.0:
        return float("nan")
    f = m / (n + m)
    need = ((n - 1) * p_hh(sigma, h) + math.log((1.0 - f) / f) / gamma) / (m - 1)
    if need >= 1.0:
        return float("nan")
    if need <= 0.0:
        return 0.0
    # erf(h / (2 sigma_c sqrt(1 - rho))) = need  ->  rho = 1 - (h / (2 sigma_c erfinv(need)))^2
    x = _erfinv(need)
    return float(max(0.0, 1.0 - (h / (2.0 * sigma_c * x)) ** 2))


def _erfinv(y: float) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def robustness_weight_threshold(f: float, sigma: float, b: float) -> float:
    """Asymptotic bias-comparison reference, clipped to 1; not a finite-sample MSE bound."""
    return float(min(1.0, median_bias_bound(f, sigma) / max(abs(b), 1e-12)))


def critical_gain(n, m, rho, b, sigma, sigma_c, h, penalty_gap: float = 0.0) -> float:
    """Positive gain at which the surrogate bias reaches the asymptotic median-bias reference.

    This is not an MSE break-even guarantee. NaN means no positive upward crossing: dc <= 0,
    W* >= 1, or the (possibly penalized) zero-gain weight already reaches W*.
    """
    f = m / (n + m)
    wstar = robustness_weight_threshold(f, sigma, b)
    ch, cc = expected_corroboration(n, m, rho, b, sigma, sigma_c, h)
    dc = cc - ch
    if wstar >= 1.0 or wstar <= 0.0 or dc <= 1e-12:
        return float("nan")
    zstar = math.log((1.0 - f) * wstar / (f * (1.0 - wstar))) + penalty_gap
    if zstar <= 0.0:
        return float("nan")
    return float(zstar / dc)


def median_bias_bound(f: float, sigma: float) -> float:
    """Population contamination bound / large-sample median-bias reference for f < 1/2.

    The historical function name is retained for callers. At finite n this Gaussian quantile
    is not an exact bias bound and its square is not an MSE bound.
    """
    if f >= 0.5:
        return float("inf")
    q = 1.0 / (2.0 * (1.0 - f))
    return float(sigma * _norm_ppf(q))


def _norm_ppf(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# estimators (vectorised over trials)
# --------------------------------------------------------------------------------------
def credibility_weights(r: np.ndarray, gamma: float, h: float, penalty: np.ndarray | None = None) -> np.ndarray:
    """r: (..., N) reports. Softmax(gamma * corroboration count), optionally times `penalty`."""
    diff = np.abs(r[..., :, None] - r[..., None, :])
    agree = (diff <= h).astype(float)
    N = r.shape[-1]
    agree[..., np.arange(N), np.arange(N)] = 0.0
    c = agree.sum(-1)
    z = gamma * (c - c.max(-1, keepdims=True))
    w = np.exp(z)
    if penalty is not None:
        w = w * penalty
    return w / w.sum(-1, keepdims=True)


def trimmed_mean(r: np.ndarray, trim: float) -> np.ndarray:
    N = r.shape[-1]
    k = int(math.floor(trim * N))
    s = np.sort(r, axis=-1)
    if k == 0:
        return s.mean(-1)
    return s[..., k : N - k].mean(-1)


def dependence_penalty(r: np.ndarray, h: float, lam: float, stat: str = "agreement") -> np.ndarray:
    """Concentration penalty from a window of reports `r` of shape (trials, T, N).

    stat="agreement" (default; the statistic of env/dependence.py): D_ik = fraction of the T rounds
    in which |r_i - r_k| <= h. stat="correlation": Pearson correlation of the residuals about the
    per-round median. Then excess_i = max_k D_ik - mean_k D_ik over k != i and
    penalty_i = exp(-lam * excess_i).
    """
    N = r.shape[-1]
    if stat == "agreement":
        agree = (np.abs(r[..., :, None] - r[..., None, :]) <= h).astype(float)  # (trials, T, N, N)
        D = agree.mean(1)
    elif stat == "correlation":
        resid = r - np.median(r, axis=-1, keepdims=True)
        x = resid - resid.mean(1, keepdims=True)
        sd = np.sqrt((x**2).sum(1)) + 1e-12
        D = np.einsum("tik,til->tkl", x, x) / (sd[:, :, None] * sd[:, None, :])
    else:
        raise ValueError(f"unknown dependence statistic {stat!r}")
    D = D.copy()
    D[:, np.arange(N), np.arange(N)] = np.nan
    top = np.nanmax(D, axis=-1)
    mean_others = np.nanmean(D, axis=-1)
    excess = np.clip(top - mean_others, 0.0, None)
    return np.exp(-lam * excess)


def expected_excess(n, m, rho, b, sigma, sigma_c, h) -> tuple[float, float]:
    """Large-window limits of the concentration statistic (agreement rate).

    Coalition member: agrees with other coalition members at rate p_CC and honest agents at p_HC,
    so excess_C -> max(p_CC, p_HC) - [(m-1) p_CC + n p_HC] / (N-1) when m > 1.
    A singleton coalition has no p_CC entry and has zero excess. Honest member: agreement rates are
    p_HH with every honest agent and p_HC with every coalition member, so the top entry equals
    the larger of the two and excess_H -> max(p_HH, p_HC) - [(n-1) p_HH + m p_HC] / (N-1).
    """
    N = n + m
    phh, pcc, phc = p_hh(sigma, h), p_cc(rho, sigma_c, h), p_hc(b, sigma, sigma_c, h)
    exc_c = max(pcc, phc) - ((m - 1) * pcc + n * phc) / (N - 1) if m > 1 else 0.0
    exc_h = max(0.0, max(phh, phc) - ((n - 1) * phh + m * phc) / (N - 1))
    return exc_h, exc_c


def defended_penalty_gap(n, m, rho, b, lam, sigma, sigma_c, h) -> float:
    """lambda * (excess_C - excess_H): the dependence-aware log-weight discount of the coalition
    relative to an honest reporter (large-window limit)."""
    eh, ec = expected_excess(n, m, rho, b, sigma, sigma_c, h)
    return float(lam * (ec - eh))


def simulate(
    n=5, m=2, rho=1.0, b=4.0, gamma=1.0, sigma=1.0, sigma_c=1.0, h=0.4, lam=4.0, T=8, trim=0.25,
    trials=4000, seed=0, theta_scale=0.0, stat="agreement",
) -> dict[str, float]:
    """Monte Carlo MSE of every estimator for the last round of a T-round window.

    Returns MSE and bias for mean / median / trimmed / credibility / credibility_dependence and the
    realized total coalition weight of the two credibility rules.
    """
    rng = np.random.default_rng(seed)
    theta = theta_scale * rng.standard_normal((trials, T, 1))
    eps = sigma * rng.standard_normal((trials, T, n))
    common = rng.standard_normal((trials, T, 1))
    idio = rng.standard_normal((trials, T, m))
    eta = sigma_c * (math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idio)
    r = np.concatenate([theta + eps, theta + b + eta], axis=-1)  # (trials, T, N)
    last = r[:, -1, :]
    truth = theta[:, -1, 0]

    est = {
        "mean": last.mean(-1),
        "median": np.median(last, axis=-1),
        "trimmed": trimmed_mean(last, trim),
    }
    w = credibility_weights(last, gamma, h)
    est["credibility"] = (w * last).sum(-1)
    # dependence-aware: concentration of each reporter's agreement over the T-round window
    pen = dependence_penalty(r, h, lam, stat)
    wd = credibility_weights(last, gamma, h, penalty=pen)
    est["credibility_dependence"] = (wd * last).sum(-1)

    out: dict[str, float] = {}
    for k, v in est.items():
        err = v - truth
        out[f"mse_{k}"] = float(np.mean(err**2))
        out[f"bias_{k}"] = float(np.mean(err))
    out["coalition_weight_credibility"] = float(w[:, n:].sum(-1).mean())
    out["coalition_weight_credibility_dependence"] = float(wd[:, n:].sum(-1).mean())
    out["mean_penalty_coalition"] = float(pen[:, n:].mean())
    out["mean_penalty_honest"] = float(pen[:, :n].mean())
    out["p_hh_empirical"] = float(((np.abs(last[:, :n, None] - last[:, None, :n]) <= h).sum((1, 2)) - n).mean() / (n * (n - 1))) if n > 1 else float("nan")
    out["p_cc_empirical"] = float(((np.abs(last[:, n:, None] - last[:, None, n:]) <= h).sum((1, 2)) - m).mean() / (m * (m - 1))) if m > 1 else float("nan")
    return out


# --------------------------------------------------------------------------------------
# sweeps and figures
# --------------------------------------------------------------------------------------
def phase_grid(fs, rhos, gamma, N_total, trials, seed, **kw):
    """Simulated and mean-field collapse maps over coalition fraction f and correlation rho.

    The coalition size is m = round(f * N_total) with n = N_total - m (so f is discrete)."""
    rows = []
    for f in fs:
        m = max(1, int(round(f * N_total)))
        n = N_total - m
        if n < 2:
            continue
        for rho in rhos:
            s = simulate(n=n, m=m, rho=rho, gamma=gamma, trials=trials, seed=seed, **kw)
            wc = mean_field_coalition_weight(n, m, rho, kw["b"], gamma, kw["sigma"], kw["sigma_c"], kw["h"])
            gap = defended_penalty_gap(n, m, rho, kw["b"], kw["lam"], kw["sigma"], kw["sigma_c"], kw["h"])
            wcd = mean_field_coalition_weight(n, m, rho, kw["b"], gamma, kw["sigma"], kw["sigma_c"], kw["h"], penalty_gap=gap)
            rows.append({"f": m / (n + m), "n": n, "m": m, "rho": rho, "gamma": gamma, **s,
                         "mean_field_coalition_weight": wc,
                         "mean_field_coalition_weight_defended": wcd,
                         "median_bias_bound": median_bias_bound(m / (n + m), kw["sigma"]),
                         "collapse_margin": collapse_condition(n, m, rho, gamma, kw["sigma"], kw["sigma_c"], kw["h"], b=kw["b"])})
    return rows


def save_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 180,
    })
    return plt


COL = {"mean": "#8e44ad", "median": "#2e9d52", "trimmed": "#2166c2", "credibility": "#e0a900",
       "credibility_dependence": "#00897b"}
LAB = {"mean": "mean", "median": "median", "trimmed": "trimmed mean",
       "credibility": "credibility (softmax corroboration)", "credibility_dependence": "dependence-aware credibility"}


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight")


def fig_phase(rows, out: Path, N_total: int, gamma: float, title_extra: str = ""):
    plt = _style()
    fs = sorted({r["f"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    def grid(key):
        G = np.full((len(fs), len(rhos)), np.nan)
        for r in rows:
            G[fs.index(r["f"]), rhos.index(r["rho"])] = r[key]
        return G
    ratio = grid("mse_credibility") / grid("mse_median")
    ratio_dep = grid("mse_credibility_dependence") / grid("mse_median")
    wc = grid("mean_field_coalition_weight")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, Z, name in ((axes[0], ratio, "credibility"), (axes[1], ratio_dep, "dependence-aware credibility")):
        # The sampled coordinates are cell centers, as they are for the contours. An imshow
        # extent set to min/max coordinates instead makes them edges and shifts the heatmap.
        im = ax.pcolormesh(rhos, fs, np.log10(Z), shading="nearest", cmap="RdBu_r", vmin=-1.5, vmax=1.5, rasterized=True)
        ax.contour(rhos, fs, Z, levels=[1.0], colors="k", linewidths=1.6)
        ax.set_xlabel(r"coalition correlation $\rho$")
        ax.set_ylabel(r"coalition fraction $f = m/(n+m)$")
        ax.set_title(f"{name}: MSE / median MSE")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10}$(MSE ratio); black = 1 (break-even)")
    # Surrogate references: asymptotic median-bias crossing (dashed) and majority weight
    # (dotted). Neither is an exact boundary for the simulated estimator's MSE.
    wstar = grid("median_bias_bound") / DEFAULTS["b"]
    bias_difference = np.ma.masked_invalid(wc - wstar)
    if bias_difference.min() <= 0.0 <= bias_difference.max():
        axes[0].contour(rhos, fs, bias_difference, levels=[0.0], colors="white", linestyles="--", linewidths=1.5)
        bias_label = "white dashed: asymptotic bias reference"
    else:
        bias_label = "No asymptotic bias crossing in grid"
    axes[0].contour(rhos, fs, wc, levels=[0.5], colors="white", linestyles=":", linewidths=1.2)
    axes[0].text(
        0.02, 0.97,
        bias_label + "\nwhite dotted: surrogate majority weight",
        transform=axes[0].transAxes, va="top", fontsize=7.5, color="white",
        bbox={"facecolor": "#263238", "alpha": 0.7, "edgecolor": "none", "pad": 3},
    )
    im = axes[2].pcolormesh(rhos, fs, wc, shading="nearest", cmap="magma", vmin=0, vmax=1, rasterized=True)
    axes[2].set_xlabel(r"coalition correlation $\rho$")
    axes[2].set_ylabel(r"$f$")
    axes[2].grid(False)
    axes[2].set_title(r"surrogate coalition weight $W_C(f,\rho)$")
    for ax in axes:
        ax.set_xlim(min(rhos), max(rhos))
        ax.set_ylim(min(fs), max(fs))
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle("Gaussian report aggregation: simulated MSE and surrogate weight\n" + rf"$N={N_total}$, $\gamma={gamma}$, $b={DEFAULTS['b']}\sigma$, $h={DEFAULTS['h']}\sigma$, $T={DEFAULTS['T']}$, $\lambda={DEFAULTS['lam']}${title_extra}; red = larger MSE than the median", fontsize=10.5, weight="bold")
    _save(fig, out)
    plt.close(fig)


def fig_gamma_rho(rows, out: Path, n: int, m: int):
    plt = _style()
    gammas = sorted({r["gamma"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    def grid(key):
        G = np.full((len(gammas), len(rhos)), np.nan)
        for r in rows:
            G[gammas.index(r["gamma"]), rhos.index(r["rho"])] = r[key]
        return G
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, key, name in ((axes[0], "mse_credibility", "credibility"), (axes[1], "mse_credibility_dependence", "dependence-aware credibility")):
        Z = grid(key) / grid("mse_median")
        im = ax.pcolormesh(rhos, gammas, np.log10(Z), shading="nearest", cmap="RdBu_r", vmin=-1.5, vmax=1.5, rasterized=True)
        ax.contour(rhos, gammas, Z, levels=[1.0], colors="k", linewidths=1.6)
        ax.set_xlabel(r"coalition correlation $\rho$")
        ax.set_ylabel(r"credibility gain $\gamma = 1/\tau$")
        ax.grid(False)
        ax.set_title(f"{name}: MSE / median MSE")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # Surrogate majority weight (dotted) and asymptotic bias reference (dashed).
    rr = np.linspace(min(rhos), max(rhos), 200)
    g_half, g_rob = [], []
    for rho in rr:
        ch, cc = expected_corroboration(n, m, rho, DEFAULTS["b"], DEFAULTS["sigma"], DEFAULTS["sigma_c"], DEFAULTS["h"])
        f = m / (n + m)
        d = cc - ch
        g_half.append(math.log((1 - f) / f) / d if d > 1e-9 else np.nan)
        g_rob.append(critical_gain(n, m, rho, DEFAULTS["b"], DEFAULTS["sigma"], DEFAULTS["sigma_c"], DEFAULTS["h"]))
    axes[0].plot(rr, g_half, "w:", lw=1.4, label=r"surrogate $W_C = 1/2$")
    if np.isfinite(g_rob).any():
        axes[0].plot(rr, g_rob, "w--", lw=1.4, label="asymptotic bias reference")
    for g_game, lab in ((1 / 1.5, "soft"), (1 / 0.15, "sharp")):
        for ax in axes:
            ax.axhline(g_game, color="#eceff1", ls="-.", lw=0.9)
        axes[1].text(1.005, g_game, f" {lab}", fontsize=7.5, color="#37474f", va="center")
    for ax in axes:
        ax.set_xlim(min(rhos), max(rhos))
        ax.set_ylim(min(gammas), max(gammas))
    axes[0].legend(loc="upper left", fontsize=7.5, labelcolor="white", frameon=True, facecolor="#263238", framealpha=0.7)
    fig.suptitle(rf"Simulated MSE ratios and surrogate references ($n={n}$, $m={m}$, $b={DEFAULTS['b']}\sigma$, $h={DEFAULTS['h']}\sigma$)", fontsize=10.5, weight="bold")
    _save(fig, out)
    plt.close(fig)


def fig_curves(out: Path, trials: int, seed: int, quick: bool):
    plt = _style()
    d = dict(DEFAULTS)
    n, m = d["n"], d["m"]
    base = dict(n=n, m=m, b=d["b"], sigma=d["sigma"], sigma_c=d["sigma_c"], h=d["h"], lam=d["lam"], T=d["T"], trim=d["trim"], trials=trials, seed=seed)
    ests = ("mean", "median", "trimmed", "credibility", "credibility_dependence")
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.9))
    # (a) vs rho
    rhos = np.linspace(0, 1, 9 if quick else 21)
    res = [simulate(**{**base, "rho": float(r), "gamma": d["gamma"]}) for r in rhos]
    for e in ests:
        axes[0].plot(rhos, [x[f"mse_{e}"] for x in res], color=COL[e], lw=1.8, label=LAB[e])
    axes[0].set_xlabel(r"coalition correlation $\rho$")
    axes[0].set_ylabel(r"MSE of $\hat\theta$ (units of $\sigma^2$)")
    axes[0].set_title(rf"vs correlation ($f={m/(n+m):.2f}$, $\gamma={d['gamma']}$)")
    rc = critical_rho(n, m, d["gamma"], d["sigma"], d["sigma_c"], d["h"])
    # (b) vs gamma at rho = 1 (soft gain ~ 1, sharp gain ~ 6.7 in the game)
    gammas = np.linspace(0, 8, 9 if quick else 33)
    res = [simulate(**{**base, "rho": 1.0, "gamma": float(g)}) for g in gammas]
    for e in ests:
        axes[1].plot(gammas, [x[f"mse_{e}"] for x in res], color=COL[e], lw=1.8)
    f = m / (n + m)
    ch, cc = expected_corroboration(n, m, 1.0, d["b"], d["sigma"], d["sigma_c"], d["h"])
    gs = math.log((1 - f) / f) / (cc - ch) if cc > ch else float("nan")
    if np.isfinite(gs) and gs <= gammas.max():
        axes[1].axvline(gs, color="k", ls=":", lw=1.2)
        axes[1].text(gs, 0.97, r"$\gamma^*_{1/2}$: surrogate $W_C=1/2$", transform=axes[1].get_xaxis_transform(), ha="right", va="top", fontsize=7,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
    for g_game, lab in ((1 / 1.5, "soft"), (1 / 0.15, "sharp")):
        if g_game <= gammas.max():
            axes[1].axvline(g_game, color="#90a4ae", ls="--", lw=1.0)
            axes[1].text(g_game, axes[1].get_ylim()[0] * 1.05, f" {lab}", fontsize=7.5, color="#546e7a", va="bottom")
    axes[1].set_xlabel(r"credibility gain $\gamma = 1/\tau$")
    axes[1].set_title(rf"vs gain ($\rho=1$, $f={f:.2f}$)")
    # (c) vs f at rho = 1 (N fixed)
    N = n + m
    fs, res = [], []
    for mm in range(1, N - 1):
        fs.append(mm / N)
        res.append(simulate(**{**base, "n": N - mm, "m": mm, "rho": 1.0, "gamma": d["gamma"]}))
    for e in ests:
        axes[2].plot(fs, [x[f"mse_{e}"] for x in res], color=COL[e], lw=1.8, marker="o", ms=3)
    axes[2].axvline(0.5, color="k", ls=":", lw=1.2)
    axes[2].set_xlabel(r"coalition fraction $f$")
    axes[2].set_title(rf"vs coalition fraction ($\rho=1$, $\gamma={d['gamma']}$, $N={N}$)")
    # (d) vs window length T at rho = 1 and rho = 0 (false-positive cost of the penalty)
    Ts = [1, 2, 4, 8, 16, 32] if quick else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    for rho, ls in ((1.0, "-"), (0.0, "--")):
        res = [simulate(**{**base, "rho": rho, "gamma": d["gamma"], "T": T}) for T in Ts]
        axes[3].plot(Ts, [x["mse_credibility_dependence"] for x in res], color=COL["credibility_dependence"], lw=1.8, ls=ls, marker="o", ms=3, label=rf"dependence-aware, $\rho={rho:.0f}$")
        axes[3].plot(Ts, [x["mse_credibility"] for x in res], color=COL["credibility"], lw=1.4, ls=ls, label=rf"credibility, $\rho={rho:.0f}$")
        axes[3].plot(Ts, [x["mse_median"] for x in res], color=COL["median"], lw=1.2, ls=ls)
    axes[3].set_xscale("log", base=2)
    axes[3].set_xlabel("window length $T$ (rounds)")
    axes[3].set_title(rf"vs window length ($f={f:.2f}$, $\gamma={d['gamma']}$, $\lambda={d['lam']}$)")
    axes[3].legend(fontsize=7)
    for ax in axes:
        ax.set_yscale("log")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(rf"Estimator error under manufactured corroboration ($n={n}$, $m={m}$, $b={d['b']}\sigma$, $h={d['h']}\sigma$, window $T={d['T']}$, $\lambda={d['lam']}$; {trials} trials per point)", fontsize=10.5, weight="bold")
    _save(fig, out)
    plt.close(fig)
    return {"critical_rho": rc, "critical_gamma_at_rho1": gs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=FIG)
    args = ap.parse_args()
    trials = args.trials or (600 if args.quick else 4000)
    d = dict(DEFAULTS)
    N = d["n"] + d["m"]
    kw = dict(b=d["b"], sigma=d["sigma"], sigma_c=d["sigma_c"], h=d["h"], lam=d["lam"], T=d["T"], trim=d["trim"])

    # 1. phase diagram over (f, rho) for N = 7 and N = 11 (finer f grid)
    for N_total in ((7,) if args.quick else (7, 21)):
        fs = sorted({max(1, int(round(f * N_total))) / N_total for f in np.linspace(0.05, 0.6, 24)})
        fs = [f for f in fs if f * N_total <= N_total - 2]
        rhos = np.linspace(0.0, 1.0, 9 if args.quick else 21)
        rows = phase_grid(fs, rhos, d["gamma"], N_total, trials, args.seed, **kw)
        save_rows(rows, args.out / f"phase_f_rho_N{N_total}.csv")
        fig_phase(rows, args.out / f"phase_diagram_f_rho_N{N_total}", N_total, d["gamma"])
    # 2. gain-correlation plane for the 5+2 game
    rows = []
    for gamma in np.linspace(0.0, 8.0, 9 if args.quick else 17):
        for rho in np.linspace(0.0, 1.0, 9 if args.quick else 21):
            s = simulate(n=d["n"], m=d["m"], rho=float(rho), gamma=float(gamma), trials=trials, seed=args.seed, **kw)
            rows.append({"gamma": float(gamma), "rho": float(rho), **s})
    save_rows(rows, args.out / "phase_gamma_rho_n5m2.csv")
    fig_gamma_rho(rows, args.out / "phase_diagram_gamma_rho_n5m2", d["n"], d["m"])
    # 3. error curves
    crit = fig_curves(args.out / "error_curves", trials, args.seed, args.quick)
    # 4. key numbers for the writeup
    summary = {
        "defaults": d,
        "p_hh": p_hh(d["sigma"], d["h"]),
        "p_cc_rho1": p_cc(1.0, d["sigma_c"], d["h"]),
        "p_hc": p_hc(d["b"], d["sigma"], d["sigma_c"], d["h"]),
        "expected_corroboration_rho1": expected_corroboration(d["n"], d["m"], 1.0, **{k: kw[k] for k in ("b", "sigma", "sigma_c", "h")}),
        "mean_field_coalition_weight_rho1": mean_field_coalition_weight(d["n"], d["m"], 1.0, d["b"], d["gamma"], d["sigma"], d["sigma_c"], d["h"]),
        "median_bias_bound": median_bias_bound(d["m"] / N, d["sigma"]),
        **crit,
        "point_rho1": simulate(n=d["n"], m=d["m"], rho=1.0, gamma=d["gamma"], trials=trials, seed=args.seed, **kw),
        "point_rho0": simulate(n=d["n"], m=d["m"], rho=0.0, gamma=d["gamma"], trials=trials, seed=args.seed, **kw),
        "point_no_attack": simulate(n=d["n"], m=d["m"], rho=0.0, gamma=d["gamma"], trials=trials, seed=args.seed, **{**kw, "b": 0.0}),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("point")}, indent=2, default=float))
    for k in ("point_no_attack", "point_rho0", "point_rho1"):
        p = summary[k]
        print(k, {e: round(p[f"mse_{e}"], 3) for e in ("mean", "median", "trimmed", "credibility", "credibility_dependence")},
              "W_C", round(p["coalition_weight_credibility"], 3), round(p["coalition_weight_credibility_dependence"], 3))
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
