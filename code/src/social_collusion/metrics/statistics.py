"""Statistical analysis plan (plan sec.17), implemented in NumPy only.

Deliberately dependency-free: bootstrap CIs over seeds, Wilson intervals over episodes, paired
differences, Benjamini-Hochberg FDR, a permutation test, and a logistic regression with
seed-clustered (sandwich) standard errors.

Honesty note carried into the docstrings and the report: the plan sketches a *mixed-effects*
logistic regression. `cluster_robust_logit` is not a GLMM - it is a fixed-effects logit with
cluster-robust standard errors by training seed. It answers the same question (is the effect
larger than seed-to-seed variation?) without adding a statsmodels dependency, and it is labelled
as such wherever it is reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int = 0
    method: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "method": self.method,
        }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap. Values are usually *per-seed* means (plan sec.17.3)."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap")
    if x.size == 1:
        v = float(stat(x))
        return Interval(v, v, v, 1, "bootstrap(n=1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    draws = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(stat(x)), float(lo), float(hi), int(x.size), "bootstrap")


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """CI for mean(a) - mean(b) when a and b are paired by seed (plan sec.17.3)."""
    x, y = np.asarray(list(a), float), np.asarray(list(b), float)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must match: {x.shape} vs {y.shape}")
    d = x - y
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion (episode-level, plan sec.17.3)."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "wilson")
    z = _z(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(float(p), float(center - half), float(center + half), int(n), "wilson")


def _z(q: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    if not 0 < q < 1:  # pragma: no cover
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        t = np.sqrt(-2 * np.log(q))
        return float((((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
                     / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1))
    if q > phigh:
        t = np.sqrt(-2 * np.log(1 - q))
        return float(-(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
                     / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1))
    t = q - 0.5
    r = t * t
    return float((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t
                 / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def normal_sf(z: float) -> float:
    """Upper-tail standard normal probability via erfc."""
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
) -> tuple[float, float]:
    """Exchangeability test on the difference of means. Returns (observed diff, p-value)."""
    x, y = np.asarray(list(a), float), np.asarray(list(b), float)
    obs = float(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = pool[: x.size].mean() - pool[x.size :].mean()
        if (alternative == "greater" and d >= obs) or (
            alternative == "two-sided" and abs(d) >= abs(obs)
        ):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def bh_fdr(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (rejected mask, adjusted p-values) (plan sec.17.5)."""
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out <= alpha, out


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


@dataclass
class LogitResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    n: int
    n_clusters: int
    converged: bool
    few_clusters: bool = False
    note: str = (
        "fixed-effects logistic regression with cluster-robust (sandwich) SEs by training seed; "
        "not a mixed-effects GLMM"
    )

    @property
    def trustworthy(self) -> bool:
        """Cluster-robust SEs need enough clusters; below that the p-values are noise."""
        return self.converged and not self.few_clusters

    def table(self) -> list[dict]:
        return [
            {
                "term": nm,
                "coef": float(c),
                "se": float(s),
                "z": float(z),
                "p": float(p),
                "odds_ratio": float(np.exp(c)),
            }
            for nm, c, s, z, p in zip(self.names, self.coef, self.se, self.z, self.p)
        ]


MIN_CLUSTERS = 5


def cluster_robust_logit(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    names: Sequence[str] | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
    ridge: float = 1e-6,
) -> LogitResult:
    """IRLS logistic regression with cluster-robust standard errors (plan sec.17.4).

    `X` must already include an intercept column. `clusters` groups rows that are not
    independent - here, the training seed. `weights` lets aggregated binomial data be fitted
    directly (one row per (cell, outcome) with its count) instead of expanding it into fake
    individual rows: duplicating rows would inflate the apparent sample size and drive the
    cluster-robust standard errors toward zero, which is exactly the kind of number that looks
    authoritative and means nothing.

    The sandwich estimator is only trustworthy with enough clusters. With fewer than
    `MIN_CLUSTERS` the result carries `few_clusters=True` and callers should not report the
    p-values; `run_experiment_matrix.py` refuses to print them.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    wt = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = wt * np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / np.clip(mu * (1 - mu), 1e-10, None)
        XtW = X.T * w
        H = XtW @ X + ridge * np.eye(k)
        new = np.linalg.solve(H, XtW @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            converged = True
            break
        beta = new

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    w = wt * np.clip(mu * (1 - mu), 1e-10, None)
    bread = np.linalg.inv((X.T * w) @ X + ridge * np.eye(k))
    resid = (wt * (y - mu))[:, None] * X
    meat = np.zeros((k, k))
    uniq = np.unique(clusters)
    for c in uniq:
        s = resid[clusters == c].sum(axis=0)[:, None]
        meat += s @ s.T
    # small-cluster correction (Cameron-Gelbach-Miller)
    g = max(int(uniq.size), 1)
    if g > 1:
        meat *= g / (g - 1)
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    zstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = np.array([2 * normal_sf(abs(float(v))) for v in zstat])
    return LogitResult(
        names=list(names) if names else [f"x{i}" for i in range(k)],
        coef=beta,
        se=se,
        z=zstat,
        p=pvals,
        n=int(wt.sum()),
        n_clusters=int(uniq.size),
        converged=converged,
        few_clusters=bool(uniq.size < MIN_CLUSTERS),
    )


def design_matrix(rows: Sequence[dict], terms: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Build [1, term...] from a list of dict rows, coercing bools to 0/1."""
    X = np.ones((len(rows), len(terms) + 1))
    for j, t in enumerate(terms):
        X[:, j + 1] = [float(r[t]) for r in rows]
    return X, ["intercept", *terms]


def summarize_comparison(
    name: str,
    treat_per_seed: Sequence[float],
    control_per_seed: Sequence[float],
    seed: int = 0,
) -> dict:
    """The standard block reported for every pre-registered comparison."""
    paired = len(treat_per_seed) == len(control_per_seed)
    diff = (
        paired_bootstrap_diff(treat_per_seed, control_per_seed, seed=seed)
        if paired
        else Interval(
            float(np.mean(treat_per_seed) - np.mean(control_per_seed)),
            float("nan"),
            float("nan"),
            0,
            "unpaired",
        )
    )
    _, pval = permutation_test(treat_per_seed, control_per_seed, seed=seed)
    return {
        "comparison": name,
        "treat": bootstrap_ci(treat_per_seed, seed=seed).as_dict(),
        "control": bootstrap_ci(control_per_seed, seed=seed).as_dict(),
        "difference": diff.as_dict(),
        "paired": paired,
        "permutation_p": pval,
        "significant": bool(diff.excludes_zero()),
        "per_seed_treat": [float(v) for v in treat_per_seed],
        "per_seed_control": [float(v) for v in control_per_seed],
    }


__all__ = [
    "Interval",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "wilson_ci",
    "permutation_test",
    "bh_fdr",
    "cohens_h",
    "cluster_robust_logit",
    "design_matrix",
    "summarize_comparison",
    "normal_sf",
    "LogitResult",
]
