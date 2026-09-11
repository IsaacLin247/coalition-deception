"""Prespecified seed-level inference for integer episode-count contrasts.

All primary effects are sums/differences of counts divided by the same fixed
1,000-episode denominator. Dynamic programming therefore computes the exact
two-sided mean sign-flip distribution without enumerating 2**72 assignments or
rounding floating-point effects. Exactness requires joint sign invariance under
the null; pairing alone does not establish that assumption.
"""
from __future__ import annotations

import hashlib
import math
from functools import reduce

import numpy as np


def exact_sign_flip(count_differences: list[int]) -> dict:
    """Return exact tail counts and p for abs(sum(d)) under independent signs."""
    if not count_differences:
        raise ValueError("At least one independent seed is required")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in count_differences):
        raise ValueError("Supply exact integer count differences, not rounded rates")
    values = [int(v) for v in count_differences]
    nonzero = [abs(v) for v in values if v]
    n = len(values)
    if not nonzero:
        return dict(p_exact=1.0, extreme_assignments=str(2**n),
                    total_assignments=str(2**n), lattice_gcd=0,
                    method="exact_integer_subset_sum_sign_flip")
    step = reduce(math.gcd, nonzero)
    weights = [v // step for v in nonzero]
    observed = abs(sum(values)) // step
    total_weight = sum(weights)
    # Python arbitrary-size integers preserve counts beyond uint64 at n >= 64.
    counts = np.zeros(total_weight + 1, dtype=object)
    counts[0] = 1
    end = 0
    for weight in weights:
        previous = counts[:end + 1].copy()
        counts[weight:weight + end + 1] += previous
        end += weight
    extremes = np.abs(2 * np.arange(total_weight + 1) - total_weight) >= observed
    extreme = int(sum(counts[extremes])) * (2 ** (n - len(nonzero)))
    assignments = 2**n
    if int(sum(counts)) * (2 ** (n - len(nonzero))) != assignments:
        raise AssertionError("Sign-flip distribution lost assignment multiplicity")
    return dict(p_exact=extreme / assignments, extreme_assignments=str(extreme),
                total_assignments=str(assignments), lattice_gcd=step,
                method="exact_integer_subset_sum_sign_flip")


def holm(p_values: list[float]) -> list[float]:
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("A complete finite family of p-values is required")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        result[i] = min(1.0, running)
    return result


def summarize_contrast(name: str, count_differences: list[int],
                       denominator: int, expected_seeds: list[int],
                       included_seeds: list[int], bootstrap_resamples: int = 20000) -> dict:
    if included_seeds != expected_seeds or len(count_differences) != len(expected_seeds):
        raise ValueError("Incomplete or reordered prescribed seed cohort")
    if len(set(expected_seeds)) != len(expected_seeds) or len(expected_seeds) < 2:
        raise ValueError("Independent, unique replicate IDs are required")
    if denominator <= 0 or bootstrap_resamples != 20000:
        raise ValueError("Invalid denominator or deviation from frozen bootstrap count")
    test = exact_sign_flip(count_differences)
    values = np.asarray(count_differences, dtype=float) / denominator
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(("wcc-validation-20260911:" + name).encode()).digest()[:8], "big")
    rng = np.random.default_rng(bootstrap_seed)
    sampled = values[rng.integers(0, len(values), (bootstrap_resamples, len(values)))].mean(axis=1)
    return dict(contrast=name, seed_ids=expected_seeds, n_seeds=len(values),
                integer_differences=[int(v) for v in count_differences],
                denominator=denominator, mean=float(values.mean()),
                sd=float(values.std(ddof=1)),
                ci95_pointwise=np.quantile(sampled, [.025, .975]).tolist(),
                bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed,
                **test)
