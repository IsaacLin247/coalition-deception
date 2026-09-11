#!/usr/bin/env python3
"""Transparent normal-theory power sensitivity for the prospective supplement.

This is statistical planning, not game simulation.  It neither reads prospective
outcomes nor runs the environment.  For a paired contrast, the independent sample
is the vector of complete training-seed differences, and ``sd`` is that vector's
population SD.  Noncentral-t power is exact for independent normal differences;
for other distributions it is a planning approximation, not a power guarantee.

Each row uses alpha / family_size, the first and most conservative Holm threshold.
Consequently it does not assume that another primary comparison will reject first.
It is not an exact power calculation for a sign-flip or bootstrap test.

Requirements: NumPy and SciPy. Example, from the repository root:
    python validation/power_analysis.py --n 64 72 --format markdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import nct, norm, t


def paired_t_power(n: int, effect: float, sd: float, alpha: float) -> float:
    """Two-sided power for a one-sample t test of complete paired differences."""
    if n < 2 or sd <= 0 or not 0 < alpha < 1:
        raise ValueError("Require n >= 2, sd > 0, and 0 < alpha < 1")
    critical = t.isf(alpha / 2, n - 1)
    noncentrality = abs(effect) * math.sqrt(n) / sd
    # Symmetry avoids a SciPy/Boost lower-tail NaN for some large positive
    # noncentralities (for example df=47, nc=sqrt(48), alpha=.05/6).
    power = float(nct.sf(critical, n - 1, noncentrality)
                  + nct.sf(critical, n - 1, -noncentrality))
    if not math.isfinite(power):
        raise ArithmeticError("noncentral-t calculation was nonfinite")
    return power


def minimum_n(effect: float, sd: float, alpha: float, power: float) -> int | None:
    for n in range(2, 10001):
        if paired_t_power(n, effect, sd, alpha) >= power:
            return n
    return None


def signflip_power_mc(n: int, effect: float, sd: float, alpha: float,
                      trials: int, signs: int, seed: int) -> dict:
    """Independent synthetic alternatives and independently drawn signs per trial.

    Round each normal difference onto the planned 1/1000 integer lattice.  The
    environment is not imported. The plus-one Monte Carlo p-value approximates
    the intended exhaustive sign-flip test; finite random-sign error remains.
    """
    if trials < 1 or signs < 1:
        raise ValueError("MC trials and sign draws must be positive")
    rng = np.random.default_rng(seed)
    rejected = 0
    for start in range(0, trials, 8):
        batch = min(8, trials - start)
        diffs = np.rint(rng.normal(effect, sd, (batch, n)) * 1000).astype(np.int64)
        observed = np.abs(diffs.sum(axis=1))
        flips = rng.integers(0, 2, (batch, signs, n), dtype=np.int8) * 2 - 1
        permuted = np.einsum("ijk,ik->ij", flips, diffs, optimize=False)
        tail = (np.abs(permuted) >= observed[:, None]).sum(axis=1)
        pvalues = (tail + 1) / (signs + 1)
        rejected += int((pvalues <= alpha).sum())
    estimate = rejected / trials
    z = float(norm.isf(.025))
    denom = 1 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / denom
    radius = z * math.sqrt(estimate * (1 - estimate) / trials
                           + z * z / (4 * trials * trials)) / denom
    return {"n": n, "effect": effect, "sd": sd, "alpha": alpha,
            "synthetic_alternatives": trials, "sign_draws_per_alternative": signs,
            "rng_seed": seed, "integer_lattice_scale": 1000,
            "rejections": rejected, "estimated_power": estimate,
            "mc_standard_error": math.sqrt(estimate * (1 - estimate) / trials),
            "mc_wilson95": [center - radius, center + radius],
            "interpretation": "MC uncertainty conditional on normal rounded alternatives; "
                              "not uncertainty about power in the real environment",
            "pvalue": "(1 + randomized_tail_count)/(1 + sign_draws)",
            "finite_sign_error": "Approximates, rather than exhaustively evaluates, "
                                 "the intended exact sign-flip test"}


def historical_reference(path: Path) -> dict:
    payload = path.read_bytes()
    rows = json.loads(payload)["rows"]
    lookup = {r["contrast"]: r for r in rows}
    references = [
        ("legacy_hypothesis_adaptation",
         "counterattack_hyp_n7_r1|hypothesis adaptation|false_ejection_rate"),
        ("f4_win_adaptation",
         "f4_tenseed_crew5|adaptation|coalition_game_win_rate"),
        ("f4_false_ejection_count_adaptation",
         "f4_tenseed_crew5|adaptation|mean_false_ejections"),
    ]
    extracted = []
    for label, key in references:
        row = lookup[key]
        diff = np.asarray(row["per_seed_differences"], dtype=float)
        sd = float(diff.std(ddof=1))
        if not math.isclose(sd, row["sd"], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Historical SD mismatch: {key}")
        extracted.append({"label": label, "contrast": key,
                          "n": len(diff), "mean": float(diff.mean()),
                          "sd": sd, "sd_times_1_5": 1.5 * sd})
    wins = lookup[references[1][1]]
    harm = lookup["f4_tenseed_crew5|adaptation|any_false_ejection_rate"]
    if wins["included_seeds"] != harm["included_seeds"]:
        raise ValueError("Historical F4 contrasts do not have matching seed order")
    diff = (np.asarray(wins["per_seed_differences"], dtype=float)
            - np.asarray(harm["per_seed_differences"], dtype=float))
    extracted.append({"label": "f4_win_minus_any_harm_adaptation",
                      "n": len(diff), "mean": float(diff.mean()),
                      "sd": float(diff.std(ddof=1)),
                      "sd_times_1_5": 1.5 * float(diff.std(ddof=1))})
    return {"file": "data/final_analysis/paired_effects.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": "Historical planning reference only; never pooled into new tests",
            "rows": extracted}


def build_report(ns: list[int], family_size: int, alpha: float, source: Path) -> dict:
    individual_alpha = alpha / family_size
    reference = historical_reference(source)
    old_sd = reference["rows"][0]["sd"]
    scenarios = [
        ("probability", .05, .10, "5 percentage points, SD .10"),
        ("probability", .05, 1.5 * old_sd,
         "5 percentage points, 1.5 times legacy hypothesis SD"),
        ("probability", .05, .15, "5 percentage points, SD .15"),
        ("probability", .10, .10, "10 percentage points, SD .10"),
        ("probability", .10, .15, "10 percentage points, SD .15"),
        ("false_ejections_per_game", .10, .20, "0.10 ejections/game, SD .20"),
        ("false_ejections_per_game", .20, .20, "0.20 ejections/game, SD .20"),
    ]
    rows = []
    for unit, effect, sd, label in scenarios:
        rows.append({
            "label": label, "unit": unit, "effect": effect, "sd": sd,
            "standardized_effect": effect / sd,
            "power": {str(n): paired_t_power(n, effect, sd, individual_alpha)
                      for n in ns},
            "minimum_n_for_80_percent": minimum_n(effect, sd, individual_alpha, .8),
            "minimum_n_for_90_percent": minimum_n(effect, sd, individual_alpha, .9),
        })
    return {
        "schema_version": 1, "scipy_version": scipy_version,
        "method": "Two-sided noncentral-t power; normal paired differences",
        "family_size": family_size, "familywise_alpha": alpha,
        "individual_alpha": individual_alpha, "sample_sizes": ns,
        "limitations": [
            "New defender variance is unknown; historical SD does not guarantee it.",
            "Power is for one contrast at the conservative first Holm threshold.",
            "It is not the probability all family members reject.",
            "Normal-theory assumptions are approximations for seed-level outcomes.",
            "Effect targets describe detectable changes, not equivalence margins.",
            "No prospective outcomes are read and no optional sample expansion is implied.",
        ],
        "historical_reference": reference, "scenarios": rows,
    }


def markdown(report: dict) -> str:
    ns = report["sample_sizes"]
    out = ["# Prospective supplement power sensitivity", "",
           f"Two-sided alpha = {report['familywise_alpha']} / "
           f"{report['family_size']} = {report['individual_alpha']:.8f}. "
           "Normal-theory noncentral-t calculation on paired seed differences.", "",
           "| Planning scenario | SD | " + " | ".join(f"n={n}" for n in ns)
           + " | Minimum n for 80% | Minimum n for 90% |",
           "|---|---:|" + "---:|" * (len(ns) + 2)]
    for row in report["scenarios"]:
        out.append("| " + row["label"] + f" | {row['sd']:.6f} | "
                   + " | ".join(f"{100 * row['power'][str(n)]:.1f}%" for n in ns)
                   + f" | {row['minimum_n_for_80_percent']}"
                   + f" | {row['minimum_n_for_90_percent']} |")
    out += ["", "Limitations:", ""]
    out += ["- " + text for text in report["limitations"]]
    out += ["", "Historical source SHA256: `"
            + report["historical_reference"]["sha256"] + "`.", ""]
    if "signflip_monte_carlo" in report:
        out += ["## Monte Carlo check of the sign-flip planning benchmark", "",
                "Independent synthetic normal alternatives rounded to the 1/1000 lattice; "
                "independent sign draws for every alternative. Confidence intervals quantify "
                "Monte Carlo precision, not uncertainty about the real study's power.", "",
                "| n | Effect | SD | Estimated power | MC SE | MC 95% interval |",
                "|---:|---:|---:|---:|---:|---:|"]
        for row in report["signflip_monte_carlo"]:
            lo, hi = row["mc_wilson95"]
            out.append(f"| {row['n']} | {row['effect']:.3f} | {row['sd']:.6f} | "
                       f"{row['estimated_power']:.3f} | {row['mc_standard_error']:.4f} | "
                       f"[{lo:.3f}, {hi:.3f}] |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[32, 48, 64, 72, 80, 96])
    parser.add_argument("--family-size", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--mc-trials", type=int, default=0,
                        help="Optional synthetic sign-flip power trials per scenario")
    parser.add_argument("--mc-signs", type=int, default=8191)
    parser.add_argument("--mc-n", type=int, default=72)
    parser.add_argument("--mc-seed", type=int, default=2026091101)
    parser.add_argument("--historical-source", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "data/final_analysis/paired_effects.json")
    args = parser.parse_args()
    if args.family_size < 1 or not args.n or any(n < 2 for n in args.n):
        parser.error("family-size must be positive and each n must be at least 2")
    report = build_report(args.n, args.family_size, args.alpha, args.historical_source)
    if args.mc_trials:
        historical_sd = report["historical_reference"]["rows"][0]["sd"]
        report["signflip_monte_carlo"] = [
            signflip_power_mc(args.mc_n, .05, sd, report["individual_alpha"],
                              args.mc_trials, args.mc_signs, args.mc_seed + i)
            for i, sd in enumerate([.10, historical_sd * 1.5, .15])]
    print(json.dumps(report, indent=2, allow_nan=False)
          if args.format == "json" else markdown(report))


if __name__ == "__main__":
    main()
