#!/usr/bin/env python3
"""Read-only independent recomputation of the completed study's paired inference.

This does not import the production analyzer, alter its tables, or resample any
experimental episodes. It reconstructs every declared contrast from retained
per-seed values and repeats the specified numerical procedures.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def verify(folder):
    def read(name):
        return json.loads((folder / name).read_text())

    status = read("status.json")
    assert status["status"] == "complete" and status["inference_enabled"]
    assert status["completed_jobs"] == status["expected_jobs"] == 240
    assert not status["problems"] and not status["missing_comparisons"]
    plan = read("planned_comparisons.json")
    rows = read("paired_effects.json")["rows"]
    assert len(plan) == len(rows) == 303
    planned = {p["name"]: p for p in plan}
    assert len(planned) == 303
    assert {r["contrast"] for r in rows} == set(planned)
    values = {}
    for row in read("per_seed.json")["rows"]:
        key = (row["group"], row["table"], row["cell"], row["metric"], row["seed"])
        assert key not in values, ("Duplicate per-seed record", key)
        values[key] = row["value"]

    families = defaultdict(list)
    for row in rows:
        spec = planned[row["contrast"]]
        assert row["complete"] and row["family"] == spec["family"]
        assert row["terms"] == spec["terms"]
        assert row["expected_seeds"] == row["included_seeds"] == spec["seeds"]
        assert row["n_seeds"] == len(spec["seeds"])
        differences = np.array([
            sum(coefficient * values[(*key, seed)] for coefficient, key in spec["terms"])
            for seed in spec["seeds"]
        ])
        assert np.isfinite(differences).all()
        np.testing.assert_allclose(differences, row["per_seed_differences"], rtol=0, atol=1e-12)
        np.testing.assert_allclose([differences.mean(), differences.std(ddof=1)],
                                   [row["mean"], row["sd"]], rtol=0, atol=1e-12)
        signs = np.array(list(itertools.product((-1., 1.), repeat=len(differences))))
        sampled_means = (signs * differences).mean(axis=1)
        exact = np.count_nonzero(np.abs(sampled_means) >= abs(differences.mean()) - 1e-12) / len(signs)
        assert abs(exact - row["p_exact"]) < 1e-12, row["contrast"]
        seed = int.from_bytes(hashlib.sha256(spec["name"].encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        samples = differences[rng.integers(len(differences), size=(20000, len(differences)))].mean(axis=1)
        np.testing.assert_allclose(np.quantile(samples, [.025, .975]), row["ci95"], rtol=0, atol=1e-12)
        families[spec["family"]].append(row)

    assert len(families) == 15
    for family in families.values():
        ordered = sorted(family, key=lambda row: row["p_exact"])
        candidates = [(len(family) - index) * row["p_exact"] for index, row in enumerate(ordered)]
        adjusted = np.minimum(1., np.maximum.accumulate(candidates))
        for row, value in zip(ordered, adjusted):
            assert row["family_size"] == len(family)
            assert abs(row["p_holm"] - value) < 1e-12, row["contrast"]

    return {
        "status": "passed", "validated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": status["source_sha256"], "completed_jobs": 240,
        "paired_contrasts": len(rows), "comparison_families": len(families),
        "family_sizes": {key: len(value) for key, value in sorted(families.items())},
        "holm_below_0_05": sum(row["p_holm"] < .05 for row in rows),
        "bootstrap_resamples_per_contrast": 20000, "numeric_absolute_tolerance": 1e-12,
        "checks": ["prespecified identities and seed cohorts", "per-seed signed expressions",
                   "paired means and sample SDs", "complete exact sign enumeration",
                   "deterministic percentile bootstrap intervals", "full-family Holm adjustment"],
        "input_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in ("status.json", "planned_comparisons.json", "per_seed.json", "paired_effects.json")},
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path(__file__).resolve().parent / "analysis")
    args = parser.parse_args()
    print(json.dumps(verify(args.analysis), indent=2, sort_keys=True))
