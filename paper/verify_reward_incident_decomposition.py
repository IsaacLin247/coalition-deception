#!/usr/bin/env python3
"""Recompute F3 reward/incident diagnostics from the released seed summaries.

This performs arithmetic on existing data and evaluates synthetic terminal-state
reward fixtures. It does not generate games, rerun policies, or train models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data/reward_incident_decomposition.json")
    args = parser.parse_args()
    repo = args.repo.resolve()
    paths = [
        "data/final_analysis/cell_summary.json",
        "data/final_analysis/per_seed.json",
        "data/final_analysis/submission_protocol.json",
        "code/scripts/train_f3_matched_cycle.py",
        "code/src/social_collusion/env/rewards.py",
        "code/src/social_collusion/metrics/outcome_metrics.py",
    ]
    summary = json.loads((repo / paths[0]).read_text())["rows"]
    per_seed = json.loads((repo / paths[1]).read_text())["rows"]
    protocol = json.loads((repo / paths[2]).read_text())
    cells = {
        "C0D0": "coalition0_vs_soft_credibility",
        "C0D1": "coalition0_vs_learned_crew1",
        "C1D1": "coalition1_vs_learned_crew1",
        "C1D0": "coalition1_vs_soft_credibility",
    }
    count_metrics = {
        "false_ejection": "false_ejection_rate",
        "incident_and_false_ejection": "any_incident_false_ejection_rate",
        "no_incident_and_false_ejection": "any_incident_free_false_ejection_rate",
        "incident": "incident_rate",
    }
    conditional_metrics = {
        "false_ejection_given_incident": "false_ejections_per_incident_meeting",
        "false_ejection_given_no_incident": "false_ejections_per_incident_free_meeting",
    }
    result_cells = []
    checked_seed_means = 0
    for crew in (3, 5, 7):
        group = f"f3_tenseed_crew{crew}"
        jobs = [j for j in protocol["jobs"] if j["name"].startswith(group + "_s")]
        assert len(jobs) == 10
        for job in jobs:
            command = job["command"]
            assert command[command.index("--single-round-objective") + 1] == "balanced_ejection"
            assert int(command[command.index("--max-rounds") + 1]) == 1
            assert int(command[command.index("--crossplay-episodes") + 1]) == 1000
        for short, cell in cells.items():
            selected = [r for r in per_seed if r["table"] == "crossplay" and r["group"] == group and r["cell"] == cell]
            summary_selected = {r["metric"]: r for r in summary if r["table"] == "crossplay" and r["group"] == group and r["cell"] == cell}
            by_seed = {seed: {r["metric"]: r["value"] for r in selected if r["seed"] == seed} for seed in range(10)}
            for metric in list(count_metrics.values()) + list(conditional_metrics.values()):
                included = [s for s in range(10) if metric in by_seed[s]]
                values = [by_seed[s][metric] for s in included]
                if metric in count_metrics.values():
                    assert summary_selected[metric]["complete"]
                    assert included == list(range(10))
                assert summary_selected[metric]["included_seeds"] == included
                assert math.isclose(statistics.mean(values), summary_selected[metric]["mean"], abs_tol=1e-12)
                checked_seed_means += 1
            seed_counts = []
            for seed, values in by_seed.items():
                counts = {}
                for name, metric in count_metrics.items():
                    count = values[metric] * 1000
                    assert math.isclose(count, round(count), abs_tol=1e-8)
                    counts[name] = round(count)
                counts["no_incident"] = 1000 - counts["incident"]
                assert counts["false_ejection"] == counts["incident_and_false_ejection"] + counts["no_incident_and_false_ejection"]
                assert counts["incident_and_false_ejection"] <= counts["incident"]
                assert counts["no_incident_and_false_ejection"] <= counts["no_incident"]
                for condition, denominator in (("incident", counts["incident"]), ("no_incident", counts["no_incident"])):
                    metric = conditional_metrics["false_ejection_given_" + condition]
                    if denominator:
                        value = counts[condition + "_and_false_ejection"] / denominator
                        assert math.isclose(value, values[metric], abs_tol=1e-12)
                    else:
                        assert metric not in values
                seed_counts.append({"seed": seed, "episodes": 1000, **counts})
            counts = {name: sum(s[name] for s in seed_counts) for name in seed_counts[0] if name not in {"seed", "episodes"}}
            total = 10000
            rates = {name: count / total for name, count in counts.items()}
            result_cells.append({
                "crew_plus_coalition": f"{crew}+2",
                "group": group,
                "cell": short,
                "source_cell": cell,
                "n_seeds": 10,
                "episodes_per_seed": 1000,
                "total_episodes": total,
                "counts": counts,
                "unconditional_rates": rates,
                "share_of_false_ejections_in_no_incident_games": counts["no_incident_and_false_ejection"] / counts["false_ejection"],
                "pooled_conditional_rates": {
                    "false_ejection_given_incident": counts["incident_and_false_ejection"] / counts["incident"],
                    "false_ejection_given_no_incident": counts["no_incident_and_false_ejection"] / counts["no_incident"],
                },
                "mean_seed_conditional_rates": {name: summary_selected[metric]["mean"] for name, metric in conditional_metrics.items()},
                "defined_seeds_for_conditional_means": {name: summary_selected[metric]["included_seeds"] for name, metric in conditional_metrics.items()},
                "per_seed_counts_reconstructed_from_published_rates": seed_counts,
            })
    deltas = []
    for crew in (3, 5, 7):
        cc = {r["cell"]: r for r in result_cells if r["crew_plus_coalition"] == f"{crew}+2"}
        delta = {metric: cc["C0D1"]["unconditional_rates"][metric] - cc["C0D0"]["unconditional_rates"][metric] for metric in count_metrics}
        assert math.isclose(delta["false_ejection"], delta["incident_and_false_ejection"] + delta["no_incident_and_false_ejection"], abs_tol=1e-12)
        deltas.append({"crew_plus_coalition": f"{crew}+2", "contrast": "C0D1 minus C0D0", "change_in_unconditional_rate": delta})

    # Synthetic final-state fixtures only: reward evaluation has no environment steps.
    sys.path.insert(0, str(repo / "code/src"))
    import numpy as np
    from social_collusion.env.enums import Role
    from social_collusion.env.rewards import terminal_rewards
    reward_fixtures = []
    roles = np.array([int(Role.CREW), int(Role.CREW), int(Role.CREW), int(Role.COALITION), int(Role.COALITION)])
    for incident_creator in (-1, 3):
        for ejection_label, ejected in (("skip", -1), ("innocent", 0), ("coalition", 3)):
            state = SimpleNamespace(config=SimpleNamespace(n_agents=5, max_rounds=1, coalition_objective="balanced_ejection"), roles=roles, incident_creator=incident_creator, ejected=ejected)
            r = terminal_rewards(state)
            expected_crew = 1 if incident_creator < 0 or ejected == 3 else (0 if ejected < 0 else -1)
            assert np.all(r[:3] == expected_crew) and np.all(r[3:] == -expected_crew)
            reward_fixtures.append({"has_incident": incident_creator >= 0, "ejection": ejection_label, "crew_reward": int(r[0]), "coalition_reward": int(r[3])})
    for j in protocol["jobs"]:
        if j["name"].startswith("f4_tenseed_crew"):
            command = j["command"]
            assert int(command[command.index("--max-rounds") + 1]) == 8
    training_source = (repo / paths[3]).read_text()
    assert 'coalition_objective=(args.single_round_objective if args.max_rounds == 1 else "survive")' in training_source
    out = {
        "purpose": "Reviewer point 2: incident decomposition and reward-objective verification for the released exploratory F3/F4 results.",
        "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths},
        "checks": {"source_mean_checks": checked_seed_means, "f3_cells": len(result_cells), "f3_seed_cells": sum(r["n_seeds"] for r in result_cells), "all_joint_decompositions_verified": True, "synthetic_terminal_reward_cases": len(reward_fixtures), "games_generated": 0, "models_trained": 0},
        "definitions": {
            "F": "An innocent is ejected in the single-meeting game.",
            "I": "The game contains a real incident.",
            "identity": "P(F) = P(F and I) + P(F and not I).",
            "unconditional_rates": "Each numerator is divided by all 10,000 episodes (ten seeds, 1,000 per seed). Equal sample sizes also make this the mean seed rate.",
            "pooled_conditional_rates": "Numerators divided by all incident/no-incident episodes across seeds; these differ from equally weighted means of seed-conditional rates.",
            "mean_seed_conditional_rates": "The published cell_summary.json equally weights defined per-seed conditional rates, omitting seeds with zero conditional denominator; their denominators differ across seeds. See defined_seeds_for_conditional_means for the explicit contributing seed lists.",
            "share_of_false_ejections_in_no_incident_games": "P(F and not I)/P(F); a descriptive composition of the observed false ejections.",
        },
        "reward_interpretation": {
            "F3": "The frozen protocol selects balanced_ejection. Crew receives +1 in every no-incident game regardless of ejection; with an incident, crew receives -1 for innocent ejection, 0 for skip, and +1 for coalition ejection. Thus the objective does not penalize innocent ejection in no-incident games, whereas the reported overall false-ejection outcome includes it.",
            "F4": "The script sets coalition_objective=survive for max_rounds=8. The reward implementation assigns terminal +/-1 by coalition victory at parity, with the exact opposite crew reward. The optimized target is victory, not minimizing innocent-ejection count or its occurrence.",
            "identification_limit": "The decomposition is descriptive. Incidence and post-incident outcomes both change with the policies; conditioning on incident selects different game populations. Neither the decomposition nor the conditional rates isolate susceptibility to testimony.",
        },
        "reward_truth_table": reward_fixtures,
        "cells": result_cells,
        "defender_change_decomposition": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "checks": out["checks"]}, indent=2))
    for row in result_cells:
        r = row["unconditional_rates"]
        print(row["crew_plus_coalition"], row["cell"], "P(F), P(F,I), P(F,!I), P(I):", *(f"{r[m]:.4f}" for m in count_metrics), "share(no-incident):", f'{100*row["share_of_false_ejections_in_no_incident_games"]:.4f}%')


if __name__ == "__main__":
    main()
