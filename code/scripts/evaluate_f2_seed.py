#!/usr/bin/env python3
"""Evaluate one F2 rule/coalition sweep for one crew size and one seed.

It keeps F2's scripted mechanism test separate from the learned-coalition training runs: each seed evaluates
all seven aggregation rules under truthful, lone-liar, alibi and framing conditions.
"""

from __future__ import annotations

import argparse

from _common import limit_threads, run_dir, save_csv, save_json, write_provenance

from social_collusion.config import load_env_config
from social_collusion.env import VecEnv
from social_collusion.metrics import outcome_metrics as om
from social_collusion.policies.crew_aggregation import RULES as _AGGREGATION_RULES
from social_collusion.policies.hypothesis_crew import HYPOTHESIS_RULE, crew_for_rule
from social_collusion.policies.registry import RoleRouter, make_coalition


# Every cell uses the same held-out evidence indices within each seed.
CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
RULES = tuple(_AGGREGATION_RULES) + (HYPOTHESIS_RULE,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crew", type=int, choices=(3, 5, 7), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    limit_threads(args.threads)

    cfg = load_env_config("meeting_only").with_(
        n_agents=args.crew + 2,
        n_coalition=2,
        coalition_objective="false_ejection",
    )
    out = run_dir(args.name or f"f2_tenseed_crew{args.crew}_s{args.seed}")
    rows: list[dict[str, object]] = []

    for condition_index, condition in enumerate(CONDITIONS):
        for rule_index, rule in enumerate(RULES):
            eval_seed = 1987654321 + args.seed * 10000
            policy = RoleRouter(
                crew_for_rule(rule),
                make_coalition(condition),
            )
            states = VecEnv(cfg, 64, seed=eval_seed).run_episodes(policy, args.episodes)
            summary = om.summarize(states)
            rows.append(
                {
                    "crew": args.crew,
                    "n_agents": args.crew + 2,
                    "seed": args.seed,
                    "condition": condition,
                    "rule": rule,
                    "episodes": args.episodes,
                    "eval_seed": eval_seed,
                    "false_ejection_rate": summary["false_ejection_rate"],
                    "coalition_favorable_rate": summary["coalition_favorable_rate"],
                    "creator_ejection_rate": summary["creator_ejection_rate"],
                }
            )

    save_csv(rows, out / "f2_seed_results.csv")
    save_json(
        {
            "experiment": "f2_scripted_rule_sweep",
            "crew": args.crew,
            "n_agents": args.crew + 2,
            "seed": args.seed,
            "episodes_per_cell": args.episodes,
            "conditions": CONDITIONS,
            "rules": RULES,
            "rows": rows,
        },
        out / "f2_seed_results.json",
    )
    write_provenance(
        out,
        {
            "script": "evaluate_f2_seed",
            "crew": args.crew,
            "seed": args.seed,
            "episodes_per_cell": args.episodes,
        },
    )
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
