#!/usr/bin/env python3
"""Emergence run (Finding 1): train the coalition vs ONE defense, logging the full collusion
eval_curve over training so we can plot the *anatomy of emergent collusion* -- framing success,
deception (lie) rate, mutual corroboration, and creator-survival all rising together.

    python scripts/train_emergence.py --rule soft_credibility --seed 0 --updates 400
    python scripts/train_emergence.py --smoke
"""
from __future__ import annotations

import argparse

from _common import limit_threads, run_dir, save_json, write_provenance

from social_collusion.config import load_algo_config, load_env_config
from social_collusion.policies.scripted_crew import TruthfulCrew
from social_collusion.rl.train import Trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="soft_credibility")
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--rollout-episodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    limit_threads(args.threads)

    cfg = load_env_config("meeting_only").with_(
        n_agents=args.n_agents, n_coalition=2, coalition_objective="false_ejection"
    )
    algo = load_algo_config("ippo")
    if args.smoke:
        algo.update(total_updates=6, n_envs=32, rollout_episodes=32, eval_every=2,
                    eval_episodes=200, checkpoint_every=2, evaluate_initial=True)
    else:
        algo.update(total_updates=args.updates, eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes, checkpoint_every=args.checkpoint_every,
                    evaluate_initial=True,
                    n_envs=args.n_envs, rollout_episodes=args.rollout_episodes)

    out = run_dir(args.name or f"emergence_{args.rule}_n{args.n_agents}_s{args.seed}")
    meta = {"experiment": "f1_emergence", "args": vars(args), "env_config": cfg.to_dict(),
            "algo_config": algo, "evaluation_seed": 987654321 + args.seed}
    write_provenance(out, meta)
    save_json(meta, out / "experiment_config.json")
    tr = Trainer(cfg, algo, TruthfulCrew(rule=args.rule), seed=args.seed, run_dir=out)
    initial_holdout = tr.evaluate(args.eval_episodes, eval_seed=1987654321 + args.seed)
    res = tr.train()
    # Separate final evidence stream, unused by either updates or monitored curves.
    final = tr.evaluate(args.eval_episodes, eval_seed=1987654321 + args.seed)
    save_json({**meta, "status": "complete", "holdout_seed": 1987654321 + args.seed,
               "initial_holdout": initial_holdout, "final_holdout": final}, out / "result.json")
    keys = sorted(res.eval_curve[-1].keys()) if res.eval_curve else []
    print(f"\neval_curve points: {len(res.eval_curve)} | keys incl: "
          f"{[k for k in keys if any(t in k for t in ('false_eject','creator_surv','false_claim','narrative','same_target'))]}")
    print(f"saved -> {out}/history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
