#!/usr/bin/env python3
"""Task 3, stage 2: adaptive counterattack against the dependence-aware defense.

Trains a *fresh* coalition (same budget, env and seed convention as C0 in the paper) against the
dependence-aware scripted crew, and -- as the paired control -- against the soft-credibility crew
it was originally trained against. Every PPO update logs, besides the usual outcome metrics, the
realized coalition pairwise dependence and penalty the defense computes on the rollout episodes,
so the training curve shows whether the coalition learns to *decorrelate* (fly under the radar)
while still colluding effectively.

Two defense placements are trained against (both use the same penalty strength):
    rule   -- crew aggregation rule `dependence_aware` (plurality vote)
    both   -- `dependence_aware` crew rule *and* `vote_aggregation=dependence_weighted`
              (the placement used inside the multi-generation loop)

Two further opponents answer the adaptive-attack question for the non-weighting aggregators
(journal revision): `mean`, the best summing rule against the scripted alibi, and `hypothesis`,
the likelihood-based hypothesis-elimination crew. A coalition trained directly against the
hypothesis crew tests whether its robustness to *scripted* corroboration survives an adversary
that is optimised against it.

After training, every coalition is cross-evaluated against every defense.

    python experiments/dependence/train_counterattack.py --n-agents 7 --seed 0
    python experiments/dependence/train_counterattack.py --n-agents 7 --seed 0 \
        --defenses soft,mean,hypothesis --name counterattack_hyp_n7_r1_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parents[1])
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from exp_common import (  # noqa: E402
    banner,
    limit_threads,
    run_dir,
    save_csv,
    save_json,
    write_provenance,
)
from train_f3_matched_cycle import coalition_game_won, load_role_checkpoint  # noqa: E402

from social_collusion.config import load_algo_config, load_env_config  # noqa: E402
from social_collusion.env import VecEnv, dependence  # noqa: E402
from social_collusion.env.enums import Role  # noqa: E402
from social_collusion.metrics import collusion_metrics as cm  # noqa: E402
from social_collusion.metrics import outcome_metrics as om  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.hypothesis_crew import HYPOTHESIS_RULE, crew_for_rule  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402
from social_collusion.rl.train import Trainer
from social_collusion.rl.torch_policy import evaluation_copy  # noqa: E402

DEFENSES = ("soft", "rule", "both")
#: every opponent the script knows; the default `--defenses` is the original triple
ALL_DEFENSES = DEFENSES + ("mean", HYPOTHESIS_RULE)
_CREW_RULE = {"soft": "soft_credibility", "rule": "dependence_aware", "both": "dependence_aware",
              "mean": "mean", HYPOTHESIS_RULE: HYPOTHESIS_RULE}


def defense_cfg(base, name: str):
    if name == "both":
        return base.with_(vote_aggregation="dependence_weighted")
    return base


def defense_crew(name: str) -> TruthfulCrew:
    if name not in _CREW_RULE:
        raise ValueError(f"unknown defense {name!r}; have {ALL_DEFENSES}")
    return crew_for_rule(_CREW_RULE[name])


def extra_metrics(states) -> dict[str, float]:
    out = dependence.summarize_dependence(states)
    d = cm.deception_rates(states)
    out["coalition_false_claim_rate"] = d.get("false_claim_rate", float("nan"))
    out["partner_defense_rate"] = cm.partner_defense_rate(states)
    out["shared_framing_rate"] = cm.shared_framing_rate(states)
    return out


def evaluate(cfg, crew, coalition, episodes, seed) -> dict[str, Any]:
    states = VecEnv(cfg, min(64, episodes), seed=seed).run_episodes(evaluation_copy(RoleRouter(crew, coalition), seed), episodes)
    out = om.summarize(states)
    out["same_target_vote_rate"] = cm.same_target_vote_rate(states)
    out["partner_defense_rate"] = cm.partner_defense_rate(states)
    out["coalition_false_claim_rate"] = cm.deception_rates(states).get("false_claim_rate", float("nan"))
    out["mean_rounds_played"] = float(np.mean([len(s.round_log) or 1 for s in states]))
    out["coalition_game_win_rate"] = float(np.mean([coalition_game_won(s) for s in states])) if cfg.max_rounds > 1 else None
    out.update({f"dep_{k}": v for k, v in dependence.summarize_dependence(states).items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--rollout-episodes", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--crossplay-episodes", type=int, default=1000)
    ap.add_argument("--dependence-penalty", type=float, default=4.0)
    ap.add_argument("--dependence-window", type=int, default=3)
    ap.add_argument("--defenses", default=",".join(DEFENSES),
                    help=f"comma-separated subset of {ALL_DEFENSES}; a fresh coalition is trained against each and cross-evaluated against all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    limit_threads(args.threads)
    if args.smoke:
        args.updates, args.n_envs, args.rollout_episodes = 3, 4, 4
        args.eval_every, args.eval_episodes, args.crossplay_episodes = 2, 16, 24
    defenses = tuple(args.defenses.split(","))
    base = load_env_config("full_short_game").with_(
        n_agents=args.n_agents, n_coalition=2, max_rounds=args.max_rounds, vote_aggregation="majority",
        coalition_objective="survive" if args.max_rounds > 1 else "balanced_ejection",
        reputation_accuracy=False, reveal_role_on_eject=False,
        dependence_penalty=args.dependence_penalty, dependence_window=args.dependence_window,
    )
    algo = dict(load_algo_config("ippo"), n_envs=args.n_envs, rollout_episodes=args.rollout_episodes,
                eval_every=args.eval_every, eval_episodes=args.eval_episodes, checkpoint_every=50,
                total_updates=args.updates)
    out = run_dir(args.name or f"counterattack_n{args.n_agents}_r{args.max_rounds}_s{args.seed}" + ("_smoke" if args.smoke else ""))
    meta = {"experiment": "dependence_adaptive_counterattack", "args": vars(args), "env_config": base.to_dict(), "algo_config": algo}
    write_provenance(out, meta)
    save_json(meta, out / "experiment_config.json")
    banner(f"Adaptive counterattack | n={args.n_agents} | rounds={args.max_rounds} | seed={args.seed} | lambda={args.dependence_penalty}")

    coalitions = {}
    for name in defenses:
        cfg = defense_cfg(base, name)
        tr = Trainer(cfg, algo, defense_crew(name), seed=args.seed, run_dir=out / f"coalition_vs_{name}",
                     device=args.device, extra_metrics=extra_metrics)
        print(f"\n[train] fresh coalition vs defense '{name}' (seed {args.seed})", flush=True)
        res = tr.train(log_every=max(1, args.updates // 8))
        coalitions[name] = load_role_checkpoint(res.checkpoint, Role.COALITION, args.device)
        hist = res.history
        first, last = hist[0], hist[-1]
        print(
            f"[train] vs {name}: FE {first['false_ejection_rate']:.3f} -> {last['false_ejection_rate']:.3f} | "
            f"coalition dependence {first.get('coalition_pair_dependence', float('nan')):.3f} -> {last.get('coalition_pair_dependence', float('nan')):.3f} | "
            f"penalty {first.get('coalition_penalty', float('nan')):.3f} -> {last.get('coalition_penalty', float('nan')):.3f}",
            flush=True,
        )

    banner("cross-evaluation: every trained coalition vs every defense")
    rows: list[dict[str, Any]] = []
    eval_seed = 1987654321 + args.seed
    for trained_vs, coal in coalitions.items():
        for name in defenses:
            cfg = defense_cfg(base, name)
            s = evaluate(cfg, defense_crew(name), coal, args.crossplay_episodes, eval_seed)
            rows.append({"seed": args.seed, "coalition_trained_vs": trained_vs, "defense": name, **s})
            print(f"  C[{trained_vs:4s}] vs {name:4s}: FE={s['false_ejection_rate']:.3f} fav={s['coalition_favorable_rate']:.3f} "
                  f"same_target={s['same_target_vote_rate']:.3f} dep={s.get('dep_coalition_pair_dependence', float('nan')):.3f} "
                  f"pen={s.get('dep_coalition_penalty', float('nan')):.3f}", flush=True)
    save_csv(rows, out / "crossplay.csv")
    save_json({**meta, "status": "complete"}, out / "result.json")
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
