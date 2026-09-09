#!/usr/bin/env python3
"""Coordination ablation: what does the learned coalition's coordination depend on?

Trains the C0 stage of the paper (a fresh coalition against the scripted soft-credibility crew,
same budget, same seeds) in the spatial game under four information conditions and evaluates
every trained coalition against the same crew:

    default                 partner identity known, learned private symbol channel on (the paper)
    no_channel              partner identity known, no private channel
    no_partner              partner identity hidden, channel on
    no_channel_no_partner   neither

Reports the F1 behavioural metrics (false ejection, same-target voting, creator survival,
false-claim rate) and, in the multi-round game, the coalition game-win rate. This is the causal
control the F1-F4 results lacked: it separates coordination that needs a private channel or
partner identity from coordination that emerges from a shared objective alone.

    python experiments/ablation/train_coalition_ablation.py --seed 0 --n-agents 7 --max-rounds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from exp_common import banner, limit_threads, run_dir, save_csv, save_json, write_provenance  # noqa: E402
from train_f3_matched_cycle import coalition_game_won, load_role_checkpoint  # noqa: E402

from social_collusion.config import load_algo_config, load_env_config  # noqa: E402
from social_collusion.env import VecEnv  # noqa: E402
from social_collusion.env.enums import Role  # noqa: E402
from social_collusion.metrics import collusion_metrics as cm  # noqa: E402
from social_collusion.metrics import communication_metrics as comm  # noqa: E402
from social_collusion.metrics import outcome_metrics as om  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402
from social_collusion.rl.train import Trainer
from social_collusion.rl.torch_policy import evaluation_copy  # noqa: E402

CONDITIONS = {
    "default": dict(),
    "no_channel": dict(enable_symbol_channel=False, symbol_mode="none"),
    "no_partner": dict(partner_known=False),
    "no_channel_no_partner": dict(enable_symbol_channel=False, symbol_mode="none", partner_known=False),
}


def evaluate(cfg, crew, coalition, episodes: int, seed: int) -> dict[str, Any]:
    states = VecEnv(cfg, min(64, episodes), seed=seed).run_episodes(evaluation_copy(RoleRouter(crew, coalition), seed), episodes)
    out = om.summarize(states)
    out["same_target_vote_rate"] = cm.same_target_vote_rate(states)
    out["same_target_vote_lift"] = cm.same_target_vote_lift(states)
    out["partner_defense_rate"] = cm.partner_defense_rate(states)
    out["shared_framing_rate"] = cm.shared_framing_rate(states)
    out["coalition_false_claim_rate"] = cm.deception_rates(states).get("false_claim_rate", float("nan"))
    # Both aliases require a real incident; absence of a creator is not creator survival.
    out["creator_survival_rate"] = float(out["creator_survival"])
    out["mean_rounds_played"] = float(np.mean([len(s.round_log) or 1 for s in states]))
    out["coalition_game_win_rate"] = float(np.mean([coalition_game_won(s) for s in states])) if cfg.max_rounds > 1 else None
    if cfg.enable_symbol_channel:
        out.update({f"comm_{k}": v for k, v in comm.summarize(states).items() if isinstance(v, (int, float))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--rollout-episodes", type=int, default=32)
    ap.add_argument("--eval-episodes", type=int, default=1000)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    limit_threads(args.threads)
    if args.smoke:
        args.updates, args.n_envs, args.rollout_episodes, args.eval_episodes = 3, 4, 4, 16
    conditions = tuple(args.conditions.split(","))

    base = load_env_config("full_short_game").with_(
        n_agents=args.n_agents, n_coalition=2, max_rounds=args.max_rounds, vote_aggregation="majority",
        coalition_objective="survive" if args.max_rounds > 1 else "balanced_ejection",
        reputation_accuracy=False, reveal_role_on_eject=False,
    )
    algo = dict(load_algo_config("ippo"), n_envs=args.n_envs, rollout_episodes=args.rollout_episodes,
                eval_every=0, eval_episodes=args.eval_episodes, checkpoint_every=50, total_updates=args.updates)
    out = run_dir(args.name or f"ablation_n{args.n_agents}_r{args.max_rounds}_s{args.seed}" + ("_smoke" if args.smoke else ""))
    meta = {"experiment": "coalition_coordination_ablation", "args": vars(args), "env_config": base.to_dict(), "algo_config": algo}
    write_provenance(out, meta)
    save_json(meta, out / "experiment_config.json")
    banner(f"Coordination ablation | n={args.n_agents} | rounds={args.max_rounds} | seed={args.seed}")

    eval_seed = 1987654321 + args.seed
    rows: list[dict[str, Any]] = []
    for name in conditions:
        cfg = base.with_(**CONDITIONS[name])
        crew = TruthfulCrew(rule="soft_credibility")
        print(f"\n[train] coalition vs soft credibility under '{name}' (seed {args.seed})", flush=True)
        res = Trainer(cfg, algo, crew, seed=args.seed, run_dir=out / f"coalition_{name}", device=args.device).train(
            log_every=max(1, args.updates // 8)
        )
        coalition = load_role_checkpoint(res.checkpoint, Role.COALITION, args.device)
        summ = evaluate(cfg, TruthfulCrew(rule="soft_credibility"), coalition, args.eval_episodes, eval_seed)
        first = res.history[0] if res.history else {}
        row = {"seed": args.seed, "condition": name, "n_agents": args.n_agents, "max_rounds": args.max_rounds,
               "initial_false_ejection_rate": first.get("false_ejection_rate", float("nan")),
               "initial_same_target_vote_rate": first.get("same_target_vote_rate", float("nan")),
               **{k: v for k, v in summ.items() if isinstance(v, (int, float)) or v is None}}
        rows.append(row)
        print(f"[eval] {name:22s} FE={summ['false_ejection_rate']:.3f} same-target={summ['same_target_vote_rate']:.3f} "
              f"creator-survival={summ['creator_survival_rate']:.3f} false-claims={summ['coalition_false_claim_rate']:.3f}"
              + (f" game-win={summ['coalition_game_win_rate']:.3f}" if summ['coalition_game_win_rate'] is not None else ""), flush=True)
    save_csv(rows, out / "result.csv")
    save_json({"experiment": "coalition_coordination_ablation", "seed": args.seed, "rows": rows}, out / "result.json")
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
