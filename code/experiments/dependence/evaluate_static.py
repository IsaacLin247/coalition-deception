#!/usr/bin/env python3
"""Task 3, stage 1: the dependence-aware defense against *static* coalitions.

Two evaluations, both in the spatial game the learned checkpoints were trained in
(`full_short_game`, single meeting unless --max-rounds says otherwise):

1. F2-style rule sweep: every crew aggregation rule (mean, median, trimmed, soft / sharp
   credibility, dependence-aware) against the scripted coalitions (truthful, lone liar, alibi,
   framer) -- does the new rule keep credibility's honest-testimony advantage while removing the
   manufactured-corroboration exploit?
2. Learned coalitions: the same rules against the trained C0 / C1 (and any later C_g) checkpoints
   of a multi-generation run -- does the defense neutralise the *learned* attack?
3. A penalty-strength sweep for the dependence-aware rule and for the dependence-weighted vote
   tally (the mechanism used inside the multi-generation loop).

    python experiments/dependence/evaluate_static.py --n-agents 7 --seed 0 \
        --checkpoints "$SOCIAL_COLLUSION_RUNS/multigen_none_n7_r8_k10_s0"

Outputs `rules.csv`, `strength_sweep.csv`, `learned.csv` under the run directory plus figures in
experiments/dependence/figures (via aggregate_dependence.py).
"""

from __future__ import annotations

import argparse
import glob
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

from social_collusion.config import load_env_config  # noqa: E402
from social_collusion.env import VecEnv, dependence  # noqa: E402
from social_collusion.env.enums import Role  # noqa: E402
from social_collusion.metrics import collusion_metrics as cm  # noqa: E402
from social_collusion.metrics import outcome_metrics as om  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.crew_aggregation import RULES as _AGGREGATION_RULES  # noqa: E402
from social_collusion.policies.hypothesis_crew import HYPOTHESIS_RULE, crew_for_rule  # noqa: E402
from social_collusion.rl.torch_policy import evaluation_copy

#: aggregation rules plus the hypothesis-elimination crew (appended: cell seeds are index-based)
RULES = tuple(_AGGREGATION_RULES) + (HYPOTHESIS_RULE,)
from social_collusion.policies.registry import make_coalition  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402

CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
STRENGTHS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)


def summarize(states, cfg) -> dict[str, float]:
    out = om.summarize(states)
    out["same_target_vote_rate"] = cm.same_target_vote_rate(states)
    out["coalition_false_claim_rate"] = cm.deception_rates(states).get("false_claim_rate", float("nan"))
    out["mean_rounds_played"] = float(np.mean([len(s.round_log) or 1 for s in states]))
    out["coalition_game_win_rate"] = float(np.mean([coalition_game_won(s) for s in states])) if cfg.max_rounds > 1 else None
    out.update({f"dep_{k}": v for k, v in dependence.summarize_dependence(states).items()})
    return out


def play(cfg, crew, coalition, episodes, seed):
    return VecEnv(cfg, min(64, episodes), seed=seed).run_episodes(
        evaluation_copy(RoleRouter(crew, coalition), seed), episodes)


def find_checkpoints(root: str | None, cfg) -> dict[str, Path]:
    """`C<g>` -> checkpoint of the coalition trained at generation g in a multigen run dir.

    Checkpoints are only usable in the game they were trained in (the observation layout depends
    on the player count, rooms, horizon and symbol channel), so a mismatch is an error here rather
    than a shape error deep inside torch.
    """
    if not root:
        return {}
    import torch

    out: dict[str, Path] = {}
    for d in sorted(glob.glob(str(Path(root) / "stage*_C*"))):
        ck = Path(d) / "checkpoint_final.pt"
        if not ck.exists():
            continue
        saved = torch.load(ck, map_location="cpu", weights_only=False)["env_config"]
        for key in ("n_agents", "n_coalition", "rooms", "free_play_turns", "enable_symbol_channel", "n_symbols"):
            mine = cfg.to_dict()[key]
            if saved.get(key) != mine:
                raise ValueError(
                    f"{ck} was trained with {key}={saved.get(key)!r} but this evaluation uses {mine!r}; "
                    "pass the multigen run for the same crew size"
                )
        if saved.get("actor_private_history", False) != cfg.actor_private_history:
            raise ValueError(f"{ck} uses a different private-history observation schema")
        out[Path(d).name.split("_")[-1]] = ck
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", default=None, help="a multigen run directory (its stage*_C* checkpoints are evaluated)")
    ap.add_argument("--max-learned-generation", type=int, default=3, help="evaluate C0..C<this> from --checkpoints")
    ap.add_argument("--dependence-window", type=int, default=3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    limit_threads(args.threads)
    if args.smoke:
        args.episodes = 24
    base = load_env_config("full_short_game").with_(
        n_agents=args.n_agents, n_coalition=2, max_rounds=args.max_rounds,
        coalition_objective="survive" if args.max_rounds > 1 else "balanced_ejection",
        vote_aggregation="majority", dependence_window=args.dependence_window,
    )
    out = run_dir(args.name or f"depstatic_n{args.n_agents}_r{args.max_rounds}_s{args.seed}" + ("_smoke" if args.smoke else ""))
    meta = {"experiment": "dependence_static_evaluation", "args": vars(args), "env_config": base.to_dict()}
    write_provenance(out, meta)
    banner(f"Dependence-aware defense vs static coalitions | n={args.n_agents} | rounds={args.max_rounds} | seed={args.seed}")

    # 1. rule x scripted-condition sweep (default strength)
    rows: list[dict[str, Any]] = []
    for ci, cond in enumerate(CONDITIONS):
        for ri, rule in enumerate(RULES):
            eval_seed = 1987654321 + args.seed * 10000
            states = play(base, crew_for_rule(rule), make_coalition(cond), args.episodes, eval_seed)
            rows.append({"seed": args.seed, "condition": cond, "rule": rule, "episodes": args.episodes, **summarize(states, base)})
            print(f"  {cond:9s} {rule:20s} FE={rows[-1]['false_ejection_rate']:.3f} fav={rows[-1]['coalition_favorable_rate']:.3f}", flush=True)
    save_csv(rows, out / "rules.csv")

    # 2. strength sweep: crew rule and vote tally, against the scripted alibi and truthful coalitions
    sweep: list[dict[str, Any]] = []
    for cond in ("alibi", "truthful"):
        for si, lam in enumerate(STRENGTHS):
            eval_seed = 1987654321 + args.seed * 10000 + 5000
            cfg_rule = base.with_(dependence_penalty=lam)
            st = play(cfg_rule, TruthfulCrew(rule="dependence_aware"), make_coalition(cond), args.episodes, eval_seed)
            sweep.append({"seed": args.seed, "condition": cond, "mechanism": "crew_rule", "strength": lam, **summarize(st, cfg_rule)})
            cfg_vote = base.with_(vote_aggregation="dependence_weighted", dependence_penalty=lam)
            st = play(cfg_vote, TruthfulCrew(rule="soft_credibility"), make_coalition(cond), args.episodes, eval_seed)
            sweep.append({"seed": args.seed, "condition": cond, "mechanism": "vote_tally", "strength": lam, **summarize(st, cfg_vote)})
            st = play(cfg_vote, TruthfulCrew(rule="dependence_aware"), make_coalition(cond), args.episodes, eval_seed)
            sweep.append({"seed": args.seed, "condition": cond, "mechanism": "both", "strength": lam, **summarize(st, cfg_vote)})
            print(f"  strength {lam:.0f} {cond:8s}: rule FE={sweep[-3]['false_ejection_rate']:.3f} | tally FE={sweep[-2]['false_ejection_rate']:.3f} | both FE={sweep[-1]['false_ejection_rate']:.3f}", flush=True)
    save_csv(sweep, out / "strength_sweep.csv")

    # 3. learned coalitions from a multigen run
    learned: list[dict[str, Any]] = []
    cks = find_checkpoints(args.checkpoints, base)
    cks = {k: v for k, v in cks.items() if int(k[1:]) <= args.max_learned_generation}
    if cks:
        banner(f"learned coalitions: {sorted(cks)}")
        for gi, (gen, ck) in enumerate(sorted(cks.items(), key=lambda kv: int(kv[0][1:]))):
            coal = load_role_checkpoint(ck, Role.COALITION)
            for ri, rule in enumerate(RULES):
                eval_seed = 1987654321 + args.seed * 10000 + 7000
                st = play(base, crew_for_rule(rule), coal, args.episodes, eval_seed)
                learned.append({"seed": args.seed, "coalition": gen, "rule": rule, "mechanism": "crew_rule", **summarize(st, base)})
            cfg_vote = base.with_(vote_aggregation="dependence_weighted")
            st = play(cfg_vote, TruthfulCrew(rule="dependence_aware"), coal, args.episodes, 1987654321 + args.seed * 10000 + 7000)
            learned.append({"seed": args.seed, "coalition": gen, "rule": "dependence_aware", "mechanism": "both", **summarize(st, cfg_vote)})
            fe = {r["rule"]: r["false_ejection_rate"] for r in learned if r["coalition"] == gen and r["mechanism"] == "crew_rule"}
            print(f"  {gen}: " + " ".join(f"{k}={v:.3f}" for k, v in fe.items()) + f" | both={learned[-1]['false_ejection_rate']:.3f}", flush=True)
        save_csv(learned, out / "learned.csv")
    save_json({**meta, "status": "complete", "checkpoints": {k: str(v) for k, v in cks.items()}}, out / "result.json")
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
