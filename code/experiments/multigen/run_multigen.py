#!/usr/bin/env python3
"""Multi-generation attacker/defender learning dynamics (one training seed per process).

    C0 -> D1 -> C1 -> D2 -> C2 -> ... -> Dk -> Ck

Every stage freezes one side and trains the other for a fixed PPO budget, exactly as the
existing C0 -> D1 -> C1 experiment does (scripts/train_f3_*.py): the same env, the same
per-stage update budget, stage seeds offset by +10000 per stage, and the same evaluation seed.
D0 is the scripted soft-credibility crew the paper's C0 was trained against.

After every stage the frozen matchup is evaluated (coalition win / false ejection / same-target
voting / false claims / game length / realized dependence), both sides' policy entropies are
measured on a fixed probe set, and the newly trained side is compared with its predecessor by
mean KL and JS divergence on that probe set and by a behavioural vote-pattern divergence. After
the last stage every C_i is played against every D_j (the (k+1) x (k+1) cross-play matrix) and
the pairwise probe-set KL between all same-side generations is recorded.

    python experiments/multigen/run_multigen.py --config experiments/multigen/config.yaml --seed 0
    python experiments/multigen/run_multigen.py --config ... --seed 0 --smoke
    python experiments/multigen/run_multigen.py --config ... --seed 0 --defense dependence

Outputs (under $SOCIAL_COLLUSION_RUNS/<name>/):
    generations.csv / generations.json   one row per stage (the per-generation record)
    crossplay.json                        C_i vs D_j matrices (win, false ejection, rounds)
    evaluation_records.json               index of per-game/per-meeting CSVs for every evaluation
    evaluation/<stage-or-pair>/            episode_metrics.csv and round_metrics.csv
    policy_matrices.json                  probe-set KL / JS between all generations, per side
    stageNN_<side><g>/history.json        PPO training history and checkpoint of that stage
    experiment_config.json, runmeta.json
"""

from __future__ import annotations

import argparse

# make experiments/ importable so every experiment shares exp_common (and scripts/_common)
import sys as _sys  # noqa: E402
import time
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parents[1])
if _EXPERIMENTS_DIR not in _sys.path:
    _sys.path.insert(0, _EXPERIMENTS_DIR)

from exp_common import (  # noqa: E402
    REPO,
    banner,
    limit_threads,
    load_yaml,
    run_dir,
    save_csv,
    save_json,
    write_provenance,
)
from train_f3_matched_cycle import (  # noqa: E402
    _episode_rows, _round_rows, load_role_checkpoint, summarize_pair,
)

from social_collusion.config import EnvConfig, load_algo_config, load_env_config  # noqa: E402
from social_collusion.env import VecEnv, dependence  # noqa: E402
from social_collusion.env.enums import Role  # noqa: E402
from social_collusion.policies.base import BasePolicy, RoleRouter  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402
from social_collusion.rl import policy_distance as pdist  # noqa: E402
from social_collusion.rl.train import Trainer  # noqa: E402
from social_collusion.rl.torch_policy import evaluation_copy

DEFAULT_CONFIG = REPO / "experiments" / "multigen" / "config.yaml"
EVAL_SEED_BASE = 1987654321  # identical to the F3/F4 launchers
STAGE_SEED_STEP = 10000  # C0: seed, D1: seed + 10000, C1: seed + 20000, ... (as in F3)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default=None, help="run directory name (default derived from the config)")
    ap.add_argument("--generations", type=int, default=None, help="override k")
    ap.add_argument("--n-agents", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--defense", choices=("none", "dependence"), default=None)
    ap.add_argument("--dependence-penalty", type=float, default=None)
    ap.add_argument("--dependence-window", type=int, default=None)
    ap.add_argument("--updates-per-stage", type=int, default=None)
    ap.add_argument("--single-round-objectives", default=None, metavar="COALITION,CREW",
                    help="reward ablation: single-meeting objectives for coalition and crew stages, e.g. "
                         "balanced_ejection,balanced_ejection (default: the config's side-specific pair)")
    ap.add_argument("--multi-round-objective", default=None, choices=("survive", "false_ejection"),
                    help="reward ablation: multi-round objective for every stage (default: survive)")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--crossplay-episodes", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny budget to verify the pipeline end to end")
    ap.add_argument("--skip-crossplay", action="store_true")
    return ap.parse_args()


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    overrides = {
        "generations": args.generations,
        "n_agents": args.n_agents,
        "max_rounds": args.max_rounds,
        "defense": args.defense,
        "dependence_penalty": args.dependence_penalty,
        "dependence_window": args.dependence_window,
        "updates_per_stage": args.updates_per_stage,
        "multi_round_objective": args.multi_round_objective,
        "eval_episodes": args.eval_episodes,
        "crossplay_episodes": args.crossplay_episodes,
        "device": args.device,
        "threads": args.threads,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.single_round_objectives is not None:
        coal_obj, crew_obj = args.single_round_objectives.split(",")
        cfg["single_round_objectives"] = {"coalition": coal_obj, "crew": crew_obj}
    cfg["seed"] = int(args.seed)
    cfg["smoke"] = bool(args.smoke)
    if args.smoke:
        cfg.update(
            generations=min(int(cfg.get("generations", 2)), 2),
            updates_per_stage=3,
            n_envs=4,
            rollout_episodes=4,
            eval_every=0,
            eval_episodes=20,
            crossplay_episodes=16,
            checkpoint_every=0,
            probe_episodes_per_policy=2,
        )
    cfg["name"] = args.name or (
        f"multigen_{cfg['defense']}_n{cfg['n_agents']}_r{cfg['max_rounds']}_k{cfg['generations']}"
        f"_s{cfg['seed']}" + ("_smoke" if args.smoke else "")
    )
    return cfg


def make_env_configs(s: dict[str, Any]) -> tuple[EnvConfig, EnvConfig, EnvConfig]:
    """(coalition-stage cfg, crew-stage cfg, evaluation cfg). They differ only in the objective."""
    vote_aggregation = "dependence_weighted" if s["defense"] == "dependence" else "majority"
    base = load_env_config(s["env"]).with_(
        n_agents=int(s["n_agents"]),
        n_coalition=int(s["n_coalition"]),
        max_rounds=int(s["max_rounds"]),
        vote_aggregation=vote_aggregation,
        reputation_accuracy=False,
        reveal_role_on_eject=False,
        dependence_penalty=float(s["dependence_penalty"]),
        dependence_window=int(s["dependence_window"]),
        dependence_top_j=int(s["dependence_top_j"]),
    )
    if int(s["max_rounds"]) > 1:
        obj = s["multi_round_objective"]
        return base.with_(coalition_objective=obj), base.with_(coalition_objective=obj), base.with_(coalition_objective=obj)
    so = s["single_round_objectives"]
    return (
        base.with_(coalition_objective=so["coalition"]),
        base.with_(coalition_objective=so["crew"]),
        base.with_(coalition_objective=so["coalition"]),
    )


def base_defender(s: dict[str, Any]) -> BasePolicy:
    rule = "dependence_aware" if s["defense"] == "dependence" else "soft_credibility"
    return TruthfulCrew(rule=rule)


def evaluate(cfg: EnvConfig, crew: BasePolicy, coalition: BasePolicy, episodes: int, seed: int, pair: str) -> tuple[dict, list]:
    env = VecEnv(cfg, min(64, max(1, episodes)), seed=seed)
    states = env.run_episodes(evaluation_copy(RoleRouter(crew, coalition), seed), episodes)
    out = summarize_pair(states, cfg, pair, seed)
    out.update({f"dep_{k}": v for k, v in dependence.summarize_dependence(states).items()})
    return out, states


def save_evaluation_records(states: list, pair: str, seed: int, out: Path,
                            evaluation_id: str, scope: str) -> dict[str, Any]:
    """Persist each game and completed meeting without retaining all matrix cells in RAM."""
    directory = out / "evaluation" / evaluation_id
    directory.mkdir(parents=True, exist_ok=True)
    episode_rows = _episode_rows(states, pair, seed)
    round_rows = _round_rows(states, pair, seed)
    episodes_path = directory / "episode_metrics.csv"
    rounds_path = directory / "round_metrics.csv"
    save_csv(episode_rows, episodes_path)
    save_csv(round_rows, rounds_path)
    return {
        "evaluation_id": evaluation_id,
        "scope": scope,
        "pair": pair,
        "eval_seed": int(seed),
        "n_episodes": len(episode_rows),
        "n_meetings": len(round_rows),
        "episode_metrics": str(episodes_path.relative_to(out)),
        "round_metrics": str(rounds_path.relative_to(out)),
    }


def main() -> int:
    args = parse_args()
    s = build_settings(args)
    limit_threads(int(s.get("threads", 1)))
    seed, k = int(s["seed"]), int(s["generations"])
    cfg_c, cfg_d, cfg_eval = make_env_configs(s)
    algo_base = load_algo_config("ippo")
    common = dict(
        n_envs=int(s["n_envs"]),
        rollout_episodes=int(s["rollout_episodes"]),
        eval_every=int(s["eval_every"]),
        checkpoint_every=int(s["checkpoint_every"]),
        eval_episodes=int(s["eval_episodes"]),
        total_updates=int(s["updates_per_stage"]),
    )
    algo = dict(algo_base, **common)
    device = str(s.get("device", "cpu"))
    out = run_dir(s["name"])
    eval_seed = EVAL_SEED_BASE + seed
    meta = {
        "experiment": "multigen_alternating_response_cycle",
        "settings": s,
        "env_config_coalition_stage": cfg_c.to_dict(),
        "env_config_crew_stage": cfg_d.to_dict(),
        "algo_config": algo,
        "eval_seed": eval_seed,
        "stage_seed_step": STAGE_SEED_STEP,
    }
    write_provenance(out, meta)
    save_json(meta, out / "experiment_config.json")
    banner(
        f"Multigen | defense={s['defense']} | n={cfg_c.n_agents} ({cfg_c.n_crew}+{cfg_c.n_coalition}) | "
        f"rounds={cfg_c.max_rounds} | k={k} | seed={seed} | {out}"
    )

    # fixed probe set (same for every seed and generation)
    t0 = time.time()
    probe = pdist.collect_probe_set(
        cfg_eval,
        pdist.default_probe_policies(cfg_eval),
        seed=int(s["probe_seed"]),
        episodes_per_policy=int(s["probe_episodes_per_policy"]),
        keep_states=True,
    )
    print(f"probe set: {len(probe)} decision points ({time.time() - t0:.1f}s)", flush=True)

    defenders: dict[int, BasePolicy] = {0: base_defender(s)}
    coalitions: dict[int, BasePolicy] = {}
    probs_c: dict[int, list[np.ndarray]] = {}
    probs_d: dict[int, list[np.ndarray]] = {0: pdist.head_probs(defenders[0], probe)}
    stage_dirs: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    stages: list[tuple[str, int]] = [("C", 0)]
    for g in range(1, k + 1):
        stages += [("D", g), ("C", g)]

    for idx, (side, g) in enumerate(stages):
        stage_seed = seed + STAGE_SEED_STEP * idx
        stage_name = f"stage{idx:02d}_{side}{g}"
        stage_out = out / stage_name
        t_stage = time.time()
        if side == "C":
            trainer = Trainer(
                cfg_c, algo, crew_policy=defenders[g], learner_role=Role.COALITION,
                seed=stage_seed, run_dir=stage_out, device=device,
                extra_metrics=dependence.summarize_dependence,
            )
            role = Role.COALITION
        else:
            trainer = Trainer(
                cfg_d, algo, crew_policy=None, opponent_policy=coalitions[g - 1],
                learner_role=Role.CREW, seed=stage_seed, run_dir=stage_out, device=device,
                extra_metrics=dependence.summarize_dependence,
            )
            role = Role.CREW
        print(f"\n[{stage_name}] training {side}{g} vs frozen {'D' if side == 'C' else 'C'}{g if side == 'C' else g - 1} (seed {stage_seed})", flush=True)
        res = trainer.train(log_every=max(1, int(s["updates_per_stage"]) // 8))
        if not res.checkpoint:
            raise RuntimeError(f"{stage_name} produced no checkpoint")
        pol = load_role_checkpoint(res.checkpoint, role, device)
        stage_dirs[f"{side}{g}"] = str(stage_out)
        train_wall = time.time() - t_stage

        # frozen matchup after this stage
        if side == "C":
            coalitions[g] = pol
            probs_c[g] = pdist.head_probs(pol, probe)
            crew_pol, coal_pol = defenders[g], coalitions[g]
            c_gen, d_gen = g, g
            prev = probs_c.get(g - 1)
        else:
            defenders[g] = pol
            probs_d[g] = pdist.head_probs(pol, probe)
            crew_pol, coal_pol = defenders[g], coalitions[g - 1]
            c_gen, d_gen = g - 1, g
            prev = probs_d.get(g - 1) if g - 1 >= 1 else None  # D0 is scripted: no KL
        pair = f"C{c_gen}_vs_D{d_gen}"
        summary, states = evaluate(cfg_eval, crew_pol, coal_pol, int(s["eval_episodes"]), eval_seed, pair)
        evaluation_records.append(save_evaluation_records(
            states, pair, eval_seed, out, stage_name, "stage",
        ))
        save_json({"evaluations": evaluation_records}, out / "evaluation_records.json")
        trained_probs = probs_c[g] if side == "C" else probs_d[g]
        dist = pdist.compare_policies(trained_probs, prev, probe, role)
        # behavioural distance to the scripted D0 is still defined for D1
        if side == "D" and g == 1:
            dist["vote_pattern_js"] = pdist.vote_pattern_divergence(trained_probs, probs_d[0], probe, role)
            dist["argmax_disagreement"] = pdist.argmax_disagreement(
                trained_probs, probs_d[0], probe, probe.rows(role=role)
            )
        ent_c = pdist.compare_policies(probs_c[c_gen], None, probe, Role.COALITION)
        ent_d = pdist.compare_policies(probs_d[d_gen], None, probe, Role.CREW)
        last = res.history[-1] if res.history else {}
        row: dict[str, Any] = {
            "stage_index": idx,
            "side": side,
            "generation": g,
            "coalition_generation": c_gen,
            "defender_generation": d_gen,
            "stage_seed": stage_seed,
            "train_wall_s": round(train_wall, 1),
            "train_final_false_ejection": last.get("false_ejection_rate"),
            "train_final_coalition_win": last.get("coalition_win_rate"),
            "train_final_ppo_entropy": last.get("entropy"),
            "train_final_same_target_vote": last.get("train_same_target_vote"),
            **{k_: v for k_, v in summary.items() if k_ not in ("pair", "eval_seed")},
            "pair": pair,
            "eval_seed": eval_seed,
            **{f"trained_{k_}": v for k_, v in dist.items()},
            "coalition_policy_entropy": ent_c["policy_entropy"],
            "coalition_vote_entropy": ent_c["vote_entropy"],
            "crew_policy_entropy": ent_d["policy_entropy"],
            "crew_vote_entropy": ent_d["vote_entropy"],
        }
        rows.append(row)
        save_csv(rows, out / "generations.csv")
        save_json({"rows": rows, "stage_dirs": stage_dirs}, out / "generations.json")
        win = summary.get("coalition_game_win_rate")
        print(
            f"[{stage_name}] {pair}: FE={summary['false_ejection_rate']:.3f} "
            f"fav={summary['coalition_favorable_rate']:.3f} "
            f"win={'NA' if win is None else f'{win:.3f}'} rounds={summary['mean_rounds_played']:.2f} "
            f"same_target={summary.get('same_target_vote_rate', float('nan')):.3f} "
            f"KL_prev={dist.get('kl_to_previous', float('nan')):.3f} "
            f"voteJS_prev={dist.get('vote_pattern_js', float('nan')):.3f} | {train_wall:.0f}s",
            flush=True,
        )

    # ---- cross-generation evaluation: every C_i vs every D_j ------------------------------
    if not args.skip_crossplay:
        banner("cross-play: every coalition generation vs every defender generation")
        keys = ("coalition_game_win_rate", "coalition_favorable_rate", "false_ejection_rate",
                "mean_rounds_played", "same_target_vote_rate", "creator_ejection_rate",
                "any_false_ejection_rate", "mean_false_ejections", "false_ejections_per_meeting")
        mats = {k_: np.full((k + 1, k + 1), np.nan) for k_ in keys}
        t0 = time.time()
        for i in range(k + 1):
            for j in range(k + 1):
                pair = f"C{i}_vs_D{j}"
                summ, states = evaluate(cfg_eval, defenders[j], coalitions[i], int(s["crossplay_episodes"]), eval_seed, pair)
                evaluation_records.append(save_evaluation_records(
                    states, pair, eval_seed, out, pair, "crossplay",
                ))
                save_json({"evaluations": evaluation_records}, out / "evaluation_records.json")
                for k_ in keys:
                    v = summ.get(k_)
                    mats[k_][i, j] = np.nan if v is None else float(v)
            print(f"  C{i} row done ({time.time() - t0:.0f}s)", flush=True)
        save_json(
            {
                "rows": "coalition generation i (0..k)",
                "cols": "defender generation j (0 = scripted base defender)",
                "episodes_per_cell": int(s["crossplay_episodes"]),
                "matrices": {k_: m.tolist() for k_, m in mats.items()},
            },
            out / "crossplay.json",
        )

    # ---- probe-set distances between all generations of each side -------------------------
    def matrix(probs: dict[int, list[np.ndarray]], role: Role, fn) -> dict[str, Any]:
        gens = sorted(probs)
        m = np.full((len(gens), len(gens)), np.nan)
        rows_sel = probe.rows(role=role)
        for a, ga in enumerate(gens):
            for b, gb in enumerate(gens):
                m[a, b] = float(fn(probs[ga], probs[gb], probe, rows_sel).mean())
        return {"generations": gens, "matrix": m.tolist()}

    learned_d = {g: p for g, p in probs_d.items() if g >= 1}
    save_json(
        {
            "coalition_kl": matrix(probs_c, Role.COALITION, pdist.kl_divergence),
            "coalition_js": matrix(probs_c, Role.COALITION, pdist.js_divergence),
            "crew_kl": matrix(learned_d, Role.CREW, pdist.kl_divergence),
            "crew_js": matrix(learned_d, Role.CREW, pdist.js_divergence),
            "probe_rows": {"coalition": int(probe.rows(role=Role.COALITION).size), "crew": int(probe.rows(role=Role.CREW).size)},
        },
        out / "policy_matrices.json",
    )
    write_provenance(out, {**meta, "status": "complete"})
    print(f"\nsaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
