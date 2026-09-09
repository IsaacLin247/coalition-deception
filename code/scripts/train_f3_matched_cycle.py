#!/usr/bin/env python3
"""Matched single-/multi-round F3 response cycle.

The same staged order is used for both round regimes so that ``max_rounds`` is the main
experimental contrast:

    scripted soft-credibility crew -> coalition_0 -> learned crew_1 -> coalition_1

All three learner stages save their histories and periodic checkpoints.  Evaluation keeps the
episode as the unit of analysis, writes per-round records when rounds exist, and stores a small
compressed replay set for later movement/meeting visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import banner, limit_threads, run_dir, save_csv, save_json, write_provenance

from social_collusion.config import load_algo_config, load_env_config
from social_collusion.env import VecEnv
from social_collusion.env.enums import Role
from social_collusion.metrics import collusion_metrics as cm
from social_collusion.metrics import communication_metrics as comm
from social_collusion.metrics import outcome_metrics as om
from social_collusion.policies.base import BasePolicy, RoleRouter
from social_collusion.policies.scripted_crew import TruthfulCrew
from social_collusion.replay import record_episode, validate_record, write as write_replay
from social_collusion.rl.train import RoleLearnerPolicy, Trainer
from social_collusion.rl.torch_policy import TorchPolicy, evaluation_copy


def load_role_checkpoint(path: str | Path, role: Role, device: str = "cpu") -> RoleLearnerPolicy:
    actor = TorchPolicy.load(path, device=device)
    return RoleLearnerPolicy([actor], separate=False, target_role=role)


def coalition_game_won(state) -> bool:
    coal = int(np.sum(state.alive & (state.roles == int(Role.COALITION))))
    crew = int(np.sum(state.alive & (state.roles == int(Role.CREW))))
    return coal > 0 and coal >= crew


def _round_rows(states: list, pair: str, eval_seed: int) -> list[dict[str, Any]]:
    """All completed meetings, with explicit innocent-ejection indicators.

    ``crew_framed`` retains the historical column: incident-gated for multi-round
    games, but any innocent ejection for the old single-meeting fallback. Use the
    explicit ``false_ejection`` columns for comparisons across round regimes.
    """
    rows: list[dict[str, Any]] = []
    for episode_index, state in enumerate(states):
        legacy = {int(record[0]): int(record[2]) for record in state.round_log}
        for meeting in om.meeting_outcomes(state):
            round_index = int(meeting["round"])
            had_incident = bool(meeting["had_incident"])
            false_ejection = bool(meeting["false_ejection"])
            framed = legacy.get(
                round_index,
                int(false_ejection and (had_incident or state.config.max_rounds == 1)),
            )
            rows.append(
                {
                    "pair": pair,
                    "eval_seed": int(eval_seed),
                    "episode_index": int(episode_index),
                    "round": int(round_index),
                    "had_incident": int(had_incident),
                    "crew_framed": int(framed),
                    "false_ejection": int(false_ejection),
                    "incident_false_ejection": int(false_ejection and had_incident),
                    "incident_free_false_ejection": int(false_ejection and not had_incident),
                    "ejected": int(meeting["ejected"]),
                    "creator_ejected": int(meeting["creator_ejected"]),
                    "any_ejection": int(meeting["any_ejection"]),
                }
            )
    return rows


def _episode_rows(states: list, pair: str, eval_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_index, state in enumerate(states):
        row = dict(om.episode_row(state))
        row.update(
            {
                "pair": pair,
                "eval_seed": int(eval_seed),
                "episode_index": int(episode_index),
                "rounds_played": int(len(state.round_log) if state.round_log else 1),
                "coalition_game_won": bool(coalition_game_won(state)),
                "total_crew_framed": int(state.crew_framed_count),
                "total_incident_rounds": int(state.incident_round_count),
                "total_creator_ejected": int(state.creator_ejected_count),
            }
        )
        rows.append(row)
    return rows


def summarize_pair(states: list, cfg, pair: str, eval_seed: int) -> dict[str, Any]:
    out: dict[str, Any] = om.summarize(states)
    out.update({k: v for k, v in cm.summarize_all(states).items() if isinstance(v, float)})
    if cfg.enable_symbol_channel:
        out.update(comm.summarize(states))
    out.update(
        {
            "pair": pair,
            "eval_seed": int(eval_seed),
            "n_agents": int(cfg.n_agents),
            "n_crew": int(cfg.n_crew),
            "n_coalition": int(cfg.n_coalition),
            "max_rounds": int(cfg.max_rounds),
            "vote_aggregation": cfg.vote_aggregation,
            "coalition_objective": cfg.coalition_objective,
            "coalition_survival_rate": float(out["coalition_favorable_rate"]),
            "coalition_game_win_rate": (
                float(np.mean([coalition_game_won(s) for s in states]))
                if cfg.max_rounds > 1
                else None
            ),
            "mean_rounds_played": float(
                np.mean([len(s.round_log) if s.round_log else 1 for s in states])
            ),
            "episodes_reaching_max_round": int(
                sum(1 for s in states if len(s.round_log) >= cfg.max_rounds)
            )
            if cfg.max_rounds > 1
            else int(len(states)),
        }
    )
    return out


def evaluate_pair(
    cfg,
    crew: BasePolicy,
    coalition: BasePolicy,
    pair: str,
    episodes: int,
    eval_seed: int,
    replay_dir: Path,
    replay_episodes: int,
    runmeta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = RoleRouter(crew, coalition)
    env = VecEnv(cfg, min(32, max(1, episodes)), seed=eval_seed)
    states = env.run_episodes(evaluation_copy(policy, eval_seed), episodes)
    summary = summarize_pair(states, cfg, pair, eval_seed)
    episode_rows = _episode_rows(states, pair, eval_seed)
    round_rows = _round_rows(states, pair, eval_seed)

    replay_rows: list[dict[str, Any]] = []
    for episode_index in range(min(replay_episodes, episodes)):
        # Serial replay is an illustrative resampling of this environment draw;
        # its actor draws differ from batched evaluation. Isolate and seed it too.
        replay_actor_seed = eval_seed + 3000000000 + episode_index
        replay_policy = evaluation_copy(RoleRouter(crew, coalition), replay_actor_seed)
        replay_policy.reset(1)
        _, record = record_episode(
            cfg,
            replay_policy,
            seed=eval_seed,
            episode_index=episode_index,
            with_observations=False,
            runmeta={**runmeta, "trajectory_source": "independent_serial_replay",
                     "actor_seed": int(replay_actor_seed)},
        )
        replay_path = replay_dir / f"{pair}_episode{episode_index}.json.gz"
        write_replay(record, replay_path, compress=True)
        validation = validate_record(record, check_determinism=True)
        replay_rows.append(
            {
                "pair": pair,
                "eval_seed": int(eval_seed),
                "episode_index": int(episode_index),
                "actor_seed": int(replay_actor_seed),
                "trajectory_source": "independent_serial_replay",
                "path": str(replay_path),
                "valid": bool(validation.ok),
                "validation_errors": " | ".join(validation.errors),
                "steps": int(len(record.steps)),
                "rounds": int(
                    sum(
                        1
                        for step in record.steps
                        for event in step.events
                        if event.get("type") == "RoundStarted"
                    )
                    + 1
                ),
            }
        )
    return summary, episode_rows, round_rows, replay_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, required=True)
    ap.add_argument("--coalition", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, choices=(1, 8), required=True)
    ap.add_argument("--initial-updates", type=int, default=300)
    ap.add_argument("--defender-updates", type=int, default=300)
    ap.add_argument("--adapted-updates", type=int, default=300)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--rollout-episodes", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--crossplay-episodes", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--replay-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--initial-only",
        action="store_true",
        help="train and evaluate coalition C0 only; do not train the defender or adapted coalition",
    )
    ap.add_argument(
        "--single-round-objective",
        choices=("false_ejection", "balanced_ejection", "risk_balanced_ejection"),
        default="false_ejection",
        help="single-round terminal reward; balanced_ejection is +1/0/-1 and risk_balanced_ejection is +1/-0.5/-1",
    )
    args = ap.parse_args()
    limit_threads(args.threads)

    if args.smoke:
        args.initial_updates = 3
        args.defender_updates = 3
        args.adapted_updates = 3
        args.n_envs = 4
        args.rollout_episodes = 4
        args.eval_every = 0
        args.eval_episodes = 20
        args.crossplay_episodes = 30
        args.checkpoint_every = 2
        args.replay_episodes = 1

    cfg = load_env_config("full_short_game").with_(
        n_agents=args.n_agents,
        n_coalition=args.coalition,
        max_rounds=args.max_rounds,
        vote_aggregation="majority",
        # A one-meeting cycle uses the F1-compatible framing objective.  In repeated games the
        # pre-registered game-level survival objective is the meaningful impostor objective.
        coalition_objective=(args.single_round_objective if args.max_rounds == 1 else "survive"),
        reputation_accuracy=False,
        reveal_role_on_eject=False,
    )
    algo_base = load_algo_config("ippo")
    common = dict(
        n_envs=args.n_envs,
        rollout_episodes=args.rollout_episodes,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        eval_episodes=args.eval_episodes,
    )

    out = run_dir(args.name)
    replay_dir = out / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    runmeta_extra = {
        "experiment": "f3_matched_single_multi_response_cycle",
        "training_order": (
            ["coalition0_vs_scripted_soft_credibility"]
            if args.initial_only
            else [
                "coalition0_vs_scripted_soft_credibility",
                "crew1_vs_frozen_coalition0",
                "coalition1_vs_frozen_crew1",
            ]
        ),
        "n_agents": int(cfg.n_agents),
        "n_crew": int(cfg.n_crew),
        "n_coalition": int(cfg.n_coalition),
        "max_rounds": int(cfg.max_rounds),
        "seed": int(args.seed),
        "checkpoint_every": int(args.checkpoint_every),
        "eval_every": int(args.eval_every),
        "eval_episodes": int(args.eval_episodes),
        "crossplay_episodes": int(args.crossplay_episodes),
        "replay_episodes_per_pair": int(args.replay_episodes),
        "single_round_objective": args.single_round_objective,
        "env_config": cfg.to_dict(),
        "algo_config": algo_base,
    }
    write_provenance(out, runmeta_extra)
    save_json(runmeta_extra, out / "experiment_config.json")
    banner(
        f"Matched F3 cycle | n={cfg.n_agents} ({cfg.n_crew}+{cfg.n_coalition}) | "
        f"rounds={cfg.max_rounds} | seed={args.seed}"
    )

    attack_out = out / "coalition0_vs_soft_credibility"
    defense_out = out / "crew1_vs_learned_coalition0"
    adapted_out = out / "coalition1_vs_learned_crew1"

    attack_algo = dict(algo_base, **common, total_updates=args.initial_updates)
    attack_result = Trainer(
        cfg,
        attack_algo,
        TruthfulCrew(rule="soft_credibility"),
        seed=args.seed,
        run_dir=attack_out,
        device=args.device,
    ).train()
    if not attack_result.checkpoint:
        raise RuntimeError("coalition0 training did not produce a checkpoint")

    coalition0 = load_role_checkpoint(attack_result.checkpoint, Role.COALITION, args.device)
    if args.initial_only:
        soft_crew = TruthfulCrew(rule="soft_credibility")
        eval_seed = 1987654321 + args.seed
        summary, episode_rows, round_rows, replay_rows = evaluate_pair(
            cfg,
            soft_crew,
            coalition0,
            "coalition0_vs_soft_credibility",
            args.crossplay_episodes,
            eval_seed,
            replay_dir,
            args.replay_episodes,
            runmeta_extra,
        )
        save_csv([summary], out / "crossplay_summary.csv")
        save_csv(episode_rows, out / "episode_metrics.csv")
        save_csv(round_rows, out / "round_metrics.csv")
        save_csv(replay_rows, out / "replay_manifest.csv")
        result = {
            **runmeta_extra,
            "status": "complete",
            "eval_seed": int(eval_seed),
            "stages": {
                "coalition0": {
                    "checkpoint": attack_result.checkpoint,
                    "final_metrics": attack_result.final_metrics,
                }
            },
            "summaries": [summary],
            "artifact_files": [
                "experiment_config.json",
                "runmeta.json",
                "crossplay_summary.csv",
                "episode_metrics.csv",
                "round_metrics.csv",
                "replay_manifest.csv",
                "coalition0_vs_soft_credibility/history.json",
            ],
        }
        save_json(result, out / "result.json")
        write_provenance(out, {**runmeta_extra, "status": "complete"})
        print(
            f"coalition0_vs_soft_credibility | FE={summary['false_ejection_rate']:.3f} | "
            f"incident={summary.get('creator_incident_rate', float('nan')):.3f}",
            flush=True,
        )
        print(f"saved -> {out}", flush=True)
        return 0

    defense_algo = dict(algo_base, **common, total_updates=args.defender_updates)
    defense_result = Trainer(
        cfg,
        defense_algo,
        crew_policy=None,
        opponent_policy=coalition0,
        learner_role=Role.CREW,
        seed=args.seed + 10000,
        run_dir=defense_out,
        device=args.device,
    ).train()
    if not defense_result.checkpoint:
        raise RuntimeError("crew1 training did not produce a checkpoint")

    crew1 = load_role_checkpoint(defense_result.checkpoint, Role.CREW, args.device)
    adapted_algo = dict(algo_base, **common, total_updates=args.adapted_updates)
    adapted_result = Trainer(
        cfg,
        adapted_algo,
        crew_policy=crew1,
        learner_role=Role.COALITION,
        seed=args.seed + 20000,
        run_dir=adapted_out,
        device=args.device,
    ).train()
    if not adapted_result.checkpoint:
        raise RuntimeError("coalition1 training did not produce a checkpoint")

    coalition1 = load_role_checkpoint(adapted_result.checkpoint, Role.COALITION, args.device)
    soft_crew = TruthfulCrew(rule="soft_credibility")
    pairs = {
        "coalition0_vs_soft_credibility": (soft_crew, coalition0),
        "coalition0_vs_learned_crew1": (crew1, coalition0),
        "coalition1_vs_learned_crew1": (crew1, coalition1),
        "coalition1_vs_soft_credibility": (soft_crew, coalition1),
    }
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    eval_seed = 1987654321 + args.seed
    for pair, (crew, coalition) in pairs.items():
        summary, eps, rounds, replays = evaluate_pair(
            cfg,
            crew,
            coalition,
            pair,
            args.crossplay_episodes,
            eval_seed,
            replay_dir,
            args.replay_episodes,
            runmeta_extra,
        )
        summaries.append(summary)
        episode_rows.extend(eps)
        round_rows.extend(rounds)
        replay_rows.extend(replays)
        print(
            f"{pair:38s} | FE={summary['false_ejection_rate']:.3f} | "
            f"survival={summary['coalition_survival_rate']:.3f} | "
            f"game_win={summary['coalition_game_win_rate'] if summary['coalition_game_win_rate'] is not None else 'NA'} | "
            f"rounds={summary['mean_rounds_played']:.2f}",
            flush=True,
        )

    save_csv(summaries, out / "crossplay_summary.csv")
    save_csv(episode_rows, out / "episode_metrics.csv")
    save_csv(round_rows, out / "round_metrics.csv")
    save_csv(replay_rows, out / "replay_manifest.csv")
    result = {
        **runmeta_extra,
        "status": "complete",
        "eval_seed": int(eval_seed),
        "stages": {
            "coalition0": {
                "checkpoint": attack_result.checkpoint,
                "final_metrics": attack_result.final_metrics,
            },
            "crew1": {
                "checkpoint": defense_result.checkpoint,
                "final_metrics": defense_result.final_metrics,
            },
            "coalition1": {
                "checkpoint": adapted_result.checkpoint,
                "final_metrics": adapted_result.final_metrics,
            },
        },
        "summaries": summaries,
        "artifact_files": [
            "experiment_config.json",
            "runmeta.json",
            "crossplay_summary.csv",
            "episode_metrics.csv",
            "round_metrics.csv",
            "replay_manifest.csv",
            "coalition0_vs_soft_credibility/history.json",
            "crew1_vs_learned_coalition0/history.json",
            "coalition1_vs_learned_crew1/history.json",
        ],
    }
    save_json(result, out / "result.json")
    write_provenance(out, {**runmeta_extra, "status": "complete"})
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
