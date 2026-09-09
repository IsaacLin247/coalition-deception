"""Record an episode into a replay file while it is played."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.action_masks import active_agents, compute_masks
from social_collusion.env.enums import Phase, Role
from social_collusion.env.observations import build_all_actor_obs
from social_collusion.env.rewards import outcome_dict
from social_collusion.env.state import GameState
from social_collusion.env.transition import reset as engine_reset
from social_collusion.env.transition import transition as engine_transition
from social_collusion.replay.schema import ReplayRecord, StepRecord
from social_collusion.seeding import episode_rng


def record_episode(
    cfg: EnvConfig,
    policy,
    seed: int,
    episode_index: int = 0,
    with_observations: bool = False,
    policy_metadata_fn=None,
    runmeta: dict[str, Any] | None = None,
) -> tuple[GameState, ReplayRecord]:
    """Play one episode and build a complete `ReplayRecord` (plan sec.12.2)."""
    # Keep policy sampling separate from environment stochasticity.  The replay stores actions,
    # not the policy RNG stream, so sharing one generator would advance the environment RNG during
    # the original play and make action-only re-simulation fail even when the engine is pure.
    env_rng = episode_rng(seed, episode_index)
    policy_rng = episode_rng(seed ^ 0x51A7E, episode_index)
    state = engine_reset(cfg, env_rng, seed=seed)
    rec = ReplayRecord(
        env_config=cfg.to_dict(),
        seed=int(seed),
        episode_index=int(episode_index),
        roles={f"A{i}": Role(int(r)).name for i, r in enumerate(state.roles)},
        initial_state={
            "positions": state.positions.tolist(),
            "alive": state.alive.tolist(),
            "tasks": state.tasks.astype(int).tolist(),
            "incident": {
                "creator": int(state.incident_creator),
                "victim": int(state.incident_victim),
                "time": int(state.incident_time),
                "room": int(state.incident_room),
            },
            "horizon": int(state.horizon),
            "state_hash": state.state_hash(),
        },
        runmeta=runmeta or {},
    )
    needs_obs = getattr(policy, "needs_obs", True) or with_observations
    while state.phase != int(Phase.TERMINAL):
        masks = compute_masks(state)
        act = active_agents(state)
        obs = build_all_actor_obs(state) if needs_obs else np.zeros((cfg.n_agents, 1), np.float32)
        a = policy.act([state], obs[None], [m[None] for m in masks], act[None], policy_rng)[0]
        step = StepRecord(
            phase=Phase(state.phase).name,
            turn=int(state.turn),
            sub_step=int(state.sub_step),
            joint_actions=np.asarray(a, dtype=int).tolist(),
            active=[bool(x) for x in act],
        )
        if with_observations:
            step.private_observations = {
                f"A{i}": obs[i].tolist() for i in range(cfg.n_agents) if act[i]
            }
        if policy_metadata_fn is not None:
            step.policy_metadata = policy_metadata_fn(state, a)
        res = engine_transition(state, a, env_rng, inplace=True, validate=True)
        state = res.state
        step.events = [e.to_dict() for e in res.events]
        step.state_hash = state.state_hash()
        step.public_observation = _public_view(state)
        rec.steps.append(step)

    rec.outcome = outcome_dict(state)
    rec.final_state_hash = state.state_hash()
    rec.event_log_hash = state.event_log_hash()
    return state, rec


def _public_view(state: GameState) -> dict[str, Any]:
    """Only what a normal observer could know - what the public replay view renders."""
    cfg = state.config
    return {
        "phase": Phase(state.phase).name,
        "alive": state.alive.tolist(),
        "reporter": int(state.reporter),
        "report_turn": int(state.report_turn),
        "marker_room": int(state.incident_room) if state.report_turn >= 0 else -1,
        "incident_time_public": cfg.public_incident_time(
            state.incident_time, state.report_turn >= 0
        ),
        "claims": [c.to_dict() for c in state.claims],
        "responses": [r.to_dict() for r in state.responses],
        "votes": state.votes.tolist(),
        "ejected": int(state.ejected),
    }


def write(record: ReplayRecord, path: str | Path, compress: bool | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress is None:
        compress = path.suffix == ".gz"
    blob = json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
    if compress:
        with gzip.open(path, "wt") as fh:
            fh.write(blob)
    else:
        with open(path, "w") as fh:
            fh.write(blob)
    return path


def write_many(records: list[ReplayRecord], path: str | Path) -> Path:
    """JSON-lines bundle - one replay per line, for shipping a whole evaluation set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as fh:  # type: ignore[operator]
        for r in records:
            fh.write(json.dumps(r.to_dict(), separators=(",", ":"), sort_keys=True) + "\n")
    return path


__all__ = ["record_episode", "write", "write_many"]
