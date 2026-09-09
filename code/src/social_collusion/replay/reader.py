"""Load replays and re-simulate them from actions alone."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import Phase
from social_collusion.env.state import GameState
from social_collusion.env.transition import reset as engine_reset
from social_collusion.env.transition import transition as engine_transition
from social_collusion.replay.schema import ReplayRecord
from social_collusion.seeding import episode_rng


def _open(path: Path, mode: str = "rt"):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def read(path: str | Path) -> ReplayRecord:
    path = Path(path)
    with _open(path) as fh:
        return ReplayRecord.from_dict(json.load(fh))


def read_many(path: str | Path) -> list[ReplayRecord]:
    path = Path(path)
    with _open(path) as fh:
        return [ReplayRecord.from_dict(json.loads(line)) for line in fh if line.strip()]


def config_of(record: ReplayRecord) -> EnvConfig:
    raw = dict(record.env_config)
    raw["rooms"] = tuple(raw["rooms"])
    raw["edges"] = tuple(tuple(e) for e in raw["edges"])
    raw.pop("adjacency", None)
    return EnvConfig(**raw)


def resimulate(record: ReplayRecord) -> tuple[GameState, list[str]]:
    """Replay from (config, seed, ordered actions) alone and return the per-step state hashes.

    This is the determinism guarantee of plan sec.12.3: nothing but the recorded actions is
    fed back in, so matching hashes prove the engine is a pure function of them.
    """
    cfg = config_of(record)
    rng = episode_rng(record.seed, record.episode_index)
    state = engine_reset(cfg, rng, seed=record.seed)
    hashes: list[str] = []
    for step in record.steps:
        a = np.asarray(step.joint_actions, dtype=np.int64)
        res = engine_transition(state, a, rng, inplace=True, validate=True)
        state = res.state
        hashes.append(state.state_hash())
    return state, hashes


def rebuild_state(record: ReplayRecord) -> GameState:
    """The terminal GameState of a recorded episode (for metrics on saved replays)."""
    state, _ = resimulate(record)
    if state.phase != int(Phase.TERMINAL):  # pragma: no cover - truncated replay
        raise ValueError("replay did not reach TERMINAL")
    return state


__all__ = ["read", "read_many", "resimulate", "rebuild_state", "config_of"]
