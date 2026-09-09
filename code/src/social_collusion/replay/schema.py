"""Replay record schema (plan sec.12.2).

A replay is self-describing: config + seed + ordered joint actions are enough to reproduce the
episode exactly, and the per-step state hashes let a reader *prove* it did (plan sec.12.3).
The renderer in `visualization/` consumes only this file - never a checkpoint, never the
training code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class StepRecord:
    phase: str
    turn: int
    sub_step: int
    joint_actions: list[list[int]]
    active: list[bool]
    events: list[dict[str, Any]] = field(default_factory=list)
    state_hash: str = ""
    public_observation: dict[str, Any] = field(default_factory=dict)
    private_observations: dict[str, Any] = field(default_factory=dict)
    #: logits / action probabilities - analysis runs only, too expensive during training
    policy_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayRecord:
    schema_version: str = SCHEMA_VERSION
    env_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    episode_index: int = 0
    roles: dict[str, str] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    runmeta: dict[str, Any] = field(default_factory=dict)
    final_state_hash: str = ""
    event_log_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ReplayRecord:
        steps = [StepRecord(**s) for s in d.get("steps", [])]
        return ReplayRecord(**{**d, "steps": steps})


REQUIRED_TOP_LEVEL = (
    "schema_version",
    "env_config",
    "seed",
    "roles",
    "steps",
    "outcome",
    "final_state_hash",
)

__all__ = ["SCHEMA_VERSION", "StepRecord", "ReplayRecord", "REQUIRED_TOP_LEVEL"]
