"""Replay validation: schema completeness, determinism, and observation-privacy auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from social_collusion.replay.reader import config_of, read, resimulate
from social_collusion.replay.schema import REQUIRED_TOP_LEVEL, SCHEMA_VERSION, ReplayRecord


@dataclass
class ValidationResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:  # pragma: no cover - display only
        head = "VALID" if self.ok else "INVALID"
        body = "".join(f"\n  ERROR: {e}" for e in self.errors)
        body += "".join(f"\n  warn:  {w}" for w in self.warnings)
        return head + body


def validate_record(record: ReplayRecord, check_determinism: bool = True) -> ValidationResult:
    res = ValidationResult()
    d = record.to_dict()
    for key in REQUIRED_TOP_LEVEL:
        if key not in d:
            res.fail(f"missing top-level field {key!r}")
    if record.schema_version != SCHEMA_VERSION:
        res.warn(f"schema version {record.schema_version} != {SCHEMA_VERSION}")
    if not record.steps:
        res.fail("replay has no steps")
        return res

    try:
        cfg = config_of(record)
    except Exception as exc:  # pragma: no cover - malformed config
        res.fail(f"env_config does not rebuild: {exc}")
        return res

    for k, step in enumerate(record.steps):
        arr = step.joint_actions
        if len(arr) != cfg.n_agents:
            res.fail(f"step {k}: joint_actions has {len(arr)} rows, expected {cfg.n_agents}")
        if len(step.active) != cfg.n_agents:
            res.fail(f"step {k}: active mask has {len(step.active)} entries")

    if check_determinism:
        try:
            state, hashes = resimulate(record)
        except Exception as exc:
            res.fail(f"resimulation raised {type(exc).__name__}: {exc}")
            return res
        if state.state_hash() != record.final_state_hash:
            res.fail("final state hash mismatch: replay is not deterministic")
        for k, (h, step) in enumerate(zip(hashes, record.steps)):
            if step.state_hash and h != step.state_hash:
                res.fail(f"state hash mismatch at step {k}")
                break
        if state.event_log_hash() != record.event_log_hash:
            res.warn("event log hash mismatch (events changed but state did not)")
    return res


def validate_file(path: str | Path, check_determinism: bool = True) -> ValidationResult:
    return validate_record(read(path), check_determinism=check_determinism)


def assert_valid(record: ReplayRecord) -> None:
    r = validate_record(record)
    if not r.ok:  # pragma: no cover - raised only on a real defect
        raise AssertionError(str(r))


__all__ = ["ValidationResult", "validate_record", "validate_file", "assert_valid"]
