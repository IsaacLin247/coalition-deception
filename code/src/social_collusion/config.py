"""Environment / algorithm configuration.

`EnvConfig` is frozen and hashable so it can be embedded in a replay record and compared
exactly when validating determinism.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class EnvConfig:
    """Frozen environment specification (research_contract.md sec.3)."""

    name: str = "meeting_only"
    n_agents: int = 5
    n_coalition: int = 2
    rooms: tuple[str, ...] = ("Hub", "Electrical", "Navigation", "Medbay")
    edges: tuple[tuple[str, str], ...] = (
        ("Hub", "Electrical"),
        ("Hub", "Navigation"),
        ("Hub", "Medbay"),
    )
    free_play_turns: int = 6
    synthetic_evidence: bool = True

    #: Number of play->meeting->vote rounds before the game ends. 1 (default) is the original
    #: single-meeting game and keeps every existing result bit-identical. With max_rounds > 1 the
    #: game loops: after a vote, if neither side has won (coalition parity / all coalition ejected),
    #: a fresh incident round begins with the survivors, and a persistent reputation vector carries
    #: across meetings (the setting where the adaptive-capture dynamics live).
    max_rounds: int = 1

    #: Give a central judge the crew's pooled private grounded evidence (in addition to testimony).
    #: Default False = the blind judge. True = the "sighted central" control that de-confounds
    #: centralization from blindness in the sighted direction (sighted-central vs sighted-distributed).
    judge_sighted: bool = False

    #: Cross-round reputation update for judge_adaptive. False (default) = AGREEMENT-based (reward who
    #: corroborates -- the gameable naive reputation). True = ACCURACY-based: reward being RIGHT
    #: (accused a revealed impostor) and punish backing a revealed impostor. The constructive defense.
    reputation_accuracy: bool = False

    enable_response_round: bool = True
    enable_symbol_channel: bool = False
    n_symbols: int = 4
    symbol_mode: str = "none"  # none | random | learned | learned_unobserved | public
    partner_known: bool = True
    shared_reward: bool = True

    #: Append the observer's complete within-round positional evidence (including
    #: observed absences) to its actor input. False preserves archived observation
    #: dimensions; new spatial runs explicitly enable this in full_short_game.yaml.
    actor_private_history: bool = False

    confidence_levels: int = 3
    allow_self_vote: bool = False
    allow_skip_vote: bool = True
    tie_ejects: bool = False
    reveal_role_on_eject: bool = False
    random_speaking_order: bool = True

    #: Who decides the ejection (the "dictator" — see env/dictator.py). "majority" is the original
    #: plurality vote (default, bit-compatible). The "judge_*" modes hand the decision to a single
    #: judge that rules from the public discussion: judge_uniform / judge_median are naive;
    #: judge_reputation_soft / judge_reputation_sharp weight speakers by corroboration and are the
    #: sophisticated rules a coordinating coalition can capture.
    vote_aggregation: str = "majority"

    #: Dependence-aware credibility (env/dependence.py). `dependence_penalty` is the strength
    #: lambda in penalty_i = exp(-lambda * excess_i); `dependence_window` is how many meetings
    #: (current round included) the pairwise agreement matrix pools over -- 1 means the current
    #: meeting only, which is all a single-meeting game has; `dependence_top_j` is how many
    #: closest partners define an agent's "small group". The defaults leave every existing
    #: rule untouched: they are only read by the `dependence_aware` crew rule, the
    #: `judge_dependence_aware` judge, and `vote_aggregation="dependence_weighted"`.
    dependence_penalty: float = 4.0
    dependence_window: int = 1
    dependence_top_j: int = 1

    #: What the (learned) coalition is rewarded for. "survive" (default) = the pre-registered
    #: coalition_favorable outcome (no coalition member ejected) — but in a single meeting that is
    #: winnable by inducing a SKIP, so a learner captures no rule in particular. "false_ejection" =
    #: the coalition is rewarded only when an innocent crew member is actually ejected, which removes
    #: the skip escape hatch and is faithful to the impostor's real goal (eliminate crew). Use it to
    #: test whether a *learned* coalition reproduces the scripted capture ranking across vote rules.
    #: "balanced_ejection" is a single-round zero-sum diagnostic: innocent ejected = +1 for the
    #: coalition, skip = 0, coalition member ejected = -1. "meeting_ejection" is the same
    #: incident-conditioned outcome for a meeting study, but no-incident episodes are neutral
    #: rather than a full-round attack failure. "risk_balanced_ejection" applies the role-relative
    #: +1 / -0.5 / -1 scores, making a skip mildly bad for both sides.
    coalition_objective: str = "survive"

    #: Does the report announce *when* the incident happened?
    #: With an unknown incident time every agent has to be cleared at every timestep, which makes
    #: honest deduction nearly impossible and lets a coordinated voting bloc win without lying -
    #: measured in tests/test_statistical_controls.py::test_incident_time_public_is_load_bearing.
    #: Publishing the time makes alibis at that instant decisive, which is what gives partner
    #: corroboration a sharp causal effect. Ablatable: set False to reproduce the degenerate case.
    incident_time_public: bool = True

    # spatial variant only
    n_tasks_per_agent: int = 2
    incident_requires_isolation: bool = True
    auto_report_at_end: bool = True
    max_evidence_retries: int = 64

    # derived, filled in __post_init__
    adjacency: tuple[tuple[int, ...], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.n_coalition >= self.n_agents:
            raise ValueError("n_coalition must be < n_agents")
        if self.n_agents - self.n_coalition < 2:
            raise ValueError("need at least 2 crew (one victim + one voter)")
        if self.symbol_mode not in {"none", "random", "learned", "learned_unobserved", "public"}:
            raise ValueError(f"unknown symbol_mode {self.symbol_mode!r}")
        if self.enable_symbol_channel and self.symbol_mode == "none":
            raise ValueError("enable_symbol_channel=True requires a symbol_mode other than 'none'")
        if self.vote_aggregation not in (
            "majority",
            "judge_uniform",
            "judge_median",
            "judge_reputation_soft",
            "judge_reputation_sharp",
            "judge_adaptive",
            "judge_dependence_aware",
            "dependence_weighted",
        ):
            raise ValueError(f"unknown vote_aggregation {self.vote_aggregation!r}")
        if self.dependence_penalty < 0:
            raise ValueError("dependence_penalty must be >= 0")
        if self.dependence_window < 1 or self.dependence_top_j < 1:
            raise ValueError("dependence_window and dependence_top_j must be >= 1")
        if self.coalition_objective not in (
            "survive",
            "false_ejection",
            "balanced_ejection",
            "meeting_ejection",
            "risk_balanced_ejection",
        ):
            raise ValueError(f"unknown coalition_objective {self.coalition_objective!r}")
        if self.synthetic_evidence and self.max_rounds > 1:
            # `_reset_round` regenerates spatial positions only; it does NOT re-sample a synthetic
            # incident, so rounds 2+ silently fall back to spatial free-play (kill-driven, and
            # untrained for a meeting_only policy). Multi-round is only valid in the spatial game.
            raise ValueError(
                "synthetic_evidence=True is not supported with max_rounds>1 (rounds 2+ have no "
                "synthetic incident regeneration); use a spatial env like full_short_game"
            )
        idx = {r: i for i, r in enumerate(self.rooms)}
        adj: list[set[int]] = [set() for _ in self.rooms]
        for a, b in self.edges:
            if a not in idx or b not in idx:
                raise ValueError(f"edge {(a, b)} references an unknown room")
            adj[idx[a]].add(idx[b])
            adj[idx[b]].add(idx[a])
        object.__setattr__(self, "adjacency", tuple(tuple(sorted(s)) for s in adj))

    # -- derived sizes -------------------------------------------------------
    @property
    def n_rooms(self) -> int:
        return len(self.rooms)

    @property
    def n_crew(self) -> int:
        return self.n_agents - self.n_coalition

    @property
    def n_times(self) -> int:
        """Number of position timesteps: t = 0..free_play_turns."""
        return self.free_play_turns + 1

    @property
    def none_agent(self) -> int:
        """Sentinel index used by *_subject / *_target / vote heads for 'no target' / SKIP."""
        return self.n_agents

    @property
    def none_room(self) -> int:
        return self.n_rooms

    @property
    def none_time(self) -> int:
        return self.n_times

    def public_incident_time(self, incident_time: int, reported: bool) -> int:
        """What the meeting is told about *when* the incident happened (-1 = not disclosed)."""
        return int(incident_time) if (self.incident_time_public and reported) else -1

    @property
    def k_symbols(self) -> int:
        return self.n_symbols if self.enable_symbol_channel else 1

    def neighbors(self, room: int) -> tuple[int, ...]:
        return self.adjacency[room]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("adjacency", None)
        d["rooms"] = list(self.rooms)
        d["edges"] = [list(e) for e in self.edges]
        return d

    def fingerprint(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def with_(self, **kwargs: Any) -> EnvConfig:
        """Return a copy with fields overridden (keeps the frozen dataclass usable)."""
        kwargs.pop("adjacency", None)
        return replace(self, **kwargs)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "PyYAML is required to load configs (pip install pyyaml), or build EnvConfig directly."
        ) from exc
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_env_config(name_or_path: str | Path, **overrides: Any) -> EnvConfig:
    """Load `configs/env/<name>.yaml` (or an explicit path) into an EnvConfig."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_ROOT / "env" / f"{name_or_path}.yaml"
    raw = _read_yaml(path)
    raw["rooms"] = tuple(raw.get("rooms", ()))
    raw["edges"] = tuple(tuple(e) for e in raw.get("edges", ()))
    raw.update(overrides)
    known = {f for f in EnvConfig.__dataclass_fields__ if f != "adjacency"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown env config keys: {sorted(unknown)}")
    return EnvConfig(**raw)


def load_algo_config(name_or_path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Algorithm configs stay plain dicts: they are swept over far more freely than env configs."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_ROOT / "algo" / f"{name_or_path}.yaml"
    cfg = _read_yaml(path)
    cfg.update(overrides)
    return cfg


def load_experiment_config(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_ROOT / "experiments" / f"{name_or_path}.yaml"
    return _read_yaml(path)
