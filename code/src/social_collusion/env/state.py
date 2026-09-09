"""Game state, claims, responses, events, and the deterministic state hash.

The state is deliberately *small and array-shaped*: the entire ground truth of an episode is
`positions` (a `(n_times, n_agents)` room-index matrix) plus the incident tuple. Every
observation, every claim truth label, and every metric is a pure function of that matrix, which
is what makes the meeting-only and spatial variants share one code path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import (
    N_HEADS,
    ClaimType,
    EventType,
    Phase,
    ResponseType,
    Role,
    TruthLabel,
)

DEAD = -1  # position sentinel


@dataclass
class Claim:
    """A structured public statement (plan sec.7.5). Fields not used by the claim type are None."""

    speaker: int
    claim_type: int
    subject: int | None = None
    room: int | None = None
    time: int | None = None
    polarity: bool = True
    confidence: int = 1
    label: int | None = None  # TruthLabel, filled by the engine at the moment it is made
    order: int = 0  # position in the speaking order
    # decomposition of the label, kept because deception metrics need them separately
    content_true: bool | None = None
    supported: bool | None = None

    def is_informative(self) -> bool:
        return self.claim_type != ClaimType.NO_INFORMATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": int(self.speaker),
            "claim_type": ClaimType(self.claim_type).name,
            "subject": None if self.subject is None else int(self.subject),
            "room": None if self.room is None else int(self.room),
            "time": None if self.time is None else int(self.time),
            "polarity": bool(self.polarity),
            "confidence": int(self.confidence),
            "label": None if self.label is None else TruthLabel(self.label).name,
            "order": int(self.order),
            "content_true": self.content_true,
            "supported": self.supported,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Claim:
        return Claim(
            speaker=d["speaker"],
            claim_type=int(ClaimType[d["claim_type"]]),
            subject=d["subject"],
            room=d["room"],
            time=d["time"],
            polarity=d["polarity"],
            confidence=d["confidence"],
            label=None if d["label"] is None else int(TruthLabel[d["label"]]),
            order=d.get("order", 0),
            content_true=d.get("content_true"),
            supported=d.get("supported"),
        )


@dataclass
class Response:
    speaker: int
    response_type: int
    target: int | None = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": int(self.speaker),
            "response_type": ResponseType(self.response_type).name,
            "target": None if self.target is None else int(self.target),
            "order": int(self.order),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Response:
        return Response(
            speaker=d["speaker"],
            response_type=int(ResponseType[d["response_type"]]),
            target=d["target"],
            order=d.get("order", 0),
        )


@dataclass
class Event:
    """Immutable typed event. Appended, never edited (plan sec.12.1)."""

    type: str
    turn: int
    phase: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "turn": int(self.turn),
            "phase": Phase(self.phase).name,
            "payload": _jsonable(self.payload),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


@dataclass
class GameState:
    """Complete ground truth of one episode.

    `positions[t, i]` is the room index of agent i at timestep t, or DEAD (-1) once i is
    inactive. Nothing else in the engine may hold positional truth: observation building,
    claim labelling and every metric read this one array.
    """

    config: EnvConfig
    seed: int

    phase: int = int(Phase.RESET)
    turn: int = 0  # free-play timestep t
    sub_step: int = 0  # index within a sequential round

    roles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    alive: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    tasks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    tasks_done: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))

    horizon: int = 0  # last valid timestep index (free-play may end early on a report)

    incident_creator: int = -1
    incident_victim: int = -1
    incident_time: int = -1
    incident_room: int = -1
    marker_found_by: int = -1
    marker_found_time: int = -1
    marker_seen: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    marker_seen_time: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    reporter: int = -1
    report_turn: int = -1
    invalid_action_attempts: int = 0

    speaking_order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    claims: list[Claim] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    votes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    symbols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    symbols_received: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    ejected: int = -1
    round: int = 0
    reputation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    crew_framed_count: int = 0  # innocents ejected in incident-bearing rounds (legacy reward counter)
    incident_round_count: int = 0  # rounds that had a real incident (denominator for kill-decoupled rates)
    creator_ejected_count: int = 0  # rounds where the incident creator was correctly ejected
    #: per-round outcome log (multi-round only): (round, had_incident, crew_framed, creator_ejected).
    #: lets us measure framing rate AS A FUNCTION OF round index (the compounding trajectory).
    round_log: list = field(default_factory=list)
    #: Every completed meeting, including single-round and incident-free meetings. Unlike
    #: round_log's legacy crew_framed field, false_ejection is not gated on an incident.
    meeting_log: list[dict[str, Any]] = field(default_factory=list)
    #: per-completed-round (agreement, opportunity) pair matrices for the dependence-aware
    #: mechanism's sliding window (env/dependence.py). Only filled when a config asks for a
    #: window longer than the current meeting; never part of the canonical state hash.
    dependence_log: list = field(default_factory=list)
    #: Which round's record was most recently appended; prevents counting the current
    #: meeting both as current and as history in post-vote/terminal diagnostics.
    dependence_logged_round: int = -1
    done: bool = False
    events: list[Event] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    # -- convenience ---------------------------------------------------------
    @property
    def n(self) -> int:
        return self.config.n_agents

    @property
    def coalition(self) -> np.ndarray:
        return np.flatnonzero(self.roles == int(Role.COALITION))

    @property
    def crew(self) -> np.ndarray:
        return np.flatnonzero(self.roles == int(Role.CREW))

    def partner_of(self, i: int) -> int:
        """The other coalition member (assumes n_coalition == 2), else -1."""
        if self.roles[i] != int(Role.COALITION):
            return -1
        others = [int(j) for j in self.coalition if j != i]
        return others[0] if len(others) == 1 else -1

    def room_at(self, i: int, t: int) -> int:
        return int(self.positions[t, i])

    def colocated(self, i: int, j: int, t: int) -> bool:
        """Both alive at t and in the same room -> i could observe j."""
        a, b = int(self.positions[t, i]), int(self.positions[t, j])
        return a != DEAD and a == b

    def add_event(self, etype: str, **payload: Any) -> None:
        self.events.append(Event(etype, self.turn, self.phase, payload))

    # -- copying / hashing ---------------------------------------------------
    def copy(self) -> GameState:
        """Deep enough copy that `transition` can be a pure function."""
        return replace(
            self,
            roles=self.roles.copy(),
            alive=self.alive.copy(),
            positions=self.positions.copy(),
            tasks=self.tasks.copy(),
            tasks_done=self.tasks_done.copy(),
            marker_seen=self.marker_seen.copy(),
            marker_seen_time=self.marker_seen_time.copy(),
            speaking_order=self.speaking_order.copy(),
            claims=[replace(c) for c in self.claims],
            responses=[replace(r) for r in self.responses],
            votes=self.votes.copy(),
            symbols=self.symbols.copy(),
            symbols_received=self.symbols_received.copy(),
            reputation=self.reputation.copy(),
            events=list(self.events),
            round_log=list(self.round_log),
            meeting_log=[dict(record) for record in self.meeting_log],
            dependence_log=[(a.copy(), o.copy()) for a, o in self.dependence_log],
            info=dict(self.info),
        )

    def canonical(self) -> dict[str, Any]:
        """Everything that defines the state, in a stable order (used for the hash)."""
        return {
            "phase": int(self.phase),
            "turn": int(self.turn),
            "sub_step": int(self.sub_step),
            "roles": self.roles.tolist(),
            "alive": self.alive.tolist(),
            "positions": self.positions.tolist(),
            "tasks": self.tasks.astype(np.int8).tolist(),
            "tasks_done": self.tasks_done.astype(np.int8).tolist(),
            "incident": [
                int(self.incident_creator),
                int(self.incident_victim),
                int(self.incident_time),
                int(self.incident_room),
            ],
            "horizon": int(self.horizon),
            "marker": [
                int(self.marker_found_by),
                int(self.marker_found_time),
                self.marker_seen.astype(np.int8).tolist(),
                self.marker_seen_time.tolist(),
            ],
            "report": [int(self.reporter), int(self.report_turn)],
            "speaking_order": self.speaking_order.tolist(),
            "claims": [c.to_dict() for c in self.claims],
            "responses": [r.to_dict() for r in self.responses],
            "votes": self.votes.tolist(),
            "symbols": self.symbols.tolist(),
            "symbols_received": self.symbols_received.tolist(),
            "ejected": int(self.ejected),
            "done": bool(self.done),
        }

    def state_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def event_log_hash(self) -> str:
        blob = json.dumps([e.to_dict() for e in self.events], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class TransitionResult:
    """What one `transition()` call produced."""

    state: GameState
    events: list[Event]
    rewards: np.ndarray
    terminated: bool
    info: dict[str, Any] = field(default_factory=dict)


def empty_joint_action(cfg: EnvConfig) -> np.ndarray:
    """All-no-op joint action: shape (n_agents, N_HEADS), every head at index 0."""
    return np.zeros((cfg.n_agents, N_HEADS), dtype=np.int64)


__all__ = [
    "DEAD",
    "Claim",
    "Response",
    "Event",
    "GameState",
    "TransitionResult",
    "empty_joint_action",
    "EventType",
]
