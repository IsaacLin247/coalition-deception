"""Turn a replay into the flat JSON the HTML viewer consumes (plan sec.14).

The viewer must never import the environment, so everything it needs - including the public /
privileged split and the counterfactual panel - is baked in here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_collusion.env.claims import to_text
from social_collusion.env.enums import ResponseType, Role, TruthLabel
from social_collusion.env.state import DEAD, GameState
from social_collusion.replay.reader import rebuild_state
from social_collusion.replay.schema import ReplayRecord


def episode_to_view(
    state: GameState, counterfactual: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = state.config
    names = [f"A{i}" for i in range(cfg.n_agents)]
    turns = []
    for t in range(state.horizon + 1):
        rooms = []
        for r, room in enumerate(cfg.rooms):
            rooms.append(
                {
                    "room": room,
                    "agents": [i for i in range(cfg.n_agents) if int(state.positions[t, i]) == r],
                }
            )
        turns.append({"t": t, "rooms": rooms})

    claims = []
    for c in state.claims:
        claims.append(
            {
                **c.to_dict(),
                "text": to_text(c, cfg, names),
                "label": None if c.label is None else TruthLabel(c.label).name,
            }
        )
    return {
        "config": {
            "name": cfg.name,
            "rooms": list(cfg.rooms),
            "edges": [list(e) for e in cfg.edges],
            "n_agents": cfg.n_agents,
            "n_times": cfg.n_times,
            "symbol_channel": cfg.enable_symbol_channel,
        },
        "seed": int(state.seed),
        "names": names,
        "public": {
            "alive": state.alive.tolist(),
            "reporter": int(state.reporter),
            "report_turn": int(state.report_turn),
            "marker_room": int(state.incident_room) if state.report_turn >= 0 else -1,
            "incident_time_public": cfg.public_incident_time(
                state.incident_time, state.report_turn >= 0
            ),
            "claims": claims,
            "responses": [
                {**r.to_dict(), "type": ResponseType(r.response_type).name}
                for r in state.responses
            ],
            "votes": state.votes.tolist(),
            "skip_index": cfg.n_agents,
            "ejected": int(state.ejected),
        },
        "privileged": {
            "roles": [Role(int(r)).name for r in state.roles],
            "coalition": [int(i) for i in state.coalition],
            "incident": {
                "creator": int(state.incident_creator),
                "victim": int(state.incident_victim),
                "time": int(state.incident_time),
                "room": int(state.incident_room),
            },
            "positions": state.positions.tolist(),
            "symbols": state.symbols.tolist(),
            "symbols_received": state.symbols_received.tolist(),
            "marker_seen": state.marker_seen.tolist(),
            "dead_marker": DEAD,
        },
        "turns": turns,
        "outcome": _outcome(state),
        "counterfactual": counterfactual,
    }


def _outcome(state: GameState) -> dict[str, Any]:
    from social_collusion.env.rewards import outcome_dict

    return outcome_dict(state)


def record_to_view(record: ReplayRecord, **kw) -> dict[str, Any]:
    return episode_to_view(rebuild_state(record), **kw)


def write_view(view: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(view, fh, indent=1)
    return path


__all__ = ["episode_to_view", "record_to_view", "write_view"]
