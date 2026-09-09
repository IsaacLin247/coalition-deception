"""Synthetic evidence generation for the meeting-only benchmark (plan Stage 2).

The meeting-only variant does *not* use a different world model: it samples the same object the
spatial variant produces - a `(n_times, n_agents)` position matrix plus one incident - directly
from the episode RNG. Everything downstream (observations, claim truth labels, metrics) is then
literally the same code, which is what makes "meeting-only vs. spatial" a controlled comparison
rather than two unrelated experiments.
"""

from __future__ import annotations

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import Role
from social_collusion.env.state import DEAD

STAY_PROB = 0.35


def assign_roles(cfg: EnvConfig, rng: np.random.Generator) -> np.ndarray:
    """Uniform random role assignment; slots are permuted every episode (plan sec.8.4)."""
    roles = np.zeros(cfg.n_agents, dtype=np.int8)
    coalition = rng.choice(cfg.n_agents, size=cfg.n_coalition, replace=False)
    roles[coalition] = int(Role.COALITION)
    return roles


def assign_tasks(cfg: EnvConfig, rng: np.random.Generator) -> np.ndarray:
    """`tasks[i, r]` - does agent i have a task in room r? (crew truth, coalition cover)"""
    tasks = np.zeros((cfg.n_agents, cfg.n_rooms), dtype=bool)
    k = min(cfg.n_tasks_per_agent, cfg.n_rooms)
    for i in range(cfg.n_agents):
        for r in rng.choice(cfg.n_rooms, size=k, replace=False):
            tasks[i, r] = True
    return tasks


def random_walk(cfg: EnvConfig, rng: np.random.Generator) -> np.ndarray:
    """Free random walk on the room graph: (n_times, n_agents) room indices."""
    pos = np.zeros((cfg.n_times, cfg.n_agents), dtype=np.int8)
    pos[0] = rng.integers(0, cfg.n_rooms, size=cfg.n_agents)
    for t in range(1, cfg.n_times):
        for i in range(cfg.n_agents):
            here = int(pos[t - 1, i])
            nb = cfg.neighbors(here)
            if not nb or rng.random() < STAY_PROB:
                pos[t, i] = here
            else:
                pos[t, i] = nb[int(rng.integers(0, len(nb)))]
    return pos


def _incident_candidates(
    cfg: EnvConfig, pos: np.ndarray, roles: np.ndarray
) -> list[tuple[int, int, int]]:
    """(t, creator, victim) triples where an incident would be valid.

    Valid = creator (coalition) and victim (crew) share a room at t, and no *other* crew member
    is present to witness it. A coalition partner in the room is allowed - that is precisely the
    situation in which a coordinated alibi becomes available.
    """
    out: list[tuple[int, int, int]] = []
    # leave at least one turn before (so ENTER/LEAVE evidence exists) and one after (so the
    # marker can be discovered by someone other than the creator)
    lo, hi = 1, cfg.n_times - 2
    for t in range(lo, hi + 1):
        for c in np.flatnonzero(roles == int(Role.COALITION)):
            room = int(pos[t, c])
            crew_here = [
                int(j)
                for j in range(cfg.n_agents)
                if roles[j] == int(Role.CREW) and int(pos[t, j]) == room
            ]
            if cfg.incident_requires_isolation and len(crew_here) != 1:
                continue
            for v in crew_here:
                out.append((int(t), int(c), int(v)))
    return out


def generate(cfg: EnvConfig, rng: np.random.Generator) -> dict:
    """Produce a complete synthetic episode ground truth.

    Returns a dict with roles, tasks, positions, the incident tuple, marker sightings, the
    reporter, and the horizon. Rejection-samples the walk until a valid incident exists
    (typically first try; the retry bound keeps determinism obvious).
    """
    roles = assign_roles(cfg, rng)
    tasks = assign_tasks(cfg, rng)

    pos = random_walk(cfg, rng)
    cands = _incident_candidates(cfg, pos, roles)
    tries = 0
    while not cands and tries < cfg.max_evidence_retries:
        pos = random_walk(cfg, rng)
        cands = _incident_candidates(cfg, pos, roles)
        tries += 1
    if not cands:
        # Deterministic fallback: pull one crew member into a coalition member's room at t=1.
        creator = int(np.flatnonzero(roles == int(Role.COALITION))[0])
        victim = int(np.flatnonzero(roles == int(Role.CREW))[0])
        t_inc = 1
        room = int(pos[t_inc, creator])
        pos[t_inc, victim] = room
        pos[max(0, t_inc - 1), victim] = room
        for j in range(cfg.n_agents):
            if roles[j] == int(Role.CREW) and j != victim and int(pos[t_inc, j]) == room:
                nb = cfg.neighbors(room)
                pos[t_inc, j] = nb[0] if nb else room
        cands = [(t_inc, creator, victim)]

    t_inc, creator, victim = cands[int(rng.integers(0, len(cands)))]
    room_inc = int(pos[t_inc, creator])

    # the victim's last recorded position is where the incident happened; then it is inactive
    pos[t_inc, victim] = room_inc
    pos[t_inc + 1 :, victim] = DEAD

    alive = np.ones(cfg.n_agents, dtype=bool)
    alive[victim] = False

    marker_seen = np.zeros(cfg.n_agents, dtype=bool)
    marker_seen_time = np.full(cfg.n_agents, -1, dtype=np.int8)
    marker_seen[creator] = True  # the creator knows, trivially
    marker_seen_time[creator] = t_inc
    for t in range(t_inc + 1, cfg.n_times):
        for i in range(cfg.n_agents):
            if alive[i] and int(pos[t, i]) == room_inc and not marker_seen[i]:
                marker_seen[i] = True
                marker_seen_time[i] = t

    # the first agent to walk in on the marker calls the meeting; crew break ties before
    # coalition, then lowest index. If nobody finds it, the meeting is called automatically.
    reporter, report_turn = -1, cfg.free_play_turns
    finders = [
        (int(marker_seen_time[i]), int(roles[i]), i)
        for i in range(cfg.n_agents)
        if marker_seen[i] and i != creator and marker_seen_time[i] >= 0
    ]
    if finders:
        finders.sort()
        report_turn, _, reporter = finders[0]

    return {
        "roles": roles,
        "tasks": tasks,
        "positions": pos,
        "alive": alive,
        "incident_creator": int(creator),
        "incident_victim": int(victim),
        "incident_time": int(t_inc),
        "incident_room": int(room_inc),
        "marker_seen": marker_seen,
        "marker_seen_time": marker_seen_time,
        "reporter": int(reporter),
        "report_turn": int(report_turn),
        "horizon": int(cfg.free_play_turns),
    }


def speaking_order(cfg: EnvConfig, alive: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    living = np.flatnonzero(alive).astype(np.int8)
    if cfg.random_speaking_order:
        rng.shuffle(living)
    return living


__all__ = [
    "assign_roles",
    "assign_tasks",
    "random_walk",
    "generate",
    "speaking_order",
    "STAY_PROB",
]
