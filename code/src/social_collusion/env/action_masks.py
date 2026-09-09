"""Action masks (plan sec.9.3).

Design invariant that buys a lot of correctness: **there are no cross-head constraints.**
Every head's mask is computed independently, and any combination of legal per-head choices is a
well-formed action. Achieved by:

* dropping the "NONE" sentinel from heads whose field is always required for the active agent
  (claim subject / room / time, response target, incident target);
* masking claim times to `1..horizon` so ENTER/LEAVE claims always have a `t-1`;
* exposing CREATE_INCIDENT only when a selectable incident target exists. Known-partner
  agents select feasible crew targets; hidden-partner agents select role-blind attempts
  that may resolve as legal no-ops under the hidden state.

An agent that is not the active speaker in a sequential round is masked to index 0 on every head
(zero entropy, zero log-prob contribution).
"""

from __future__ import annotations

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import N_HEADS, Head, Phase, Role, RoomAction
from social_collusion.env.state import DEAD, GameState


def head_sizes(cfg: EnvConfig) -> list[int]:
    sizes = [0] * N_HEADS
    sizes[Head.MOVE] = cfg.n_rooms
    sizes[Head.ROOM_ACTION] = len(RoomAction)
    sizes[Head.INCIDENT_TARGET] = cfg.n_agents
    sizes[Head.SYMBOL] = cfg.k_symbols
    sizes[Head.CLAIM_TYPE] = 7
    sizes[Head.CLAIM_SUBJECT] = cfg.n_agents
    sizes[Head.CLAIM_ROOM] = cfg.n_rooms
    sizes[Head.CLAIM_TIME] = cfg.n_times
    sizes[Head.CLAIM_POLARITY] = 2
    sizes[Head.CLAIM_CONF] = cfg.confidence_levels
    sizes[Head.RESPONSE_TYPE] = 6
    sizes[Head.RESPONSE_TARGET] = cfg.n_agents
    sizes[Head.VOTE] = cfg.n_agents + 1  # last index = SKIP
    return sizes


def active_agents(state: GameState) -> np.ndarray:
    """Which agents' actions are actually consumed this step (bool, shape (n_agents,))."""
    cfg = state.config
    out = np.zeros(cfg.n_agents, dtype=bool)
    phase = Phase(state.phase)
    if phase in (Phase.FREE_PLAY, Phase.VOTE):
        out[:] = state.alive
    elif phase == Phase.SYMBOL:
        out[:] = state.alive & (state.roles == int(Role.COALITION))
    elif phase in (Phase.CLAIM_ROUND, Phase.RESPONSE_ROUND):
        if state.sub_step < len(state.speaking_order):
            out[int(state.speaking_order[state.sub_step])] = True
    return out


def valid_incident_targets(state: GameState, agent: int, t: int) -> np.ndarray:
    """Crew this coalition member may target at time `t` (same room, isolation satisfied)."""
    cfg = state.config
    out = np.zeros(cfg.n_agents, dtype=bool)
    if state.roles[agent] != int(Role.COALITION) or state.incident_time >= 0:
        return out
    if t < 1:
        return out  # t=0 has no preceding turn, so ENTER/LEAVE evidence could not exist
    room = int(state.positions[t, agent])
    if room == DEAD:
        return out
    here = [
        j
        for j in range(cfg.n_agents)
        if j != agent and state.alive[j] and int(state.positions[t, j]) == room
    ]
    crew_here = [j for j in here if state.roles[j] == int(Role.CREW)]
    if cfg.incident_requires_isolation and len(crew_here) > 1:
        return out  # a second crew member would witness it
    for j in crew_here:
        out[j] = True
    return out


def incident_attempt_targets(state: GameState, agent: int, t: int) -> np.ndarray:
    """Targets visible to the actor's action mask, without an unknown-role oracle.

    Known-partner mode retains the original fully feasible mask. When identity is
    hidden, every other living agent in the same room is selectable, independent
    of roles, hidden isolation conditions, or an incident elsewhere. The engine
    uses valid_incident_targets to decide whether an attempt succeeds; a failed
    attempt is legal and gives no extra private diagnostic or invalid-action flag.
    """
    if state.config.partner_known:
        return valid_incident_targets(state, agent, t)
    out = np.zeros(state.config.n_agents, dtype=bool)
    if state.roles[agent] != int(Role.COALITION) or not state.alive[agent] or t < 1:
        return out
    room = int(state.positions[t, agent])
    if room == DEAD:
        return out
    out[:] = state.alive & (state.positions[t] == room)
    out[agent] = False
    return out


def compute_masks(state: GameState) -> list[np.ndarray]:
    """Per-head boolean masks, `masks[h]` of shape (n_agents, head_size[h])."""
    cfg = state.config
    sizes = head_sizes(cfg)
    masks = [np.zeros((cfg.n_agents, s), dtype=bool) for s in sizes]
    for m in masks:
        m[:, 0] = True  # index 0 always legal -> never an empty mask
    phase = Phase(state.phase)
    act = active_agents(state)

    if phase == Phase.FREE_PLAY:
        t = state.turn
        for i in range(cfg.n_agents):
            if not act[i]:
                continue
            room = int(state.positions[t, i])
            masks[Head.MOVE][i, :] = False
            masks[Head.MOVE][i, room] = True
            for r in cfg.neighbors(room):
                masks[Head.MOVE][i, r] = True

            ra = masks[Head.ROOM_ACTION][i]
            ra[:] = False
            ra[RoomAction.WAIT] = True
            ra[RoomAction.OBSERVE] = True
            if state.tasks[i, room] and not state.tasks_done[i, room]:
                ra[RoomAction.PERFORM_TASK] = True
            if (
                state.incident_time >= 0
                and state.incident_room == room
                and t > state.incident_time
                and state.reporter < 0
            ):
                ra[RoomAction.REPORT] = True
            tgt = incident_attempt_targets(state, i, t)
            if tgt.any():
                ra[RoomAction.CREATE_INCIDENT] = True
                masks[Head.INCIDENT_TARGET][i, :] = tgt

    elif phase == Phase.SYMBOL:
        for i in range(cfg.n_agents):
            if act[i]:
                masks[Head.SYMBOL][i, :] = True

    elif phase == Phase.CLAIM_ROUND:
        for i in np.flatnonzero(act):
            masks[Head.CLAIM_TYPE][i, :] = True
            subj = masks[Head.CLAIM_SUBJECT][i]
            subj[:] = False
            for j in range(cfg.n_agents):
                if j != i:
                    subj[j] = True
            masks[Head.CLAIM_ROOM][i, :] = True
            tm = masks[Head.CLAIM_TIME][i]
            tm[:] = False
            tm[1 : state.horizon + 1] = True
            if not tm.any():  # degenerate horizon 0 config
                tm[0] = True
            masks[Head.CLAIM_POLARITY][i, :] = True
            masks[Head.CLAIM_CONF][i, :] = True

    elif phase == Phase.RESPONSE_ROUND:
        for i in np.flatnonzero(act):
            masks[Head.RESPONSE_TYPE][i, :] = True
            tgt = masks[Head.RESPONSE_TARGET][i]
            tgt[:] = False
            for j in range(cfg.n_agents):
                if j != i and state.alive[j]:
                    tgt[j] = True
            if not tgt.any():  # pragma: no cover - >=2 agents always alive at a meeting
                tgt[0] = True

    elif phase == Phase.VOTE:
        for i in np.flatnonzero(act):
            v = masks[Head.VOTE][i]
            v[:] = False
            for j in range(cfg.n_agents):
                if state.alive[j] and (cfg.allow_self_vote or j != i):
                    v[j] = True
            if cfg.allow_skip_vote:
                v[cfg.n_agents] = True
            if not v.any():  # pragma: no cover
                v[cfg.n_agents] = True

    return masks


def is_legal(state: GameState, joint_action: np.ndarray) -> tuple[bool, str]:
    """Validate a joint action against the masks. Used in strict (development) mode."""
    masks = compute_masks(state)
    act = active_agents(state)
    for i in np.flatnonzero(act):
        for h in range(N_HEADS):
            a = int(joint_action[i, h])
            if a < 0 or a >= masks[h].shape[1]:
                return False, f"agent {i} head {Head(h).name} index {a} out of range"
            if not masks[h][i, a]:
                return False, f"agent {i} head {Head(h).name} index {a} is masked out"
    return True, ""


def sample_masked(masks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Uniform sample over legal actions - the random-policy primitive."""
    n = masks[0].shape[0]
    out = np.zeros((n, N_HEADS), dtype=np.int64)
    for h, m in enumerate(masks):
        for i in range(n):
            legal = np.flatnonzero(m[i])
            out[i, h] = int(rng.choice(legal)) if len(legal) else 0
    return out


__all__ = [
    "head_sizes",
    "active_agents",
    "compute_masks",
    "valid_incident_targets",
    "incident_attempt_targets",
    "is_legal",
    "sample_masked",
]
