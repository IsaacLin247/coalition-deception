"""Actor observations and the privileged critic state (plan sec.8).

Two hard rules enforced here and tested in `tests/test_observation_privacy.py`:

1. An actor observation is a function of **public information plus that agent's own private
   history only**. Ground-truth roles, the incident tuple, other agents' private observations and
   other agents' symbols never enter it. `tests/test_observation_privacy.py` mutates hidden state
   and asserts the observation vector is bit-identical.
2. Every block is a **named slice** (`OBS_SPEC.slices`), so feature-ablation analyses
   (plan sec.23.4) zero a block by name rather than by a magic offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import ClaimType, Phase, ResponseType, Role
from social_collusion.env.knowledge import ObserverKnowledge, observed_horizon
from social_collusion.env.state import DEAD, GameState


@dataclass
class ObservationSpec:
    """Layout of the flat actor observation."""

    cfg: EnvConfig
    slices: dict[str, slice] = field(default_factory=dict)
    size: int = 0

    @staticmethod
    def build(cfg: EnvConfig) -> ObservationSpec:
        n, R, T1, K = cfg.n_agents, cfg.n_rooms, cfg.n_times, cfg.k_symbols
        blocks = [
            ("phase", len(Phase)),
            ("self_id", n),
            ("self_role", 2),
            ("partner_mask", n),
            ("self_room", R + 1),
            ("alive", n),
            ("last_seen", n * (R + 1)),
            ("since_seen", n),
            ("marker", 3),
            ("tasks", 2 * R),
            ("public_report", (n + 1) + 1 + (R + 1) + (T1 + 1)),
            ("transcript", n * (n + 7 + (n + 1) + (R + 1) + (T1 + 1) + 3)),
            ("responses", n * (n + 6 + (n + 1) + 1)),
            ("votes", n * (n + 2)),
            ("symbol", 2 * (K + 1)),
            ("order", 2 * (n + 1)),
        ]
        if cfg.actor_private_history:
            # Per time: observed flag, own-room one-hot, co-located agents.
            # Append rather than reorder to retain every legacy named slice.
            blocks.append(("private_history", T1 * (1 + R + n)))
        spec = ObservationSpec(cfg=cfg)
        off = 0
        for name, size in blocks:
            spec.slices[name] = slice(off, off + size)
            off += size
        spec.size = off
        return spec

    def block(self, obs: np.ndarray, name: str) -> np.ndarray:
        return obs[..., self.slices[name]]

    def ablate(self, obs: np.ndarray, names: list[str]) -> np.ndarray:
        """Return a copy with the named blocks zeroed (feature-ablation analysis)."""
        out = np.array(obs, copy=True)
        for nm in names:
            out[..., self.slices[nm]] = 0.0
        return out


_SPEC_CACHE: dict[EnvConfig, ObservationSpec] = {}


def get_spec(cfg: EnvConfig) -> ObservationSpec:
    # EnvConfig is a frozen dataclass of hashable fields, so it keys the cache directly.
    # (Hashing it is ~100x cheaper than serialising it, and this runs once per observation.)
    spec = _SPEC_CACHE.get(cfg)
    if spec is None:
        spec = ObservationSpec.build(cfg)
        _SPEC_CACHE[cfg] = spec
    return spec


def obs_dim(cfg: EnvConfig) -> int:
    return get_spec(cfg).size


# --------------------------------------------------------------------------------------
# private memory derived from own co-location only
# --------------------------------------------------------------------------------------
def last_seen(state: GameState, agent: int) -> tuple[np.ndarray, np.ndarray]:
    """(room, time) of the last moment `agent` could actually see each other agent.

    Room is -1 and time is -1 when never seen. Legacy actors retain only this
    summary; actor_private_history additionally preserves all positional facts.
    """
    cfg = state.config
    rooms = np.full(cfg.n_agents, -1, dtype=np.int64)
    times = np.full(cfg.n_agents, -1, dtype=np.int64)
    upto = observed_horizon(state)
    for t in range(upto + 1):
        me = int(state.positions[t, agent])
        if me == DEAD:
            continue
        for j in range(cfg.n_agents):
            if j == agent:
                rooms[j], times[j] = me, t
                continue
            if int(state.positions[t, j]) == me:
                rooms[j], times[j] = me, t
    return rooms, times


# --------------------------------------------------------------------------------------
# actor observation
# --------------------------------------------------------------------------------------
def build_actor_obs(state: GameState, agent: int, out: np.ndarray | None = None) -> np.ndarray:
    cfg = state.config
    spec = get_spec(cfg)
    n, R, T1, K = cfg.n_agents, cfg.n_rooms, cfg.n_times, cfg.k_symbols
    o = np.zeros(spec.size, dtype=np.float32) if out is None else out
    o[:] = 0.0
    S = spec.slices

    o[S["phase"]][int(state.phase)] = 1.0
    o[S["self_id"]][agent] = 1.0
    o[S["self_role"]][int(state.roles[agent])] = 1.0

    if cfg.partner_known and state.roles[agent] == int(Role.COALITION):
        for j in state.coalition:
            if int(j) != agent:
                o[S["partner_mask"]][int(j)] = 1.0

    room_now = int(state.positions[min(state.turn, T1 - 1), agent])
    o[S["self_room"]][R if room_now == DEAD else room_now] = 1.0
    o[S["alive"]][:] = state.alive.astype(np.float32)

    rooms, times = last_seen(state, agent)
    ls = o[S["last_seen"]].reshape(n, R + 1)
    for j in range(n):
        ls[j, R if rooms[j] < 0 else rooms[j]] = 1.0
    ss = o[S["since_seen"]]
    for j in range(n):
        ss[j] = 1.0 if times[j] < 0 else float(state.turn - times[j]) / max(1, T1)

    mk = o[S["marker"]]
    mk[0] = float(state.marker_seen[agent])
    mk[1] = (
        float(state.marker_seen_time[agent]) / max(1, T1) if state.marker_seen[agent] else 0.0
    )
    mk[2] = float(state.marker_seen[agent] and room_now == state.incident_room)

    tk = o[S["tasks"]].reshape(2, R)
    tk[0] = state.tasks[agent].astype(np.float32)
    tk[1] = state.tasks_done[agent].astype(np.float32)

    pr = o[S["public_report"]]
    pr[n if state.reporter < 0 else state.reporter] = 1.0
    pr[n + 1] = float(max(state.report_turn, 0)) / max(1, T1)
    reported = state.report_turn >= 0
    marker_room_public = state.incident_room if reported else -1
    pr[n + 2 + (R if marker_room_public < 0 else marker_room_public)] = 1.0
    t_pub = cfg.public_incident_time(state.incident_time, reported)
    pr[n + 2 + (R + 1) + (T1 if t_pub < 0 else t_pub)] = 1.0

    slot = n + 7 + (n + 1) + (R + 1) + (T1 + 1) + 3
    tr = o[S["transcript"]].reshape(n, slot)
    for c in state.claims:
        k = int(c.order)
        if k >= n:  # pragma: no cover - order is bounded by the speaking order length
            continue
        row, off = tr[k], 0
        row[off + int(c.speaker)] = 1.0
        off += n
        row[off + int(c.claim_type)] = 1.0
        off += 7
        row[off + (n if c.subject is None else int(c.subject))] = 1.0
        off += n + 1
        row[off + (R if c.room is None else int(c.room))] = 1.0
        off += R + 1
        row[off + (T1 if c.time is None else int(c.time))] = 1.0
        off += T1 + 1
        row[off] = float(bool(c.polarity))
        row[off + 1] = float(c.confidence) / max(1, cfg.confidence_levels - 1)
        row[off + 2] = 1.0  # this slot has been filled

    rslot = n + 6 + (n + 1) + 1
    rr = o[S["responses"]].reshape(n, rslot)
    for r in state.responses:
        k = int(r.order)
        if k >= n:  # pragma: no cover
            continue
        row, off = rr[k], 0
        row[off + int(r.speaker)] = 1.0
        off += n
        row[off + int(r.response_type)] = 1.0
        off += 6
        row[off + (n if r.target is None else int(r.target))] = 1.0
        off += n + 1
        row[off] = 1.0

    vv = o[S["votes"]].reshape(n, n + 2)
    for j in range(n):
        v = int(state.votes[j])
        if v >= 0:
            vv[j, v] = 1.0
            vv[j, n + 1] = 1.0

    sy = o[S["symbol"]]
    own, recv = int(state.symbols[agent]), int(state.symbols_received[agent])
    sy[K if own < 0 else own] = 1.0
    sy[(K + 1) + (K if recv < 0 else recv)] = 1.0

    orr = o[S["order"]]
    pos_in_order = np.flatnonzero(state.speaking_order == agent)
    orr[n if len(pos_in_order) == 0 else int(pos_in_order[0])] = 1.0
    orr[(n + 1) + min(int(state.sub_step), n)] = 1.0

    if cfg.actor_private_history:
        history = o[S["private_history"]].reshape(T1, 1 + R + n)
        knowledge = ObserverKnowledge.build(state, agent)
        for t in range(observed_horizon(state) + 1):
            room = int(knowledge.present[t])
            if room == DEAD:
                continue
            history[t, 0] = 1.0
            history[t, 1 + room] = 1.0
            history[t, 1 + R:] = (knowledge.known[t] >= 0).astype(np.float32)
    return o


def build_all_actor_obs(state: GameState) -> np.ndarray:
    cfg = state.config
    out = np.zeros((cfg.n_agents, obs_dim(cfg)), dtype=np.float32)
    for i in range(cfg.n_agents):
        build_actor_obs(state, i, out[i])
    return out


# --------------------------------------------------------------------------------------
# privileged critic state (plan sec.8.2) - never fed to an actor
# --------------------------------------------------------------------------------------
def critic_dim(cfg: EnvConfig) -> int:
    n, R, T1, K = cfg.n_agents, cfg.n_rooms, cfg.n_times, cfg.k_symbols
    return (
        len(Phase)
        + 2 * n  # true roles (one-hot per agent)
        + n * T1 * (R + 1)  # true positions
        + n  # alive
        + (n + 1) + (R + 1) + (T1 + 1)  # incident creator / room / time
        + n  # marker seen
        + n * (K + 1)  # all private symbols
        + n * 8  # per-agent claim summary
        + n  # votes cast flag
    )


def build_critic_state(state: GameState) -> np.ndarray:
    cfg = state.config
    n, R, T1, K = cfg.n_agents, cfg.n_rooms, cfg.n_times, cfg.k_symbols
    o = np.zeros(critic_dim(cfg), dtype=np.float32)
    off = 0
    o[off + int(state.phase)] = 1.0
    off += len(Phase)
    for i in range(n):
        o[off + 2 * i + int(state.roles[i])] = 1.0
    off += 2 * n
    for i in range(n):
        for t in range(T1):
            p = int(state.positions[t, i])
            o[off + (i * T1 + t) * (R + 1) + (R if p == DEAD else p)] = 1.0
    off += n * T1 * (R + 1)
    o[off : off + n] = state.alive.astype(np.float32)
    off += n
    o[off + (n if state.incident_creator < 0 else state.incident_creator)] = 1.0
    off += n + 1
    o[off + (R if state.incident_room < 0 else state.incident_room)] = 1.0
    off += R + 1
    o[off + (T1 if state.incident_time < 0 else state.incident_time)] = 1.0
    off += T1 + 1
    o[off : off + n] = state.marker_seen.astype(np.float32)
    off += n
    for i in range(n):
        s = int(state.symbols[i])
        o[off + i * (K + 1) + (K if s < 0 else s)] = 1.0
    off += n * (K + 1)
    for c in state.claims:
        base = off + int(c.speaker) * 8
        o[base + 0] = 1.0
        o[base + 1] = float(c.claim_type) / 6.0
        o[base + 2] = float(bool(c.content_true))
        o[base + 3] = float(bool(c.supported))
        o[base + 4] = float(c.claim_type == int(ClaimType.NO_INFORMATION))
        o[base + 5] = float(c.subject is not None)
        o[base + 6] = float(c.confidence) / max(1, cfg.confidence_levels - 1)
        o[base + 7] = float(bool(c.polarity))
    off += n * 8
    o[off : off + n] = (state.votes >= 0).astype(np.float32)
    return o


def scramble_hidden_state(
    state: GameState, observer: int, rng: np.random.Generator, scramble_roles: bool = True
) -> GameState:
    """Randomise everything `observer` must not be able to see, and nothing else.

    Used by the privacy test: `build_actor_obs(state, i)` must be bit-identical before and after.
    Deliberately *not* scrambled, because they are legitimately knowable: the observer's own
    role and trajectory, the alive mask (a missing agent is public), the reported marker room,
    the disclosed incident time, and the public transcript itself.

    `scramble_roles=False` is for a coalition observer under `partner_known`: being told your
    partner's identity in a 2-member coalition determines everyone else's role by elimination,
    so roles are not hidden state for that observer and scrambling them would test the wrong
    thing.
    """
    st = state.copy()
    cfg = st.config
    others = [j for j in range(cfg.n_agents) if j != observer]
    if scramble_roles:
        for j in others:
            st.roles[j] = int(rng.integers(0, 2))
    # who actually did it, and the private symbols of everyone else
    st.incident_creator = int(rng.choice(others))
    for j in others:
        st.symbols[j] = int(rng.integers(-1, max(1, cfg.k_symbols)))
        st.symbols_received[j] = int(rng.integers(-1, max(1, cfg.k_symbols)))
        st.marker_seen[j] = bool(rng.integers(0, 2))
        st.marker_seen_time[j] = int(rng.integers(-1, cfg.n_times))
    # ground-truth annotations attached to public claims
    for c in st.claims:
        c.label = int(rng.integers(0, 5))
        c.content_true = bool(rng.integers(0, 2))
        c.supported = bool(rng.integers(0, 2))
    return st


def response_summary(state: GameState) -> dict[str, int]:
    """Small helper used by metrics/renderers."""
    out = {rt.name: 0 for rt in ResponseType}
    for r in state.responses:
        out[ResponseType(r.response_type).name] += 1
    return out


__all__ = [
    "ObservationSpec",
    "get_spec",
    "obs_dim",
    "critic_dim",
    "build_actor_obs",
    "build_all_actor_obs",
    "build_critic_state",
    "last_seen",
    "response_summary",
    "scramble_hidden_state",
]
