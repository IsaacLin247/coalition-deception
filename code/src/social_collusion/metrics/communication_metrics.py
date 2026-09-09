"""Emergent-communication metrics (plan sec.16.5, sec.23).

The separation this module enforces is the one Eccles et al. [R10] insist on:

* **positive signalling** - does the symbol carry information about the sender's private state?
  (`symbol_state_mi`)
* **positive listening** - does the receiver's behaviour actually depend on it?
  (`receiver_action_mi`, and the interventional `positive_listening_score` in
  `counterfactuals.py`, because a decodable symbol may still be unused)

A protocol that scores high on the first and zero on the second is *not* communication, and the
report says so.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from social_collusion.env.enums import Role
from social_collusion.env.state import DEAD, GameState


# --------------------------------------------------------------------------------------
# discrete information theory (plug-in with an optional Miller-Madow correction)
# --------------------------------------------------------------------------------------
def entropy(x: Sequence[int], correct: bool = True) -> float:
    x = np.asarray(list(x))
    if x.size == 0:
        return float("nan")
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    if correct:
        h += (len(p) - 1) / (2 * x.size * np.log(2))
    return max(h, 0.0)


def mutual_information(x: Sequence[int], y: Sequence[int], correct: bool = True) -> float:
    x = np.asarray(list(x))
    y = np.asarray(list(y))
    if x.size == 0 or x.size != y.size:
        return float("nan")
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xs.size, ys.size))
    np.add.at(joint, (xi, yi), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = float((joint[mask] * np.log2(joint[mask] / (px @ py)[mask])).sum())
    if correct:
        df = (xs.size - 1) * (ys.size - 1)
        mi -= df / (2 * x.size * np.log(2))
    return max(mi, 0.0)


def normalized_mi(x: Sequence[int], y: Sequence[int]) -> float:
    hx, hy = entropy(x), entropy(y)
    denom = min(hx, hy)
    return float(mutual_information(x, y) / denom) if denom > 1e-9 else 0.0


# --------------------------------------------------------------------------------------
# per-episode extraction
# --------------------------------------------------------------------------------------
def _sender_receiver(state: GameState) -> tuple[int, int] | None:
    """(sender, receiver) = (partner, creator). The partner is the one holding private position
    information the creator needs; that is the direction the channel is useful in."""
    coal = [int(i) for i in state.coalition]
    if len(coal) != 2 or state.incident_creator < 0:
        return None
    creator = int(state.incident_creator)
    partner = [i for i in coal if i != creator]
    return (partner[0], creator) if partner else None


def channel_records(states: Sequence[GameState]) -> dict[str, list[int]]:
    """Aligned per-episode variables for the information-theoretic quantities."""
    out: dict[str, list[int]] = {
        "symbol": [],
        "sender_room": [],
        "incident_room": [],
        "frame_target": [],
        "sender_next_claim_room": [],
        "receiver_claim_room": [],
        "receiver_vote": [],
        "sender_vote": [],
        "receiver_claim_type": [],
    }
    for s in states:
        sr = _sender_receiver(s)
        if sr is None:
            continue
        sender, receiver = sr
        sym = int(s.symbols[sender])
        if sym < 0:
            continue
        t_star = int(s.incident_time) if s.config.incident_time_public else max(1, s.horizon - 1)
        t_star = min(max(t_star, 0), s.config.n_times - 1)
        room = int(s.positions[t_star, sender])
        out["symbol"].append(sym)
        out["sender_room"].append(room if room != DEAD else s.config.n_rooms)
        out["incident_room"].append(int(s.incident_room))
        crew_alive = [j for j in range(s.config.n_agents) if s.roles[j] == int(Role.CREW) and s.alive[j]]
        out["frame_target"].append(crew_alive[0] if crew_alive else -1)
        sc = [c for c in s.claims if c.speaker == sender]
        rc = [c for c in s.claims if c.speaker == receiver]
        out["sender_next_claim_room"].append(
            int(sc[0].room) if sc and sc[0].room is not None else s.config.n_rooms
        )
        out["receiver_claim_room"].append(
            int(rc[0].room) if rc and rc[0].room is not None else s.config.n_rooms
        )
        out["receiver_claim_type"].append(int(rc[0].claim_type) if rc else -1)
        out["receiver_vote"].append(int(s.votes[receiver]))
        out["sender_vote"].append(int(s.votes[sender]))
    return out


def summarize(states: Sequence[GameState]) -> dict[str, float]:
    """The plan sec.16.5 block."""
    rec = channel_records(states)
    if not rec["symbol"]:
        return {"channel_utilization": 0.0}
    sym = rec["symbol"]
    k = states[0].config.k_symbols
    used = len(set(sym))
    out = {
        "symbol_entropy": entropy(sym),
        "symbol_entropy_max": float(np.log2(max(k, 1))),
        "channel_utilization": used / max(k, 1),
        # positive signalling: does the symbol encode the sender's private state?
        "symbol_state_mi_sender_room": mutual_information(sym, rec["sender_room"]),
        "symbol_state_mi_incident_room": mutual_information(sym, rec["incident_room"]),
        "symbol_state_mi_frame_target": mutual_information(sym, rec["frame_target"]),
        # does it predict the sender's own next action? (a private note-to-self, not communication)
        "symbol_next_action_mi": mutual_information(sym, rec["sender_next_claim_room"]),
        # positive listening (observational): does the receiver's behaviour co-vary with it?
        "receiver_action_mi_claim_room": mutual_information(sym, rec["receiver_claim_room"]),
        "receiver_action_mi_claim_type": mutual_information(sym, rec["receiver_claim_type"]),
        "receiver_action_mi_vote": mutual_information(sym, rec["receiver_vote"]),
        "message_redundancy": normalized_mi(rec["sender_vote"], rec["receiver_vote"]),
        "n_episodes_with_symbol": float(len(sym)),
    }
    out["protocol_alignment"] = (
        out["symbol_state_mi_sender_room"] / max(out["symbol_entropy"], 1e-9)
    )
    return out


def symbol_semantics(states: Sequence[GameState]) -> dict[int, dict[str, float]]:
    """Per-symbol conditional distributions (plan sec.23.1) - the interpretability table."""
    rec = channel_records(states)
    out: dict[int, dict[str, float]] = {}
    syms = sorted(set(rec["symbol"]))
    for sy in syms:
        idx = [i for i, v in enumerate(rec["symbol"]) if v == sy]
        rooms = Counter(rec["sender_room"][i] for i in idx)
        total = max(1, len(idx))
        entry = {"count": float(len(idx)), "share": len(idx) / max(1, len(rec["symbol"]))}
        for r, c in rooms.items():
            entry[f"p_sender_room_{r}"] = c / total
        entry["modal_sender_room"] = float(rooms.most_common(1)[0][0]) if rooms else float("nan")
        entry["modal_sender_room_purity"] = (
            rooms.most_common(1)[0][1] / total if rooms else float("nan")
        )
        out[int(sy)] = entry
    return out


def cross_play_matrix(results: dict[tuple[int, int], float]) -> np.ndarray:
    """Reshape {(coalition_seed, crew_seed): metric} into a matrix for the dashboard."""
    if not results:
        return np.zeros((0, 0))
    rows = sorted({a for a, _ in results})
    cols = sorted({b for _, b in results})
    m = np.full((len(rows), len(cols)), np.nan)
    for (a, b), v in results.items():
        m[rows.index(a), cols.index(b)] = v
    return m


def protocol_consistency(per_seed_semantics: Sequence[dict[int, dict[str, float]]]) -> float:
    """Do independently trained seeds converge on the same symbol->room mapping?

    Reported because emergent protocols routinely overfit their training partner [R15]; a low
    value means the protocol is a private convention, not a discovered code.
    """
    if len(per_seed_semantics) < 2:
        return float("nan")
    maps = []
    for sem in per_seed_semantics:
        maps.append({s: v.get("modal_sender_room", np.nan) for s, v in sem.items()})
    keys = set.intersection(*[set(m) for m in maps]) if maps else set()
    if not keys:
        return 0.0
    agree, total = 0, 0
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            for k in keys:
                total += 1
                agree += int(maps[i][k] == maps[j][k])
    return float(agree / total) if total else float("nan")


__all__ = [
    "entropy",
    "mutual_information",
    "normalized_mi",
    "channel_records",
    "summarize",
    "symbol_semantics",
    "cross_play_matrix",
    "protocol_consistency",
]
