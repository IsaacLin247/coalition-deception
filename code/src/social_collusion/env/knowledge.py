"""Observer-relative knowledge: what one agent can actually verify.

This is the bridge between ground truth and what a listener may legitimately reason with. It is
used by the scripted crew, by the deception metrics ("was this lie catchable?") and by the
counterfactual machinery. It never exposes anything the observer could not know.

An observer's knowledge has a very specific shape, which is why the checks below are exact and
cheap (no general SAT call needed):

* positive facts - `LOC(j, t) = r` for every agent j the observer was co-located with at t;
* negative facts - `LOC(j, t) != my_room_at_t` for every agent j it did *not* see while present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from social_collusion.env.claims import claim_constraints
from social_collusion.env.state import DEAD, Claim, GameState

UNKNOWN = -1


def observed_horizon(state: GameState) -> int:
    """Last elapsed position index the observer can consult in the current round."""
    return min(
        state.turn,
        state.horizon if state.horizon > 0 else state.turn,
        state.config.n_times - 1,
    )


@dataclass
class ObserverKnowledge:
    """Everything agent `observer` can verify from its own trajectory."""

    state: GameState
    observer: int
    known: np.ndarray  # (n_times, n_agents) room index or UNKNOWN
    present: np.ndarray  # (n_times,) observer's own room or DEAD

    @staticmethod
    def build(state: GameState, observer: int) -> ObserverKnowledge:
        cfg = state.config
        T1, n = cfg.n_times, cfg.n_agents
        known = np.full((T1, n), UNKNOWN, dtype=np.int64)
        present = np.full(T1, DEAD, dtype=np.int64)
        upto = observed_horizon(state)
        for t in range(upto + 1):
            me = int(state.positions[t, observer])
            present[t] = me
            if me == DEAD:
                continue
            known[t, observer] = me
            for j in range(n):
                if j != observer and int(state.positions[t, j]) == me:
                    known[t, j] = me
        return ObserverKnowledge(state, observer, known, present)

    # -- queries ---------------------------------------------------------
    def knows_location(self, j: int, t: int) -> bool:
        return self.known[t, j] != UNKNOWN

    def knows_absent(self, j: int, t: int, room: int) -> bool:
        """Certain that j was NOT in `room` at t."""
        if self.known[t, j] != UNKNOWN:
            return int(self.known[t, j]) != room
        # observer was in `room` and did not see j there
        return int(self.present[t]) == room

    def caught_false(self, claim: Claim) -> bool:
        """Does this claim contradict something the observer directly knows?

        Exact for the observer's fact shape; no SAT call, so it is cheap enough for the
        training loop.
        """
        for c in claim_constraints(claim):
            kind = c[0]
            if kind == "eq_const":
                _, a, t, r = c
                if not (0 <= t < self.known.shape[0]):
                    return True
                if self.known[t, a] != UNKNOWN and int(self.known[t, a]) != r:
                    return True
                if int(self.present[t]) == r and self.known[t, a] == UNKNOWN:
                    return True  # observer was there and did not see a
            elif kind == "neq_const":
                _, a, t, r = c
                if 0 <= t < self.known.shape[0] and int(self.known[t, a]) == r:
                    return True
            elif kind == "eq_pair":
                _, a, b, t = c
                ra, rb = int(self.known[t, a]), int(self.known[t, b])
                if ra != UNKNOWN and rb != UNKNOWN and ra != rb:
                    return True
                if ra != UNKNOWN and int(self.present[t]) == ra and rb == UNKNOWN:
                    return True
                if rb != UNKNOWN and int(self.present[t]) == rb and ra == UNKNOWN:
                    return True
            elif kind == "neq_pair":
                _, a, b, t = c
                ra, rb = int(self.known[t, a]), int(self.known[t, b])
                if ra != UNKNOWN and rb != UNKNOWN and ra == rb:
                    return True
            elif kind == "marker":
                if self.state.marker_seen[self.observer] and int(c[1]) != self.state.incident_room:
                    return True
        return False

    def corroborates(self, claim: Claim) -> bool:
        """Does the observer directly confirm every positional constraint of the claim?"""
        cons = claim_constraints(claim)
        if not cons:
            return False
        confirmed = False
        for c in cons:
            if c[0] == "eq_const":
                _, a, t, r = c
                if not (0 <= t < self.known.shape[0]) or self.known[t, a] == UNKNOWN:
                    return False
                if int(self.known[t, a]) != r:
                    return False
                confirmed = True
            elif c[0] == "eq_pair":
                _, a, b, t = c
                ra, rb = int(self.known[t, a]), int(self.known[t, b])
                if ra == UNKNOWN or rb == UNKNOWN or ra != rb:
                    return False
                confirmed = True
        return confirmed


def scene_window(state: GameState) -> range:
    """The publicly known times at which the incident could have happened.

    With `incident_time_public` (the default) the report is forensic and pins the instant, so the
    window is the single timestep `incident_time` and an alibi at that instant is decisive.
    Without it the window is the whole elapsed episode, nobody can be cleared, and honest
    deduction collapses - that ablation is kept because it is a result, not dead code.
    """
    if state.config.incident_time_public and state.incident_time >= 0 and state.report_turn >= 0:
        return range(int(state.incident_time), int(state.incident_time) + 1)
    hi = state.report_turn if state.report_turn >= 0 else state.horizon
    return range(1, max(1, hi) + 1)


def own_scene_exposure(state: GameState, observer: int) -> np.ndarray:
    """How often the observer personally saw each agent in the marker room during the window."""
    cfg = state.config
    k = ObserverKnowledge.build(state, observer)
    out = np.zeros(cfg.n_agents, dtype=np.float32)
    room = state.incident_room
    if room < 0:
        return out
    for t in scene_window(state):
        if t >= cfg.n_times:
            break
        for j in range(cfg.n_agents):
            if k.known[t, j] != UNKNOWN and int(k.known[t, j]) == room:
                out[j] += 1.0
    return out


__all__ = ["ObserverKnowledge", "scene_window", "own_scene_exposure", "UNKNOWN"]
