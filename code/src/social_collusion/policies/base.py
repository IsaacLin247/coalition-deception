"""Policy base classes and role-routing composition.

Every policy - scripted, random or neural - implements the same batched `act()` so they are
freely interchangeable. `RoleRouter` is what makes the whole experiment matrix expressible:
"crew = scripted contradiction voter, coalition = learned PPO actor" is one object.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from social_collusion.env.enums import N_HEADS, Role
from social_collusion.env.state import GameState


class BasePolicy:
    name = "base"
    #: scripted policies read the GameState directly, so the runner can skip building the
    #: (comparatively expensive) flat observation vectors for them
    needs_obs = False

    def act(
        self,
        states: Sequence[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """(B, n_agents, N_HEADS) integer actions. Only `active` entries are consumed."""
        B, n = active.shape
        out = np.zeros((B, n, N_HEADS), dtype=np.int64)
        for b, st in enumerate(states):
            for i in np.flatnonzero(active[b]):
                out[b, i] = self.act_single(st, int(i), [m[b, i] for m in masks], rng)
        return out

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        """Per-agent action; `masks[h]` is a 1-D boolean array for head h."""
        raise NotImplementedError

    def reset(self, batch_size: int) -> None:  # recurrent policies override
        pass


def first_legal(mask: np.ndarray, preferred: int | None = None) -> int:
    """Preferred index if legal, else the lowest legal index (never returns an illegal action)."""
    if preferred is not None and 0 <= preferred < len(mask) and mask[preferred]:
        return int(preferred)
    legal = np.flatnonzero(mask)
    return int(legal[0]) if len(legal) else 0


def sample_legal(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal)) if len(legal) else 0


class RoleRouter(BasePolicy):
    """Dispatch each agent to the crew policy or the coalition policy by its true role."""

    name = "role_router"

    def __init__(self, crew: BasePolicy, coalition: BasePolicy):
        self.crew = crew
        self.coalition = coalition
        self.needs_obs = crew.needs_obs or coalition.needs_obs

    def act(
        self,
        states: Sequence[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        crew_active = np.zeros_like(active)
        coal_active = np.zeros_like(active)
        for b, st in enumerate(states):
            is_coal = st.roles == int(Role.COALITION)
            crew_active[b] = active[b] & ~is_coal
            coal_active[b] = active[b] & is_coal
        a = np.zeros((*active.shape, N_HEADS), dtype=np.int64)
        if crew_active.any():
            a = np.where(crew_active[..., None], self.crew.act(states, obs, masks, crew_active, rng), a)
        if coal_active.any():
            a = np.where(
                coal_active[..., None], self.coalition.act(states, obs, masks, coal_active, rng), a
            )
        return a

    def reset(self, batch_size: int) -> None:
        self.crew.reset(batch_size)
        self.coalition.reset(batch_size)


class AgentRouter(BasePolicy):
    """Dispatch by explicit agent index - used for partner-replacement counterfactuals."""

    name = "agent_router"

    def __init__(self, default: BasePolicy, overrides: dict[int, BasePolicy] | None = None):
        self.default = default
        self.overrides = overrides or {}
        self.needs_obs = default.needs_obs or any(p.needs_obs for p in self.overrides.values())

    def act(
        self,
        states: Sequence[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        a = self.default.act(states, obs, masks, active, rng)
        for idx, pol in self.overrides.items():
            sel = np.zeros_like(active)
            sel[:, idx] = active[:, idx]
            if sel.any():
                a = np.where(sel[..., None], pol.act(states, obs, masks, sel, rng), a)
        return a

    def reset(self, batch_size: int) -> None:
        self.default.reset(batch_size)
        for p in self.overrides.values():
            p.reset(batch_size)


class CreatorRouter(BasePolicy):
    """Dispatch coalition members by whether they created the incident.

    This is a *training-harness* split, not something an agent observes: it lets Stage 4 hold the
    partner fixed to a scripted honest policy while only the incident creator learns, so any gain
    is attributable to one speaker rather than to coordination. Crew are routed to `crew`.
    """

    name = "creator_router"

    def __init__(self, crew: BasePolicy, creator: BasePolicy, partner: BasePolicy):
        self.crew = crew
        self.creator = creator
        self.partner = partner
        self.needs_obs = any(p.needs_obs for p in (crew, creator, partner))

    def act(
        self,
        states: Sequence[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        crew_sel = np.zeros_like(active)
        creator_sel = np.zeros_like(active)
        partner_sel = np.zeros_like(active)
        for b, st in enumerate(states):
            is_coal = st.roles == int(Role.COALITION)
            crew_sel[b] = active[b] & ~is_coal
            for i in np.flatnonzero(active[b] & is_coal):
                if int(i) == int(st.incident_creator):
                    creator_sel[b, i] = True
                else:
                    partner_sel[b, i] = True
        out = np.zeros((*active.shape, N_HEADS), dtype=np.int64)
        for sel, pol in ((crew_sel, self.crew), (creator_sel, self.creator), (partner_sel, self.partner)):
            if sel.any():
                out = np.where(sel[..., None], pol.act(states, obs, masks, sel, rng), out)
        return out

    def reset(self, batch_size: int) -> None:
        for p in (self.crew, self.creator, self.partner):
            p.reset(batch_size)


__all__ = [
    "BasePolicy",
    "RoleRouter",
    "AgentRouter",
    "CreatorRouter",
    "first_legal",
    "sample_legal",
]
