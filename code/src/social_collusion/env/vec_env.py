"""Single-episode driver and a lockstep vectorised runner.

RLlib's multi-agent vectorisation has documented limitations for setups like this one, and the
environment is tiny, so a local batched runner is both simpler and more predictable
(plan sec.11.1). `VecEnv` keeps B independent `GameState`s and steps them together so policy
inference is one batched forward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.action_masks import active_agents, compute_masks, head_sizes
from social_collusion.env.enums import N_HEADS, Phase
from social_collusion.env.observations import build_all_actor_obs, build_critic_state, obs_dim
from social_collusion.env.state import GameState
from social_collusion.env.transition import reset as env_reset
from social_collusion.env.transition import transition
from social_collusion.seeding import episode_rng


class Policy(Protocol):
    """What every actor - scripted, random, or neural - must provide."""

    def act(
        self,
        states: list[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return an (B, n_agents, N_HEADS) integer action array. Only entries where
        `active[b, i]` is True are consumed by the engine."""
        ...


def clamp_to_masks(action: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Project an arbitrary action onto the legal set (used by the PettingZoo wrapper).

    Returns the clamped action and the number of entries that had to be changed - the
    "fraction of invalid-action attempts before masking" efficiency metric (plan sec.16.7).
    """
    out = np.array(action, copy=True)
    n_invalid = 0
    n_agents = masks[0].shape[0]
    for h, m in enumerate(masks):
        for i in range(n_agents):
            a = int(out[i, h])
            if a < 0 or a >= m.shape[1] or not m[i, a]:
                legal = np.flatnonzero(m[i])
                out[i, h] = int(legal[0]) if len(legal) else 0
                n_invalid += 1
    return out, n_invalid


# --------------------------------------------------------------------------------------
# single episode
# --------------------------------------------------------------------------------------
def play_episode(
    cfg: EnvConfig,
    policy: Policy | Callable,
    seed: int,
    episode_index: int = 0,
    record: bool = False,
    validate: bool = True,
) -> tuple[GameState, dict[str, Any]]:
    """Play one full episode. Returns the terminal state and an optional step trace."""
    rng = episode_rng(seed, episode_index)
    state = env_reset(cfg, rng, seed=seed)
    trace: dict[str, Any] = {"steps": []} if record else {}
    guard = 0
    while state.phase != int(Phase.TERMINAL):
        guard += 1
        if guard > 512:  # pragma: no cover - phase machine is finite
            raise RuntimeError("episode did not terminate; phase machine bug")
        masks = compute_masks(state)
        act = active_agents(state)
        obs = build_all_actor_obs(state)
        a = policy.act([state], obs[None], [m[None] for m in masks], act[None], rng)[0]
        if record:
            trace["steps"].append(
                {
                    "phase": Phase(state.phase).name,
                    "turn": int(state.turn),
                    "sub_step": int(state.sub_step),
                    "joint_actions": np.asarray(a).tolist(),
                    "active": act.tolist(),
                }
            )
        res = transition(state, a, rng, inplace=True, validate=validate)
        state = res.state
        if record:
            trace["steps"][-1]["events"] = [e.to_dict() for e in res.events]
    return state, trace


# --------------------------------------------------------------------------------------
# vectorised runner
# --------------------------------------------------------------------------------------
class VecEnv:
    """B independent episodes stepped in lockstep.

    Episodes have different lengths only in the spatial variant (an early report ends free
    play), so `alive_mask` marks which sub-environments are still running.
    """

    def __init__(self, cfg: EnvConfig, batch_size: int, seed: int, episode_offset: int = 0):
        self.cfg = cfg
        self.B = batch_size
        self.seed = seed
        self.episode_counter = episode_offset
        self.obs_dim = obs_dim(cfg)
        self.head_sizes = head_sizes(cfg)
        self.states: list[GameState] = []
        self.rngs: list[np.random.Generator] = []
        self.episode_ids: list[int] = []
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.states, self.rngs, self.episode_ids = [], [], []
        for _ in range(self.B):
            self._new_episode()

    def _new_episode(self, slot: int | None = None) -> None:
        idx = self.episode_counter
        self.episode_counter += 1
        rng = episode_rng(self.seed, idx)
        st = env_reset(self.cfg, rng, seed=self.seed)
        if slot is None:
            self.states.append(st)
            self.rngs.append(rng)
            self.episode_ids.append(idx)
        else:
            self.states[slot] = st
            self.rngs[slot] = rng
            self.episode_ids[slot] = idx

    # -- observation side ------------------------------------------------
    def observe(self, needed: bool = True) -> np.ndarray:
        """Flat actor observations. `needed=False` returns a zero array without doing the work -
        scripted policies read the GameState directly and never look at it."""
        out = np.zeros((self.B, self.cfg.n_agents, self.obs_dim), dtype=np.float32)
        if not needed:
            return out
        for b, st in enumerate(self.states):
            out[b] = build_all_actor_obs(st)
        return out

    def critic_states(self) -> np.ndarray:
        return np.stack([build_critic_state(st) for st in self.states])

    def masks(self) -> list[np.ndarray]:
        per_env = [compute_masks(st) for st in self.states]
        return [np.stack([pe[h] for pe in per_env]) for h in range(N_HEADS)]

    def active(self) -> np.ndarray:
        return np.stack([active_agents(st) for st in self.states])

    def phases(self) -> np.ndarray:
        return np.array([st.phase for st in self.states], dtype=np.int64)

    def done(self) -> np.ndarray:
        return np.array([st.phase == int(Phase.TERMINAL) for st in self.states], dtype=bool)

    # -- stepping --------------------------------------------------------
    def step(
        self, actions: np.ndarray, validate: bool = False
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Advance every non-terminal sub-environment by one phase-step."""
        rewards = np.zeros((self.B, self.cfg.n_agents), dtype=np.float32)
        dones = np.zeros(self.B, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.B)]
        for b, st in enumerate(self.states):
            if st.phase == int(Phase.TERMINAL):
                dones[b] = True
                continue
            res = transition(st, actions[b], self.rngs[b], inplace=True, validate=validate)
            self.states[b] = res.state
            rewards[b] = res.rewards
            dones[b] = res.terminated
            infos[b] = res.info
        return rewards, dones, infos

    def autoreset(self) -> list[GameState]:
        """Replace terminal sub-environments with fresh episodes; return the finished states."""
        finished = []
        for b, st in enumerate(self.states):
            if st.phase == int(Phase.TERMINAL):
                finished.append(st)
                self._new_episode(slot=b)
        return finished

    def run_episodes(
        self, policy: Policy, n_episodes: int, validate: bool = False
    ) -> list[GameState]:
        """Evaluate fixed episode indices, completing each vector batch.

        Stopping after the first N asynchronous completions censors long episodes at
        the boundary. Keep slot/index order and finish the final batch before taking
        its first remaining indices, independently of their outcomes or durations.
        """
        if n_episodes < 0:
            raise ValueError("n_episodes must be nonnegative")
        out: list[GameState] = []
        rng = np.random.default_rng(self.seed ^ 0x5EED)
        needs_obs = getattr(policy, "needs_obs", True)
        reset_policy = getattr(policy, "reset", None)
        if reset_policy is not None:
            reset_policy(self.B)
        completed = self.done()
        while len(out) < n_episodes:
            obs, masks, act = self.observe(needs_obs), self.masks(), self.active()
            a = policy.act(self.states, obs, masks, act, rng)
            _, dones, _ = self.step(a, validate=validate)
            mark_done = getattr(policy, "mark_done", None)
            if mark_done is not None:
                mark_done(dones & ~completed)
            completed |= dones
            if completed.all():
                remaining = n_episodes - len(out)
                out.extend(self.autoreset()[:remaining])
                completed[:] = False
                if reset_policy is not None:
                    reset_policy(self.B)
        return out


__all__ = ["VecEnv", "play_episode", "clamp_to_masks", "Policy"]
