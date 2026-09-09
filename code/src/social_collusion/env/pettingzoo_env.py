"""PettingZoo ParallelEnv wrapper (plan Task 7).

The engine underneath is framework-free; this module is a thin adapter so that the official
`pettingzoo.test.parallel_api_test` can run in CI [R1][R21].

Two adapter-level decisions, both documented because they are visible to users:

* Sequential meeting turns are expressed inside the parallel API by masking every non-speaking
  agent down to a single legal no-op, rather than by shrinking `env.agents`. That keeps the agent
  set stable, which is what the API test expects.
* PettingZoo semantics allow any action inside the declared space, so the wrapper *clamps*
  illegal actions to legal ones and counts them in `infos[agent]["invalid_action_attempts"]`.
  The native engine (`transition(..., validate=True)`) still raises on illegal actions.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

try:
    from gymnasium import spaces
    from pettingzoo.utils.env import ParallelEnv
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pettingzoo and gymnasium are required for the PettingZoo wrapper: "
        "pip install 'social-collusion[env]'"
    ) from exc

from social_collusion.config import EnvConfig, load_env_config
from social_collusion.env.action_masks import active_agents, compute_masks, head_sizes
from social_collusion.env.enums import Phase
from social_collusion.env.observations import build_actor_obs, obs_dim
from social_collusion.env.transition import reset as engine_reset
from social_collusion.env.transition import transition as engine_transition
from social_collusion.env.vec_env import clamp_to_masks
from social_collusion.seeding import episode_rng


class SocialCollusionParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "social_collusion_v0", "is_parallelizable": True}

    def __init__(self, config: EnvConfig | str = "meeting_only", render_mode: str | None = None):
        self.cfg = config if isinstance(config, EnvConfig) else load_env_config(config)
        self.render_mode = render_mode
        self.possible_agents = [f"A{i}" for i in range(self.cfg.n_agents)]
        self.agents: list[str] = list(self.possible_agents)
        self._sizes = head_sizes(self.cfg)
        self._obs_dim = obs_dim(self.cfg)
        self.state_obj = None
        self._rng = np.random.default_rng(0)
        self._episode = 0
        self._seed = 0

    # -- spaces ----------------------------------------------------------
    @functools.cache  # noqa: B019 - spaces are immutable per env instance
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.MultiDiscrete(np.array(self._sizes, dtype=np.int64))

    # -- helpers ---------------------------------------------------------
    def _index(self, agent: str) -> int:
        return int(agent[1:])

    def _obs_for(self, agent: str) -> np.ndarray:
        return build_actor_obs(self.state_obj, self._index(agent))

    def _infos(self, extra: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        masks = compute_masks(self.state_obj)
        act = active_agents(self.state_obj)
        infos: dict[str, dict[str, Any]] = {}
        for a in self.agents:
            i = self._index(a)
            infos[a] = {
                "action_mask": [m[i].astype(np.int8) for m in masks],
                "active": bool(act[i]),
                "phase": Phase(self.state_obj.phase).name,
            }
            if extra and a in extra:
                infos[a].update(extra[a])
        return infos

    # -- API -------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
            self._episode = 0
        self._rng = episode_rng(self._seed, self._episode)
        self._episode += 1
        self.state_obj = engine_reset(self.cfg, self._rng, seed=self._seed)
        # Every possible agent starts in `agents`, including the incident victim, which is then
        # terminated on the first step. PettingZoo expects the reset agent set to be the full
        # one and every agent to leave via a termination, so this keeps the API test clean.
        self.agents = list(self.possible_agents)
        obs = {a: self._obs_for(a) for a in self.agents}
        return obs, self._infos()

    def step(self, actions: dict[str, Any]):
        cfg = self.cfg
        joint = np.zeros((cfg.n_agents, len(self._sizes)), dtype=np.int64)
        for a, act in actions.items():
            joint[self._index(a)] = np.asarray(act, dtype=np.int64)
        masks = compute_masks(self.state_obj)
        joint, n_invalid = clamp_to_masks(joint, masks)

        res = engine_transition(self.state_obj, joint, self._rng, inplace=True, validate=False)
        self.state_obj = res.state
        terminated = res.terminated

        obs = {a: self._obs_for(a) for a in self.agents}
        rewards = {a: float(res.rewards[self._index(a)]) for a in self.agents}
        terminations = {
            a: bool(terminated or not self.state_obj.alive[self._index(a)]) for a in self.agents
        }
        truncations = {a: False for a in self.agents}
        infos = self._infos(
            {a: {"invalid_action_attempts": n_invalid} for a in self.agents}
        )
        if terminated:
            infos = {a: {**infos[a], **res.info} for a in self.agents}
        self.agents = [a for a in self.agents if not terminations[a]]
        return obs, rewards, terminations, truncations, infos

    def render(self):
        from social_collusion.env.render_text import render_state

        text = render_state(self.state_obj, privileged=False)
        if self.render_mode == "human":  # pragma: no cover - interactive
            print(text)
        return text

    def close(self):  # pragma: no cover - nothing to release
        pass

    def state(self) -> np.ndarray:
        from social_collusion.env.observations import build_critic_state

        return build_critic_state(self.state_obj)


def parallel_env(config: EnvConfig | str = "meeting_only", **kwargs) -> SocialCollusionParallelEnv:
    return SocialCollusionParallelEnv(config, **kwargs)


def env(config: EnvConfig | str = "meeting_only", **kwargs):
    """AEC view, via PettingZoo's standard conversion."""
    from pettingzoo.utils.conversions import parallel_to_aec

    return parallel_to_aec(parallel_env(config, **kwargs))


__all__ = ["SocialCollusionParallelEnv", "parallel_env", "env"]
