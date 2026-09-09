"""Uniform-over-legal-actions policy (plan sec.15.1) - the null baseline for every comparison."""

from __future__ import annotations

import numpy as np

from social_collusion.env.enums import N_HEADS
from social_collusion.env.state import GameState
from social_collusion.policies.base import BasePolicy, sample_legal


class RandomPolicy(BasePolicy):
    name = "random"

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        return np.array([sample_legal(m, rng) for m in masks], dtype=np.int64)


class NoOpPolicy(BasePolicy):
    """Always index 0 on every head: WAIT / NO_INFORMATION / NEUTRAL / vote for the first
    legal target. Used as the 'neutral' control action in counterfactual interventions."""

    name = "noop"

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        out = np.zeros(N_HEADS, dtype=np.int64)
        for h, m in enumerate(masks):
            if not m[0]:
                legal = np.flatnonzero(m)
                out[h] = int(legal[0]) if len(legal) else 0
        return out


class SkipVotePolicy(NoOpPolicy):
    """Neutral in every phase and always SKIP - the strictest 'does nothing' control."""

    name = "skip"

    def act_single(
        self, state: GameState, agent: int, masks: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        from social_collusion.env.enums import Head

        out = super().act_single(state, agent, masks, rng)
        skip = state.config.n_agents
        if skip < len(masks[Head.VOTE]) and masks[Head.VOTE][skip]:
            out[Head.VOTE] = skip
        return out


__all__ = ["RandomPolicy", "NoOpPolicy", "SkipVotePolicy"]
