"""Rollout storage with per-(episode, agent) return computation and phase-stratified batching.

Plan sec.14.5 is the reason for the stratification: almost every scientifically interesting
decision (the claim, the vote) happens once per episode while free-play steps dominate the
count, so uniform minibatching starves the meeting heads of gradient. `phase_stratified_batches`
guarantees each minibatch carries the same phase mix as the full rollout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from social_collusion.env.enums import N_HEADS


@dataclass
class Rollout:
    obs: list[np.ndarray] = field(default_factory=list)
    critic_state: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    masks: list[list[np.ndarray]] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    phase: list[int] = field(default_factory=list)
    ep_id: list[int] = field(default_factory=list)
    agent: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)

    def add(self, obs, critic_state, action, masks, logp, value, phase, ep_id, agent) -> None:
        self.obs.append(obs)
        self.critic_state.append(critic_state)
        self.actions.append(action)
        self.masks.append(masks)
        self.logp.append(float(logp))
        self.value.append(float(value))
        self.phase.append(int(phase))
        self.ep_id.append(int(ep_id))
        self.agent.append(int(agent))
        self.reward.append(0.0)

    def __len__(self) -> int:
        return len(self.obs)

    def assign_terminal(self, ep_id: int, rewards: np.ndarray) -> None:
        """Terminal reward is credited to the *last* stored step of each agent in this episode."""
        last: dict[int, int] = {}
        for k, (e, a) in enumerate(zip(self.ep_id, self.agent)):
            if e == ep_id:
                last[a] = k
        for a, k in last.items():
            self.reward[k] = float(rewards[a])

    # -- tensor views ----------------------------------------------------
    def finalize(self, gamma: float = 1.0, gae_lambda: float = 0.95) -> dict[str, np.ndarray]:
        n = len(self)
        adv = np.zeros(n, dtype=np.float32)
        ret = np.zeros(n, dtype=np.float32)
        value = np.asarray(self.value, dtype=np.float32)
        reward = np.asarray(self.reward, dtype=np.float32)

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k, (e, a) in enumerate(zip(self.ep_id, self.agent)):
            groups[(e, a)].append(k)
        for idxs in groups.values():
            gae = 0.0
            for pos in reversed(range(len(idxs))):
                k = idxs[pos]
                nxt = 0.0 if pos == len(idxs) - 1 else value[idxs[pos + 1]]
                nonterminal = 0.0 if pos == len(idxs) - 1 else 1.0
                delta = reward[k] + gamma * nxt * nonterminal - value[k]
                gae = delta + gamma * gae_lambda * nonterminal * gae
                adv[k] = gae
                ret[k] = gae + value[k]

        head_sizes = [self.masks[0][h].shape[-1] for h in range(N_HEADS)] if n else []
        masks = [
            np.stack([self.masks[k][h] for k in range(n)]).astype(bool) for h in range(N_HEADS)
        ]
        return {
            "obs": np.stack(self.obs).astype(np.float32),
            "critic_state": np.stack(self.critic_state).astype(np.float32),
            "actions": np.stack(self.actions).astype(np.int64),
            "masks": masks,
            "head_sizes": head_sizes,
            "logp": np.asarray(self.logp, dtype=np.float32),
            "value": value,
            "adv": adv,
            "ret": ret,
            "phase": np.asarray(self.phase, dtype=np.int64),
            "agent": np.asarray(self.agent, dtype=np.int64),
            "ep_id": np.asarray(self.ep_id, dtype=np.int64),
            "reward": reward,
        }


def phase_stratified_batches(
    phase: np.ndarray, n_minibatches: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Split indices so every minibatch has the same phase composition (plan sec.14.5)."""
    parts: list[list[np.ndarray]] = [[] for _ in range(n_minibatches)]
    for p in np.unique(phase):
        idx = np.flatnonzero(phase == p)
        rng.shuffle(idx)
        for j, chunk in enumerate(np.array_split(idx, n_minibatches)):
            parts[j].append(chunk)
    out = [np.concatenate(p) if p else np.zeros(0, dtype=int) for p in parts]
    for a in out:
        rng.shuffle(a)
    return [a for a in out if a.size]


def uniform_batches(n: int, n_minibatches: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return [a for a in np.array_split(idx, n_minibatches) if a.size]


__all__ = ["Rollout", "phase_stratified_batches", "uniform_batches"]
