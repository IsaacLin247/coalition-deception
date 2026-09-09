"""Actor-critic networks with factored, masked categorical heads (plan sec.14.1).

One design detail does a lot of work: a head whose mask has exactly one legal action
contributes `log 1 = 0` to the log-probability and `0` to the entropy automatically. Because
`action_masks.compute_masks` collapses every phase-irrelevant head to index 0, the "only
phase-relevant heads contribute to the log probability and entropy" requirement of plan sec.9.2
falls out of masking instead of needing bookkeeping that can silently go wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

NEG_INF = -1e9


def mlp(sizes: Sequence[int], act=nn.ReLU, out_act=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class MultiHeadActor(nn.Module):
    """Shared torso -> optional GRU -> one categorical head per action component."""

    def __init__(
        self,
        obs_dim: int,
        head_sizes: Sequence[int],
        hidden_dim: int = 128,
        use_gru: bool = False,
        gru_dim: int = 128,
    ):
        super().__init__()
        self.head_sizes = list(head_sizes)
        self.total = int(sum(head_sizes))
        self.use_gru = use_gru
        self.torso = mlp([obs_dim, hidden_dim, hidden_dim], out_act=nn.ReLU)
        feat = hidden_dim
        if use_gru:
            self.gru = nn.GRU(hidden_dim, gru_dim, batch_first=False)
            feat = gru_dim
        self.head = nn.Linear(feat, self.total)
        self.feat_dim = feat
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def features(self, obs: torch.Tensor, hxs: torch.Tensor | None = None):
        z = self.torso(obs)
        if self.use_gru:
            # obs: (T, B, D) when training on sequences, (B, D) when acting
            if z.dim() == 2:
                z, hxs = self.gru(z.unsqueeze(0), hxs)
                z = z.squeeze(0)
            else:
                z, hxs = self.gru(z, hxs)
        return z, hxs

    def logits(self, obs: torch.Tensor, hxs: torch.Tensor | None = None):
        z, hxs = self.features(obs, hxs)
        return self.head(z), hxs

    def split(self, flat: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(flat, self.head_sizes, dim=-1))


class Critic(nn.Module):
    """Value head. `state_dim` is the privileged critic state for MAPPO, the actor obs for IPPO."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim, 1])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        head_sizes: Sequence[int],
        critic_dim: int | None = None,
        hidden_dim: int = 128,
        use_gru: bool = False,
        gru_dim: int = 128,
    ):
        super().__init__()
        self.actor = MultiHeadActor(obs_dim, head_sizes, hidden_dim, use_gru, gru_dim)
        self.critic = Critic(critic_dim if critic_dim is not None else obs_dim, hidden_dim)
        self.centralized = critic_dim is not None
        self.head_sizes = list(head_sizes)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_log_probs(
    flat_logits: torch.Tensor,
    head_sizes: Sequence[int],
    masks: Sequence[torch.Tensor],
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-prob of `actions` and summed entropy, over masked categorical heads.

    `masks[h]` has shape (..., head_sizes[h]); `actions` has shape (..., n_heads).
    """
    parts = torch.split(flat_logits, list(head_sizes), dim=-1)
    logp = torch.zeros(flat_logits.shape[:-1], device=flat_logits.device)
    ent = torch.zeros_like(logp)
    for h, (lg, m) in enumerate(zip(parts, masks)):
        lg = lg.masked_fill(~m, NEG_INF)
        logits = lg - lg.logsumexp(dim=-1, keepdim=True)
        p = logits.exp()
        a = actions[..., h].unsqueeze(-1)
        logp = logp + logits.gather(-1, a).squeeze(-1)
        ent = ent - (p * torch.where(m, logits, torch.zeros_like(logits))).sum(-1)
    return logp, ent


def sample_actions(
    flat_logits: torch.Tensor,
    head_sizes: Sequence[int],
    masks: Sequence[torch.Tensor],
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one action per head under the masks. Returns (actions, logp, entropy)."""
    parts = torch.split(flat_logits, list(head_sizes), dim=-1)
    acts, logps, ents = [], [], []
    for lg, m in zip(parts, masks):
        lg = lg.masked_fill(~m, NEG_INF)
        logits = lg - lg.logsumexp(dim=-1, keepdim=True)
        p = logits.exp()
        if deterministic:
            a = p.argmax(dim=-1)
        else:
            flat = p.reshape(-1, p.shape[-1])
            idx = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
            a = idx.reshape(p.shape[:-1])
        acts.append(a)
        logps.append(logits.gather(-1, a.unsqueeze(-1)).squeeze(-1))
        ents.append(-(p * torch.where(m, logits, torch.zeros_like(logits))).sum(-1))
    return torch.stack(acts, dim=-1), torch.stack(logps, dim=-1).sum(-1), torch.stack(ents, dim=-1).sum(-1)


__all__ = [
    "MultiHeadActor",
    "Critic",
    "ActorCritic",
    "masked_log_probs",
    "sample_actions",
    "mlp",
]
