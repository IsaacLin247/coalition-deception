"""Clipped PPO update shared by IPPO and MAPPO (plan sec.14.4).

IPPO and MAPPO differ here in exactly one place - what the critic is fed - so they are one
implementation with a flag rather than two files that drift apart. `centralized_critic=True`
feeds the privileged state of `observations.build_critic_state`; the actor never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from social_collusion.rl.networks import ActorCritic, masked_log_probs
from social_collusion.rl.rollout_buffer import phase_stratified_batches, uniform_batches


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    entropy_coef_final: float = 0.005
    grad_clip: float = 0.5
    ppo_epochs: int = 8
    n_minibatches: int = 4
    normalize_advantages: bool = True
    stratify_minibatches_by_phase: bool = True
    target_kl: float | None = 0.05

    @staticmethod
    def from_dict(d: dict) -> PPOConfig:
        fields = {f for f in PPOConfig.__dataclass_fields__}
        return PPOConfig(**{k: v for k, v in d.items() if k in fields})


class PPOLearner:
    def __init__(self, model: ActorCritic, cfg: PPOConfig, device: str = "cpu"):
        if model.actor.use_gru:
            # The update below re-runs the actor on shuffled minibatches with a zero hidden
            # state, which is correct for a feed-forward policy and silently wrong for a
            # recurrent one (it destroys the very memory the GRU is for). Sequence-batched
            # BPTT is not implemented, so refuse rather than train something broken. The plan's
            # own guidance is to add recurrence only after non-recurrent policies pass their
            # sanity checks, and the meeting-only observation already carries the full public
            # transcript, so it is not needed for Stages 2-7.
            raise NotImplementedError(
                "PPOLearner does not support recurrent actors: the minibatch update would reset "
                "the GRU state. Set use_gru=false, or implement sequence-batched updates first."
            )
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    def entropy_coef(self, progress: float) -> float:
        c = self.cfg
        return float(c.entropy_coef + (c.entropy_coef_final - c.entropy_coef) * np.clip(progress, 0, 1))

    def update(self, batch: dict[str, np.ndarray], progress: float = 0.0, seed: int = 0) -> dict[str, float]:
        cfg = self.cfg
        dev = self.device
        n = batch["obs"].shape[0]
        if n == 0:  # pragma: no cover
            return {}
        obs = torch.as_tensor(batch["obs"], device=dev)
        critic_in = torch.as_tensor(
            batch["critic_state"] if self.model.centralized else batch["obs"], device=dev
        )
        actions = torch.as_tensor(batch["actions"], device=dev)
        masks = [torch.as_tensor(m, device=dev) for m in batch["masks"]]
        old_logp = torch.as_tensor(batch["logp"], device=dev)
        adv_np = batch["adv"]
        if cfg.normalize_advantages and adv_np.std() > 1e-8:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = torch.as_tensor(adv_np, device=dev)
        ret = torch.as_tensor(batch["ret"], device=dev)
        head_sizes = batch["head_sizes"]

        rng = np.random.default_rng(seed)
        ent_coef = self.entropy_coef(progress)
        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "approx_kl": 0.0}
        n_updates = 0
        stop = False
        for _ in range(cfg.ppo_epochs):
            batches = (
                phase_stratified_batches(batch["phase"], cfg.n_minibatches, rng)
                if cfg.stratify_minibatches_by_phase
                else uniform_batches(n, cfg.n_minibatches, rng)
            )
            for idx_np in batches:
                idx = torch.as_tensor(idx_np, device=dev, dtype=torch.long)
                logits, _ = self.model.actor.logits(obs[idx], None)
                logp, ent = masked_log_probs(
                    logits, head_sizes, [m[idx] for m in masks], actions[idx]
                )
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                p1 = ratio * a
                p2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a
                policy_loss = -torch.min(p1, p2).mean()
                value = self.model.critic(critic_in[idx])
                value_loss = 0.5 * (value - ret[idx]).pow(2).mean()
                entropy = ent.mean()
                loss = policy_loss + cfg.value_coef * value_loss - ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - (logp - old_logp[idx])).mean())
                    logs["policy_loss"] += float(policy_loss)
                    logs["value_loss"] += float(value_loss)
                    logs["entropy"] += float(entropy)
                    logs["clip_frac"] += float(((ratio - 1).abs() > cfg.clip_eps).float().mean())
                    logs["approx_kl"] += approx_kl
                n_updates += 1
                if cfg.target_kl is not None and approx_kl > cfg.target_kl * 1.5:
                    stop = True
                    break
            if stop:
                break
        for k in logs:
            logs[k] /= max(1, n_updates)
        logs["entropy_coef"] = ent_coef
        logs["n_minibatch_updates"] = float(n_updates)
        logs["early_stopped"] = float(stop)
        return logs


__all__ = ["PPOConfig", "PPOLearner"]
