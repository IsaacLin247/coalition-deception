"""Neural policy that satisfies the same `act()` contract as every scripted baseline.

Because it is interchangeable with the scripted policies, the entire counterfactual battery -
partner replacement, statement override, symbol scrambling - works on a trained checkpoint with
no extra code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from social_collusion.config import EnvConfig
from social_collusion.env.action_masks import head_sizes
from social_collusion.env.observations import critic_dim, obs_dim
from social_collusion.env.state import GameState
from social_collusion.policies.base import BasePolicy
from social_collusion.rl.networks import ActorCritic, sample_actions


def evaluation_copy(policy: BasePolicy, seed: int) -> BasePolicy:
    """Isolate evaluation from training and reproducibly seed every neural actor.

    Routers contain policies in attributes or lists. Copying them also isolates recurrent
    memory and scripted state; evaluation must not advance a learner's sampling stream.
    """
    import copy

    out = copy.deepcopy(policy)
    seen: set[int] = set()
    actors: list[TorchPolicy] = []

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, TorchPolicy):
            actors.append(obj)
        elif isinstance(obj, BasePolicy):
            for value in vars(obj).values():
                if isinstance(value, BasePolicy):
                    visit(value)
                elif isinstance(value, (list, tuple, dict)):
                    for child in (value.values() if isinstance(value, dict) else value):
                        if isinstance(child, BasePolicy):
                            visit(child)

    visit(out)
    for index, actor in enumerate(actors):
        actor.generator.manual_seed(int(seed) * 100 + index)
        actor.model.eval()
    return out


class TorchPolicy(BasePolicy):
    """Wraps an `ActorCritic`. Stateless unless the actor uses a GRU."""

    name = "torch"
    needs_obs = True

    def __init__(
        self,
        cfg: EnvConfig,
        model: ActorCritic | None = None,
        hidden_dim: int = 128,
        use_gru: bool = False,
        gru_dim: int = 128,
        centralized_critic: bool = False,
        device: str = "cpu",
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.head_sizes = head_sizes(cfg)
        self.deterministic = deterministic
        if model is None:
            model = ActorCritic(
                obs_dim(cfg),
                self.head_sizes,
                critic_dim=critic_dim(cfg) if centralized_critic else None,
                hidden_dim=hidden_dim,
                use_gru=use_gru,
                gru_dim=gru_dim,
            )
        self.model = model.to(self.device)
        # the sampling generator must live on the same device as the logits (torch.multinomial
        # rejects a CPU generator for CUDA tensors), so --device cuda actually works
        self.generator = torch.Generator(device=self.device.type).manual_seed(seed)
        self.hidden: torch.Tensor | None = None
        self.last: dict[str, np.ndarray] = {}

    # -- BasePolicy ------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        self.hidden = None

    def mark_done(self, done: np.ndarray) -> None:
        """Zero the recurrent state of sub-environments whose episode just ended.

        The vectorised runner auto-resets, so without this the GRU would carry memory across an
        episode boundary. No-op for the default non-recurrent configuration.
        """
        if self.hidden is None or not self.model.actor.use_gru:
            return
        n = self.cfg.n_agents
        d = np.repeat(np.asarray(done, dtype=bool), n)
        self.hidden[:, torch.as_tensor(d, device=self.hidden.device), :] = 0.0

    @torch.no_grad()
    def act(
        self,
        states: Sequence[GameState],
        obs: np.ndarray,
        masks: list[np.ndarray],
        active: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        B, n = active.shape
        flat_obs = torch.as_tensor(obs.reshape(B * n, -1), dtype=torch.float32, device=self.device)
        flat_masks = [
            torch.as_tensor(m.reshape(B * n, -1), dtype=torch.bool, device=self.device)
            for m in masks
        ]
        if self.model.actor.use_gru and (
            self.hidden is None or self.hidden.shape[1] != B * n
        ):
            self.hidden = torch.zeros(
                1, B * n, self.model.actor.feat_dim, device=self.device
            )
        logits, self.hidden = self.model.actor.logits(flat_obs, self.hidden)
        a, logp, ent = sample_actions(
            logits, self.head_sizes, flat_masks, self.deterministic, self.generator
        )
        self.last = {
            "logp": logp.cpu().numpy().reshape(B, n),
            "entropy": ent.cpu().numpy().reshape(B, n),
        }
        return a.cpu().numpy().reshape(B, n, len(self.head_sizes))

    @torch.no_grad()
    def value(self, critic_input: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(critic_input, dtype=torch.float32, device=self.device)
        return self.model.critic(x).cpu().numpy()

    @torch.no_grad()
    def action_probs(self, state: GameState, agent: int) -> list[np.ndarray]:
        """Per-head probabilities - saved into replay `policy_metadata` for analysis runs."""
        from social_collusion.env.action_masks import compute_masks
        from social_collusion.env.observations import build_actor_obs

        o = torch.as_tensor(build_actor_obs(state, agent)[None], dtype=torch.float32)
        masks = compute_masks(state)
        logits, _ = self.model.actor.logits(o, None)
        parts = torch.split(logits, self.head_sizes, dim=-1)
        out = []
        for h, lg in enumerate(parts):
            m = torch.as_tensor(masks[h][agent][None], dtype=torch.bool)
            lg = lg.masked_fill(~m, -1e9)
            out.append(torch.softmax(lg, dim=-1)[0].cpu().numpy())
        return out

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "head_sizes": self.head_sizes,
                "env_config": self.cfg.to_dict(),
                "centralized": self.model.centralized,
                "use_gru": self.model.actor.use_gru,
                "hidden_dim": self.model.actor.torso[0].out_features,
                "gru_dim": self.model.actor.feat_dim,
                "extra": extra or {},
            },
            path,
        )
        return path

    @staticmethod
    def load(path: str | Path, cfg: EnvConfig | None = None, device: str = "cpu", **kw) -> TorchPolicy:
        blob = torch.load(path, map_location=device, weights_only=False)
        if cfg is None:
            raw = dict(blob["env_config"])
            raw["rooms"] = tuple(raw["rooms"])
            raw["edges"] = tuple(tuple(e) for e in raw["edges"])
            raw.pop("adjacency", None)
            cfg = EnvConfig(**raw)
        pol = TorchPolicy(
            cfg,
            hidden_dim=int(blob.get("hidden_dim", 128)),
            gru_dim=int(blob.get("gru_dim", 128)),
            centralized_critic=bool(blob.get("centralized", False)),
            use_gru=bool(blob.get("use_gru", False)),
            device=device,
            **kw,
        )
        pol.model.load_state_dict(blob["state_dict"])
        pol.model.eval()
        return pol


class FrozenCopy(TorchPolicy):
    """A deep copy used as the 'frozen earlier checkpoint' counterfactual control."""

    name = "torch_frozen"

    def __init__(self, source: TorchPolicy):
        import copy

        super().__init__(
            source.cfg,
            model=copy.deepcopy(source.model),
            centralized_critic=source.model.centralized,
            use_gru=source.model.actor.use_gru,
            device=str(source.device),
        )
        self.model.eval()


__all__ = ["TorchPolicy", "FrozenCopy", "evaluation_copy"]
