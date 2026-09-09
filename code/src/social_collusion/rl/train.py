"""Training loop for staged role learning in the social-collusion environment.

The default remains coalition learning for backwards compatibility.  F3 can additionally train the
crew first and then pass its checkpoint back as a frozen opponent while the coalition learns. One
shared actor is used by default; `separate_actors` gives members of the learner role independent
networks as an optional specialization ablation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.enums import Role
from social_collusion.env.state import GameState
from social_collusion.env.vec_env import VecEnv
from social_collusion.policies.base import BasePolicy
from social_collusion.rl.ppo import PPOConfig, PPOLearner
from social_collusion.rl.rollout_buffer import Rollout
from social_collusion.rl.torch_policy import TorchPolicy, evaluation_copy


def role_rank(state: GameState, agent: int, role: Role) -> int:
    """Position of an agent within a role's index-sorted member list.

    The routing is only a training convenience.  It does not alter observations or reveal the
    role to the actor; it just supports the separate-actor ablation for either side of the game.
    """
    members = [int(i) for i in np.flatnonzero(state.roles == int(role))]
    return members.index(int(agent)) if int(agent) in members else 0


def coalition_rank(state: GameState, agent: int) -> int:
    """Backward-compatible coalition-specific rank helper."""
    return role_rank(state, agent, Role.COALITION)


class RoleLearnerPolicy(BasePolicy):
    """One or two TorchPolicy actors routed by the selected role's rank.

    This is deliberately usable as a frozen opponent.  A learned crew checkpoint therefore
    participates in every movement, report, testimony, and vote decision while the other side is
    being trained, instead of silently falling back to the scripted crew.
    """

    name = "role_learner"
    needs_obs = True

    def __init__(self, actors: Sequence[TorchPolicy], separate: bool, target_role: Role):
        self.actors = list(actors)
        self.separate = separate
        self.target_role = Role(target_role)
        self.last_actor_idx: np.ndarray | None = None

    def act(self, states, obs, masks, active, rng):
        if not self.separate:
            return self.actors[0].act(states, obs, masks, active, rng)
        out = None
        idx_map = np.zeros(active.shape, dtype=np.int64)
        for b, st in enumerate(states):
            for i in range(st.config.n_agents):
                # The historical separate-actors option creates two networks.  Coalition has two
                # members; for a larger crew role, reuse them by rank so every learner still gets
                # a valid action instead of silently receiving the all-zero no-op action.
                idx_map[b, i] = role_rank(st, i, self.target_role) % len(self.actors)
        for k, actor in enumerate(self.actors):
            sel = active & (idx_map == k)
            a = actor.act(states, obs, masks, active, rng)
            out = a if out is None else np.where(sel[..., None], a, out)
        self.last_actor_idx = idx_map
        return out

    def actor_for(self, state: GameState, agent: int) -> int:
        return (role_rank(state, agent, self.target_role) % len(self.actors)) if self.separate else 0

    def reset(self, batch_size: int) -> None:
        for a in self.actors:
            a.reset(batch_size)

    def mark_done(self, done: np.ndarray) -> None:
        for a in self.actors:
            a.mark_done(done)


class CoalitionPolicy(RoleLearnerPolicy):
    """Backward-compatible coalition learner routed by coalition rank."""

    name = "coalition_learner"

    def __init__(self, actors: Sequence[TorchPolicy], separate: bool):
        super().__init__(actors, separate, Role.COALITION)


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    eval_curve: list[dict] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None


class Trainer:
    def __init__(
        self,
        cfg: EnvConfig,
        algo: dict,
        crew_policy: BasePolicy | None,
        seed: int = 0,
        run_dir: str | Path | None = None,
        separate_actors: bool = False,
        device: str = "cpu",
        eval_crew_policy: BasePolicy | None = None,
        learner_role: Role = Role.COALITION,
        opponent_policy: BasePolicy | None = None,
        eval_opponent_policy: BasePolicy | None = None,
        extra_metrics: Callable[[list[GameState]], dict[str, float]] | None = None,
    ):
        from social_collusion.seeding import seed_everything

        seed_everything(seed)
        self.cfg = cfg
        self.algo = dict(algo)
        self.seed = seed
        self.learner_role = Role(learner_role)
        if self.learner_role == Role.COALITION:
            if crew_policy is None:
                raise ValueError("coalition training requires crew_policy as the frozen opponent")
            self.crew = crew_policy
            self.opponent = crew_policy
            self.eval_opponent = eval_crew_policy or crew_policy
        else:
            if opponent_policy is None:
                raise ValueError("crew training requires opponent_policy as the frozen coalition")
            self.crew = None
            self.opponent = opponent_policy
            self.eval_opponent = eval_opponent_policy or opponent_policy
        # Keep the old attribute for callers that inspect the training condition.
        self.eval_crew = eval_crew_policy or crew_policy
        self.device = device
        self.separate = separate_actors
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        #: optional per-update summary of the finished rollout episodes, merged into `history`
        #: (e.g. the realized coalition dependence while the coalition learns to evade a defense)
        self.extra_metrics = extra_metrics

        #: Stage 4 only: freeze the non-creator coalition member to this scripted policy, so a
        #: single learned speaker can be isolated. `None` means both members learn.
        self.frozen_partner: BasePolicy | None = None
        centralized = bool(algo.get("centralized_critic", False))
        n_actors = 2 if separate_actors else 1
        self.actors = [
            TorchPolicy(
                cfg,
                hidden_dim=int(algo.get("hidden_dim", 128)),
                use_gru=bool(algo.get("use_gru", False)),
                gru_dim=int(algo.get("gru_dim", 128)),
                centralized_critic=centralized,
                device=device,
                seed=seed * 100 + k,
            )
            for k in range(n_actors)
        ]
        self.policy = (
            CoalitionPolicy(self.actors, separate_actors)
            if self.learner_role == Role.COALITION
            else RoleLearnerPolicy(self.actors, separate_actors, Role.CREW)
        )
        ppo_cfg = PPOConfig.from_dict(algo)
        self.learners = [PPOLearner(a.model, ppo_cfg, device=device) for a in self.actors]
        self.ppo_cfg = ppo_cfg
        self.n_envs = int(algo.get("n_envs", 256))
        self.rollout_episodes = int(algo.get("rollout_episodes", 256))
        self.total_updates = int(algo.get("total_updates", 200))
        self.vec = VecEnv(cfg, self.n_envs, seed=seed)
        self.env_steps = 0
        self.episodes_done = 0
        self.updates_done = 0

    # -- rollout ---------------------------------------------------------
    def collect(self) -> tuple[list[Rollout], list[GameState]]:
        """Collect complete batches of episodes for episodic, terminal-reward PPO.

        ``rollout_episodes`` is a total episode target across the vector, rounded up to a
        whole batch. Finished slots remain inactive until their batch finishes: resetting
        them immediately would leave unfinished trajectories when the target is reached.
        Such trajectories cannot be treated as terminal or carried across a policy update.
        """
        buffers = [Rollout() for _ in self.actors]
        finished: list[GameState] = []
        self.policy.reset(self.n_envs)
        self.opponent.reset(self.n_envs)
        centralized = self.actors[0].model.centralized
        completed = np.zeros(self.n_envs, dtype=bool)
        while len(finished) < self.rollout_episodes:
            states = self.vec.states
            obs = self.vec.observe(True)
            masks = self.vec.masks()
            active = self.vec.active()
            ep_ids = list(self.vec.episode_ids)

            is_learner = np.stack([(st.roles == int(self.learner_role)) for st in states])
            learner_active = active & is_learner
            opponent_active = active & ~is_learner
            frozen_active = np.zeros_like(active)
            if self.frozen_partner is not None:
                if self.learner_role != Role.COALITION:
                    raise ValueError("frozen_partner is only supported when training the coalition")
                for b, st in enumerate(states):
                    for i in np.flatnonzero(learner_active[b]):
                        if int(i) != int(st.incident_creator):
                            frozen_active[b, i] = True
                learner_active &= ~frozen_active

            a_learn = self.policy.act(states, obs, masks, active, np.random.default_rng(0))
            logp = self.actors[0].last.get("logp")
            a_opponent = self.opponent.act(
                states, obs, masks, opponent_active, np.random.default_rng(0)
            )
            actions = np.where(learner_active[..., None], a_learn, a_opponent)
            if self.frozen_partner is not None and frozen_active.any():
                a_frozen = self.frozen_partner.act(
                    states, obs, masks, frozen_active, np.random.default_rng(0)
                )
                actions = np.where(frozen_active[..., None], a_frozen, actions)

            critic_states = (
                self.vec.critic_states() if centralized else None
            )
            values = [
                a.value(critic_states if centralized else obs.reshape(-1, obs.shape[-1]))
                for a in self.actors
            ]
            for b in range(self.n_envs):
                for i in np.flatnonzero(learner_active[b]):
                    k = self.policy.actor_for(states[b], int(i))
                    cs = critic_states[b] if centralized else obs[b, i]
                    v = values[k][b] if centralized else values[k][b * self.cfg.n_agents + i]
                    lp = logp[b, i] if logp is not None else 0.0
                    if self.separate:
                        lp = self.actors[k].last["logp"][b, i]
                    buffers[k].add(
                        obs[b, i], cs, actions[b, i], [m[b, i] for m in masks],
                        lp, v, states[b].phase, ep_ids[b], int(i),
                    )

            rewards, dones, _ = self.vec.step(actions, validate=False)
            self.env_steps += int(active.sum())
            newly_done = dones & ~completed
            for b in np.flatnonzero(newly_done):
                for buf in buffers:
                    buf.assign_terminal(ep_ids[b], rewards[b])
            self.policy.mark_done(newly_done)
            mark_done = getattr(self.opponent, "mark_done", None)
            if mark_done is not None:
                mark_done(newly_done)
            completed |= dones
            if completed.all():
                finished.extend(self.vec.autoreset())
                completed[:] = False
        self.episodes_done += len(finished)
        return buffers, finished

    # -- training --------------------------------------------------------
    def train(self, log_every: int = 10) -> TrainResult:
        from social_collusion.metrics import collusion_metrics as cm
        from social_collusion.metrics import outcome_metrics as om

        res = TrainResult()
        t0 = time.time()
        eval_every = int(self.algo.get("eval_every", 25))
        ckpt_every = int(self.algo.get("checkpoint_every", 100))
        if log_every < 1:
            raise ValueError("log_every must be at least 1")
        if self.algo.get("evaluate_initial", False):
            ev = self.evaluate(int(self.algo.get("eval_episodes", 1000)))
            ev.update(update=-1, updates_completed=0)
            res.eval_curve.append(ev)
            if self.run_dir:
                self.save(self.run_dir / "checkpoint_0.pt")
        for update in range(self.total_updates):
            buffers, finished = self.collect()
            progress = update / max(1, self.total_updates - 1)
            logs: dict[str, float] = {}
            for k, (buf, learner) in enumerate(zip(buffers, self.learners)):
                if len(buf) == 0:  # pragma: no cover
                    continue
                batch = buf.finalize(self.ppo_cfg.gamma, self.ppo_cfg.gae_lambda)
                out = learner.update(batch, progress=progress, seed=self.seed * 1000 + update)
                for kk, vv in out.items():
                    logs[f"actor{k}_{kk}" if self.separate else kk] = vv
            self.updates_done = update + 1
            summary = om.summarize(finished)
            outcome_label = "W_C" if self.cfg.max_rounds > 1 else "S_C"
            row = {
                "update": update,
                "updates_completed": self.updates_done,
                "env_steps": self.env_steps,
                "episodes": self.episodes_done,
                "wall_s": round(time.time() - t0, 1),
                **{k: round(float(v), 5) for k, v in logs.items()},
                **{k: round(float(v), 5) for k, v in summary.items()},
                "train_same_target_vote": round(float(cm.same_target_vote_rate(finished)), 4),
            }
            if self.extra_metrics is not None:
                row.update(
                    {k: round(float(v), 5) for k, v in self.extra_metrics(finished).items()}
                )
            res.history.append(row)
            if update % log_every == 0 or update == self.total_updates - 1:
                print(
                    f"  upd {update:4d} | eps {self.episodes_done:7d} | "
                    f"F_E {summary['false_ejection_rate']:.3f} | {outcome_label} {summary['coalition_win_rate']:.3f} | "
                    f"ent {logs.get('entropy', float('nan')):.3f} | {row['wall_s']}s"
                )
            if eval_every and (update + 1) % eval_every == 0:
                ev = self.evaluate(int(self.algo.get("eval_episodes", 1000)))
                ev["update"] = update
                ev["updates_completed"] = self.updates_done
                res.eval_curve.append(ev)
            if self.run_dir and ckpt_every and (update + 1) % ckpt_every == 0:
                self.save(self.run_dir / f"checkpoint_{update + 1}.pt")
            if self.run_dir and ((eval_every and (update + 1) % eval_every == 0)
                                 or (ckpt_every and (update + 1) % ckpt_every == 0)):
                self._write_history(res)
        res.final_metrics = self.evaluate(int(self.algo.get("eval_episodes", 2000)))
        if self.run_dir:
            res.checkpoint = str(self.save(self.run_dir / "checkpoint_final.pt"))
            self._write_history(res)
        return res

    def _write_history(self, result: TrainResult) -> None:
        path = self.run_dir / "history.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"history": result.history, "eval_curve": result.eval_curve,
                                         "final_metrics": result.final_metrics,
                                         "updates_completed": self.updates_done}, indent=2))
        temporary.replace(path)

    # -- evaluation ------------------------------------------------------
    def eval_policy(self) -> BasePolicy:
        """The policy as it is actually played: role-routed, with the frozen partner if any."""
        from social_collusion.policies.base import CreatorRouter, RoleRouter

        if self.learner_role == Role.COALITION:
            if self.frozen_partner is not None:
                return CreatorRouter(self.eval_opponent, self.policy, self.frozen_partner)
            return RoleRouter(self.eval_opponent, self.policy)
        return RoleRouter(self.policy, self.eval_opponent)

    def eval_states(self, n_episodes: int, eval_seed: int | None = None) -> list[GameState]:
        # Training starts at episode zero under self.seed. Reusing that seed would
        # evaluate the same evidence/role draws already collected for optimization.
        seed = 987654321 + self.seed if eval_seed is None else eval_seed
        ve = VecEnv(self.cfg, min(self.n_envs, 128), seed=seed)
        return ve.run_episodes(evaluation_copy(self.eval_policy(), seed), n_episodes)

    def evaluate(self, n_episodes: int = 1000, eval_seed: int | None = None) -> dict[str, Any]:
        from social_collusion.metrics import collusion_metrics as cm
        from social_collusion.metrics import communication_metrics as comm
        from social_collusion.metrics import outcome_metrics as om

        states = self.eval_states(n_episodes, eval_seed)
        out: dict[str, Any] = om.summarize(states)
        out.update({k: v for k, v in cm.summarize_all(states).items() if isinstance(v, float)})
        if self.cfg.enable_symbol_channel:
            out.update(comm.summarize(states))
        out["suspect_id_accuracy"] = om.suspect_identification_accuracy(states)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        for k, a in enumerate(self.actors):
            p = path if len(self.actors) == 1 else path.with_name(f"{path.stem}_actor{k}{path.suffix}")
            a.save(p, extra={"seed": self.seed, "algo": self.algo, "updates": self.updates_done,
                             "episodes": self.episodes_done, "env_steps": self.env_steps})
        return path


def make_crew(name: str, cfg: EnvConfig, listener_path: str | Path | None = None) -> BasePolicy:
    """Crew policy for a condition: 'scripted' | 'bayesian' | 'listener' | 'random'."""
    from social_collusion.policies.registry import make_crew as _mk

    if name == "listener":
        from social_collusion.policies.belief_listener import (
            BeliefListenerNet,
            BeliefListenerPolicy,
        )

        if listener_path is None:
            raise ValueError("crew='listener' requires listener_path")
        return BeliefListenerPolicy(BeliefListenerNet.load(listener_path))
    return _mk({"scripted": "truthful", "bayesian": "bayesian", "random": "random"}.get(name, name))


__all__ = [
    "Trainer",
    "TrainResult",
    "RoleLearnerPolicy",
    "CoalitionPolicy",
    "role_rank",
    "coalition_rank",
    "make_crew",
]
