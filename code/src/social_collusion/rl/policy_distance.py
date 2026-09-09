"""Distances between successive policies on a fixed probe set of states.

The multi-generation study (experiments/multigen) needs to say whether the alternating
attacker / defender process converges, oscillates, or keeps moving. Win rates alone cannot
separate "the policy stopped changing" from "the policy changed but the opponent changed too", so
every generation is compared with its predecessor on the *same* fixed set of decision points:

* `kl_divergence`   mean KL(pi_new || pi_old) over the masked categorical heads (summed over
                     heads, averaged over probe rows where an agent of the given role acts);
* `js_divergence`   the symmetric, bounded counterpart (log base 2, in [0, n_heads]);
* `vote_pattern_divergence`  a behavioural distance: Jensen-Shannon divergence between the two
                     policies' vote distributions on VOTE-phase probe rows, after mapping every
                     vote target to a *canonical* category (incident creator / other coalition
                     member / crew / skip), so that it is invariant to the per-episode
                     re-randomisation of agent slots;
* `argmax_disagreement`  fraction of probe rows on which the greedy actions differ.

The probe set is generated once from scripted play (a mixture of coalition strategies plus
random play for coverage) under a fixed seed and reused across generations and training seeds.
Only the actor's forward pass is needed, so scripted policies can also be compared (their
distributions are the one-hot of their deterministic action).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env.action_masks import active_agents, compute_masks, head_sizes
from social_collusion.env.enums import N_HEADS, Head, Phase, Role
from social_collusion.env.observations import build_all_actor_obs
from social_collusion.env.state import GameState
from social_collusion.env.transition import reset as env_reset
from social_collusion.env.transition import transition
from social_collusion.policies.base import BasePolicy
from social_collusion.seeding import episode_rng

VOTE_CATEGORIES = ("creator", "other_coalition", "crew", "skip")
EPS = 1e-8


@dataclass
class ProbeSet:
    cfg: EnvConfig
    obs: np.ndarray  # (N, obs_dim) float32
    masks: list[np.ndarray]  # per head, (N, head_size) bool
    role: np.ndarray  # (N,) role of the acting agent
    phase: np.ndarray  # (N,)
    agent: np.ndarray  # (N,) acting agent index
    #: canonical category of each vote index for this row (-1 = not a legal / meaningful target)
    vote_category: np.ndarray  # (N, n_agents + 1) int
    states: list[GameState] = field(default_factory=list)  # snapshot per row (scripted policies)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def rows(self, role: Role | None = None, phase: Phase | None = None) -> np.ndarray:
        sel = np.ones(len(self), dtype=bool)
        if role is not None:
            sel &= self.role == int(role)
        if phase is not None:
            sel &= self.phase == int(phase)
        return np.flatnonzero(sel)


def _vote_categories(state: GameState, agent: int) -> np.ndarray:
    cfg = state.config
    out = np.full(cfg.n_agents + 1, -1, dtype=np.int64)
    for j in range(cfg.n_agents):
        if j == agent or not state.alive[j]:
            continue
        if j == state.incident_creator:
            out[j] = 0
        elif state.roles[j] == int(Role.COALITION):
            out[j] = 1
        else:
            out[j] = 2
    if cfg.allow_skip_vote:
        out[cfg.n_agents] = 3
    return out


def collect_probe_set(
    cfg: EnvConfig,
    policies: Sequence[BasePolicy],
    seed: int = 4242,
    episodes_per_policy: int = 16,
    keep_states: bool = True,
) -> ProbeSet:
    """Play `episodes_per_policy` episodes under each policy and record every decision point.

    A decision point is (state, active agent). Episode RNGs derive from `seed` exactly as
    `VecEnv` does, so the probe set is reproducible and identical across training seeds.
    """
    obs_rows: list[np.ndarray] = []
    mask_rows: list[list[np.ndarray]] = [[] for _ in range(N_HEADS)]
    roles: list[int] = []
    phases: list[int] = []
    agents: list[int] = []
    cats: list[np.ndarray] = []
    states: list[GameState] = []
    ep = 0
    for pol in policies:
        pol.reset(1)
        for _ in range(episodes_per_policy):
            rng = episode_rng(seed, ep)
            ep += 1
            st = env_reset(cfg, rng, seed=seed)
            guard = 0
            while st.phase != int(Phase.TERMINAL):
                guard += 1
                if guard > 2048:  # pragma: no cover
                    raise RuntimeError("probe episode did not terminate")
                masks = compute_masks(st)
                act = active_agents(st)
                obs = build_all_actor_obs(st)
                for i in np.flatnonzero(act):
                    obs_rows.append(obs[i].copy())
                    for h in range(N_HEADS):
                        mask_rows[h].append(masks[h][i].copy())
                    roles.append(int(st.roles[i]))
                    phases.append(int(st.phase))
                    agents.append(int(i))
                    cats.append(_vote_categories(st, int(i)))
                    if keep_states:
                        states.append(st.copy())
                a = pol.act([st], obs[None], [m[None] for m in masks], act[None], rng)[0]
                st = transition(st, a, rng, inplace=True, validate=False).state
    return ProbeSet(
        cfg=cfg,
        obs=np.stack(obs_rows).astype(np.float32),
        masks=[np.stack(mask_rows[h]).astype(bool) for h in range(N_HEADS)],
        role=np.asarray(roles, dtype=np.int64),
        phase=np.asarray(phases, dtype=np.int64),
        agent=np.asarray(agents, dtype=np.int64),
        vote_category=np.stack(cats),
        states=states,
    )


def default_probe_policies(cfg: EnvConfig) -> list[BasePolicy]:
    """Scripted mixture used for the fixed probe set: covers honest, deceptive and random play."""
    from social_collusion.policies.registry import make_pair

    return [
        make_pair("truthful", "alibi"),
        make_pair("truthful", "framer"),
        make_pair("truthful", "lone_liar"),
        make_pair("truthful", "truthful"),
        make_pair("random", "random"),
    ]


# --------------------------------------------------------------------------------------
# per-head probabilities
# --------------------------------------------------------------------------------------
def torch_head_probs(policy, probe: ProbeSet, batch: int = 2048) -> list[np.ndarray]:
    """Masked per-head action probabilities of a `TorchPolicy` (or RoleLearnerPolicy) on the probe."""
    import torch

    actor_pol = getattr(policy, "actors", None)
    tp = actor_pol[0] if actor_pol else policy  # RoleLearnerPolicy -> first (shared) actor
    sizes = tp.head_sizes
    out = [np.zeros((len(probe), s), dtype=np.float64) for s in sizes]
    with torch.no_grad():
        for start in range(0, len(probe), batch):
            sl = slice(start, min(start + batch, len(probe)))
            o = torch.as_tensor(probe.obs[sl], dtype=torch.float32, device=tp.device)
            logits, _ = tp.model.actor.logits(o, None)
            for h, lg in enumerate(torch.split(logits, sizes, dim=-1)):
                m = torch.as_tensor(probe.masks[h][sl], dtype=torch.bool, device=tp.device)
                lg = lg.masked_fill(~m, -1e9)
                out[h][sl] = torch.softmax(lg, dim=-1).cpu().numpy()
    return out


def scripted_head_probs(policy: BasePolicy, probe: ProbeSet, seed: int = 0) -> list[np.ndarray]:
    """One-hot per-head distributions of a scripted policy's action on each probe row."""
    if not probe.states:
        raise ValueError("scripted_head_probs needs a probe set collected with keep_states=True")
    sizes = head_sizes(probe.cfg)
    out = [np.zeros((len(probe), s), dtype=np.float64) for s in sizes]
    rng = np.random.default_rng(seed)
    for r, st in enumerate(probe.states):
        masks = [probe.masks[h][r] for h in range(N_HEADS)]
        a = policy.act_single(st, int(probe.agent[r]), masks, rng)
        for h in range(N_HEADS):
            out[h][r, int(a[h])] = 1.0
    return out


def head_probs(policy: BasePolicy, probe: ProbeSet) -> list[np.ndarray]:
    if hasattr(policy, "actors") or hasattr(policy, "model"):
        return torch_head_probs(policy, probe)
    return scripted_head_probs(policy, probe)


# --------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------
def _kl_rows(p: np.ndarray, q: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.where(m, p, 0.0)
    q = np.where(m, np.clip(q, EPS, None), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(np.clip(p, EPS, None)) - np.log(q)), 0.0)
    return terms.sum(axis=-1)


def kl_divergence(P: list[np.ndarray], Q: list[np.ndarray], probe: ProbeSet, rows=None) -> np.ndarray:
    """Per-row KL(P || Q) summed over heads (heads with one legal action contribute 0)."""
    rows = np.arange(len(probe)) if rows is None else np.asarray(rows)
    out = np.zeros(rows.size, dtype=np.float64)
    for h in range(N_HEADS):
        out += _kl_rows(P[h][rows], Q[h][rows], probe.masks[h][rows])
    return out


def js_divergence(P: list[np.ndarray], Q: list[np.ndarray], probe: ProbeSet, rows=None) -> np.ndarray:
    """Per-row Jensen-Shannon divergence (base 2) summed over heads; each head lies in [0, 1]."""
    rows = np.arange(len(probe)) if rows is None else np.asarray(rows)
    out = np.zeros(rows.size, dtype=np.float64)
    for h in range(N_HEADS):
        m = probe.masks[h][rows]
        p, q = P[h][rows], Q[h][rows]
        mid = 0.5 * (p + q)
        out += 0.5 * (_kl_rows(p, mid, m) + _kl_rows(q, mid, m)) / np.log(2.0)
    return out


def entropy(P: list[np.ndarray], probe: ProbeSet, rows=None) -> np.ndarray:
    """Per-row policy entropy (nats) summed over heads."""
    rows = np.arange(len(probe)) if rows is None else np.asarray(rows)
    out = np.zeros(rows.size, dtype=np.float64)
    for h in range(N_HEADS):
        p = np.where(probe.masks[h][rows], P[h][rows], 0.0)
        out -= np.where(p > 0, p * np.log(np.clip(p, EPS, None)), 0.0).sum(axis=-1)
    return out


def vote_category_distribution(P: list[np.ndarray], probe: ProbeSet, rows) -> np.ndarray:
    """(len(rows), 4) distribution over canonical vote categories for VOTE-phase rows."""
    rows = np.asarray(rows)
    pv = P[Head.VOTE][rows]
    cats = probe.vote_category[rows]
    out = np.zeros((rows.size, len(VOTE_CATEGORIES)), dtype=np.float64)
    for c in range(len(VOTE_CATEGORIES)):
        out[:, c] = np.where(cats == c, pv, 0.0).sum(axis=-1)
    total = out.sum(axis=-1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def vote_pattern_divergence(
    P: list[np.ndarray], Q: list[np.ndarray], probe: ProbeSet, role: Role
) -> float:
    """Mean JS divergence (base 2) between canonical vote-category distributions on VOTE rows."""
    rows = probe.rows(role=role, phase=Phase.VOTE)
    if rows.size == 0:
        return float("nan")
    a = vote_category_distribution(P, probe, rows)
    b = vote_category_distribution(Q, probe, rows)
    m = np.ones_like(a, dtype=bool)
    mid = 0.5 * (a + b)
    js = 0.5 * (_kl_rows(a, mid, m) + _kl_rows(b, mid, m)) / np.log(2.0)
    return float(js.mean())


def argmax_disagreement(P: list[np.ndarray], Q: list[np.ndarray], probe: ProbeSet, rows=None) -> float:
    """Fraction of probe rows on which the greedy joint action differs on at least one head."""
    rows = np.arange(len(probe)) if rows is None else np.asarray(rows)
    differ = np.zeros(rows.size, dtype=bool)
    for h in range(N_HEADS):
        differ |= P[h][rows].argmax(axis=-1) != Q[h][rows].argmax(axis=-1)
    return float(differ.mean()) if rows.size else float("nan")


def compare_policies(
    new: BasePolicy | list[np.ndarray],
    old: BasePolicy | list[np.ndarray] | None,
    probe: ProbeSet,
    role: Role,
) -> dict[str, float]:
    """All distances between `new` and `old` for the rows where `role` acts, plus new's entropy.

    `new` / `old` may be policies or precomputed head-probability lists; `old=None` yields only the
    entropy block (used for generation 0).
    """
    P = new if isinstance(new, list) else head_probs(new, probe)
    rows = probe.rows(role=role)
    out = {
        "probe_rows": float(rows.size),
        "policy_entropy": float(entropy(P, probe, rows).mean()) if rows.size else float("nan"),
        "vote_entropy": float(
            entropy(P, probe, probe.rows(role=role, phase=Phase.VOTE)).mean()
        ) if probe.rows(role=role, phase=Phase.VOTE).size else float("nan"),
    }
    if old is None:
        return out
    Q = old if isinstance(old, list) else head_probs(old, probe)
    out.update(
        {
            "kl_to_previous": float(kl_divergence(P, Q, probe, rows).mean()),
            "js_to_previous": float(js_divergence(P, Q, probe, rows).mean()),
            "vote_pattern_js": vote_pattern_divergence(P, Q, probe, role),
            "argmax_disagreement": argmax_disagreement(P, Q, probe, rows),
        }
    )
    return out


__all__ = [
    "ProbeSet",
    "VOTE_CATEGORIES",
    "collect_probe_set",
    "default_probe_policies",
    "head_probs",
    "torch_head_probs",
    "scripted_head_probs",
    "kl_divergence",
    "js_divergence",
    "entropy",
    "vote_category_distribution",
    "vote_pattern_divergence",
    "argmax_disagreement",
    "compare_policies",
]
