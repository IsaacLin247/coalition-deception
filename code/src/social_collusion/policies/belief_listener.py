"""Stage 3: a supervised belief listener - the stable, interpretable audience.

Architecture choice with a scientific reason: the model scores **each candidate independently**
from features expressed *relative to that candidate*, and pools the transcript with a DeepSets
sum/max. So permutation of agent slots is an exact symmetry of the network, not something it has
to learn and might fail at. The plan's acceptance gate "avoid relying on fixed identities" is
therefore satisfied *architecturally*, and `evaluate()` verifies it numerically rather than
taking it on faith. Speaking order is exposed as one scalar feature, so order-dependence remains
measurable (and ablatable) instead of being silently designed away.

The feature extractor is pure NumPy so it can be tested, inspected and used by the scripted
policies without importing torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from social_collusion.env.claims import implied_speaker_room, implied_subject_room
from social_collusion.env.enums import N_HEADS, Head, Phase, ResponseType
from social_collusion.env.knowledge import ObserverKnowledge, scene_window
from social_collusion.env.state import GameState
from social_collusion.policies.base import first_legal
from social_collusion.policies.scripted_crew import ContradictionVoter

N_CLAIM_TYPES = 7
N_RESPONSE_TYPES = 6
CLAIM_FEATURES = N_CLAIM_TYPES + 12
RESPONSE_FEATURES = N_RESPONSE_TYPES + 3
OWN_FEATURES = 7
GLOBAL_FEATURES = 4
FEATURE_DIM = OWN_FEATURES + 2 * CLAIM_FEATURES + 2 * RESPONSE_FEATURES + GLOBAL_FEATURES


def listener_features(state: GameState, listener: int) -> tuple[np.ndarray, np.ndarray]:
    """(features, candidate_mask) with features of shape (n_agents, FEATURE_DIM).

    Everything is computed from `ObserverKnowledge` plus the public transcript, so the listener
    can be dropped into the environment without leaking privileged state.
    """
    cfg = state.config
    n = cfg.n_agents
    k = ObserverKnowledge.build(state, listener)
    window = [t for t in scene_window(state) if t < cfg.n_times]
    marker_room = state.incident_room if state.report_turn >= 0 else -1
    T1 = cfg.n_times

    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)

    # -- own evidence, per candidate
    for j in range(n):
        mask[j] = bool(state.alive[j] and j != listener)
        at_scene = sum(1 for t in window if k.known[t, j] != -1 and int(k.known[t, j]) == marker_room)
        elsewhere = sum(
            1 for t in window if k.known[t, j] != -1 and int(k.known[t, j]) != marker_room
        )
        unknown = sum(1 for t in window if k.known[t, j] == -1)
        seen_any = sum(1 for t in range(T1) if k.known[t, j] != -1)
        feats[j, 0] = float(at_scene > 0)
        feats[j, 1] = float(elsewhere > 0)
        feats[j, 2] = float(unknown == len(window))
        feats[j, 3] = seen_any / max(1, T1)
        feats[j, 4] = float(state.alive[j])
        feats[j, 5] = float(j == listener)
        feats[j, 6] = float(j == state.reporter)

    # -- transcript, pooled per candidate (sum and max)
    off_sum = OWN_FEATURES
    off_max = OWN_FEATURES + CLAIM_FEATURES
    n_claims = max(1, len(state.claims))
    for c in state.claims:
        sp = implied_speaker_room(c)
        sub = implied_subject_room(c)
        caught = float(k.caught_false(c))
        for j in range(n):
            v = np.zeros(CLAIM_FEATURES, dtype=np.float32)
            v[int(c.claim_type)] = 1.0
            o = N_CLAIM_TYPES
            v[o + 0] = float(bool(c.polarity))
            v[o + 1] = float(c.confidence) / max(1, cfg.confidence_levels - 1)
            v[o + 2] = float(c.room is not None and c.room == marker_room)
            v[o + 3] = float(c.time is not None and c.time in window)
            v[o + 4] = float(c.speaker == j)
            v[o + 5] = float(c.subject == j)
            v[o + 6] = float(c.speaker == listener)
            v[o + 7] = float(c.subject == listener)
            if sub is not None and sub[0] == j and sub[1] in window:
                v[o + 8] = float(sub[2] == marker_room)
                v[o + 9] = float(sub[2] != marker_room)
            if sp is not None and c.speaker == j and sp[0] in window:
                v[o + 10] = float(sp[1] == marker_room)
            v[o + 11] = caught * float(c.speaker == j)
            feats[j, off_sum : off_sum + CLAIM_FEATURES] += v / n_claims
            feats[j, off_max : off_max + CLAIM_FEATURES] = np.maximum(
                feats[j, off_max : off_max + CLAIM_FEATURES], v
            )

    # -- responses, pooled per candidate
    off_rs = OWN_FEATURES + 2 * CLAIM_FEATURES
    off_rm = off_rs + RESPONSE_FEATURES
    n_resp = max(1, len(state.responses))
    for r in state.responses:
        for j in range(n):
            v = np.zeros(RESPONSE_FEATURES, dtype=np.float32)
            v[int(r.response_type)] = 1.0
            o = N_RESPONSE_TYPES
            v[o + 0] = float(r.target == j)
            v[o + 1] = float(r.speaker == j)
            v[o + 2] = float(r.speaker == listener)
            feats[j, off_rs : off_rs + RESPONSE_FEATURES] += v / n_resp
            feats[j, off_rm : off_rm + RESPONSE_FEATURES] = np.maximum(
                feats[j, off_rm : off_rm + RESPONSE_FEATURES], v
            )

    # -- global context, broadcast to every candidate
    off_g = off_rs + 2 * RESPONSE_FEATURES
    listener_at_scene = float(
        any(k.known[t, listener] != -1 and int(k.known[t, listener]) == marker_room for t in window)
    )
    feats[:, off_g + 0] = float(state.alive.sum()) / n
    feats[:, off_g + 1] = float(max(state.report_turn, 0)) / max(1, T1)
    feats[:, off_g + 2] = listener_at_scene
    feats[:, off_g + 3] = float(marker_room >= 0)
    return feats, mask


FEATURE_BLOCKS = {
    "own_evidence": slice(0, OWN_FEATURES),
    "transcript": slice(OWN_FEATURES, OWN_FEATURES + 2 * CLAIM_FEATURES),
    "responses": slice(
        OWN_FEATURES + 2 * CLAIM_FEATURES,
        OWN_FEATURES + 2 * CLAIM_FEATURES + 2 * RESPONSE_FEATURES,
    ),
    "global": slice(OWN_FEATURES + 2 * CLAIM_FEATURES + 2 * RESPONSE_FEATURES, FEATURE_DIM),
}


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------
@dataclass
class ListenerDataset:
    X: np.ndarray  # (N, n_agents, FEATURE_DIM)
    mask: np.ndarray  # (N, n_agents)
    y: np.ndarray  # (N,) index of the true incident creator
    meta: list[dict]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def split(self, frac: float, seed: int = 0) -> tuple[ListenerDataset, ListenerDataset]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        cut = int(len(self) * frac)
        a, b = idx[:cut], idx[cut:]
        return (
            ListenerDataset(self.X[a], self.mask[a], self.y[a], [self.meta[i] for i in a]),
            ListenerDataset(self.X[b], self.mask[b], self.y[b], [self.meta[i] for i in b]),
        )


def build_dataset(states: Sequence[GameState], crew_only: bool = True) -> ListenerDataset:
    """One training example per (episode, living listener)."""
    from social_collusion.env.enums import Role

    Xs, Ms, ys, meta = [], [], [], []
    for s in states:
        if s.incident_creator < 0:
            continue
        for j in range(s.config.n_agents):
            if not s.alive[j] or j == s.incident_creator:
                continue
            if crew_only and s.roles[j] != int(Role.CREW):
                continue
            f, m = listener_features(s, j)
            if not m.any():  # pragma: no cover
                continue
            Xs.append(f)
            Ms.append(m)
            ys.append(int(s.incident_creator))
            meta.append({"seed": int(s.seed), "listener": j, "n_claims": len(s.claims)})
    if not Xs:  # pragma: no cover
        return ListenerDataset(np.zeros((0, 0, FEATURE_DIM)), np.zeros((0, 0), bool), np.zeros(0, int), [])
    return ListenerDataset(np.stack(Xs), np.stack(Ms), np.array(ys, dtype=np.int64), meta)


# --------------------------------------------------------------------------------------
# model (torch, imported lazily)
# --------------------------------------------------------------------------------------
def _torch():
    import torch

    return torch


class BeliefListenerNet:
    """Per-candidate scorer. Thin wrapper so the rest of the package never imports torch."""

    def __init__(self, hidden: int = 128, feature_dim: int = FEATURE_DIM, seed: int = 0):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.feature_dim = feature_dim

    def parameters(self):
        return self.net.parameters()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def logits(self, X, mask):
        torch = self.torch
        if not torch.is_tensor(X):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        z = self.net(X).squeeze(-1)
        return z.masked_fill(~mask, -1e9)

    def predict_proba(self, X, mask) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return torch.softmax(self.logits(X, mask), dim=-1).cpu().numpy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {"state_dict": self.net.state_dict(), "feature_dim": self.feature_dim}, path
        )
        return path

    @staticmethod
    def load(path: str | Path, hidden: int = 128) -> BeliefListenerNet:
        torch = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        m = BeliefListenerNet(hidden=hidden, feature_dim=int(blob["feature_dim"]))
        m.net.load_state_dict(blob["state_dict"])
        m.net.eval()
        return m


def train(
    ds: ListenerDataset,
    val: ListenerDataset | None = None,
    hidden: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[BeliefListenerNet, list[dict]]:
    torch = _torch()
    model = BeliefListenerNet(hidden=hidden, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.as_tensor(ds.X, dtype=torch.float32)
    M = torch.as_tensor(ds.mask, dtype=torch.bool)
    y = torch.as_tensor(ds.y, dtype=torch.long)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        model.net.train()
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model.logits(X[idx], M[idx])
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(idx)
        model.net.eval()
        rec = {"epoch": ep, "train_loss": total / max(1, n)}
        if val is not None and len(val):
            p = model.predict_proba(val.X, val.mask)
            rec["val_acc"] = float((p.argmax(axis=1) == val.y).mean())
            rec["val_nll"] = float(-np.log(np.clip(p[np.arange(len(val)), val.y], 1e-12, None)).mean())
        history.append(rec)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return model, history


# --------------------------------------------------------------------------------------
# evaluation (plan Stage 3 acceptance gate)
# --------------------------------------------------------------------------------------
def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def evaluate(model: BeliefListenerNet, ds: ListenerDataset) -> dict[str, float]:
    """Accuracy, calibration, chance floor, and the two required ablations."""
    p = model.predict_proba(ds.X, ds.mask)
    acc = float((p.argmax(axis=1) == ds.y).mean())
    chance = float(np.mean([1.0 / max(1, m.sum()) for m in ds.mask]))
    out = {
        "accuracy": acc,
        "chance": chance,
        "lift_over_chance": acc - chance,
        "nll": float(-np.log(np.clip(p[np.arange(len(ds)), ds.y], 1e-12, None)).mean()),
        "ece": expected_calibration_error(p, ds.y),
        "n": float(len(ds)),
    }
    for name, sl in FEATURE_BLOCKS.items():
        Xa = ds.X.copy()
        Xa[:, :, sl] = 0.0
        pa = model.predict_proba(Xa, ds.mask)
        out[f"accuracy_without_{name}"] = float((pa.argmax(axis=1) == ds.y).mean())
        out[f"drop_without_{name}"] = acc - out[f"accuracy_without_{name}"]
    # identity permutation: an exact symmetry of the architecture, verified numerically
    rng = np.random.default_rng(0)
    perm = np.stack([rng.permutation(ds.X.shape[1]) for _ in range(len(ds))])
    Xp = np.take_along_axis(ds.X, perm[:, :, None], axis=1)
    Mp = np.take_along_axis(ds.mask, perm, axis=1)
    yp = np.array([int(np.flatnonzero(perm[i] == ds.y[i])[0]) for i in range(len(ds))])
    pp = model.predict_proba(Xp, Mp)
    out["accuracy_identity_permuted"] = float((pp.argmax(axis=1) == yp).mean())
    out["identity_permutation_gap"] = acc - out["accuracy_identity_permuted"]
    return out


def acceptance_gate(metrics: dict[str, float]) -> dict[str, bool]:
    """The plan's Stage-3 gate, as explicit booleans."""
    return {
        "beats_chance": metrics["accuracy"] > metrics["chance"] + 0.05,
        "uses_evidence": metrics["drop_without_own_evidence"] > 0.01,
        "uses_transcript": metrics["drop_without_transcript"] > 0.01,
        "identity_robust": abs(metrics["identity_permutation_gap"]) < 0.01,
    }


# --------------------------------------------------------------------------------------
# policy wrapper
# --------------------------------------------------------------------------------------
class BeliefListenerPolicy(ContradictionVoter):
    """Crew policy whose vote comes from the trained listener (conditions E8/E9)."""

    name = "belief_listener"

    def __init__(
        self,
        model: BeliefListenerNet,
        speaker: ContradictionVoter | None = None,
        threshold: float = 0.0,
        **kw,
    ):
        super().__init__(**kw)
        self.model = model
        from social_collusion.policies.scripted_crew import TruthfulCrew

        self.speaker = speaker or TruthfulCrew()
        self.threshold = threshold

    def posterior(self, state: GameState, voter: int) -> np.ndarray:
        f, m = listener_features(state, voter)
        return self.model.predict_proba(f[None], m[None])[0]

    def vote_index(self, state: GameState, voter: int, rng: np.random.Generator) -> int:
        cfg = state.config
        p = self.posterior(state, voter)
        if not np.isfinite(p).any() or p.sum() <= 0:  # pragma: no cover
            return cfg.n_agents
        if self.stochastic:
            return int(rng.choice(len(p), p=p / p.sum()))
        best = int(np.argmax(p))
        if p[best] < self.threshold and cfg.allow_skip_vote:
            return cfg.n_agents
        return best

    def act_single(self, state, agent, masks, rng):
        if Phase(state.phase) == Phase.VOTE:
            out = np.zeros(N_HEADS, dtype=np.int64)
            for h, m in enumerate(masks):
                if not m[0]:
                    out[h] = first_legal(m)
            out[Head.VOTE] = first_legal(masks[Head.VOTE], self.vote_index(state, agent, rng))
            return out
        if Phase(state.phase) == Phase.RESPONSE_ROUND:
            out = np.zeros(N_HEADS, dtype=np.int64)
            for h, m in enumerate(masks):
                if not m[0]:
                    out[h] = first_legal(m)
            p = self.posterior(state, agent)
            best = int(np.argmax(p))
            if p[best] > 1.5 / max(1, int(state.alive.sum()) - 1):
                out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
                out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], best)
            return out
        return self.speaker.act_single(state, agent, masks, rng)


__all__ = [
    "FEATURE_DIM",
    "FEATURE_BLOCKS",
    "listener_features",
    "ListenerDataset",
    "build_dataset",
    "BeliefListenerNet",
    "train",
    "evaluate",
    "acceptance_gate",
    "expected_calibration_error",
    "BeliefListenerPolicy",
]
