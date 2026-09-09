#!/usr/bin/env python3
"""Which public channels make a pair look dependent? Per-channel agreement rates for coalition
pairs, honest pairs and cross pairs, for scripted and learned coalitions.

The dependence-aware mechanism pools four channels (claim corroboration, vouching, support
responses, same-target votes). This diagnostic separates them so the RESULTS can say *why* the
penalty lands where it lands: e.g. honest co-witnesses genuinely vouch for each other (a dyad the
mechanism cannot distinguish from a manufactured alibi), whereas a learned coalition coordinates
through votes without mutual vouching.

    python experiments/dependence/channel_diagnostic.py --n-agents 7 \
        --checkpoint results/counterattack_n7_r1_s0/coalition_vs_soft/checkpoint_final.pt \
        --checkpoint results/counterattack_n7_r1_s0/coalition_vs_both/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parents[1])
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from exp_common import save_csv  # noqa: E402
from train_f3_matched_cycle import load_role_checkpoint  # noqa: E402

from social_collusion.config import load_env_config  # noqa: E402
from social_collusion.env import VecEnv, dependence  # noqa: E402
from social_collusion.env.claims import ClaimType, is_compatible  # noqa: E402
from social_collusion.env.enums import ResponseType, Role  # noqa: E402
from social_collusion.env.knowledge import scene_window  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.registry import make_coalition  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402

CHANNELS = ("corroboration", "vouching", "support", "votes")


def channel_matrices(state) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-channel (agree, opportunity) matrices, same definitions as env/dependence.py."""
    cfg = state.config
    n = cfg.n_agents
    alive = dependence.voting_population(state)
    living = [i for i in range(n) if alive[i]]
    out = {c: (np.zeros((n, n)), np.zeros((n, n))) for c in CHANNELS}

    def add(ch, a, b, hit):
        ag, op = out[ch]
        op[a, b] += 1
        op[b, a] += 1
        if hit:
            ag[a, b] += 1
            ag[b, a] += 1

    by_speaker: dict[int, list] = {}
    for c in state.claims:
        if c.claim_type != int(ClaimType.NO_INFORMATION):
            by_speaker.setdefault(int(c.speaker), []).append(c)
    speakers = sorted(s for s in by_speaker if alive[s])
    for i, a in enumerate(speakers):
        for b in speakers[i + 1:]:
            add("corroboration", a, b, is_compatible([*by_speaker[a], *by_speaker[b]], cfg))
    marker = state.incident_room if state.report_turn >= 0 else -1
    if marker >= 0 and state.claims:
        window = [t for t in scene_window(state) if t < cfg.n_times]
        vouched = set()
        for c in state.claims:
            j = dependence._vouches_for(c, marker, window)
            if j is not None and j != int(c.speaker):
                vouched.add((min(int(c.speaker), j), max(int(c.speaker), j)))
        for a, b in dependence._pairs(living):
            add("vouching", a, b, (a, b) in vouched)
    if state.responses:
        supported = set()
        for r in state.responses:
            if r.target is not None and r.response_type in (int(ResponseType.DEFEND), int(ResponseType.AGREE_WITH)):
                s, t = int(r.speaker), int(r.target)
                if s != t:
                    supported.add((min(s, t), max(s, t)))
        for a, b in dependence._pairs(living):
            add("support", a, b, (a, b) in supported)
    voters = [i for i in living if state.votes[i] >= 0]
    skip = cfg.n_agents
    for a, b in dependence._pairs(voters):
        va, vb = int(state.votes[a]), int(state.votes[b])
        add("votes", a, b, va == vb and va != skip)
    return out


def summarize(states) -> dict[str, float]:
    acc = {(ch, kind): [0.0, 0.0] for ch in CHANNELS for kind in ("coalition", "crew", "cross")}
    pen = {"coalition": [], "crew": []}
    for s in states:
        mats = channel_matrices(s)
        alive = dependence.voting_population(s)
        coal = [int(i) for i in s.coalition if alive[i]]
        crew = [int(i) for i in s.crew if alive[i]]
        for ch, (ag, op) in mats.items():
            for a, b in dependence._pairs(coal):
                acc[(ch, "coalition")][0] += ag[a, b]
                acc[(ch, "coalition")][1] += op[a, b]
            for a, b in dependence._pairs(crew):
                acc[(ch, "crew")][0] += ag[a, b]
                acc[(ch, "crew")][1] += op[a, b]
            for a in coal:
                for b in crew:
                    acc[(ch, "cross")][0] += ag[a, b]
                    acc[(ch, "cross")][1] += op[a, b]
        p = dependence.agent_penalties(s, include_votes=True)
        pen["coalition"].extend(p[i] for i in coal)
        pen["crew"].extend(p[i] for i in crew)
    out = {f"{ch}_{kind}": (v[0] / v[1] if v[1] else float("nan")) for (ch, kind), v in acc.items()}
    out["penalty_coalition"] = float(np.mean(pen["coalition"]))
    out["penalty_crew"] = float(np.mean(pen["crew"]))
    out.update({f"overall_{k}": v for k, v in dependence.summarize_dependence(states).items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--checkpoint", action="append", default=[], help="coalition checkpoint(s); label = parent dir name")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_env_config("full_short_game").with_(n_agents=args.n_agents, n_coalition=2, max_rounds=1, coalition_objective="balanced_ejection")
    coalitions = {"scripted truthful": make_coalition("truthful"), "scripted alibi": make_coalition("alibi"), "scripted framer": make_coalition("framer")}
    for ck in args.checkpoint:
        coalitions[f"learned {Path(ck).parent.name}"] = load_role_checkpoint(ck, Role.COALITION)
    rows = []
    for name, coal in coalitions.items():
        states = VecEnv(cfg, 64, seed=987654321 + args.seed).run_episodes(RoleRouter(TruthfulCrew(rule="soft_credibility"), coal), args.episodes)
        s = summarize(states)
        rows.append({"coalition": name, "n_agents": args.n_agents, **s})
        print(f"{name:32s} " + " ".join(f"{ch[:5]} C/H/X={s[f'{ch}_coalition']:.2f}/{s[f'{ch}_crew']:.2f}/{s[f'{ch}_cross']:.2f}" for ch in CHANNELS) + f" | penalty C/H={s['penalty_coalition']:.2f}/{s['penalty_crew']:.2f}", flush=True)
    if args.out:
        save_csv(rows, args.out)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
