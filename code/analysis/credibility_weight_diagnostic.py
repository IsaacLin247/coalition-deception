#!/usr/bin/env python3
"""Where does the credibility rule's behaviour come from? A mechanism diagnostic for F2.

Runs the scripted F2 match-ups (meeting-only game, ten crew sizes x conditions) and records,
for every honest voter in every meeting, the quantities the analytical model reasons about:

* the corroboration count (number of other speakers whose claims are jointly satisfiable with
  the speaker's) for coalition speakers and for honest speakers;
* the realised soft-credibility softmax weight placed on the coalition's testimony, against the
  coalition's share of the speakers (the "uniform" weight the mean rule gives it);
* the fraction of speakers with no informative claim, and how much weight the rule leaves them;
* how often a voter personally catches a coalition speaker lying;
* whether the mean and the soft-credibility rule would pick a different most-suspected agent
  from the same record (the decision-relevant difference between the two rules).

It then runs a *rule counterfactual*: the same match-ups with the alibi-witness term of the
suspicion model switched off (a speaker's testimony that places another agent away from the
scene no longer lowers that agent's suspicion). If the manufactured-corroboration exploit of
F2 runs through vouching rather than through the softmax weights, it disappears here.

    python analysis/credibility_weight_diagnostic.py --episodes 500 --out analysis/credibility_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import limit_threads, save_csv  # noqa: E402

from social_collusion.config import load_env_config  # noqa: E402
from social_collusion.env import VecEnv  # noqa: E402
from social_collusion.env.enums import ClaimType, Role  # noqa: E402
from social_collusion.env.knowledge import ObserverKnowledge, scene_window  # noqa: E402
from social_collusion.metrics import outcome_metrics as om  # noqa: E402
from social_collusion.policies import crew_aggregation as ca  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.registry import make_coalition  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402

CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
CREWS = (3, 5, 7)


def voter_records(state, rule_tau: float) -> list[dict[str, float]]:
    # The terminal state has already removed the ejected player. Reconstruct the
    # population at the vote, so an ejected honest voter still contributes and
    # its evidence/candidacy is evaluated before its own ejection.
    state = state.copy()
    state.alive[state.votes >= 0] = True
    cfg = state.config
    coalition = set(int(j) for j in state.coalition)
    out = []
    for voter in range(cfg.n_agents):
        if int(state.roles[voter]) != int(Role.CREW) or not state.alive[voter]:
            continue
        k = ObserverKnowledge.build(state, voter)
        sig, spoke = ca.testimony_signals(state, voter, k)
        cred = ca.speaker_credibility(state, voter, k)
        idx = np.flatnonzero(spoke)
        if idx.size == 0:
            continue
        w = ca._softmax(cred[idx] / rule_tau)
        is_coal = np.array([i in coalition for i in idx])
        informative = np.zeros(cfg.n_agents, dtype=bool)
        caught = np.zeros(cfg.n_agents, dtype=bool)
        for c in state.claims:
            if c.claim_type != int(ClaimType.NO_INFORMATION):
                informative[c.speaker] = True
            if k.caught_false(c):
                caught[c.speaker] = True
        rec = {
            "n_speakers": float(idx.size),
            "coalition_share": float(is_coal.mean()),
            "coalition_weight_soft": float(w[is_coal].sum()) if is_coal.any() else float("nan"),
            "cred_coalition": float(cred[idx][is_coal].mean()) if is_coal.any() else float("nan"),
            "cred_honest": float(cred[idx][~is_coal].mean()) if (~is_coal).any() else float("nan"),
            "uninformative_share": float((~informative[idx]).mean()),
            "weight_on_uninformative": float(w[~informative[idx]].sum()),
            "coalition_caught": float(caught[idx][is_coal].any()) if is_coal.any() else float("nan"),
        }
        own = ca.own_evidence(state, voter, k)
        s_mean = own + ca.aggregate_testimony(sig, spoke, "mean")
        s_soft = own + ca.aggregate_testimony(sig, spoke, "soft_credibility", credibility=cred)
        for s in (s_mean, s_soft):
            s[voter] = -np.inf
            s[~state.alive] = -np.inf
        rec["argmax_differs"] = float(int(np.argmax(s_mean)) != int(np.argmax(s_soft)))
        rec["mean_top_is_coalition"] = float(int(np.argmax(s_mean)) in coalition)
        rec["soft_top_is_coalition"] = float(int(np.argmax(s_soft)) in coalition)
        rec["mean_skips"] = float(float(np.max(s_mean)) < 0.5)
        rec["soft_skips"] = float(float(np.max(s_soft)) < 0.5)
        out.append(rec)
    return out


def self_placement_records(state) -> dict[str, float]:
    """Does the incident creator's own testimony place it in the incident room at the incident time?"""
    from social_collusion.env.claims import implied_speaker_room

    creator = int(state.incident_creator)
    marker = state.incident_room if state.report_turn >= 0 else -1
    window = set(t for t in scene_window(state) if t < state.config.n_times)
    if creator < 0 or marker < 0:
        return {}
    at_scene = 0.0
    spoke = 0.0
    for c in state.claims:
        if int(c.speaker) != creator:
            continue
        spoke = 1.0
        sp = implied_speaker_room(c)
        if sp is not None and sp[0] in window and sp[1] == marker:
            at_scene = 1.0
    return {"creator_spoke": spoke, "creator_self_placed_at_scene": at_scene}


def run_cell(crew: int, condition: str, rule: str, episodes: int, seed: int):
    cfg = load_env_config("meeting_only").with_(n_agents=crew + 2, n_coalition=2, coalition_objective="false_ejection")
    policy = RoleRouter(TruthfulCrew(rule=rule), make_coalition(condition))
    states = VecEnv(cfg, 64, seed=seed).run_episodes(policy, episodes)
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, default=REPO / "analysis" / "credibility_diagnostic")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--weights-only", action="store_true", help="recompute voter weights without rerunning rule counterfactuals")
    args = ap.parse_args()
    limit_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. realised weights and corroboration counts under the soft-credibility crew
    rows = []
    for crew in CREWS:
        for ci, cond in enumerate(CONDITIONS):
            states = run_cell(crew, cond, "soft_credibility", args.episodes, args.seed + 100 * crew + ci)
            recs = [r for s in states for r in voter_records(s, ca.TAU_SOFT)]
            keys = recs[0].keys()
            row = {"crew": crew, "condition": cond, "n_voter_records": len(recs)}
            for key in keys:
                vals = np.array([r[key] for r in recs], dtype=float)
                row[key] = float(np.nanmean(vals))
            sp = [self_placement_records(s) for s in states]
            sp = [d for d in sp if d]
            for key in ("creator_spoke", "creator_self_placed_at_scene"):
                row[key] = float(np.mean([d[key] for d in sp])) if sp else float("nan")
            row["false_ejection_rate"] = om.summarize(states)["false_ejection_rate"]
            rows.append(row)
            print(f"crew {crew} {cond:9s}: cred C={row['cred_coalition']:.2f} H={row['cred_honest']:.2f} "
                  f"| coalition weight {row['coalition_weight_soft']:.3f} vs share {row['coalition_share']:.3f} "
                  f"| uninformative {row['uninformative_share']:.2f} (weight {row['weight_on_uninformative']:.3f}) "
                  f"| argmax differs {row['argmax_differs']:.2f} | FE {row['false_ejection_rate']:.3f}", flush=True)
    save_csv(rows, args.out / "realised_weights.csv")
    weights_metadata = {
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "population": "all pre-ejection honest voters; dead candidates excluded from top picks",
    }
    if args.weights_only:
        # Keep any archived rule-counterfactual results and their provenance;
        # only the realized-weight block was recomputed by this command.
        summary_path = args.out / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary.update(realised_weights=rows, realised_weights_metadata=weights_metadata)
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"saved realized weights -> {args.out}")
        return 0

    # 2. rule counterfactual: alibi-witness term off
    cf_rows = []
    default_w = ca.W_ALIBI_WITNESS
    for vouching in ("on", "off"):
        ca.W_ALIBI_WITNESS = default_w if vouching == "on" else 0.0
        for crew in CREWS:
            for ci, cond in enumerate(CONDITIONS):
                for ri, rule in enumerate(("mean", "median", "soft_credibility")):
                    states = run_cell(crew, cond, rule, args.episodes, args.seed + 100 * crew + ci + 10 * ri)
                    summ = om.summarize(states)
                    cf_rows.append({
                        "vouching": vouching, "crew": crew, "condition": cond, "rule": rule,
                        "false_ejection_rate": summ["false_ejection_rate"],
                        "creator_ejection_rate": summ.get("creator_ejection_rate", float("nan")),
                        "coalition_favorable_rate": summ.get("coalition_favorable_rate", float("nan")),
                    })
                    print(f"vouching {vouching} crew {crew} {cond:9s} {rule:16s} FE={summ['false_ejection_rate']:.3f}", flush=True)
    ca.W_ALIBI_WITNESS = default_w
    save_csv(cf_rows, args.out / "vouching_counterfactual.csv")

    summary = {
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "realised_weights": rows,
        "realised_weights_metadata": weights_metadata,
        "vouching_counterfactual": cf_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
