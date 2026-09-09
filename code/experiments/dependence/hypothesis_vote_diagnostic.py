#!/usr/bin/env python3
"""How a coalition trained against the hypothesis-elimination crew gets innocents ejected (F8).

For every `counterattack_hyp_n<N>_r1_s*` run, the coalitions trained against soft credibility,
the mean and the hypothesis crew are replayed against each of the three rules, and every meeting
is decomposed:

* how the ballots split (crew votes for the ejected agent, crew skips, coalition votes), and
  whether the ejection was *coalition-decided*: without the two coalition ballots the ejected
  agent would not have held the plurality;
* what the coalition said: false-claim rate and the fraction of no-information claims;
* what the crew could infer: each honest voter's top membership posterior and whether the true
  coalition members hold the top two places.

    python experiments/dependence/hypothesis_vote_diagnostic.py --n-agents 7 --episodes 500

Writes `experiments/dependence/figures/hypothesis_vote_diagnostic_n<N>.{csv,json}`.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENTS_DIR = str(Path(__file__).resolve().parents[1])
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from exp_common import limit_threads, results_root, save_csv, save_json  # noqa: E402
from train_f3_matched_cycle import load_role_checkpoint  # noqa: E402

from social_collusion.config import EnvConfig  # noqa: E402
from social_collusion.env import VecEnv  # noqa: E402
from social_collusion.env.enums import ClaimType, Role, TruthLabel  # noqa: E402
from social_collusion.policies.base import RoleRouter  # noqa: E402
from social_collusion.policies.hypothesis_crew import coalition_posterior, crew_for_rule  # noqa: E402
from social_collusion.rl.torch_policy import evaluation_copy  # noqa: E402

RULE = {"soft": "soft_credibility", "mean": "mean", "hypothesis": "hypothesis"}


def checkpoint_env_config(coalition, n_agents: int) -> EnvConfig:
    """Replay each checkpoint under its saved environment and observation layout.

    In particular, archived checkpoints predate the full private-history block.
    Loading today's YAML for their environment would silently change the condition
    and give their actor an observation of the wrong width.
    """
    cfg = coalition.actors[0].cfg
    if any(actor.cfg != cfg for actor in coalition.actors):
        raise ValueError("Diagnostic requires actors with the same saved environment config")
    if cfg.n_agents != n_agents or cfg.n_coalition != 2 or cfg.max_rounds != 1:
        raise ValueError(
            "Hypothesis vote diagnostic requires a saved single-round, two-member "
            f"coalition environment with n_agents={n_agents}; got "
            f"n_agents={cfg.n_agents}, n_coalition={cfg.n_coalition}, "
            f"max_rounds={cfg.max_rounds}"
        )
    return cfg


def decompose(states) -> dict[str, float]:
    n_meet = n_inc = 0
    fe = fe_inc = fe_noinc = coalition_decided_fe = crew_led_fe = 0
    creator_ej = 0
    crew_skip = crew_ballots = 0
    coal_same = coal_pairs = 0
    coal_claims = coal_false = coal_noinfo = 0
    top_post, both_top2, creator_top = [], [], []
    for terminal in states:
        # Ejection updates alive after ballots have been cast. Reconstruct the voting
        # population from recorded ballots so the ejected player's vote and inference
        # are counted, and creator likelihoods see the pre-ejection population.
        s = terminal.copy()
        voted = s.votes >= 0
        s.alive[voted] = True
        cfg = s.config
        if s.votes.size == 0:
            continue
        n_meet += 1
        had_incident = s.incident_creator >= 0
        n_inc += int(had_incident)
        creator_ej += int(had_incident and s.ejected == s.incident_creator)
        skip = cfg.n_agents
        crew = [i for i in range(cfg.n_agents) if s.roles[i] == int(Role.CREW) and voted[i]]
        coal = [int(i) for i in s.coalition if voted[i]]
        for i in crew:
            if s.votes[i] >= 0:
                crew_ballots += 1
                crew_skip += int(s.votes[i] == skip)
        cv = [int(s.votes[i]) for i in coal if s.votes[i] >= 0]
        if len(cv) == 2:
            coal_pairs += 1
            coal_same += int(cv[0] == cv[1] and cv[0] != skip)
        for c in s.claims:
            if s.roles[c.speaker] == int(Role.COALITION):
                coal_claims += 1
                coal_noinfo += int(c.claim_type == int(ClaimType.NO_INFORMATION))
                coal_false += int(c.label is not None and int(c.label) == int(TruthLabel.FALSE))
        if s.ejected >= 0 and s.roles[s.ejected] == int(Role.CREW):
            fe += 1
            fe_inc += int(had_incident)
            fe_noinc += int(not had_incident)
            # tally without the coalition's ballots
            tally = np.zeros(cfg.n_agents + 1)
            for i in crew:
                if s.votes[i] >= 0:
                    tally[int(s.votes[i])] += 1
            cand = tally[: cfg.n_agents]
            ejected_crew_votes = cand[s.ejected]
            others = np.delete(cand, s.ejected)
            crew_alone_ejects = ejected_crew_votes > 0 and ejected_crew_votes > others.max()
            if crew_alone_ejects:
                crew_led_fe += 1
            else:
                coalition_decided_fe += 1
        for v in crew:
            member, creator = coalition_posterior(s, v)
            if not had_incident:
                top_post.append(float(np.max(member)))
                continue
            order = np.argsort(-member)
            top_post.append(float(member[order[0]]))
            both_top2.append(float(set(int(x) for x in order[:2]) == set(int(x) for x in s.coalition)))
            if creator.sum() > 0:
                creator_top.append(float(int(np.argmax(creator)) == int(s.incident_creator)))
    d = max(n_meet, 1)
    return {
        "meetings": n_meet,
        "incident_rate": n_inc / d,
        "false_ejection_rate": fe / d,
        "false_ejection_with_incident": fe_inc / max(n_inc, 1),
        "false_ejection_without_incident": fe_noinc / max(n_meet - n_inc, 1),
        "creator_ejection_rate": creator_ej / d,
        "creator_ejection_per_incident": creator_ej / max(n_inc, 1),
        "coalition_decided_false_ejections": coalition_decided_fe / max(fe, 1),
        "crew_led_false_ejections": crew_led_fe / max(fe, 1),
        "coalition_decided_fe_rate": coalition_decided_fe / d,
        "crew_skip_rate": crew_skip / max(crew_ballots, 1),
        "coalition_same_target_rate": coal_same / max(coal_pairs, 1),
        "coalition_false_claim_rate": coal_false / max(coal_claims, 1),
        "coalition_no_information_rate": coal_noinfo / max(coal_claims, 1),
        "crew_top_membership_posterior": float(np.mean(top_post)) if top_post else float("nan"),
        "crew_top2_is_coalition": float(np.mean(both_top2)) if both_top2 else float("nan"),
        "crew_creator_posterior_correct": float(np.mean(creator_top)) if creator_top else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-agents", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    limit_threads(args.threads)
    root = args.runs or results_root()
    rows: list[dict[str, Any]] = []
    for d in sorted(glob.glob(str(root / f"counterattack_hyp_n{args.n_agents}_r1_s*"))):
        d = Path(d)
        if not (d / "crossplay.csv").exists():
            continue
        seed = int(d.name.rsplit("_s", 1)[1])
        for trained in ("soft", "mean", "hypothesis"):
            ck = d / f"coalition_vs_{trained}" / "checkpoint_final.pt"
            if not ck.exists():
                continue
            coal = load_role_checkpoint(ck, Role.COALITION, "cpu")
            base = checkpoint_env_config(coal, args.n_agents)
            for against in ("soft", "mean", "hypothesis"):
                crew = crew_for_rule(RULE[against])
                eval_seed = 987654321 + seed
                policy = evaluation_copy(RoleRouter(crew, coal), eval_seed)
                states = VecEnv(base, min(64, args.episodes), seed=eval_seed).run_episodes(policy, args.episodes)
                rec = {"seed": seed, "coalition_trained_vs": trained, "evaluated_vs": against, **decompose(states)}
                rows.append(rec)
                print(f"s{seed} C[{trained:10s}] vs {against:10s}: FE {rec['false_ejection_rate']:.3f} (incident rate {rec['incident_rate']:.2f}; FE with/without incident {rec['false_ejection_with_incident']:.2f}/{rec['false_ejection_without_incident']:.2f}) | coalition-decided {rec['coalition_decided_false_ejections']:.2f} | crew skip {rec['crew_skip_rate']:.2f} | same-target {rec['coalition_same_target_rate']:.2f} | false claims {rec['coalition_false_claim_rate']:.2f} | no-info {rec['coalition_no_information_rate']:.2f} | top posterior {rec['crew_top_membership_posterior']:.2f} | top2=coalition {rec['crew_top2_is_coalition']:.2f}", flush=True)
    out = args.out
    save_csv(rows, out / f"hypothesis_vote_diagnostic_n{args.n_agents}.csv")
    summary: dict[str, Any] = {}
    for trained in ("soft", "mean", "hypothesis"):
        for against in ("soft", "mean", "hypothesis"):
            sel = [r for r in rows if r["coalition_trained_vs"] == trained and r["evaluated_vs"] == against]
            if not sel:
                continue
            summary[f"C[{trained}] vs {against}"] = {
                "n_seeds": len(sel),
                **{k: {"mean": float(np.mean([r[k] for r in sel])), "sd": float(np.std([r[k] for r in sel], ddof=1)) if len(sel) > 1 else 0.0}
                   for k in sel[0] if k not in ("seed", "coalition_trained_vs", "evaluated_vs")},
            }
    save_json(summary, out / f"hypothesis_vote_diagnostic_n{args.n_agents}.json")
    print("\n== means across seeds ==")
    for cell, v in summary.items():
        print(f"{cell:28s} n={v['n_seeds']}: FE {v['false_ejection_rate']['mean']:.3f} (incident {v['incident_rate']['mean']:.2f}; FE with/without {v['false_ejection_with_incident']['mean']:.2f}/{v['false_ejection_without_incident']['mean']:.2f}; creator ejected {v['creator_ejection_rate']['mean']:.2f}) | coalition-decided share {v['coalition_decided_false_ejections']['mean']:.2f} | crew skip {v['crew_skip_rate']['mean']:.2f} | same-target {v['coalition_same_target_rate']['mean']:.2f} | false claims {v['coalition_false_claim_rate']['mean']:.2f} | no-info {v['coalition_no_information_rate']['mean']:.2f} | top posterior {v['crew_top_membership_posterior']['mean']:.2f} | top2=coalition {v['crew_top2_is_coalition']['mean']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
