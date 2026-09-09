#!/usr/bin/env python3
"""Frozen-source F8 replay diagnostic. No training, source mutation, or inferential tests.

Freeze this script/protocol after smoke validation. Production refuses to start unless every
prescribed N=7/N=9 hypothesis job is complete and its provenance/checkpoints validate. The nine
cells share reset draws and per-episode sampling seeds, not policy-dependent spatial histories.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RULES = {"soft": "soft_credibility", "mean": "mean", "hypothesis": "hypothesis"}
SEEDS = {7: tuple(range(10)), 9: tuple(range(5))}
EPISODES = 500
EVIDENCE_BASE = 2987654321
API = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def job_hash(job):
    return digest_json({key: value for key, value in job.items() if key != "worker"})


def required_jobs(study):
    by_name = {job["name"]: job for job in study["jobs"]}
    if len(by_name) != len(study["jobs"]):
        raise ValueError("Duplicate job names in study protocol")
    names = [f"counterattack_hyp_n{n}_r1_s{seed}" for n, seeds in SEEDS.items() for seed in seeds]
    missing = set(names) - by_name.keys()
    if missing:
        raise ValueError(f"Study omits prescribed jobs: {sorted(missing)}")
    jobs = [by_name[name] for name in names]
    for job in jobs:
        expected = {f"coalition_vs_{rule}": 400 for rule in RULES}
        if job["expected_stages"] != expected or job["family"] != "hypothesis":
            raise ValueError(f"Unexpected stage design: {job['name']}")
    return jobs


def verify_source(source, study):
    source = Path(source).resolve()
    manifest = study["sources"]
    if not manifest or digest_json(manifest) != study["source_sha256"]:
        raise ValueError("Study source-manifest fingerprint is invalid")
    expected = set(manifest)
    actual = {"pyproject.toml"}
    for directory in ("src", "scripts", "configs", "experiments"):
        actual.update(path.relative_to(source).as_posix() for path in (source / directory).rglob("*")
                      if path.is_file() and path.suffix in (".py", ".yaml", ".yml")
                      and "__pycache__" not in path.parts)
    if actual != expected:
        raise ValueError(f"Frozen source inventory differs: added={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    for name, expected_hash in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha_file(path) != expected_hash:
            raise ValueError(f"Frozen source mismatch: {name}")


def load_api(source):
    """Import exclusively from the independently verified immutable package."""
    global API
    source = Path(source).resolve()
    if API is not None:
        if API["source"] != source:
            raise ValueError("Cannot switch source packages within one process")
        return API
    if any(name == "social_collusion" or name.startswith("social_collusion.") for name in sys.modules):
        raise ValueError("Start a fresh process; social_collusion was imported before source isolation")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source / "src"), str(source / "scripts"), str(source / "experiments")]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from social_collusion.config import EnvConfig
    from social_collusion.env import build_all_actor_obs, compute_masks, reset, transition
    from social_collusion.env.action_masks import active_agents
    from social_collusion.env.enums import ClaimType, Phase, Role, TruthLabel
    from social_collusion.policies.base import RoleRouter
    from social_collusion.policies.hypothesis_crew import coalition_posterior, crew_for_rule
    from social_collusion.rl.torch_policy import evaluation_copy
    from social_collusion.seeding import episode_rng
    from train_f3_matched_cycle import load_role_checkpoint
    for name, module in list(sys.modules.items()):
        if name == "social_collusion" or name.startswith("social_collusion.") or name in ("train_f3_matched_cycle", "_common"):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError(f"Imported mutable source: {name} from {module.__file__}")
    API = locals().copy()
    return API


def cfg_from_dict(raw):
    raw = dict(raw)
    raw.pop("adjacency", None)
    raw["rooms"] = tuple(raw["rooms"])
    raw["edges"] = tuple(tuple(edge) for edge in raw["edges"])
    return API["EnvConfig"](**raw)


def validate_checkpoint_blob(blob, meta, seed, update, path, reference=None):
    if (blob["env_config"] != meta["env_config"] or blob["extra"]["seed"] != seed
            or blob["extra"]["updates"] != update or blob["extra"]["algo"] != meta["algo_config"]):
        raise ValueError(f"Checkpoint metadata mismatch: {path}")
    if not blob["state_dict"] or not all(bool(value.isfinite().all()) for value in blob["state_dict"].values()):
        raise ValueError(f"Nonfinite/empty checkpoint weights: {path}")
    if reference is not None:
        if blob["state_dict"].keys() != reference["state_dict"].keys() or any(
                value.shape != reference["state_dict"][key].shape or value.dtype != reference["state_dict"][key].dtype
                for key, value in blob["state_dict"].items()):
            raise ValueError(f"Checkpoint tensor keys/shapes/dtypes differ from final: {path}")


def validate_run(run, n, seed, study, job=None, smoke=False):
    """Validate completion, run provenance, saved configuration, budgets and final weights."""
    run = Path(run)
    meta = read_json(run / "experiment_config.json")
    result = read_json(run / "result.json")
    provenance = read_json(run / "runmeta.json")
    args = meta["args"]
    if bool(args.get("smoke")) != smoke:
        raise ValueError(f"Smoke/production input mismatch: {run}")
    if args["n_agents"] != n or args["seed"] != seed or args["defenses"] != ",".join(RULES):
        raise ValueError(f"Unexpected checkpoint experiment: {run}")
    if result.get("status") != "complete" or not (run / "crossplay.csv").is_file():
        raise ValueError(f"Incomplete counterattack run: {run}")
    if result["env_config"] != meta["env_config"] or provenance["env_config"] != meta["env_config"]:
        raise ValueError(f"Run environment metadata disagree: {run}")
    if result["args"] != args or provenance["args"] != args or result["algo_config"] != meta["algo_config"]:
        raise ValueError(f"Run experiment metadata disagree: {run}")
    with (run / "crossplay.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    expected_cells = {(trained, against) for trained in RULES for against in RULES}
    if (len(cells) != 9 or {(r["coalition_trained_vs"], r["defense"]) for r in cells} != expected_cells
            or any(int(row["seed"]) != seed or float(row["n_episodes"]) != (24 if smoke else 1000)
                   or not math.isfinite(float(row["false_ejection_rate"])) for row in cells)):
        raise ValueError(f"Incomplete/incorrect crossplay cells: {run}")
    if not smoke:
        marker = read_json(run.parent / "_control" / f"{job['name']}.complete.json")
        if (marker.get("name") != job["name"] or marker.get("returncode") != 0
                or marker.get("artifacts_valid") is not True
                or marker.get("source_sha256") != study["source_sha256"]
                or marker.get("job_sha256") != job_hash(job)
                or provenance.get("source_sha256") != study["source_sha256"]):
            raise ValueError(f"Completion/provenance fingerprint mismatch: {run}")
        if args.get("name") != job["name"]:
            raise ValueError(f"Run name differs from prescribed job: {run}")
    cfg = cfg_from_dict(meta["env_config"])
    if (cfg.n_agents != n or cfg.n_coalition != 2 or cfg.max_rounds != 1
            or cfg.vote_aggregation != "majority" or cfg.tie_ejects or not cfg.actor_private_history
            or cfg.synthetic_evidence or cfg.reveal_role_on_eject):
        raise ValueError(f"Unexpected replay environment: {run}")
    budget = 3 if smoke else 400
    if meta["algo_config"]["total_updates"] != budget:
        raise ValueError(f"Unexpected update budget: {run}")
    torch = API["torch"]
    records = {}
    for trained in RULES:
        stage = run / f"coalition_vs_{trained}"
        history = read_json(stage / "history.json")
        if (history.get("updates_completed") != budget or len(history.get("history", [])) != budget
                or not history.get("final_metrics")):
            raise ValueError(f"Incomplete history: {stage}")
        final = stage / "checkpoint_final.pt"
        blob = torch.load(final, map_location="cpu", weights_only=True)
        validate_checkpoint_blob(blob, meta, seed, budget, final)
        checkpoint_hashes = {"checkpoint_final.pt": sha_file(final)}
        if not smoke:
            for update in range(job["checkpoint_interval"], budget + 1, job["checkpoint_interval"]):
                path = stage / f"checkpoint_{update}.pt"
                checkpoint_hashes[path.name] = sha_file(path)
                last = torch.load(path, map_location="cpu", weights_only=True)
                validate_checkpoint_blob(last, meta, seed, update, path, reference=blob)
            if last["state_dict"].keys() != blob["state_dict"].keys() or any(
                    not torch.equal(value, last["state_dict"][key]) for key, value in blob["state_dict"].items()):
                raise ValueError(f"Final checkpoint differs from final scheduled weights: {final}")
        actor = API["load_role_checkpoint"](final, API["Role"].COALITION, "cpu")
        if any(a.cfg != cfg for a in actor.actors):
            raise ValueError(f"Checkpoint loader changed saved environment: {final}")
        records[trained] = {"path": str(final.resolve()), "sha256": checkpoint_hashes["checkpoint_final.pt"],
                            "checkpoint_hashes": checkpoint_hashes, "updates": budget,
                            "history_sha256": sha_file(stage / "history.json")}
    return cfg, records


def gate_all(results, study):
    """No globbing, partial-seed analysis, or per-cell skipping is permitted."""
    checked, errors = {}, []
    for job in required_jobs(study):
        name = job["name"]
        n = int(name.split("_n", 1)[1].split("_", 1)[0])
        seed = int(name.rsplit("_s", 1)[1])
        try:
            cfg, checkpoints = validate_run(Path(results) / name, n, seed, study, job=job)
            checked[name] = {"n_agents": n, "seed": seed, "job_sha256": job_hash(job),
                             "config": cfg.to_dict(), "checkpoints": checkpoints}
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("Production diagnostic is gated; all 15 jobs must pass:\n" + "\n".join(errors))
    return checked


def pending_markers(results, study):
    """Cheap dependency check only; the full artifact/checkpoint gate still runs afterward."""
    reasons = {}
    for job in required_jobs(study):
        name = job["name"]
        path = Path(results) / "_control" / f"{name}.complete.json"
        try:
            marker = read_json(path)
            if (marker.get("name") != name or marker.get("returncode") != 0
                    or marker.get("artifacts_valid") is not True
                    or marker.get("source_sha256") != study["source_sha256"]
                    or marker.get("job_sha256") != job_hash(job)):
                reasons[name] = f"Completion marker does not match the frozen job: {path}"
        except (OSError, ValueError) as error:
            reasons[name] = str(error)
    return reasons


def write_status(args, status, **details):
    # This sidecar is intentionally outside --out: waiting must not occupy the result directory.
    path = args.out.resolve().with_name(args.out.name + ".status.json")
    write_json(path, {"status": status, "smoke": args.smoke, "pid": os.getpid(),
                      "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "output": str(args.out.resolve()), "script_sha256": sha_file(__file__),
                      "diagnostic_protocol_sha256": (sha_file(args.protocol)
                                                      if not args.smoke and args.protocol.exists() else None),
                      **details})


def wait_for_markers(args, study):
    start = time.monotonic()
    while True:
        pending = pending_markers(args.results, study)
        if not pending:
            return
        write_status(args, "waiting", missing_jobs=sorted(pending), blocking_reasons=pending,
                     waiting_seconds=time.monotonic()-start)
        print(json.dumps({"status":"waiting", "remaining_jobs":len(pending),
                          "missing_jobs":sorted(pending)}), flush=True)
        time.sleep(args.poll_seconds)


def plurality(votes, n):
    tally = [0] * n
    for vote in votes:
        if 0 <= vote < n:
            tally[vote] += 1
    top = max(tally, default=0)
    winners = [agent for agent, count in enumerate(tally) if count == top]
    return (winners[0] if top > 0 and len(winners) == 1 else -1), tally


def ballot_metrics(roles, votes, ejected):
    """Fixed-ballot necessity and sufficiency are distinct counterfactual questions."""
    n = len(roles)
    crew = [agent for agent in range(n) if roles[agent] == 0 and votes[agent] >= 0]
    coalition = [agent for agent in range(n) if roles[agent] == 1 and votes[agent] >= 0]
    actual, tally = plurality(votes, n)
    if actual != ejected:
        raise ValueError("Recorded ejection disagrees with the complete plurality tally")
    crew_only, crew_tally = plurality([votes[a] for a in crew], n)
    coalition_only, coalition_tally = plurality([votes[a] for a in coalition], n)
    false_ejection = ejected >= 0 and roles[ejected] == 0
    return {"actual_tally": tally, "crew_only_tally": crew_tally, "coalition_only_tally": coalition_tally,
            "crew_only_ejected": crew_only, "coalition_only_ejected": coalition_only,
            "false_ejection": int(false_ejection),
            "necessary_for_observed_false_ejection": int(false_ejection and crew_only != ejected),
            "sufficient_alone_for_observed_false_ejection": int(false_ejection and coalition_only == ejected),
            "crew_only_any_false_ejection": int(crew_only >= 0 and roles[crew_only] == 0),
            "crew_votes_for_ejected": crew_tally[ejected] if ejected >= 0 else 0,
            "coalition_votes_for_ejected": coalition_tally[ejected] if ejected >= 0 else 0,
            "crew_ballots": len(crew), "crew_skips": sum(votes[a] == n for a in crew),
            "coalition_pairs": int(len(coalition) == 2),
            "coalition_same_target": int(len(coalition) == 2 and votes[coalition[0]] == votes[coalition[1]] < n)}


def episode_seeds(n, seed, episode):
    evidence = EVIDENCE_BASE + 10000 * seed + n
    raw = f"vote-diagnostic-actor:{evidence}:{episode}".encode()
    actor = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**61)
    # evaluation_copy multiplies this seed by 100; keep it within the torch seed domain.
    return evidence, actor % (2**56)


def play_and_record(cfg, coalition, against, n, seed, episode):
    a, np = API, API["np"]
    evidence_seed, actor_seed = episode_seeds(n, seed, episode)
    env_rng = a["episode_rng"](evidence_seed, episode)
    policy_rng = np.random.default_rng(actor_seed)
    state = a["reset"](cfg, env_rng, seed=evidence_seed)
    initial_hash = state.state_hash()
    policy = a["evaluation_copy"](a["RoleRouter"](a["crew_for_rule"](RULES[against]), coalition), actor_seed)
    policy.reset(1)
    before_vote = None
    steps = []
    for _ in range(512):
        if state.done:
            break
        if state.phase == int(a["Phase"].VOTE):
            if before_vote is not None:
                raise ValueError("Expected exactly one meeting")
            before_vote = state.copy()
        masks = a["compute_masks"](state)
        active = a["active_agents"](state)
        action = policy.act([state], a["build_all_actor_obs"](state)[None],
                            [mask[None] for mask in masks], active[None], policy_rng)[0]
        steps.append({"phase": int(state.phase), "turn": int(state.turn), "sub_step": int(state.sub_step),
                      "actions": action.tolist()})
        state = a["transition"](state, action, env_rng, inplace=True, validate=True).state
    if not state.done or before_vote is None:
        raise ValueError("Diagnostic episode failed to complete its single meeting")
    roles, votes = state.roles.tolist(), state.votes.tolist()
    scored = []
    for voter in range(n):
        if roles[voter] != int(a["Role"].CREW) or votes[voter] < 0:
            continue
        member, creator = a["coalition_posterior"](before_vote, voter)
        order = sorted(range(n), key=lambda j: (-member[j], j))
        positive = bool(np.any(member))
        if positive and not np.isclose(member.sum(), cfg.n_coalition):
            raise ValueError("Membership scores do not normalize")
        scored.append({"agent": voter, "member_score": member.tolist(), "creator_score": creator.tolist(),
                       "top_membership_score": float(max(member)), "zero_score_view": int(not positive),
                       "top2_is_coalition": int(set(order[:2]) == set(np.flatnonzero(state.roles == 1)))
                       if positive and state.incident_creator >= 0 else None,
                       "creator_top_correct": int(int(np.argmax(creator)) == state.incident_creator)
                       if creator.sum() > 0 else None})
    coal_claims = [claim for claim in state.claims if roles[claim.speaker] == int(a["Role"].COALITION)]
    had_incident = state.incident_creator >= 0
    record = {"episode_index": episode, "evidence_seed": evidence_seed, "actor_seed": actor_seed,
              "initial_state_hash": initial_hash, "roles": roles, "votes": votes, "ejected": int(state.ejected),
              "had_incident": int(had_incident), "creator_ejected": int(had_incident and state.ejected == state.incident_creator),
              "coalition_claims": len(coal_claims),
              "coalition_false_claims": sum(claim.label == int(a["TruthLabel"].FALSE) for claim in coal_claims),
              "coalition_noinfo_claims": sum(claim.claim_type == int(a["ClaimType"].NO_INFORMATION) for claim in coal_claims),
              "scores": scored, "pre_vote_state": before_vote.canonical(),
              "pre_vote_public_events": [event.to_dict() for event in before_vote.events
                                         if event.type in ("AgentEjected", "RoundStarted")],
              "terminal_state_hash": state.state_hash(), "steps": steps,
              **ballot_metrics(roles, votes, int(state.ejected))}
    if state.invalid_action_attempts:
        raise ValueError("Replay attempted invalid actions")
    return record


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def summarize_records(records):
    count = len(records)
    total = lambda key: sum(row[key] for row in records)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    fe, incidents = total("false_ejection"), total("had_incident")
    scores = [score for row in records for score in row["scores"]]
    return {"episodes": count, "incidents": incidents, "false_ejections": fe,
            "crew_ballots": total("crew_ballots"), "coalition_claims": total("coalition_claims"),
            "incident_rate": ratio(incidents, count), "false_ejection_rate": ratio(fe, count),
            "false_ejection_per_incident": ratio(sum(row["false_ejection"] for row in records if row["had_incident"]), incidents),
            "false_ejection_without_incident": ratio(sum(row["false_ejection"] for row in records if not row["had_incident"]), count-incidents),
            "creator_ejection_rate": ratio(total("creator_ejected"), count),
            "creator_ejection_per_incident": ratio(total("creator_ejected"), incidents),
            "coalition_ballots_necessary_share_of_fe": ratio(total("necessary_for_observed_false_ejection"), fe),
            "coalition_ballots_sufficient_alone_share_of_fe": ratio(total("sufficient_alone_for_observed_false_ejection"), fe),
            "crew_only_any_false_ejection_rate": ratio(total("crew_only_any_false_ejection"), count),
            "crew_skip_rate": ratio(total("crew_skips"), total("crew_ballots")),
            "coalition_same_target_rate": ratio(total("coalition_same_target"), total("coalition_pairs")),
            "coalition_false_claim_rate": ratio(total("coalition_false_claims"), total("coalition_claims")),
            "coalition_noinfo_claim_rate": ratio(total("coalition_noinfo_claims"), total("coalition_claims")),
            "crew_top_membership_score": mean_or_none([s["top_membership_score"] for s in scores]),
            "crew_zero_score_view_rate": mean_or_none([s["zero_score_view"] for s in scores]),
            "crew_top2_is_coalition_per_incident": mean_or_none([s["top2_is_coalition"] for s in scores]),
            "crew_creator_score_top_correct": mean_or_none([s["creator_top_correct"] for s in scores])}


def summarize_seeds(rows):
    grouped = {}
    for row in rows:
        key = f"n{row['n_agents']}|{row['trained']}|{row['against']}"
        grouped.setdefault(key, []).append(row)
    result = {}
    for key, cell in sorted(grouped.items()):
        metrics = {}
        for metric in cell[0]["metrics"]:
            values = [r["metrics"][metric] for r in cell if r["metrics"][metric] is not None]
            metrics[metric] = {"mean": mean_or_none(values), "sd": statistics.stdev(values) if len(values)>1 else None,
                               "defined_seeds": len(values), "total_seeds": len(cell)}
        result[key] = {"seeds": sorted(row["seed"] for row in cell), "metrics": metrics}
    return result


def freeze_protocol(args, study):
    path = args.protocol.resolve()
    frozen_script = path.with_name("replicate_vote_diagnostic_frozen.py")
    if path.exists() or frozen_script.exists() or path.with_suffix(".sha256").exists():
        raise ValueError("Refusing to overwrite frozen diagnostic files")
    design = {"kind": "post-audit descriptive F8 ballot replay", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script_sha256": sha_file(__file__), "study_protocol_sha256": sha_file(args.study_protocol),
              "source_sha256": study["source_sha256"], "episodes_per_cell": EPISODES,
              "evidence_seed_base": EVIDENCE_BASE, "seeds": {str(n): list(seeds) for n,seeds in SEEDS.items()},
              "rules": RULES, "required_jobs": {j["name"]: job_hash(j) for j in required_jobs(study)},
              "evaluation": "CPU serial episodes; same reset draws and per-episode actor seeds across nine cells; isolated copies",
              "inference": "Descriptive per-seed ratios and across-seed mean/sample SD only; no p-values or confidence intervals",
              "zero_denominator": "null; across-seed conditional summaries state the number of defined seeds",
              "counterfactuals": "Hold recorded ballots fixed; crew-only necessity and coalition-only sufficiency are distinct"}
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, frozen_script)
    write_json(path, design)
    path.with_suffix(".sha256").write_text(sha_file(path) + "\n")
    print(json.dumps({"protocol": str(path), "sha256": sha_file(path), "script": str(frozen_script)}, indent=2))


def verify_protocol(args, study):
    path = args.protocol
    if sha_file(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Diagnostic protocol hash mismatch")
    plan = read_json(path)
    if (plan["script_sha256"] != sha_file(__file__) or plan["study_protocol_sha256"] != sha_file(args.study_protocol)
            or plan["source_sha256"] != study["source_sha256"] or plan["episodes_per_cell"] != EPISODES
            or plan["required_jobs"] != {j["name"]: job_hash(j) for j in required_jobs(study)}):
        raise ValueError("Diagnostic script/protocol/study mismatch")
    return plan


def run_diagnostic(args, study, checked, episodes):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty diagnostic output directory")
    start = time.monotonic()
    manifest = {"smoke": args.smoke, "script_sha256": sha_file(__file__), "source_sha256": study["source_sha256"],
                "study_protocol_sha256": sha_file(args.study_protocol),
                "diagnostic_protocol_sha256": None if args.smoke else sha_file(args.protocol),
                "episodes_per_cell": episodes, "checked_inputs": checked,
                "python": sys.version, "numpy": API["np"].__version__, "torch": API["torch"].__version__}
    write_json(out / "inputs.json", manifest)
    rows, raw_hashes = [], {}
    for name, run in checked.items():
        n, seed, cfg = run["n_agents"], run["seed"], cfg_from_dict(run["config"])
        reset_hashes = None
        for trained, ck in run["checkpoints"].items():
            if sha_file(ck["path"]) != ck["sha256"]:
                raise ValueError("Checkpoint changed after integrity gate")
            coalition = API["load_role_checkpoint"](ck["path"], API["Role"].COALITION, "cpu")
            original_rng = [a.generator.get_state().clone() for a in coalition.actors]
            for against in RULES:
                cell = f"n{n}_s{seed}_{trained}_vs_{against}"
                records = []
                raw = out / "raw" / f"{cell}.jsonl.gz"
                raw.parent.mkdir(exist_ok=True)
                with gzip.open(raw, "wt", encoding="utf-8") as stream:
                    for episode in range(episodes):
                        record = play_and_record(cfg, coalition, against, n, seed, episode)
                        record.update(n_agents=n, training_seed=seed, trained=trained, against=against,
                                      checkpoint_sha256=ck["sha256"])
                        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                        records.append(record)
                hashes = [record["initial_state_hash"] for record in records]
                if reset_hashes is None:
                    reset_hashes = hashes
                elif reset_hashes != hashes:
                    raise ValueError("Nine-cell common initial evidence check failed")
                for actor, before in zip(coalition.actors, original_rng):
                    if not API["torch"].equal(before, actor.generator.get_state()):
                        raise ValueError("Diagnostic advanced checkpoint policy RNG")
                raw_hashes[str(raw.relative_to(out))] = sha_file(raw)
                metrics = summarize_records(records)
                rows.append({"n_agents": n, "seed": seed, "trained": trained, "against": against,
                             "raw": str(raw.relative_to(out)), "metrics": metrics})
                write_json(out / "per_seed.json", rows)
                print(json.dumps({"cell": cell, "episodes": episodes, "false_ejection_rate": metrics["false_ejection_rate"]}), flush=True)
    write_json(out / "summary.json", {"smoke": args.smoke, "inference": "descriptive mean and sample SD only",
                                       "cells": summarize_seeds(rows)})
    write_json(out / "complete.json", {**manifest, "cells": len(rows), "episodes": episodes*len(rows),
                                         "raw_sha256": raw_hashes, "per_seed_sha256": sha_file(out / "per_seed.json"),
                                         "summary_sha256": sha_file(out / "summary.json"),
                                         "elapsed_seconds": time.monotonic()-start})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "run"))
    parser.add_argument("--source", type=Path, default=HERE.parents[1])
    parser.add_argument("--study-protocol", type=Path, default=HERE.parents[1] / "submission_protocol.json")
    parser.add_argument("--protocol", type=Path, default=HERE / "vote_diagnostic_protocol.json")
    parser.add_argument("--results", type=Path, default=HERE.parents[1] / "results/submission_20260909")
    parser.add_argument("--out", type=Path, default=HERE / "vote_diagnostic_results")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--wait", action="store_true", help="Wait for all matching completion markers before the full production gate")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.wait and (args.action != "run" or args.smoke):
        raise ValueError("--wait is only available for production run")
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("Polling interval must be 1–60 seconds")
    try:
        study = read_json(args.study_protocol)
        verify_source(args.source, study)
        required_jobs(study)
        if args.action == "freeze":
            if args.smoke:
                raise ValueError("Smoke execution must not freeze a production protocol")
            freeze_protocol(args, study)
            return 0
        if not args.smoke:
            verify_protocol(args, study)
            if args.episodes not in (None, EPISODES) or args.smoke_run:
                raise ValueError("Production uses exactly 500 episodes per cell and prescribed jobs")
        if args.action == "run" and args.out.exists() and any(args.out.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty diagnostic output directory")
        if args.wait:
            wait_for_markers(args, study)
            # Dependency waiting may span hours. Recheck immutability before importing code.
            verify_source(args.source, study)
            verify_protocol(args, study)
        load_api(args.source)
        if args.smoke:
            if args.action != "run" or args.smoke_run is None or "smoke" not in args.out.name.lower():
                raise ValueError("Smoke requires run, --smoke-run, and a clearly named smoke output")
            if args.smoke_run.resolve().is_relative_to(args.results.resolve()) or args.out.resolve().is_relative_to(args.results.resolve()):
                raise ValueError("Smoke must not read or write the production results tree")
            meta = read_json(args.smoke_run / "experiment_config.json")
            n, seed = meta["args"]["n_agents"], meta["args"]["seed"]
            cfg, checkpoints = validate_run(args.smoke_run, n, seed, study, smoke=True)
            checked = {"explicit_smoke": {"n_agents": n, "seed": seed, "config": cfg.to_dict(), "checkpoints": checkpoints}}
            episodes = 3 if args.episodes is None else args.episodes
            if not 1 <= episodes <= 20:
                raise ValueError("Smoke is bounded to 1–20 episodes per cell")
        else:
            checked, episodes = gate_all(args.results, study), EPISODES
        if args.action == "check":
            print(json.dumps({"all_required_jobs_valid": len(checked), "ready": True}, indent=2))
            return 0
        write_status(args, "running", validated_jobs=len(checked), episodes_per_cell=episodes)
        run_diagnostic(args, study, checked, episodes)
        write_status(args, "completed", completion_sha256=sha_file(args.out / "complete.json"))
    except Exception as error:
        if args.action == "run":
            write_status(args, "failed", error=str(error), traceback=traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
