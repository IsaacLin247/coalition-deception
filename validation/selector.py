#!/usr/bin/env python3
"""Select the prespecified defense from complete, source-pinned development jobs.

No prospective final outcomes are read. Selection ties and capture floors use
exact rational arithmetic on raw episode counts, not rounded displayed rates.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

from jobs import digest
from runner import completed, load_protocol, sha256, utc, verify_source, write_json


def rank_candidates(metadata: dict, counts: dict, definitions: dict) -> dict:
    """Pure prespecified selection over complete pooled development counts."""
    order = metadata["candidate_order"]
    tasks = metadata["task_names"]
    truth = metadata["truthful_task"]
    baseline = metadata["baseline_defense"]
    fallback = metadata["fallback_defense"]
    if (len(order) != len(set(order)) or len(tasks) != len(set(tasks))
            or truth not in tasks or baseline not in order or fallback not in order):
        raise ValueError("invalid candidate/task selection metadata")
    if set(counts) != set(order) or set(definitions) != set(order):
        raise ValueError("selection requires every prespecified candidate")
    raw = {}
    for candidate in order:
        if set(counts[candidate]) != set(tasks):
            raise ValueError(f"incomplete task coverage: {candidate}")
        task_scores = {}
        for task in tasks:
            cell = counts[candidate][task]
            n = cell["episodes"]
            if type(n) is not int or n <= 0:
                raise ValueError("invalid episode denominator")
            if any(type(cell[k]) is not int or not 0 <= cell[k] <= n
                   for k in ("false_ejections", "creator_ejections", "no_ejections", "incidents")):
                raise ValueError("invalid episode count")
            task_scores[task] = {"false_ejection": Fraction(cell["false_ejections"], n),
                                 "creator_capture": Fraction(cell["creator_ejections"], n)}
        raw[candidate] = {
            "tasks": task_scores,
            "truthful_creator_capture": task_scores[truth]["creator_capture"],
            "worst_task_false_ejection": max(x["false_ejection"] for x in task_scores.values()),
            "mean_false_ejection": sum(x["false_ejection"] for x in task_scores.values()) / len(tasks),
            "mean_creator_capture": sum(x["creator_capture"] for x in task_scores.values()) / len(tasks),
        }
    truth_floor = max(Fraction(str(metadata["minimum_truthful_creator_capture"])),
                      raw[baseline]["truthful_creator_capture"]
                      - Fraction(str(metadata["maximum_truthful_capture_loss"])))
    mean_floor = (raw[baseline]["mean_creator_capture"]
                  - Fraction(str(metadata["maximum_overall_capture_loss"])))
    eligible = [c for c in order
                if raw[c]["truthful_creator_capture"] >= truth_floor
                and raw[c]["mean_creator_capture"] >= mean_floor]
    def key(candidate):
        r = raw[candidate]
        return (r["worst_task_false_ejection"], r["mean_false_ejection"],
                -r["truthful_creator_capture"], order.index(candidate))
    chosen = min(eligible, key=key) if eligible else fallback
    candidates = []
    for candidate in order:
        r = raw[candidate]
        candidates.append({
            "candidate": candidate, "defense": definitions[candidate],
            "eligible": candidate in eligible,
            "truthful_capture_gate_pass": r["truthful_creator_capture"] >= truth_floor,
            "overall_capture_gate_pass": r["mean_creator_capture"] >= mean_floor,
            "truthful_creator_capture": float(r["truthful_creator_capture"]),
            "mean_creator_capture": float(r["mean_creator_capture"]),
            "worst_task_false_ejection": float(r["worst_task_false_ejection"]),
            "mean_false_ejection": float(r["mean_false_ejection"]),
            "selection_key_exact": [str(x) for x in key(candidate)],
            "tasks": [{"task": task, **counts[candidate][task],
                       "false_ejection_rate": float(r["tasks"][task]["false_ejection"]),
                       "creator_ejection_rate": float(r["tasks"][task]["creator_capture"])}
                      for task in tasks],
        })
    return {"selected_candidate": chosen, "selected_defense": definitions[chosen],
            "decision": "eligible_minimax" if eligible else "fallback_no_eligible_candidate",
            "eligible_candidates": eligible,
            "truthful_capture_floor": float(truth_floor),
            "overall_capture_floor": float(mean_floor),
            "capture_floors_exact": [str(truth_floor), str(mean_floor)],
            "candidate_scores": candidates,
            "selection_arithmetic": "Exact rational rates from complete raw episode counts"}


def select(protocol_path: Path, results_root: Path) -> dict:
    protocol = load_protocol(protocol_path)
    verify_source(protocol)
    protocol_sha = sha256(protocol_path)
    design = protocol["design"]
    metadata = design["selection"]
    order = metadata["candidate_order"]
    tasks = metadata["task_names"]
    devjobs = [j for j in design["jobs"] if j["phase"] == "development"]
    smoke_only = design.get("smoke_only") is True
    if smoke_only:
        expected_seeds = metadata["development_seeds"]
        expected_episodes = metadata["episodes_per_cell"]
        if (not expected_seeds or len(expected_seeds) != len(set(expected_seeds))
                or type(expected_episodes) is not int or expected_episodes <= 0):
            raise ValueError("smoke selection requires explicit distinct seeds and episode count")
    else:
        expected_seeds = list(range(100, 108))
        expected_episodes = 1000
        if (metadata.get("development_seeds", expected_seeds) != expected_seeds
                or metadata.get("episodes_per_cell", expected_episodes) != expected_episodes):
            raise ValueError("production selection must retain the fixed eight seeds and 1000 episodes")
    if (len(devjobs) != len(expected_seeds)
            or sorted(j["seed"] for j in devjobs) != sorted(expected_seeds)):
        raise ValueError("development jobs do not match the complete prespecified seed set")
    if any(j["kind"] != "static" or j["episodes"] != expected_episodes for j in devjobs):
        raise ValueError("selection requires static development cells with the exact planned episode count")
    counts = {c: {task: dict(episodes=0, false_ejections=0, creator_ejections=0,
                            no_ejections=0, incidents=0) for task in tasks} for c in order}
    definitions = None
    batches, inputs = [], []
    for job in sorted(devjobs, key=lambda j: j["seed"]):
        directory = results_root / job["name"]
        if not completed(job, directory, protocol_sha, None, deep=True):
            raise ValueError(f"development job is incomplete or failed provenance checks: {job['name']}")
        marker_path = directory / "complete.json"
        marker = json.loads(marker_path.read_text())
        if marker["source_sha256"] != protocol["source_sha256"]:
            raise ValueError("development completion source differs from protocol")
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        for key, wanted in (("phase", "development"), ("seed", job["seed"]),
                            ("source_sha256", protocol["source_sha256"]),
                            ("protocol_sha256", protocol_sha)):
            if result.get(key) != wanted:
                raise ValueError(f"development result {key} mismatch")
        here = {spec["name"]: spec for spec in job["defenses"] if spec["name"] in order}
        if set(here) != set(order) or (definitions is not None and definitions != here):
            raise ValueError("candidate factory definitions differ between development jobs")
        definitions = here
        if {o["name"] for o in job["opponents"]} != set(tasks):
            raise ValueError("development opponents differ from prespecified task set")
        seen = set()
        for entry in result["evaluations"]:
            candidate, task = entry["defense"], entry["opponent"]
            if candidate not in order:
                continue  # Optional contextual mean/soft baselines are not candidates.
            if task not in tasks or (candidate, task) in seen:
                raise ValueError("duplicate or unexpected selection cell")
            seen.add((candidate, task))
            cell_path = directory / entry["path"]
            if not cell_path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("evaluation path leaves its job directory")
            cell = json.loads(cell_path.read_text())
            if entry["summary"] != cell["summary"]:
                raise ValueError("result and raw cell summaries disagree")
            episodes = cell["episodes"]
            counts_here = dict(episodes=len(episodes), false_ejections=0,
                               creator_ejections=0, no_ejections=0, incidents=0)
            for episode in episodes:
                for metric, target in (("false_ejection", "false_ejections"),
                                       ("creator_ejected", "creator_ejections"),
                                       ("no_ejection", "no_ejections"),
                                       ("had_incident", "incidents")):
                    if type(episode.get(metric)) is not bool:
                        raise ValueError(f"missing boolean raw {metric}")
                    counts_here[target] += int(episode[metric])
                if episode["creator_ejected"] and not episode["had_incident"]:
                    raise ValueError("creator capture without an incident")
                if episode["no_ejection"] and (episode["false_ejection"] or episode["creator_ejected"]):
                    raise ValueError("inconsistent ejection outcomes")
            for metric, total in (("false_ejection_rate", "false_ejections"),
                                  ("creator_ejection_rate", "creator_ejections"),
                                  ("no_ejection_rate", "no_ejections")):
                expected = counts_here[total] / counts_here["episodes"]
                reported = cell["summary"][metric]
                if (type(reported) not in (int, float) or not math.isfinite(reported)
                        or abs(reported - expected) > 1e-12):
                    raise ValueError(f"raw count does not reproduce {metric}")
            for k, value in counts_here.items():
                counts[candidate][task][k] += value
            batches.append({"job": job["name"], "seed": job["seed"],
                            "candidate": candidate, "task": task,
                            "environment": entry.get("environment", cell.get("environment")),
                            **counts_here})
            inputs.append({"path": f"{job['name']}/{entry['path']}", "sha256": sha256(cell_path)})
        if seen != {(c, task) for c in order for task in tasks}:
            raise ValueError("selection coverage is incomplete")
        inputs += [{"path": f"{job['name']}/{p.name}", "sha256": sha256(p)}
                   for p in (result_path, marker_path)]
    decision = rank_candidates(metadata, counts, definitions)
    return {"schema_version": 1, "status": "frozen", "smoke_only": smoke_only, "created_utc": utc(),
            "protocol_sha256": protocol_sha, "source_sha256": protocol["source_sha256"],
            "selection_metadata": metadata, "selection_metadata_sha256": digest(metadata),
            "development_seeds": expected_seeds, "input_artifacts": inputs,
            "per_batch_counts": batches, **decision,
            "interpretation": "Development-selected heuristic decision rule; not a calibrated "
                              "posterior or a claim of defense quality. No final outcomes used."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", "--results", dest="results_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.protocol, args.results_root)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        old_content = {k: v for k, v in existing.items() if k != "created_utc"}
        new_content = {k: v for k, v in result.items() if k != "created_utc"}
        if old_content != new_content:
            raise ValueError("existing selection differs from independently reverified development decision")
        result = existing
    else:
        write_json(args.out, result, exclusive=True)
    print(json.dumps({k: result[k] for k in
                      ("status", "selected_candidate", "decision", "eligible_candidates")}, indent=2))


if __name__ == "__main__":
    main()
