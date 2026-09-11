"""Structural checks for the separately frozen defense validation supplement.

Scientific budgets and contrasts belong in the design JSON, never in queue defaults.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = 1
KINDS = {"static", "adaptive", "matched_cycle"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_design(design: dict) -> None:
    if design.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported design schema_version")
    if not isinstance(design.get("study"), str) or not design["study"]:
        raise ValueError("design requires a study identifier")
    jobs = design.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("design requires an explicit nonempty jobs list")
    names = []
    for job in jobs:
        name = job.get("name", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"unsafe job name: {name!r}")
        names.append(name)
        if job.get("kind") not in KINDS or job.get("phase") not in ("development", "final", "smoke"):
            raise ValueError(f"{name}: unknown kind or phase")
        for key in ("seed", "eval_seed", "n_agents", "max_rounds", "episodes", "eval_batch_size"):
            if not isinstance(job.get(key), int) or job[key] < (0 if key.endswith("seed") else 1):
                raise ValueError(f"{name}: invalid {key}")
        if job["n_agents"] < 5:
            raise ValueError(f"{name}: at least five agents required")
        if job["kind"] in ("static", "adaptive"):
            defenses = job.get("defenses", [])
            if not defenses or any(not isinstance(d, dict) or not re.fullmatch(r"[a-zA-Z0-9_-]+", d.get("name", "")) for d in defenses):
                raise ValueError(f"{name}: explicit safely named defense specifications required")
            if len({d["name"] for d in defenses}) != len(defenses):
                raise ValueError(f"{name}: duplicate defense names")
        if job["kind"] == "static":
            opponents = job.get("opponents", [])
            if not opponents or len({o.get("name") for o in opponents}) != len(opponents):
                raise ValueError(f"{name}: unique explicit opponents required")
            for opponent in opponents:
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", opponent.get("name", "")):
                    raise ValueError(f"{name}: unsafe opponent name")
                if opponent.get("kind") == "scripted":
                    if opponent.get("condition") not in ("truthful", "lone_liar", "alibi", "framer"):
                        raise ValueError(f"{name}: unknown scripted condition")
                elif opponent.get("kind") == "checkpoint":
                    inp = opponent.get("input", {})
                    if not isinstance(inp.get("path"), str) or not re.fullmatch(r"[a-f0-9]{64}", inp.get("sha256", "")):
                        raise ValueError(f"{name}: pinned checkpoint input required")
                    if inp["path"].startswith(("/", "\\")) or ".." in inp["path"].replace("\\", "/").split("/"):
                        raise ValueError(f"{name}: checkpoint path must stay inside package")
                else:
                    raise ValueError(f"{name}: unknown opponent kind")
        if job["kind"] != "static":
            for key in ("updates", "n_envs", "rollout_episodes", "checkpoint_every", "monitor_episodes"):
                if not isinstance(job.get(key), int) or job[key] <= 0:
                    raise ValueError(f"{name}: positive {key} required")
            if job["rollout_episodes"] % job["n_envs"]:
                raise ValueError(f"{name}: rollout episodes must be exact whole vector batches")
            if job["updates"] % job["checkpoint_every"]:
                raise ValueError(f"{name}: checkpoint interval must divide update budget")
            if job.get("device") not in ("cpu", "cuda"):
                raise ValueError(f"{name}: device must be explicit")
        if job.get("environment", "full_short_game") not in ("full_short_game", "meeting_only"):
            raise ValueError(f"{name}: unsupported environment")
    if len(set(names)) != len(names):
        raise ValueError("duplicate job names")
    by_name = {j["name"]: j for j in jobs}
    for job in jobs:
        if any(dep not in by_name for dep in job.get("dependencies", [])):
            raise ValueError(f"{job['name']}: unknown dependency")
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting:
            raise ValueError("cyclic job dependencies")
        if name in visited:
            return
        visiting.add(name)
        for dep in by_name[name].get("dependencies", []):
            visit(dep)
        visiting.remove(name)
        visited.add(name)
    for name in names:
        visit(name)


def resolve_job(job: dict, selection: dict | None) -> dict:
    """Resolve an explicitly marked selected defense without changing job identity.

The selector must write ``selected_defense`` as a complete factory specification.
The named output slot is kept stable for predeclared statistical contrasts.
"""
    resolved = json.loads(json.dumps(job))
    for index, defense in enumerate(resolved.get("defenses", [])):
        if defense.get("selection"):
            if selection is None or not isinstance(selection.get("selected_defense"), dict):
                raise ValueError("job requires a frozen selected_defense")
            if set(defense) - {"name", "selection"}:
                raise ValueError("selected defense placeholder cannot override selected parameters")
            resolved["defenses"][index] = {**selection["selected_defense"], "name": defense["name"]}
    return resolved


def build_design(inputs: dict, metadata: dict | None = None) -> dict:
    """Materialize the agreed supplement; no outcome-dependent defaults.

The original published seeds are used only as fixed development opponents. All
final learner and evidence seeds are fresh and every listed seed is required.
"""
    candidates = [dict(name=f"hypothesis_{tie}_{round(threshold * 100):03d}",
                       tie_break=tie, threshold=threshold, honesty=0.5)
                  for tie in ("index", "random", "skip") for threshold in (0.35, 0.50, 0.65, 0.80)]
    baseline = next(d for d in candidates if d["name"] == "hypothesis_index_050")
    random_tie = next(d for d in candidates if d["name"] == "hypothesis_random_050")
    skip_tie = next(d for d in candidates if d["name"] == "hypothesis_skip_050")
    references = [{"name": "soft", "rule": "soft_credibility"}, {"name": "mean", "rule": "mean"}]
    scripts = [dict(name=name, kind="scripted", condition=name, environment="meeting_only",
                    env_overrides={"coalition_objective": "false_ejection"})
               for name in ("truthful", "lone_liar", "alibi", "framer")]
    common = dict(n_agents=7, max_rounds=1, episodes=1000, eval_batch_size=32,
                  environment="full_short_game", env_overrides={})
    training = dict(updates=400, n_envs=32, rollout_episodes=32,
                    checkpoint_every=50, monitor_every=50, monitor_episodes=200, device="cpu")
    jobs = []
    for seed in range(100, 108):
        attacks = [a for a in inputs["attacks"] if a["development_seed"] == seed]
        if len(attacks) != 3 or {a["name"] for a in attacks} != {"legacy_mean", "legacy_soft", "legacy_hypothesis"}:
            raise ValueError(f"development seed {seed}: exactly three expected pinned attacks required")
        learned = [dict(name=a["name"].replace("legacy_", "trained_"), kind="checkpoint",
                        input={"path": a["path"], "sha256": a["sha256"]},
                        environment=a["environment"], env_overrides=a["env_overrides"])
                   for a in sorted(attacks, key=lambda a: ("legacy_mean", "legacy_soft", "legacy_hypothesis").index(a["name"]))]
        jobs.append(dict(**common, name=f"dev_s{seed}", kind="static", phase="development", seed=seed,
                         eval_seed=2987654321 + seed, defenses=candidates + references,
                         opponents=scripts + learned))
    for seed in range(1000, 1072):
        selected = {"name": "selected", "selection": True}
        jobs.append(dict(**{**common, "environment": "meeting_only", "env_overrides": {"coalition_objective": "false_ejection"}},
                         name=f"static_s{seed}", kind="static", phase="final", seed=seed,
                         eval_seed=1987654321 + seed, defenses=[baseline, random_tie, skip_tie, selected] + references, opponents=scripts))
        jobs.append(dict(**{**common, "env_overrides": {"dependence_penalty": 4.0, "dependence_window": 3}},
                         **training, name=f"adaptive_s{seed}", kind="adaptive", phase="final", seed=seed,
                         eval_seed=1987654321 + seed, defenses=[references[0], baseline, random_tie, skip_tie, selected]))
        jobs.append(dict(**{**common, "max_rounds": 8}, **training,
                         name=f"cycle_s{seed}", kind="matched_cycle", phase="final", seed=seed,
                         eval_seed=1987654321 + seed))
    design = dict(schema_version=1, study="post_review_defense_validation_20260911", jobs=jobs,
                  selection={"candidate_order": [d["name"] for d in candidates],
                             "task_names": [s["name"] for s in scripts] + ["trained_mean", "trained_soft", "trained_hypothesis"],
                             "truthful_task": "truthful", "baseline_defense": baseline["name"],
                             "minimum_truthful_creator_capture": .20, "maximum_truthful_capture_loss": .05,
                             "maximum_overall_capture_loss": .05, "fallback_defense": baseline["name"],
                             "aggregation": "equal_seed_then_equal_task", "task_count": 7},
                  analysis={"seed_unit": "independent training seed or scripted evaluation evidence batch",
                            "final_seed_count": 72, "episodes_per_cell": 1000,
                            "inclusion": "All planned seeds; no data-dependent stopping or exclusions.",
                            "final_outcome_blinding": "No final numerical outcome inspection until all 216 final jobs verify."})
    if metadata:
        if {"schema_version", "study", "jobs", "selection"} & metadata.keys():
            raise ValueError("metadata cannot override frozen runtime settings or selection")
        design.update(metadata)
    validate_design(design)
    return design


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Create explicit jobs before freezing any computation.")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata", type=Path)
    args = ap.parse_args()
    value = build_design(json.loads(args.inputs.read_text()), json.loads(args.metadata.read_text()) if args.metadata else None)
    with args.out.open("x", newline="\n") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
