#!/usr/bin/env python3
"""Freeze, run and verify the separate post-review validation supplement.

All scientific settings are explicit protocol inputs. Run computational jobs on the
authorized desktop; importing or checking this module performs no simulation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
sys.path.insert(0, str(HERE))
from jobs import canonical_bytes, digest, resolve_job, validate_design


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_manifest(root: Path = ROOT) -> dict[str, str]:
    paths = [root / "code" / "pyproject.toml"]
    for directory in ("src", "scripts", "configs", "experiments"):
        paths.extend(p for p in (root / "code" / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
                     and "__pycache__" not in p.parts)
    # Runtime modules only: editing a report does not change the simulation source.
    paths.extend(p for p in (root / "validation").glob("*.py") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def verify_source(protocol: dict) -> None:
    current = source_manifest()
    if current != protocol["sources"] or digest(current) != protocol["source_sha256"]:
        changed = sorted(k for k in current.keys() | protocol["sources"].keys()
                         if current.get(k) != protocol["sources"].get(k))
        raise ValueError(f"frozen source differs: {changed}")


def verify_inputs(design: dict) -> None:
    for job in design["jobs"]:
        for opponent in job.get("opponents", []):
            if opponent["kind"] == "checkpoint":
                inp = opponent["input"]
                path = ROOT / inp["path"]
                if not path.is_file() or sha256(path) != inp["sha256"]:
                    raise ValueError(f"missing or changed checkpoint input: {inp['path']}")


def freeze(design_path: Path, protocol_path: Path) -> dict:
    design = json.loads(design_path.read_text())
    validate_design(design)
    verify_inputs(design)
    sources = source_manifest()
    protocol = {"schema_version": 1, "created_utc": utc(), "design": design,
                "design_sha256": sha256(design_path), "sources": sources,
                "source_sha256": digest(sources),
                "interpretation": "Prospectively frozen supplement after review; historical data remain exploratory."}
    write_json(protocol_path, protocol, exclusive=True)
    return protocol


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    validate_design(protocol["design"])
    if digest(protocol["sources"]) != protocol["source_sha256"]:
        raise ValueError("invalid source manifest digest")
    return protocol


def load_selection(path: Path | None, protocol_path: Path, protocol: dict, phase: str) -> tuple[dict | None, str | None]:
    if phase != "final":
        return None, None
    if path is None:
        raise ValueError("final jobs require --selection, including a no-selection decision if applicable")
    selection = json.loads(path.read_text())
    if selection.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("selection belongs to another protocol")
    if selection.get("status") != "frozen":
        raise ValueError("selection must be frozen before final jobs")
    if selection.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("selection source differs from frozen protocol")
    return selection, sha256(path)


def json_safe(value):
    """Represent undefined conditional metrics as null; never remove observations."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def artifact_manifest(directory: Path) -> dict[str, dict]:
    return {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()
            and p.name not in ("complete.json", "failed.json") and not p.name.endswith(".tmp")}


def validate_artifacts(job: dict, directory: Path) -> None:
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("result is not complete")
    if result.get("job") != job:
        raise ValueError("result job differs from frozen resolved job")
    records = result.get("evaluations", [])
    expected = (len(job["defenses"]) * len(job["opponents"]) if job["kind"] == "static"
                else len(job["defenses"]) ** 2 if job["kind"] == "adaptive" else 4)
    if len(records) != expected or len({r["cell"] for r in records}) != expected:
        raise ValueError("missing or duplicate evaluation cells")
    if job["kind"] == "static":
        expected_names = {f"{o['name']}__{d['name']}" for o in job["opponents"] for d in job["defenses"]}
    elif job["kind"] == "adaptive":
        expected_names = {f"{c['name']}__{d['name']}" for c in job["defenses"] for d in job["defenses"]}
    else:
        expected_names = {f"{c}__{d}" for c in ("C0", "C1") for d in ("D0", "D1")}
    if {r["cell"] for r in records} != expected_names:
        raise ValueError("evaluation cells differ from planned conditions")
    for record in records:
        if record["path"] != f"{record['cell']}.json":
            raise ValueError("invalid evaluation artifact path")
        data = json.loads((directory / record["path"]).read_text())
        episodes = data["episodes"]
        if len(episodes) != job["episodes"] or [r["episode_index"] for r in episodes] != list(range(job["episodes"])):
            raise ValueError("evaluation episode indices are not the complete planned set")
        if data["summary"]["n_episodes"] != job["episodes"]:
            raise ValueError("summary episode count mismatch")
        if any(r["eval_seed"] != job["eval_seed"] for r in episodes):
            raise ValueError("evaluation seed mismatch")
        if any(r.get("false_ejection") not in (True, False) for r in episodes):
            raise ValueError("missing primary episode outcome")
        fe = sum(r["false_ejection"] for r in episodes) / job["episodes"]
        if abs(fe - data["summary"]["false_ejection_rate"]) > 1e-12:
            raise ValueError("summary differs from raw primary outcomes")
        for summary_key, raw_key in (("creator_ejection_rate", "creator_ejected"),
                                     ("any_false_ejection_rate", "any_false_ejection"),
                                     ("coalition_game_win_rate", "coalition_game_won"),
                                     ("mean_false_ejections", "total_false_ejections"),
                                     ("no_ejection_rate", "no_ejection")):
            if data["summary"].get(summary_key) is not None:
                if any(r.get(raw_key) is None for r in episodes):
                    raise ValueError(f"missing raw outcome: {raw_key}")
                raw_mean = sum(r[raw_key] for r in episodes) / len(episodes)
                if abs(raw_mean - data["summary"][summary_key]) > 1e-12:
                    raise ValueError(f"summary differs from raw outcomes: {summary_key}")
        if "defense_decisions" in data and job["max_rounds"] == 1:
            expected_votes = {(r["episode_index"], voter): int(vote)
                              for r in episodes for voter, vote in enumerate(r["votes"])
                              if vote >= 0 and r["roles"][voter] == 0}
            decisions = data["defense_decisions"]
            observed = {(r["episode_index"], r["voter"]): r for r in decisions}
            if len(observed) != len(decisions) or set(observed) != set(expected_votes):
                raise ValueError("decision records differ from retained honest ballots")
            for key, vote in expected_votes.items():
                row = observed[key]
                target = row["n_agents"] if row["skip"] else row["target"]
                if target != vote or row["eval_seed"] != job["eval_seed"] or "_state_token" in row:
                    raise ValueError("decision audit does not match the actual retained ballot")
    stages = result.get("stages", [])
    expected_stages = 0 if job["kind"] == "static" else len(job["defenses"]) if job["kind"] == "adaptive" else 3
    if len(stages) != expected_stages:
        raise ValueError("missing training stages")
    if job["kind"] == "adaptive":
        expected_stage_seeds = {f"coalition_vs_{d['name']}": job["seed"] for d in job["defenses"]}
    elif job["kind"] == "matched_cycle":
        expected_stage_seeds = {"coalition0_vs_soft": job["seed"],
                                "crew1_vs_coalition0": job["seed"] + 10000,
                                "coalition1_vs_crew1": job["seed"] + 20000}
    else:
        expected_stage_seeds = {}
    if len({s["path"] for s in stages}) != len(stages) or {s["path"] for s in stages} != set(expected_stage_seeds):
        raise ValueError("training stage identities differ from frozen design")
    for stage in stages:
        stage_dir = directory / stage["path"]
        history = json.loads((stage_dir / "history.json").read_text())
        budget = stage["updates"]
        if budget != job["updates"] or stage["seed"] != expected_stage_seeds[stage["path"]]:
            raise ValueError("training stage seed or budget differs from frozen design")
        if history.get("updates_completed") != budget or len(history.get("history", [])) != budget:
            raise ValueError("training did not complete exact frozen update budget")
        for index, row in enumerate(history["history"], 1):
            if row["updates_completed"] != index or row["episodes"] != index * job["rollout_episodes"]:
                raise ValueError("inexact completed training episode or update budget")
        checkpoints = ["checkpoint_0.pt", "checkpoint_final.pt"]
        checkpoints += [f"checkpoint_{u}.pt" for u in range(job["checkpoint_every"], budget + 1, job["checkpoint_every"])]
        for name in checkpoints:
            if not (stage_dir / name).is_file() or not (stage_dir / name).stat().st_size:
                raise ValueError(f"missing checkpoint: {stage['path']}/{name}")


def completed(job: dict, directory: Path, protocol_sha: str, selection_sha: str | None, deep: bool = True) -> bool:
    try:
        marker = json.loads((directory / "complete.json").read_text())
        if (marker["protocol_sha256"] != protocol_sha or marker["selection_sha256"] != selection_sha
                or marker["job_sha256"] != digest(job) or marker["returncode"] != 0):
            return False
        validate_artifacts(job, directory)
        if deep and marker["artifacts"] != artifact_manifest(directory):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def research_imports():
    for sub in ("src", "scripts", "experiments"):
        sys.path.insert(0, str(CODE / sub))


def make_config(job):
    from social_collusion.config import load_env_config
    overrides = dict(n_agents=job["n_agents"], n_coalition=2, max_rounds=job["max_rounds"],
                     vote_aggregation="majority", reputation_accuracy=False, reveal_role_on_eject=False,
                     coalition_objective="survive" if job["max_rounds"] > 1 else "balanced_ejection")
    overrides.update(job.get("env_overrides", {}))
    if "rooms" in overrides:
        overrides["rooms"] = tuple(overrides["rooms"])
    if "edges" in overrides:
        overrides["edges"] = tuple(tuple(edge) for edge in overrides["edges"])
    overrides.pop("adjacency", None)
    return load_env_config(job.get("environment", "full_short_game")).with_(**overrides)


def defense_policy(spec: dict, seed: int, record_decisions: bool = False):
    if spec.get("rule") in ("soft_credibility", "mean"):
        from social_collusion.policies.scripted_crew import TruthfulCrew
        return TruthfulCrew(rule=spec["rule"])
    from defense_policy import make_defense
    policy = make_defense(spec, seed=seed, record_decisions=record_decisions)
    policy.validation_spec = spec
    return policy


def retained_decisions(records: list[dict], states: list, eval_seed: int) -> list[dict]:
    """Join decision records to retained game identities, omitting excess batch slots."""
    retained = {id(state): index for index, state in enumerate(states)}
    decisions = []
    for original in records:
        token = original.get("_state_token")
        if token not in retained:
            continue
        row = {k: v for k, v in original.items() if k != "_state_token"}
        row.update(episode_index=retained[token], eval_seed=eval_seed)
        decisions.append(row)
    return decisions


def evaluate_exact(cfg, crew, coalition, job: dict, cell: str, output: Path) -> dict:
    """Complete every fixed-index vector batch; no fastest-finish inclusion rule."""
    import numpy as np
    from social_collusion.env import VecEnv
    from social_collusion.env.enums import Phase
    from social_collusion.policies.base import RoleRouter
    from social_collusion.rl.torch_policy import evaluation_copy
    from train_f3_matched_cycle import _episode_rows, _round_rows, summarize_pair
    policy = evaluation_copy(RoleRouter(crew, coalition), job["eval_seed"])
    states = VecEnv(cfg, min(job["eval_batch_size"], job["episodes"]), seed=job["eval_seed"]).run_episodes(policy, job["episodes"])
    if len(states) != job["episodes"] or any(s.phase != int(Phase.TERMINAL) for s in states):
        raise RuntimeError("evaluation did not return exactly the planned terminal games")
    rows = _episode_rows(states, cell, job["eval_seed"])
    for row, state in zip(rows, states):
        row.update(roles=np.asarray(state.roles).tolist(), votes=np.asarray(state.votes).tolist(),
                   alive=np.asarray(state.alive).tolist(), ejected=int(state.ejected))
    summary = summarize_pair(states, cfg, cell, job["eval_seed"])
    payload = {"cell": cell, "environment": cfg.to_dict(), "summary": summary, "episodes": rows,
               "meetings": _round_rows(states, cell, job["eval_seed"])}
    # Policies can expose decision-time audit records; the evaluated copy owns them.
    evaluated_crew = getattr(policy, "crew", None)
    if evaluated_crew is not None and hasattr(evaluated_crew, "audit_records"):
        payload["defense_decisions"] = retained_decisions(evaluated_crew.audit_records, states, job["eval_seed"])
    write_json(output, json_safe(payload))
    return {"cell": cell, "path": output.name, "summary": json_safe(summary)}


def training_algo(job: dict) -> dict:
    from social_collusion.config import load_algo_config
    return dict(load_algo_config("ippo"), total_updates=job["updates"], n_envs=job["n_envs"],
                rollout_episodes=job["rollout_episodes"], eval_every=job.get("monitor_every", 0),
                eval_episodes=job["monitor_episodes"], checkpoint_every=job["checkpoint_every"],
                evaluate_initial=True)


def execute_research(job: dict, directory: Path) -> dict:
    research_imports()
    from _common import limit_threads
    from social_collusion.policies.registry import make_coalition
    from social_collusion.policies.scripted_crew import TruthfulCrew
    from social_collusion.env.enums import Role
    from social_collusion.rl.train import Trainer
    from train_f3_matched_cycle import load_role_checkpoint
    limit_threads(1)
    cfg = make_config(job)
    result = {"job": job, "phase": job["phase"], "seed": job["seed"], "status": "running", "evaluations": [], "stages": [],
              "environment": cfg.to_dict(), "started_utc": utc()}
    def evaluate(crew, coalition, cell, evaluation_cfg=None, opponent=None, defense=None):
        print(f"EVALUATE {cell}: {job['episodes']} fixed-index episodes", flush=True)
        record = evaluate_exact(evaluation_cfg or cfg, crew, coalition, job, cell, directory / f"{cell}.json")
        record.update(opponent=opponent, defense=defense)
        result["evaluations"].append(record)
    def train(name, crew, role=Role.COALITION, opponent=None, seed=None):
        stage_dir = directory / name
        stage_seed = job["seed"] if seed is None else seed
        trainer = Trainer(cfg, training_algo(job), crew_policy=crew, seed=stage_seed,
                          learner_role=role, opponent_policy=opponent, run_dir=stage_dir,
                          device=job["device"])
        # Explicit monitored opponent separates stochastic streams from training.
        if role == Role.COALITION and hasattr(crew, "validation_spec"):
            trainer.eval_opponent = defense_policy(crew.validation_spec, 987654321 + stage_seed)
            trainer.eval_crew = trainer.eval_opponent
        print(f"TRAIN {name}: {job['updates']} updates, seed {stage_seed}", flush=True)
        trained = trainer.train(log_every=max(1, job["updates"] // 20))
        result["stages"].append({"path": name, "updates": job["updates"], "seed": stage_seed,
                                 "checkpoint": f"{name}/checkpoint_final.pt"})
        return load_role_checkpoint(trained.checkpoint, role, job["device"])
    if job["kind"] == "static":
        for opponent in job["opponents"]:
            opponent_job = {**job, "environment": opponent.get("environment", job.get("environment", "full_short_game")),
                            "env_overrides": {**job.get("env_overrides", {}), **opponent.get("env_overrides", {})}}
            opponent_cfg = make_config(opponent_job)
            if opponent["kind"] == "scripted":
                coalition = make_coalition(opponent["condition"])
            else:
                inp = opponent["input"]
                checkpoint = ROOT / inp["path"]
                if sha256(checkpoint) != inp["sha256"]:
                    raise ValueError("development attack checkpoint hash mismatch")
                coalition = load_role_checkpoint(checkpoint, Role.COALITION, "cpu")
                if coalition.actors[0].cfg != opponent_cfg:
                    raise ValueError(f"checkpoint environment mismatch for {opponent['name']}")
            for spec in job["defenses"]:
                evaluate(defense_policy(spec, job["eval_seed"], record_decisions=True), coalition,
                         f"{opponent['name']}__{spec['name']}", opponent_cfg,
                         opponent=opponent["name"], defense=spec["name"])
    elif job["kind"] == "adaptive":
        coalitions = {}
        for spec in job["defenses"]:
            coalitions[spec["name"]] = train(f"coalition_vs_{spec['name']}", defense_policy(spec, job["seed"]))
        for trained_vs, coalition in coalitions.items():
            for spec in job["defenses"]:
                evaluate(defense_policy(spec, job["eval_seed"], record_decisions=True), coalition,
                         f"{trained_vs}__{spec['name']}", opponent=trained_vs, defense=spec["name"])
    else:
        c0 = train("coalition0_vs_soft", TruthfulCrew(rule="soft_credibility"))
        d1 = train("crew1_vs_coalition0", None, Role.CREW, c0, job["seed"] + 10000)
        c1 = train("coalition1_vs_crew1", d1, seed=job["seed"] + 20000)
        for cname, coalition in (("C0", c0), ("C1", c1)):
            for dname, crew in (("D0", TruthfulCrew(rule="soft_credibility")), ("D1", d1)):
                evaluate(crew, coalition, f"{cname}__{dname}", opponent=cname, defense=dname)
    result.update(status="complete", completed_utc=utc())
    write_json(directory / "result.json", json_safe(result))
    return result


def run_one(protocol_path: Path, results: Path, name: str, selection_path: Path | None) -> None:
    protocol = load_protocol(protocol_path)
    verify_source(protocol)
    verify_inputs(protocol["design"])
    original = next(j for j in protocol["design"]["jobs"] if j["name"] == name)
    selection, selection_sha = load_selection(selection_path, protocol_path, protocol, original["phase"])
    job = resolve_job(original, selection)
    protocol_sha = sha256(protocol_path)
    directory = results / name
    if completed(job, directory, protocol_sha, selection_sha):
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"refusing to overwrite incomplete run: {directory}; preserve it under _attempts before retry")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run.json", {"job": job, "protocol_sha256": protocol_sha,
               "selection_sha256": selection_sha, "source_sha256": protocol["source_sha256"],
               "started_utc": utc(), "host": socket.gethostname(), "python": sys.version,
               "platform": platform.platform(), "pid": os.getpid()})
    try:
        result = execute_research(job, directory)
        result.update(protocol_sha256=protocol_sha, source_sha256=protocol["source_sha256"], selection_sha256=selection_sha)
        write_json(directory / "result.json", json_safe(result))
        verify_source(protocol)
        validate_artifacts(job, directory)
        write_json(directory / "complete.json", {"job_sha256": digest(job), "protocol_sha256": protocol_sha,
                   "source_sha256": protocol["source_sha256"], "selection_sha256": selection_sha,
                   "returncode": 0, "completed_utc": utc(), "artifacts": artifact_manifest(directory)}, exclusive=True)
    except BaseException as error:
        write_json(directory / "failed.json", {"failed_utc": utc(), "type": type(error).__name__, "error": str(error)})
        raise


def queue(protocol_path: Path, results: Path, phase: str, slots: int, selection_path: Path | None) -> None:
    if slots < 1:
        raise ValueError("slots must be positive")
    protocol = load_protocol(protocol_path)
    verify_source(protocol)
    verify_inputs(protocol["design"])
    selection, selection_sha = load_selection(selection_path, protocol_path, protocol, phase)
    protocol_sha = sha256(protocol_path)
    all_jobs = {j["name"]: j for j in protocol["design"]["jobs"]}
    jobs = [resolve_job(j, selection) for j in all_jobs.values() if j["phase"] == phase]
    controls = results / "_control"
    logs = results / "_logs"
    controls.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    lock = controls / f"{phase}.lock"
    write_json(lock, {"pid": os.getpid(), "hostname": socket.gethostname(), "started_utc": utc()}, exclusive=True)
    pending = [j for j in jobs if not completed(j, results / j["name"], protocol_sha, selection_sha)]
    running, failures = {}, []
    def status():
        write_json(controls / f"{phase}.json", {"updated_utc": utc(), "phase": phase,
                   "protocol_sha256": protocol_sha, "selection_sha256": selection_sha,
                   "total": len(jobs), "completed": len(jobs) - len(pending) - len(running) - len(failures),
                   "pending": [j["name"] for j in pending], "running": [{"name": name, "pid": item[0].pid} for name, item in running.items()],
                   "failed": failures})
    try:
        while pending or running:
            for job in pending[:]:
                if len(running) >= slots:
                    break
                ready = True
                for dep in job.get("dependencies", []):
                    dep_job = all_jobs[dep]
                    dep_selection = selection if dep_job["phase"] == "final" else None
                    if not completed(resolve_job(dep_job, dep_selection), results / dep, protocol_sha,
                                     selection_sha if dep_job["phase"] == "final" else None):
                        ready = False
                if not ready:
                    continue
                verify_source(protocol)
                pending.remove(job)
                command = [sys.executable, "-u", str(HERE / "runner.py"), "job", "--protocol", str(protocol_path),
                           "--results", str(results), "--job", job["name"]]
                if selection_path:
                    command += ["--selection", str(selection_path)]
                log = (logs / f"{job['name']}.log").open("a", buffering=1)
                log.write(json.dumps({"started_utc": utc(), "command": command}) + "\n")
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[job["name"]] = (process, job, log)
                print("START", job["name"], process.pid, flush=True)
            for name, (process, job, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                del running[name]
                if process.returncode != 0 or not completed(job, results / name, protocol_sha, selection_sha):
                    failures.append({"name": name, "returncode": process.returncode})
                    print("FAILED", name, process.returncode, flush=True)
                else:
                    print("COMPLETE", name, flush=True)
            status()
            if pending and not running:
                available = any(all(completed(resolve_job(all_jobs[d], selection if all_jobs[d]["phase"] == "final" else None),
                                              results / d, protocol_sha,
                                              selection_sha if all_jobs[d]["phase"] == "final" else None)
                                    for d in j.get("dependencies", [])) for j in pending)
                if not available:
                    raise RuntimeError("pending jobs have unfulfilled or failed dependencies")
            if running:
                time.sleep(5)
        if failures:
            raise RuntimeError(f"{len(failures)} jobs failed; all seeds remain required")
    finally:
        for process, _, log in running.values():
            process.terminate()
        for process, _, log in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        for name, (process, _, _) in running.items():
            failures.append({"name": name, "returncode": process.returncode, "reason": "queue_interrupted"})
        running.clear()
        status()
        lock.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("freeze", "run", "job", "status", "verify"))
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--design", type=Path)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--phase", choices=("development", "final", "smoke"), default="development")
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--job")
    ap.add_argument("--selection", type=Path)
    args = ap.parse_args()
    args.protocol = args.protocol.resolve()
    if args.action == "freeze":
        if args.design is None:
            ap.error("freeze requires --design")
        protocol = freeze(args.design, args.protocol)
        print(json.dumps({"protocol_sha256": sha256(args.protocol), "source_sha256": protocol["source_sha256"],
                          "jobs": len(protocol["design"]["jobs"])}, indent=2))
        return 0
    if args.results is None:
        ap.error("--results is required")
    args.results = args.results.resolve()
    if args.selection:
        args.selection = args.selection.resolve()
    if args.action == "run":
        queue(args.protocol, args.results, args.phase, args.slots, args.selection)
    elif args.action == "job":
        if not args.job:
            ap.error("job requires --job")
        run_one(args.protocol, args.results, args.job, args.selection)
    else:
        protocol = load_protocol(args.protocol)
        selection, selection_sha = load_selection(args.selection, args.protocol, protocol, args.phase)
        jobs = [resolve_job(j, selection) for j in protocol["design"]["jobs"] if j["phase"] == args.phase]
        checked = {j["name"]: completed(j, args.results / j["name"], sha256(args.protocol), selection_sha,
                                      deep=args.action == "verify") for j in jobs}
        print(json.dumps({"total": len(jobs), "complete": sum(checked.values()),
                          "incomplete": [n for n, valid in checked.items() if not valid], "jobs": checked}, indent=2))
        if args.action == "verify" and not all(checked.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
