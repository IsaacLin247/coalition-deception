#!/usr/bin/env python3
"""Fresh-only replication tables; see audit/submission/ANALYSIS_PLAN.md.

Final output requires every frozen-protocol job and comparison seed. --interim
emits descriptive status only and suppresses all inferential results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np

try:
    from .checkpoint_integrity import validate_checkpoints, validation_code_hashes
except ImportError:
    from checkpoint_integrity import validate_checkpoints, validation_code_hashes

REPO = Path(__file__).resolve().parents[1]
RULES = ("mean", "median", "trimmed", "sharp_credibility", "soft_credibility", "dependence_aware", "hypothesis")
CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
ABLATIONS = ("default", "no_channel", "no_partner", "no_channel_no_partner")
PAIRS = ("coalition0_vs_soft_credibility", "coalition0_vs_learned_crew1",
         "coalition1_vs_learned_crew1", "coalition1_vs_soft_credibility")
METRICS = ("false_ejection_rate", "coalition_favorable_rate", "coalition_game_win_rate",
           "no_ejection_rate", "creator_survival", "creator_ejection_rate", "incident_rate",
           "same_target_vote_rate", "coalition_false_claim_rate", "mean_rounds_played",
           "any_false_ejection_rate", "mean_false_ejections", "false_ejections_per_meeting",
           "any_incident_false_ejection_rate", "mean_incident_false_ejections",
           "false_ejections_per_incident_meeting", "any_incident_free_false_ejection_rate",
           "mean_incident_free_false_ejections", "false_ejections_per_incident_free_meeting")
Key = tuple[str, str, str, str]  # group, table, cell, metric


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def exported_artifact_path(folder, name):
    """Resolve a Windows/POSIX export within its job directory on either host."""
    if not isinstance(name, str) or not name or "\x00" in name or ":" in name:
        raise ValueError(f"Unsafe relative export path: {name!r}")
    parts = name.replace("\\", "/").split("/")
    # Empty components also reject rooted paths and UNC shares. Check before
    # Path normalization so explicit traversal and drive-relative paths fail.
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe relative export path: {name!r}")
    root = Path(folder).resolve()
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Export path escapes job directory: {name!r}")
    return path


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                         for k, v in row.items()} for row in rows)


def finite(value):
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (ValueError, TypeError):
        return None


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return dict(n_seeds=len(values), mean=float(values.mean()) if len(values) else None,
                sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def exact_mean_sign_flip(values):
    d = np.asarray(values, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Sign-flip input must include every finite paired seed")
    if len(d) > 20:
        raise ValueError("Exact enumeration is limited to 20 independent seeds")
    threshold = abs(d.mean()) - 1e-12
    return sum(abs(np.mean(d * signs)) >= threshold
               for signs in itertools.product((-1., 1.), repeat=len(d))) / 2 ** len(d)


def paired_bootstrap(values, identifier, resamples=20000):
    d = np.asarray(values, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Bootstrap requires at least two complete finite paired seeds")
    seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def holm(values):
    """Missing tests retain their place in the declared family as p=1."""
    order = sorted(range(len(values)), key=lambda i: 1. if values[i] is None else values[i])
    adjusted, previous = [None] * len(values), 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(values) - rank) * (1. if values[i] is None else values[i])))
        adjusted[i] = previous if values[i] is not None else None
    return adjusted


def restricted_response_gaps(matrix, episodes):
    """Finite observed-pool gaps with simultaneous bounded-cell MC bands."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        raise ValueError("Response gaps require a complete square finite matrix")
    if (m < 0).any() or (m > 1).any() or episodes <= 0:
        raise ValueError("Response gap MC bands require probabilities and positive episode counts")
    epsilon = float(np.sqrt(np.log(2 * m.size / .05) / (2 * episodes)))
    low, high = np.maximum(0., m - epsilon), np.minimum(1., m + epsilon)
    return [dict(generation=k, gap=float(m[:, k].max() - m[k, :].min()),
                 best_observed_coalition=int(m[:, k].argmax()),
                 best_observed_defender=int(m[k, :].argmin()),
                 mc_low=float(max(0., low[:, k].max() - high[k, :].min())),
                 mc_high=float(min(1., high[:, k].max() - low[k, :].min())),
                 simultaneous_cell_epsilon=epsilon, episodes_per_cell=episodes,
                 pool_size=m.shape[0]) for k in range(m.shape[0])]


def group_seed(name):
    match = re.fullmatch(r"(.+)_s(\d+)", name)
    if not match:
        raise ValueError(f"Unrecognized protocol job name: {name}")
    return match[1], int(match[2])


def scope(group):
    rounds = 8 if group.startswith("f4_") or "_r8" in group else 1
    n = re.search(r"_n(\d+)", group)
    crew = re.search(r"_crew(\d+)", group)
    k = re.search(r"_k(\d+)", group)
    return dict(rounds=rounds, n=int(n[1]) if n else int(crew[1]) + 2,
                k=int(k[1]) if k else None)


def primary(rounds):
    return ("false_ejection_rate",) if rounds == 1 else ("any_false_ejection_rate", "coalition_game_win_rate")


@dataclass
class Comparison:
    family: str
    name: str
    terms: list[tuple[float, Key]]
    seeds: list[int]


def planned_comparisons(jobs):
    groups, families = defaultdict(set), {}
    for job in jobs:
        group, seed = group_seed(job["name"])
        groups[group].add(seed)
        families[group] = job["family"]
    comparisons = []

    def contrast(family, name, terms):
        seeds = sorted(set.intersection(*(groups[key[0]] for _, key in terms)))
        comparisons.append(Comparison(family, name, terms, seeds))

    def pair(family, group, table, a, b, metric, suffix="", other_group=None):
        other = other_group or group
        contrast(family, f"{group}|{suffix or (a+' minus '+b)}|{metric}",
                 [(1., (group, table, a, metric)), (-1., (other, table, b, metric))])

    for group in sorted(groups):
        family, info = families[group], scope(group)
        rounds, k = info["rounds"], info["k"]
        if family == "f1":
            for metric in ("false_ejection_rate", "coalition_favorable_rate"):
                pair("f1_learning", group, "holdout", "final", "initial", metric)
        elif family in ("f2", "static"):
            for condition in CONDITIONS:
                for rule in RULES:
                    cell = f"{condition}|{rule}"
                    if family == "f2" and condition != "truthful":
                        pair("f2_condition_effects", group, "rules", cell, f"truthful|{rule}", "false_ejection_rate")
                    if rule != "soft_credibility" and (family == "static" or condition in ("alibi", "framer")):
                        pair("static_rules" if family == "static" else "f2_rule_effects", group,
                             "rules", cell, f"{condition}|soft_credibility", "false_ejection_rate")
            if family == "static":
                for generation in range(4):
                    for rule, mechanism in [(r, "crew_rule") for r in RULES if r != "soft_credibility"] + [("dependence_aware", "both")]:
                        pair("static_learned", group, "learned", f"C{generation}|{rule}|{mechanism}",
                             f"C{generation}|soft_credibility|crew_rule", "false_ejection_rate")
        elif family in ("f3", "f4"):
            metrics = primary(rounds) + (("mean_false_ejections",) if rounds == 8 else ())
            for metric in metrics:
                for a, b, label in [(PAIRS[0], PAIRS[1], "defender effect"),
                                    (PAIRS[2], PAIRS[1], "adaptation"), (PAIRS[2], PAIRS[0], "net change")]:
                    pair(f"{family}_cycle", group, "crossplay", a, b, metric, label)
        elif family == "budget":
            reference = f"f{3 if rounds == 1 else 4}_tenseed_crew5"
            for metric in primary(rounds):
                for cell in PAIRS[1:3]:
                    pair("budget", group, "crossplay", cell, cell, metric,
                         f"{cell} extended minus 400", reference)
        elif family == "ablation":
            for metric in primary(rounds) + ("same_target_vote_rate", "coalition_false_claim_rate", "creator_survival"):
                for condition in ABLATIONS[1:]:
                    pair("ablation" if metric in primary(rounds) else "ablation_behavior",
                         group, "ablation", condition, "default", metric)
        elif family in ("hypothesis", "counterattack"):
            defenders = ("mean", "hypothesis") if family == "hypothesis" else ("rule", "both")
            label = "hypothesis_adaptation" if family == "hypothesis" else "dependence_counterattack"
            for defense in defenders:
                for a, b, name in [(f"{defense}|{defense}", f"soft|{defense}", "adaptation"),
                                   (f"{defense}|{defense}", "soft|soft", "matched adaptive difference"),
                                   (f"{defense}|soft", "soft|soft", "transfer")]:
                    pair(label, group, "crossplay", a, b, "false_ejection_rate", f"{defense} {name}")
            if family == "hypothesis":
                pair(label, group, "crossplay", "hypothesis|hypothesis", "mean|mean", "false_ejection_rate")
        elif family in ("multigen", "dependence", "reward"):
            if family != "reward":
                for metric in primary(rounds):
                    names = ["matrix_adaptation_gain", "late_minus_early"] + (["parity_gap"] if k >= 4 else [])
                    for name in names:
                        contrast("loop_dynamics", f"{group}|{name}|{metric}", [(1., (group, f"loop_h{k}", name, metric))])
            if family in ("dependence", "reward"):
                reference = f"multigen_none_n7_r{rounds}_k10"
                metrics = primary(rounds) + (("coalition_favorable_rate",) if rounds == 1 else ())
                for metric in metrics:
                    for name in ("c_stage_mean", "d_stage_mean", "matrix_adaptation_gain"):
                        pair("dependence_loop" if family == "dependence" else "reward_controls", group,
                             f"loop_h{k}", name, name, metric, f"{name} minus reference h{k}", reference)
    return comparisons, {g: sorted(s) for g, s in groups.items()}


class Dataset:
    def __init__(self):
        self.cells = defaultdict(dict)
        self.rows, self.gaps = [], []

    def add(self, job, table, cell, metric, value, artifact):
        value = finite(value)
        if value is None:
            return
        group, seed = group_seed(job["name"])
        key = (group, table, cell, metric)
        if seed in self.cells[key]:
            raise ValueError(f"Duplicate seed/cell: {job['name']} {key}")
        self.cells[key][seed] = value
        self.rows.append(dict(job=job["name"], group=group, seed=seed, table=table,
                              cell=cell, metric=metric, value=value, artifact=str(artifact)))

    def metrics(self, job, table, cell, row, artifact):
        for metric in METRICS:
            self.add(job, table, cell, metric, row.get(metric), artifact)


def validate_job(job, root, fingerprint, checkpoint_records=None):
    folder = root / job["name"]
    marker = read_json(root / "_control" / f"{job['name']}.complete.json")
    job_hash = hashlib.sha256(json.dumps({k: v for k, v in job.items() if k != "worker"}, sort_keys=True).encode()).hexdigest()
    if (marker.get("returncode") != 0 or marker.get("source_sha256") != fingerprint
            or marker.get("job_sha256") != job_hash):
        raise ValueError("Completion marker does not match frozen source/success")
    for artifact in job["artifacts"]:
        path = folder / artifact
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing required artifact: {artifact}")
        if path.suffix == ".json":
            read_json(path)
    metadata = read_json(folder / "runmeta.json")
    if metadata.get("source_sha256") != fingerprint:
        raise ValueError("Run provenance does not match frozen source")
    for stage, budget in job.get("expected_stages", {}).items():
        directory = folder / stage
        history = read_json(directory / "history.json")
        if history.get("updates_completed") != budget:
            raise ValueError(f"Stage {stage} did not complete its planned {budget} updates")
        for update in range(job["checkpoint_interval"], budget + 1, job["checkpoint_interval"]):
            if not (directory / f"checkpoint_{update}.pt").is_file():
                raise ValueError(f"Missing retained checkpoint: {stage}/{update}")
    for history_path in folder.rglob("history.json"):
        history = read_json(history_path)
        if not history.get("final_metrics") or not history.get("updates_completed"):
            raise ValueError(f"Unfinished history: {history_path}")
        if not (history_path.parent / "checkpoint_final.pt").is_file():
            raise ValueError(f"Missing final checkpoint: {history_path.parent}")
        if len(history["history"]) != history["updates_completed"]:
            raise ValueError(f"Update count/history length mismatch: {history_path}")
    records = validate_checkpoints(job, folder)
    if checkpoint_records is not None:
        checkpoint_records.extend(records)
    return folder


def episode_summary(rows):
    """Independently verify advertised harm estimands from retained game rows."""
    def column(name):
        def number(value):
            if value in ("True", True):
                return 1.
            if value in ("False", False):
                return 0.
            return float(value)
        return np.asarray([number(row[name]) for row in rows], dtype=float)
    if not rows:
        raise ValueError("Empty episode evaluation")
    counts, meetings = column("total_false_ejections"), column("total_meetings")
    return dict(false_ejection_rate=float(column("false_ejection").mean()),
                coalition_favorable_rate=float(column("coalition_favorable").mean()),
                coalition_game_win_rate=float(column("coalition_game_won").mean()),
                any_false_ejection_rate=float((counts > 0).mean()),
                mean_false_ejections=float(counts.mean()),
                false_ejections_per_meeting=float(counts.sum() / meetings.sum()))


def check_episode_summary(advertised, rows, rounds):
    recomputed = episode_summary(rows)
    for metric, value in recomputed.items():
        if metric == "coalition_game_win_rate" and rounds == 1:
            continue
        if finite(advertised.get(metric)) is None or not np.isclose(float(advertised[metric]), value, rtol=0, atol=1e-10):
            raise ValueError(f"Episode records disagree with {metric}: {advertised.get(metric)} vs {value}")


def require_cells(rows, fields, expected):
    expected = set(expected)
    found = [tuple(str(row[key]) for key in fields) for row in rows]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError(f"Incomplete/duplicate {fields} cells; missing={expected-set(found)}")


def check_identity(row, expected, label):
    """Bind saved numerical/label fields to the intended seed and evaluation cell."""
    for key, value in expected.items():
        actual = row.get(key)
        if isinstance(value, (int, float)):
            try:
                valid = float(actual) == value
            except (TypeError, ValueError):
                valid = False
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"{label} identity mismatch: {key}={actual!r}, expected {value!r}")


def command_integer(job, flag, default):
    command = job.get("command", [])
    return int(command[command.index(flag) + 1]) if flag in command else default


def check_episode_identity(rows, pair, eval_seed):
    for episode, row in enumerate(rows):
        check_identity(row, dict(pair=pair, eval_seed=eval_seed, episode_index=episode), "Raw episode")


def ingest(job, folder, data):
    group, seed = group_seed(job["name"])
    family, info = job["family"], scope(group)
    if family == "f1":
        path = folder / "result.json"
        result = read_json(path)
        if result.get("holdout_seed") != 1987654321 + seed:
            raise ValueError("F1 endpoints do not use the independent final evidence stream")
        for endpoint in ("initial", "final"):
            data.metrics(job, "holdout", endpoint, result[f"{endpoint}_holdout"], path)
        history_path = folder / "history.json"
        for row in read_json(history_path)["eval_curve"]:
            x = int(row.get("updates_completed", int(row["update"]) + 1))
            data.metrics(job, "monitored_curve", str(x), row, history_path)
    elif family in ("f3", "f4", "budget"):
        path = folder / "crossplay_summary.csv"
        rows = read_csv(path)
        require_cells(rows, ["pair"], [(p,) for p in PAIRS])
        episode_groups = defaultdict(list)
        for record in read_csv(folder / "episode_metrics.csv"):
            episode_groups[record["pair"]].append(record)
        for row in rows:
            if int(float(row["eval_seed"])) != 1987654321 + seed:
                raise ValueError("Cycle crossplay does not use final holdout evidence")
            check_identity(row, dict(n_agents=info["n"], n_crew=info["n"] - 2, n_coalition=2,
                max_rounds=info["rounds"], n_episodes=command_integer(job, "--crossplay-episodes", 1000)), "Cycle summary")
            for metric in primary(info["rounds"]) + ("mean_false_ejections",):
                if finite(row.get(metric)) is None:
                    raise ValueError(f"Missing repaired cumulative outcome {metric}")
            if len(episode_groups[row["pair"]]) != int(float(row["n_episodes"])):
                raise ValueError(f"Incorrect episode denominator for {row['pair']}")
            check_episode_identity(episode_groups[row["pair"]], row["pair"], 1987654321 + seed)
            check_episode_summary(row, episode_groups[row["pair"]], info["rounds"])
            data.metrics(job, "crossplay", row["pair"], row, path)
    elif family in ("hypothesis", "counterattack"):
        path = folder / "crossplay.csv"
        rows = read_csv(path)
        defenses = ("soft", "mean", "hypothesis") if family == "hypothesis" else ("soft", "rule", "both")
        require_cells(rows, ["coalition_trained_vs", "defense"], itertools.product(defenses, repeat=2))
        for row in rows:
            check_identity(row, dict(seed=seed, n_episodes=command_integer(job, "--crossplay-episodes", 1000)), "Counterattack cell")
            data.metrics(job, "crossplay", f"{row['coalition_trained_vs']}|{row['defense']}", row, path)
    elif family == "ablation":
        path = folder / "result.json"
        rows = read_json(path)["rows"]
        require_cells(rows, ["condition"], [(c,) for c in ABLATIONS])
        for row in rows:
            check_identity(row, dict(seed=seed, n_agents=info["n"], max_rounds=info["rounds"],
                n_episodes=command_integer(job, "--eval-episodes", 1000)), "Ablation cell")
            data.metrics(job, "ablation", row["condition"], row, path)
    elif family in ("f2", "static"):
        path = folder / ("f2_seed_results.csv" if family == "f2" else "rules.csv")
        rows = read_csv(path)
        require_cells(rows, ["condition", "rule"], itertools.product(CONDITIONS, RULES))
        for row in rows:
            identity = dict(seed=seed, episodes=command_integer(job, "--episodes", 1000))
            if family == "f2":
                identity.update(crew=info["n"] - 2, n_agents=info["n"], eval_seed=1987654321 + seed * 10000)
            check_identity(row, identity, "Scripted-rule cell")
            data.metrics(job, "rules", f"{row['condition']}|{row['rule']}", row, path)
        if family == "static":
            path = folder / "learned.csv"
            rows = read_csv(path)
            expected = [(f"C{g}", r, "crew_rule") for g in range(4) for r in RULES]
            expected += [(f"C{g}", "dependence_aware", "both") for g in range(4)]
            require_cells(rows, ["coalition", "rule", "mechanism"], expected)
            for row in rows:
                check_identity(row, dict(seed=seed, n_episodes=command_integer(job, "--episodes", 1000)), "Static learned cell")
                data.metrics(job, "learned", f"{row['coalition']}|{row['rule']}|{row['mechanism']}", row, path)
            path = folder / "strength_sweep.csv"
            for row in read_csv(path):
                check_identity(row, dict(seed=seed, n_episodes=command_integer(job, "--episodes", 1000)), "Static strength cell")
                data.metrics(job, "strength_sweep", f"{row['condition']}|{row['mechanism']}|{row['strength']}", row, path)
    else:
        ingest_loop(job, folder, data)


def ingest_loop(job, folder, data):
    group, seed = group_seed(job["name"])
    info = scope(group)
    k = info["k"]
    path = folder / "generations.json"
    rows = read_json(path)["rows"]
    require_cells(rows, ["side", "generation"], [("C", "0")] + [(side, str(g)) for g in range(1, k + 1) for side in ("D", "C")])
    indexed = {(r["side"], int(r["generation"])): r for r in rows}
    for row in rows:
        generation = int(row["generation"])
        index = 2 * generation if row["side"] == "C" else 2 * generation - 1
        check_identity(row, dict(stage_index=index, stage_seed=seed + 10000 * index,
            n_agents=info["n"], n_crew=info["n"] - 2, n_coalition=2, max_rounds=info["rounds"],
            eval_seed=1987654321 + seed, n_episodes=command_integer(job, "--eval-episodes", 1000)), "Multi-generation stage")
        data.metrics(job, "stages", f"{row['side']}{row['generation']}", row, path)
    crosspath = folder / "crossplay.json"
    payload = read_json(crosspath)
    matrices = payload["matrices"]
    evaluation_records = read_json(folder / "evaluation_records.json")["evaluations"]
    expected_records = [("stage", f"stage{2*g:02d}_C{g}") for g in range(k + 1)]
    expected_records += [("stage", f"stage{2*g-1:02d}_D{g}") for g in range(1, k + 1)]
    expected_records += [("crossplay", f"C{i}_vs_D{j}") for i, j in itertools.product(range(k + 1), repeat=2)]
    require_cells(evaluation_records, ["scope", "evaluation_id"], expected_records)
    for record in evaluation_records:
        eval_seed = 1987654321 + seed
        if record["scope"] == "stage":
            match = re.fullmatch(r"stage\d+_([CD])(\d+)", record["evaluation_id"])
            generation = int(match[2])
            pair = f"C{generation if match[1] == 'C' else generation - 1}_vs_D{generation}"
            count = command_integer(job, "--eval-episodes", 1000)
        else:
            pair = record["evaluation_id"]
            count = command_integer(job, "--crossplay-episodes", 500)
        check_identity(record, dict(pair=pair, eval_seed=eval_seed, n_episodes=count), "Multi-generation evaluation index")
        episode_path = exported_artifact_path(folder, record["episode_metrics"])
        meeting_path = exported_artifact_path(folder, record["round_metrics"])
        episodes = read_csv(episode_path)
        meetings = read_csv(meeting_path)
        check_episode_identity(episodes, pair, eval_seed)
        for meeting in meetings:
            check_identity(meeting, dict(pair=pair, eval_seed=eval_seed), "Raw meeting")
        if len(episodes) != record["n_episodes"] or len(meetings) != record["n_meetings"]:
            raise ValueError(f"Raw record count mismatch: {record['evaluation_id']}")
        if record["scope"] == "stage":
            match = re.fullmatch(r"stage\d+_([CD])(\d+)", record["evaluation_id"])
            advertised = indexed[match[1], int(match[2])]
        else:
            match = re.fullmatch(r"C(\d+)_vs_D(\d+)", record["pair"])
            i, j = int(match[1]), int(match[2])
            advertised = {metric: values[i][j] for metric, values in matrices.items()}
            if len(episodes) != int(payload["episodes_per_cell"]):
                raise ValueError("Crossplay episodes_per_cell disagrees with raw records")
        check_episode_summary(advertised, episodes, info["rounds"])
    for metric in primary(info["rounds"]):
        if metric not in matrices:
            raise ValueError(f"Missing primary crossplay matrix {metric}")
    for metric, raw in matrices.items():
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape != (k + 1, k + 1):
            raise ValueError(f"Matrix shape mismatch: {metric}")
        if not np.isfinite(matrix).all():
            if metric == "coalition_game_win_rate" and info["rounds"] == 1 and np.isnan(matrix).all():
                continue
            raise ValueError(f"Incomplete matrix: {metric}")
        for i, j in itertools.product(range(k + 1), repeat=2):
            data.add(job, "crossplay", f"C{i}|D{j}", metric, matrix[i, j], crosspath)
        # Fixed reference/control horizons; never choose a horizon from outcomes.
        for horizon in sorted(set([k] + [h for h in (2, 4) if h < k])):
            m = matrix[:horizon + 1, :horizon + 1]
            c = np.asarray([finite(indexed["C", g].get(metric)) for g in range(1, horizon + 1)], dtype=float)
            d = np.asarray([finite(indexed["D", g].get(metric)) for g in range(1, horizon + 1)], dtype=float)
            if not np.isfinite(c).all() or not np.isfinite(d).all():
                raise ValueError(f"Missing stage metric: {metric}")
            window = min(3, horizon // 2)
            descriptors = dict(c_stage_mean=c.mean(), d_stage_mean=d.mean(),
                               matrix_adaptation_gain=np.mean([m[g, g] - m[g - 1, g] for g in range(1, horizon + 1)]))
            if window:
                descriptors["late_minus_early"] = c[-window:].mean() - c[:window].mean()
            same, opposite = [], []
            for i, j in itertools.product(range(1, horizon + 1), repeat=2):
                if abs(i - j) >= 2:
                    (same if (i - j) % 2 == 0 else opposite).append(m[i, j])
            if same and opposite:
                descriptors["parity_gap"] = np.mean(same) - np.mean(opposite)
            for name, value in descriptors.items():
                data.add(job, f"loop_h{horizon}", name, metric, value, crosspath)
        if metric in primary(info["rounds"]):
            for gap in restricted_response_gaps(matrix, int(payload["episodes_per_cell"])):
                data.gaps.append(dict(job=job["name"], group=group, seed=seed, metric=metric, **gap))
                data.add(job, "restricted_pool_gap", str(gap["generation"]), metric, gap["gap"], crosspath)
    distance_path = folder / "policy_matrices.json"
    for metric, record in read_json(distance_path).items():
        if not isinstance(record, dict) or "matrix" not in record:
            continue
        generations = record["generations"]
        matrix = np.asarray(record["matrix"], dtype=float)
        if matrix.shape != (len(generations), len(generations)) or not np.isfinite(matrix).all():
            raise ValueError(f"Incomplete policy-distance matrix: {metric}")
        for i, j in itertools.product(range(len(generations)), repeat=2):
            data.add(job, "policy_distance", f"{generations[i]}|{generations[j]}", metric, matrix[i, j], distance_path)


def comparison_results(comparisons, data, infer):
    output = []
    for comp in comparisons:
        available = [s for s in comp.seeds if all(s in data.cells[key] for _, key in comp.terms)]
        diffs = [sum(coefficient * data.cells[key][seed] for coefficient, key in comp.terms) for seed in available]
        complete = bool(comp.seeds) and available == comp.seeds
        record = dict(family=comp.family, contrast=comp.name, expected_seeds=comp.seeds,
                      included_seeds=available, complete=complete, per_seed_differences=diffs,
                      terms=comp.terms, **mean_sd(diffs), ci95=None, p_exact=None, p_holm=None)
        if infer and complete:
            record.update(ci95=paired_bootstrap(diffs, comp.name), p_exact=exact_mean_sign_flip(diffs))
        output.append(record)
    families = defaultdict(list)
    for i, row in enumerate(output):
        families[row["family"]].append(i)
    for family, indices in families.items():
        adjusted = holm([output[i]["p_exact"] for i in indices])
        for i, p in zip(indices, adjusted):
            output[i].update(p_holm=p, family_size=len(indices))
    return output


def summarize_cells(data, expected):
    rows = []
    for (group, table, cell, metric), values in sorted(data.cells.items()):
        seeds = sorted(values)
        label = "terminal-meeting false ejection" if metric == "false_ejection_rate" and scope(group)["rounds"] > 1 else metric
        rows.append(dict(group=group, table=table, cell=cell, metric=metric, metric_label=label,
                         expected_seeds=expected[group], included_seeds=seeds,
                         complete=seeds == expected[group], **mean_sd([values[s] for s in seeds])))
    return rows


def matched_reference_summaries(comparisons, data):
    """Supplementary reference means use the contrast's common planned seeds."""
    output, seen = [], set()
    for comp in comparisons:
        if comp.family not in ("reward_controls", "dependence_loop", "budget"):
            continue
        for _, key in comp.terms:
            identity = (key, tuple(comp.seeds))
            if identity in seen:
                continue
            seen.add(identity)
            values = data.cells[key]
            seeds = [s for s in comp.seeds if s in values]
            output.append(dict(group=key[0], table=key[1], cell=key[2], metric=key[3],
                               expected_seeds=comp.seeds, included_seeds=seeds,
                               complete=seeds == comp.seeds, **mean_sd([values[s] for s in seeds])))
    return output


def plot_curves(summary, out, interim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for group in sorted({r["group"] for r in summary if r["table"] == "monitored_curve"}):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for metric in ("false_ejection_rate", "coalition_favorable_rate"):
            rows = sorted([r for r in summary if r["group"] == group and r["table"] == "monitored_curve" and r["metric"] == metric], key=lambda r: int(r["cell"]))
            if not rows:
                continue
            x, y = np.array([int(r["cell"]) for r in rows]), np.array([r["mean"] for r in rows])
            sd = np.array([np.nan if r["sd"] is None else r["sd"] for r in rows])
            ax.plot(x, y, label=metric.replace("_", " "))
            ax.fill_between(x, y - sd, y + sd, alpha=.15)
        ax.set(xlabel="Completed PPO updates", ylabel="Probability (mean ± seed SD)", ylim=(-.03, 1.03),
               title=("INTERIM: " if interim else "") + f"Monitored evaluation, n={scope(group)['n']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{group}_curve.pdf")
        plt.close(fig)


def archive_previous_outputs(out):
    """Preserve, but remove stale interpretation files from the active directory."""
    names = ("per_seed", "cell_summary", "paired_effects", "matched_reference_summary", "restricted_pool_gaps")
    paths = [out / f"{name}.{extension}" for name in names for extension in ("csv", "json")]
    paths += [out / "README.md", *out.glob("*_curve.pdf")]
    existing = [p for p in paths if p.exists()]
    if existing:
        backup = out / "superseded" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True)
        for path in existing:
            path.rename(backup / path.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO / "submission_protocol.json")
    parser.add_argument("--runs", type=Path, default=REPO / "results/submission_20260909")
    parser.add_argument("--out", type=Path, default=REPO / "audit/submission/analysis")
    parser.add_argument("--interim", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    protocol = read_json(args.protocol)
    if protocol.get("study") != "post_audit_replication_20260909":
        raise ValueError("Only the post-audit replication protocol is accepted")
    jobs, fingerprint = protocol["jobs"], protocol["source_sha256"]
    comparisons, expected = planned_comparisons(jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    # Define the signed comparison family manifest before opening outcome files.
    write_json(args.out / "planned_comparisons.json", [asdict(c) for c in comparisons])
    data, problems, completed, checkpoint_records = Dataset(), [], [], []
    for job in jobs:
        try:
            folder = validate_job(job, args.runs, fingerprint, checkpoint_records)
            partial = Dataset()
            ingest(job, folder, partial)
            for key, values in partial.cells.items():
                data.cells[key].update(values)
            data.rows.extend(partial.rows)
            data.gaps.extend(partial.gaps)
            completed.append(job["name"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(dict(job=job["name"], error=str(error)))
    preliminary = comparison_results(comparisons, data, infer=False)
    missing_contrasts = [r["contrast"] for r in preliminary if not r["complete"]]
    ready = not problems and not missing_contrasts
    status = dict(status="complete" if ready and not args.interim else "INTERIM / FINAL INFERENCE BLOCKED",
                  inference_enabled=ready and not args.interim, expected_jobs=len(jobs),
                  completed_jobs=len(completed), completed=completed, problems=problems,
                  missing_comparisons=missing_contrasts, source_sha256=fingerprint,
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  validation_code_sha256=validation_code_hashes(),
                  protocol_path=str(args.protocol.resolve()), runs_path=str(args.runs.resolve()))
    write_json(args.out / "status.json", status)
    write_json(args.out / "checkpoint_integrity.json", dict(
        source_sha256=fingerprint, records=checkpoint_records, validation_code_sha256=validation_code_hashes(),
        note="Finite parameters and architecture, planned seed/environment/algorithm/budget, retained history counters, and final/last-update equality checked. Learner roles are derived from the frozen driver, not embedded in checkpoints. Opponent identity is not embedded; checkpoint-only checks cannot exclude a same-seed, same-configuration opponent swap. Source/job markers and raw records supply operational run provenance."))
    archive_previous_outputs(args.out)
    if not ready and not args.interim:
        # Never leave old final tables beside a new blocked status: write final
        # analyses into a fresh directory, or explicitly request an interim run.
        print(f"FINAL INFERENCE BLOCKED: {len(completed)}/{len(jobs)} jobs, {len(missing_contrasts)} incomplete contrasts. See {args.out / 'status.json'}")
        return 2
    inference = ready and not args.interim
    summaries = summarize_cells(data, expected)
    contrasts = comparison_results(comparisons, data, infer=inference)
    if not inference:
        for gap in data.gaps:
            gap.update(mc_low=None, mc_high=None)
    for name, rows in (("per_seed", data.rows), ("cell_summary", summaries),
                       ("paired_effects", contrasts), ("matched_reference_summary", matched_reference_summaries(comparisons, data)),
                       ("restricted_pool_gaps", data.gaps)):
        write_csv(args.out / f"{name}.csv", [dict(analysis_status=status["status"], **row) for row in rows])
        write_json(args.out / f"{name}.json", dict(status=status["status"], rows=rows))
    if args.plots:
        plot_curves(summaries, args.out, args.interim)
    (args.out / "README.md").write_text(
        f"# {status['status']}\n\n{len(completed)}/{len(jobs)} planned jobs included. "
        "All inference is paired at the seed level; SD describes between-seed variability. "
        "Intervals are pointwise seed-bootstrap 95% CIs; p_holm uses the complete declared family. "
        "Monitored curves are descriptive and are separate from initial/final holdout endpoints.\n\n"
        "Restricted response gaps use only the finite observed policy pool; their Monte Carlo "
        "bands do not establish true exploitability, equilibrium, or convergence. "
        "See audit/submission/ANALYSIS_PLAN.md and status.json for assumptions and completeness.\n")
    print(f"{status['status']}: {len(summaries)} descriptive cells; {len(contrasts)} planned contrasts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
