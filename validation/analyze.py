#!/usr/bin/env python3
"""Gate the complete prospective supplement and analyze six fixed primary contrasts.

All effects are derived from raw integer episode/meeting outcomes. An incomplete
cohort emits status counts only. Importing this module performs no simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner
from inference import holm, summarize_contrast
from jobs import canonical_bytes, digest, resolve_job

FINAL_SEEDS = list(range(1000, 1072))
DENOMINATOR = 1000
KINDS = ("static", "adaptive", "matched_cycle")
PRIMARY = (
    ("P1", "Truthful false ejection: selected minus original hypothesis", (
        ("static", "truthful__selected", "false_ejection", 1),
        ("static", "truthful__hypothesis_index_050", "false_ejection", -1))),
    ("P2", "Adaptation against selected defense: selected-trained minus soft-trained", (
        ("adaptive", "selected__selected", "false_ejection", 1),
        ("adaptive", "soft__selected", "false_ejection", -1))),
    ("P3", "Matched vulnerability: selected defense/attack minus original hypothesis defense/attack", (
        ("adaptive", "selected__selected", "false_ejection", 1),
        ("adaptive", "hypothesis_index_050__hypothesis_index_050", "false_ejection", -1))),
    ("P4", "Eight-round coalition victory: C1 minus C0 against D1", (
        ("matched_cycle", "C1__D1", "coalition_game_won", 1),
        ("matched_cycle", "C0__D1", "coalition_game_won", -1))),
    ("P5", "Eight-round innocent-ejection count: C1 minus C0 against D1", (
        ("matched_cycle", "C1__D1", "total_false_ejections", 1),
        ("matched_cycle", "C0__D1", "total_false_ejections", -1))),
    ("P6", "Eight-round victory change minus any-innocent-ejection change", (
        ("matched_cycle", "C1__D1", "coalition_game_won", 1),
        ("matched_cycle", "C0__D1", "coalition_game_won", -1),
        ("matched_cycle", "C1__D1", "any_false_ejection", -1),
        ("matched_cycle", "C0__D1", "any_false_ejection", 1))),
)
SUMMARY_METRICS = {
    "false_ejection": "false_ejection_rate",
    "any_false_ejection": "any_false_ejection_rate",
    "total_false_ejections": "mean_false_ejections",
    "creator_ejected": "creator_ejection_rate",
    "no_ejection": "no_ejection_rate",
    "coalition_game_won": "coalition_game_win_rate",
}
PRIMARY_FAMILY = "post_review_validation"


def read_json(path: Path) -> dict:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON literal {value}: {path}")

    return json.loads(path.read_text(), object_pairs_hook=object_pairs, parse_constant=invalid)


def integer(value, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}: expected integer in [{minimum}, {maximum}]")
    return value


def boolean(value, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label}: expected raw boolean, not an imputed or rounded rate")
    return value


def equal_rate(actual, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError(f"{label}: missing or nonfinite summary")
    if abs(actual - expected) > 1e-12:
        raise ValueError(f"{label}: summary does not match raw integer outcomes")


def required_cells(job: dict) -> set[str]:
    names = [d["name"] for d in job.get("defenses", [])]
    if job["kind"] == "static":
        return {f"{o['name']}__{d}" for o in job["opponents"] for d in names}
    if job["kind"] == "adaptive":
        return {f"{c}__{d}" for c in names for d in names}
    return {f"{c}__{d}" for c in ("C0", "C1") for d in ("D0", "D1")}


def analysis_settings(design: dict) -> tuple[list[int], int, list[int], bool]:
    """A separately flagged engineering smoke can exercise the same complete gate."""
    metadata = design.get("analysis", {})
    smoke = design.get("smoke_only") is True
    if smoke != (metadata.get("smoke_only") is True):
        raise ValueError("engineering smoke requires explicit flags in design and analysis metadata")
    if smoke:
        seeds = metadata.get("final_seeds", sorted({j["seed"] for j in design["jobs"] if j["phase"] == "final"}))
        denominator = metadata["episodes_per_cell"]
        if len(seeds) != metadata["final_seed_count"] or len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError("smoke seed cohort is not explicit and unique")
        development_seeds = design["selection"]["development_seeds"]
    else:
        seeds, denominator, development_seeds = FINAL_SEEDS, DENOMINATOR, list(range(100, 108))
        scientific = design.get("scientific_design", {})
        if scientific.get("final_seeds", seeds) != seeds:
            raise ValueError("production cohort differs from the planned 72 fresh seeds")
        if metadata.get("episodes_per_cell", denominator) != denominator:
            raise ValueError("production denominator differs from 1,000 episodes")
    inference = design.get("primary_inference", {})
    if inference.get("episode_denominator", denominator) != denominator or inference.get("lattice_scale", denominator) != denominator:
        raise ValueError("inference denominator differs from frozen evaluation budget")
    if inference.get("family_size", 6) != 6:
        raise ValueError("primary family must retain all six contrasts")
    if inference.get("family", PRIMARY_FAMILY) != PRIMARY_FAMILY:
        raise ValueError("primary family identity differs from frozen metadata")
    planned = inference.get("primary_contrasts")
    if planned is not None:
        if [c["id"] for c in planned] != [name for name, _, _ in PRIMARY]:
            raise ValueError("primary contrast IDs differ from the fixed six")
        for declared, (_, _, terms) in zip(planned, PRIMARY):
            expected = [[sign, cell, SUMMARY_METRICS[metric]] for _, cell, metric, sign in terms]
            if declared["terms"] != expected or declared["group"] != terms[0][0]:
                raise ValueError("primary contrast algebra differs from frozen metadata")
    return sorted(seeds), denominator, sorted(development_seeds), smoke


def verify_final_grid(jobs: list[dict], seeds: list[int] = FINAL_SEEDS,
                      denominator: int = DENOMINATOR) -> None:
    """The entire 72-seed cohort in each of three kinds is mandatory."""
    if len(jobs) != 3 * len(seeds):
        raise ValueError("final design must contain every prescribed seed in each of three kinds")
    for kind in KINDS:
        cohort = [j for j in jobs if j["kind"] == kind]
        if sorted(j["seed"] for j in cohort) != seeds:
            raise ValueError(f"{kind}: missing, duplicate, or substituted prescribed seed")
        for job in cohort:
            if job["episodes"] != denominator or job["n_agents"] != 7:
                raise ValueError("primary cohort requires seven agents and the fixed episode denominator")
            if job["max_rounds"] != (8 if kind == "matched_cycle" else 1):
                raise ValueError(f"{kind}: wrong round regime")
            for _, _, terms in PRIMARY:
                for term_kind, cell, _, _ in terms:
                    if term_kind == kind and cell not in required_cells(job):
                        raise ValueError(f"{kind}: missing prespecified primary cell {cell}")


def resolved_environment(job: dict, cell: str) -> dict:
    """Rebuild every declared cell configuration without creating a game state."""
    if cell not in required_cells(job):
        raise ValueError("evaluation cell is absent from the resolved job design")
    resolved = job
    if job["kind"] == "static":
        opponent_name = cell.rsplit("__", 1)[0]
        opponent = next(o for o in job["opponents"] if o["name"] == opponent_name)
        resolved = {
            **job,
            "environment": opponent.get("environment", job.get("environment", "full_short_game")),
            "env_overrides": {**job.get("env_overrides", {}), **opponent.get("env_overrides", {})},
        }
    # Configuration loading imports no trainer and runs no environment transition.
    code_source = str(runner.CODE / "src")
    if code_source not in sys.path:
        sys.path.insert(0, code_source)
    return runner.make_config(resolved).to_dict()


def verify_cell(payload: dict, job: dict, cell: str, summary: dict | None = None,
                expected_denominator: int = DENOMINATOR) -> dict:
    """Independently reconstruct ballot winners, meeting counts, and game endpoints."""
    n = job["n_agents"]
    rounds = job["max_rounds"]
    denominator = job["episodes"]
    if denominator != expected_denominator:
        raise ValueError("final primary denominator differs from frozen analysis settings")
    if payload.get("cell") != cell:
        raise ValueError("payload cell identity differs from result index")
    environment = payload["environment"]
    if canonical_bytes(environment) != canonical_bytes(resolved_environment(job, cell)):
        raise ValueError("full cell environment differs from the resolved frozen job/opponent configuration")
    if environment["n_agents"] != n or environment["max_rounds"] != rounds:
        raise ValueError("cell environment population/round regime mismatch")
    if environment.get("vote_aggregation") != "majority":
        raise ValueError("supplement must retain the frozen plurality tally")
    episodes = payload["episodes"]
    if len(episodes) != denominator:
        raise ValueError("raw episode count differs from fixed denominator")
    if [row["episode_index"] for row in episodes] != list(range(denominator)):
        raise ValueError("raw episode indices are missing, duplicated, or reordered")
    by_episode = defaultdict(list)
    for row in payload["meetings"]:
        idx = integer(row["episode_index"], "meeting episode index", 0, denominator - 1)
        if row["pair"] != cell or row["eval_seed"] != job["eval_seed"]:
            raise ValueError("meeting identity/seed mismatch")
        by_episode[idx].append(row)
    counts = Counter({metric: 0 for metric in SUMMARY_METRICS})
    honest_votes = honest_skips = 0
    expected_decisions = {}
    for idx, row in enumerate(episodes):
        if row["pair"] != cell or row["eval_seed"] != job["eval_seed"]:
            raise ValueError("episode identity/seed mismatch")
        roles, votes, alive = row["roles"], row["votes"], row["alive"]
        if any(len(vector) != n for vector in (roles, votes, alive)):
            raise ValueError("raw role/ballot/alive vector length mismatch")
        roles = [integer(role, "role", 0, 1) for role in roles]
        votes = [integer(vote, "vote", -1, n) for vote in votes]
        alive = [boolean(value, "alive status") for value in alive]
        if sum(roles) != 2:
            raise ValueError("raw role vector must contain exactly two coalition members")
        ejected = integer(row["ejected"], "ejected", -1, n - 1)
        if ejected >= 0 and alive[ejected]:
            raise ValueError("ejected candidate remains alive")
        before_vote = [value or j == ejected for j, value in enumerate(alive)]
        tally = [0] * n
        for voter, vote in enumerate(votes):
            if (vote >= 0) != before_vote[voter]:
                raise ValueError("ballots do not match the electorate before terminal ejection")
            if 0 <= vote < n:
                if not before_vote[vote] or (vote == voter and not environment["allow_self_vote"]):
                    raise ValueError("raw ballot targets an illegal candidate")
                tally[vote] += 1
            if vote == n and not environment["allow_skip_vote"]:
                raise ValueError("raw skip ballot is illegal under configuration")
            if roles[voter] == 0 and vote >= 0:
                honest_votes += 1
                honest_skips += vote == n
                expected_decisions[idx, voter] = vote
        maximum = max(tally)
        winners = [j for j, count in enumerate(tally) if count == maximum]
        reconstructed_ejected = (winners[0] if maximum > 0 and
            (len(winners) == 1 or environment["tie_ejects"]) else -1)
        if reconstructed_ejected != ejected:
            raise ValueError("raw plurality tally disagrees with reported ejection")
        terminal_fe = ejected >= 0 and roles[ejected] == 0
        creator = integer(row["incident_creator"], "incident creator", -1, n - 1)
        if creator >= 0 and roles[creator] != 1:
            raise ValueError("raw incident creator is not a coalition member")
        creator_ejected = creator >= 0 and ejected == creator
        coal_alive = sum(a and r == 1 for a, r in zip(alive, roles))
        crew_alive = sum(a and r == 0 for a, r in zip(alive, roles))
        game_won = coal_alive > 0 and coal_alive >= crew_alive
        meetings = by_episode[idx]
        if not 1 <= len(meetings) <= rounds or [m["round"] for m in meetings] != list(range(len(meetings))):
            raise ValueError("meeting sequence is missing, duplicated, or reordered")
        meeting_fe = 0
        previously_ejected = set()
        for meeting in meetings:
            expelled = integer(meeting["ejected"], "meeting ejected", -1, n - 1)
            if expelled >= 0:
                if expelled in previously_ejected or alive[expelled]:
                    raise ValueError("meeting ejects a player twice or leaves them alive")
                previously_ejected.add(expelled)
            innocent = expelled >= 0 and roles[expelled] == 0
            if integer(meeting["false_ejection"], "meeting false ejection", 0, 1) != int(innocent):
                raise ValueError("meeting false-ejection label disagrees with raw role/ejection")
            if integer(meeting["any_ejection"], "meeting any ejection", 0, 1) != int(expelled >= 0):
                raise ValueError("meeting any-ejection label disagrees with raw ejection")
            meeting_fe += innocent
        if meetings[-1]["ejected"] != ejected:
            raise ValueError("terminal ejection disagrees with final meeting")
        if integer(row["total_meetings"], "total meetings", 1, rounds) != len(meetings):
            raise ValueError("episode meeting count disagrees with complete meeting rows")
        if integer(row["total_false_ejections"], "innocent-ejection count", 0, rounds) != meeting_fe:
            raise ValueError("episode innocent-ejection count disagrees with raw meeting rows")
        reconstructed = {
            "false_ejection": terminal_fe,
            "any_false_ejection": bool(meeting_fe),
            "total_false_ejections": meeting_fe,
            "creator_ejected": creator_ejected,
            "no_ejection": ejected < 0,
            "coalition_game_won": game_won,
        }
        for metric, value in reconstructed.items():
            if metric != "total_false_ejections" and boolean(row[metric], metric) != value:
                raise ValueError(f"{metric}: recorded outcome disagrees with independent reconstruction")
            counts[metric] += int(value)
    for source_summary in [payload["summary"], *([] if summary is None else [summary])]:
        equal_rate(source_summary["n_episodes"], denominator, "n_episodes")
        for metric, summary_key in SUMMARY_METRICS.items():
            if metric == "coalition_game_won" and rounds == 1:
                if source_summary[summary_key] is not None:
                    raise ValueError("single-meeting victory summary must remain undefined")
                continue
            equal_rate(source_summary[summary_key], counts[metric] / denominator, summary_key)
    decisions = payload.get("defense_decisions", [])
    defense_name = cell.rsplit("__", 1)[1]
    defense_spec = next((d for d in job.get("defenses", []) if d["name"] == defense_name), None)
    needs_decisions = rounds == 1 and defense_spec is not None and "rule" not in defense_spec
    if needs_decisions or decisions:
        if len(decisions) != honest_votes:
            raise ValueError("decision audit count differs from raw honest electorate")
        seen_decisions = set()
        for decision in decisions:
            if "_state_token" in decision:
                raise ValueError("transient state identity must not enter retained data")
            idx = integer(decision["episode_index"], "decision episode index", 0, denominator - 1)
            voter = integer(decision["voter"], "decision voter", 0, n - 1)
            key = idx, voter
            if key not in expected_decisions or key in seen_decisions:
                raise ValueError("decision audit omits, duplicates, or misidentifies an honest voter")
            seen_decisions.add(key)
            if decision["round"] != 0 or decision["eval_seed"] != job["eval_seed"]:
                raise ValueError("decision audit does not identify the expected pre-ejection meeting")
            vote = expected_decisions[key]
            skip = boolean(decision["skip"], "audit skip")
            if skip != (vote == n) or decision["target"] != (None if vote == n else vote):
                raise ValueError("decision audit target/skip disagrees with raw honest ballot")
            if defense_spec is not None and (
                    decision["threshold"] != defense_spec["threshold"] or
                    decision["honesty"] != defense_spec["honesty"] or
                    decision["tie_break"] != defense_spec["tie_break"]):
                raise ValueError("decision audit parameters differ from resolved defense")
            tied = decision["tied_candidates"]
            if len(tied) != decision["tie_count"] or len(set(tied)) != len(tied):
                raise ValueError("decision audit tied-candidate list is inconsistent")
        if seen_decisions != set(expected_decisions):
            raise ValueError("decision audit lacks a complete pre-ejection honest electorate")
    return {
        "kind": job["kind"], "seed": job["seed"], "job": job["name"], "cell": cell,
        "denominator": denominator, "counts": dict(counts),
        "rates": {key: value / denominator for key, value in counts.items()},
        "terminal_honest_votes": honest_votes, "terminal_honest_skips": honest_skips,
        "terminal_honest_skip_rate": honest_skips / honest_votes if honest_votes else None,
        "decision_records": len(decisions),
        "final_tie_records": sum(integer(d["tie_count"], "tie count", 0, n) > 1 for d in decisions),
        "source_rows": denominator, "meeting_rows": sum(map(len, by_episode.values())),
    }


def primary_results(rows: list[dict], expected_seeds: list[int] = FINAL_SEEDS,
                    denominator: int = DENOMINATOR, t_sensitivity: bool = False) -> list[dict]:
    index = {}
    for row in rows:
        key = (row["kind"], row["seed"], row["cell"])
        if key in index:
            raise ValueError("duplicate seed/cell in primary analysis")
        if row["denominator"] != denominator:
            raise ValueError("primary denominators differ")
        if row["seed"] not in expected_seeds:
            raise ValueError("unplanned seed in primary cohort")
        for metric, count in row["counts"].items():
            integer(count, metric, 0, denominator * (8 if metric == "total_false_ejections" else 1))
        index[key] = row
    results = []
    for name, description, terms in PRIMARY:
        differences = []
        for seed in expected_seeds:
            try:
                difference = sum(sign * index[kind, seed, cell]["counts"][metric]
                                 for kind, cell, metric, sign in terms)
            except KeyError as error:
                raise ValueError(f"{name}: incomplete primary seed/cell cohort") from error
            differences.append(difference)
        result = summarize_contrast(name, differences, denominator, expected_seeds, expected_seeds)
        result.update(description=description, family=PRIMARY_FAMILY, family_size=6,
                      terms=[dict(kind=k, cell=c, metric=m, sign=s) for k, c, m, s in terms])
        results.append(result)
    adjusted = holm([row["p_exact"] for row in results])
    for row, p in zip(results, adjusted):
        row.update(p_holm=p, reject_holm_005=p < .05)
    if t_sensitivity:
        from scipy.stats import ttest_1samp
        for row in results:
            values = np.asarray(row["integer_differences"], dtype=float)
            if np.all(values == values[0]):
                p = 1.0 if values[0] == 0 else 0.0
                statistic = 0.0 if values[0] == 0 else None
            else:
                test = ttest_1samp(values, 0)
                p, statistic = float(test.pvalue), float(test.statistic)
            row.update(t_sensitivity_p=p, t_sensitivity_statistic=statistic,
                       t_sensitivity_interpretation="Sensitivity only; same six complete paired seed differences")
        for row, p in zip(results, holm([r["t_sensitivity_p"] for r in results])):
            row["t_sensitivity_p_holm"] = p
    return results


def descriptive_cells(rows: list[dict], expected_seeds: list[int] = FINAL_SEEDS) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["kind"], row["cell"]].append(row)
    result = []
    for (kind, cell), records in sorted(grouped.items()):
        if sorted(r["seed"] for r in records) != expected_seeds:
            raise ValueError("descriptive cell lacks full final seed cohort")
        for metric in SUMMARY_METRICS:
            if metric == "coalition_game_won" and kind != "matched_cycle":
                continue
            values = [r["rates"][metric] for r in records]
            result.append(dict(kind=kind, cell=cell, metric=metric, n_seeds=len(values),
                               mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                               interpretation="descriptive; no additional hypothesis test"))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refuse empty results table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def analyze(protocol_path: Path, results_path: Path, selection_path: Path | None,
            output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status="incomplete", inference_enabled=False, checked_utc=runner.utc(),
                  expected_final_jobs=216, complete_final_jobs=0, complete_development_jobs=0,
                  problems=[], incomplete_jobs=[])
    try:
        protocol = runner.load_protocol(protocol_path)
        runner.verify_source(protocol)
        runner.verify_inputs(protocol["design"])
        protocol_sha = runner.sha256(protocol_path)
        status.update(protocol_sha256=protocol_sha, source_sha256=protocol["source_sha256"])
        expected_seeds, denominator, development_seeds, smoke = analysis_settings(protocol["design"])
        status.update(expected_final_jobs=3 * len(expected_seeds), smoke_only=smoke,
                      expected_development_jobs=len(development_seeds))
        development = [j for j in protocol["design"]["jobs"] if j["phase"] == "development"]
        if sorted(j["seed"] for j in development) != development_seeds:
            raise ValueError("all prespecified development jobs are required")
        finals = [j for j in protocol["design"]["jobs"] if j["phase"] == "final"]
        verify_final_grid(finals, expected_seeds, denominator)
        for job in development:
            if runner.completed(job, results_path / job["name"], protocol_sha, None):
                status["complete_development_jobs"] += 1
            else:
                status["incomplete_jobs"].append(job["name"])
        selection, selection_sha = runner.load_selection(selection_path, protocol_path, protocol, "final")
        status["selection_sha256"] = selection_sha
        resolved = [resolve_job(job, selection) for job in finals]
        for job in resolved:
            if runner.completed(job, results_path / job["name"], protocol_sha, selection_sha):
                status["complete_final_jobs"] += 1
            else:
                status["incomplete_jobs"].append(job["name"])
        if status["incomplete_jobs"]:
            raise ValueError("the full frozen development and final cohort is not complete")
        from selector import select
        recomputed_selection = select(protocol_path, results_path)
        if ({k: v for k, v in selection.items() if k != "created_utc"} !=
                {k: v for k, v in recomputed_selection.items() if k != "created_utc"}):
            raise ValueError("frozen selection does not reproduce the prospectively declared development rule")
        frozen_time = dt.datetime.fromisoformat(selection["created_utc"])
        rows = []
        for job in resolved:
            directory = results_path / job["name"]
            result = read_json(directory / "result.json")
            run = read_json(directory / "run.json")
            marker = read_json(directory / "complete.json")
            if any(record["source_sha256"] != protocol["source_sha256"] for record in (run, marker)):
                raise ValueError("job source identity differs from frozen protocol")
            if dt.datetime.fromisoformat(run["started_utc"]) < frozen_time:
                raise ValueError("final training/evaluation began before development selection was frozen")
            evaluations = result["evaluations"]
            if {r["cell"] for r in evaluations} != required_cells(job) or len(evaluations) != len(required_cells(job)):
                raise ValueError("job cell set differs from prospectively declared crossplay")
            for record in evaluations:
                path = (directory / record["path"]).resolve()
                if directory.resolve() not in path.parents:
                    raise ValueError("evaluation path escapes the frozen job directory")
                rows.append(verify_cell(read_json(path), job, record["cell"], record["summary"], denominator))
        t_sensitivity = bool(protocol["design"].get("primary_inference", {}).get("sensitivity"))
        primary = primary_results(rows, expected_seeds, denominator, t_sensitivity)
        cells = descriptive_cells(rows, expected_seeds)
        chosen = {k: v for k, v in selection["selected_defense"].items() if k != "name"}
        original = next(d for d in resolved[0]["defenses"] if d["name"] == "hypothesis_index_050")
        selected_is_original = chosen == {k: v for k, v in original.items() if k != "name"}
        for filename, records in (("per_seed", rows), ("descriptive_cells", cells), ("primary", primary)):
            runner.write_json(output / f"{filename}.json", records)
            write_csv(output / f"{filename}.csv", records)
        lines = ["# Engineering smoke test ONLY" if smoke else "# Completed prospective validation supplement", "",
            f"All {len(expected_seeds)} prescribed seeds completed each of the static, adaptive, and eight-round jobs. "
            f"All {len(development_seeds)} development jobs, source pins, completion manifests, raw ballots, and recorded "
            f"meeting outcomes passed validation. Each final cell contains exactly {denominator} episodes.", "",
            "The six two-sided tests below form one prespecified Holm family. The exact integer "
            "sign-flip distribution requires joint sign invariance under the null; pairing alone does "
            "not establish that assumption. Confidence intervals are pointwise 20,000-resample seed "
            "bootstrap intervals and do not have familywise coverage.", "",
            "| Test | Signed mean difference | Pointwise 95% CI | Exact p | Holm p |",
            "| --- | ---: | --- | ---: | ---: |"]
        for row in primary:
            lo, hi = row["ci95_pointwise"]
            lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                         f"{row['p_exact']:.6g} | {row['p_holm']:.6g} |")
        lines += ["", *[f"- **{name}:** {description}." for name, description, _ in PRIMARY], "",
            "P5 is measured in innocent ejections per game; the other contrasts use probability "
            "differences. P3 compares defense-plus-matched-training procedures, not a fixed-attack "
            "causal intervention. P6 tests the difference between two outcome changes; it does not "
            "establish equivalence of any-false-ejection rates. Non-rejection is not evidence of "
            "equivalence. All extra cell summaries are descriptive.", "",
            "The chosen defense and every development candidate remain in the selection receipt. "
            "A software validation pass is not itself a judgment that the manuscript is ready for submission.", ""]
        if t_sensitivity:
            lines += ["Paired-t sensitivity p-values (same six-member Holm adjustment): " +
                      "; ".join(f"{r['contrast']}={r['t_sensitivity_p_holm']:.6g}" for r in primary) + ".", ""]
        if selected_is_original:
            lines += ["Development selected the original index/0.50 defense. The selected and original "
                      "slots therefore describe the same defense parameters. Zero P1 or P3 contrasts "
                      "from duplicated identical procedures are structural identities, not evidence "
                      "of equivalence between different defenses.", ""]
        if smoke:
            lines += ["These are engineering checks with deliberately reduced budgets. They are excluded "
                      "from scientific results, defense claims, and the production seed cohort.", ""]
        (output / "results.md").write_text("\n".join(lines))
        status.update(status="complete", inference_enabled=True, primary_contrasts=6,
                      selected_is_original=selected_is_original,
                      per_seed_cells=len(rows), descriptive_rows=len(cells),
                      raw_episode_rows=sum(r["source_rows"] for r in rows),
                      raw_meeting_rows=sum(r["meeting_rows"] for r in rows),
                      results_sha256=runner.sha256(output / "primary.json"))
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ImportError) as error:
        status["problems"].append(f"{type(error).__name__}: {error}")
        # Remove only this analyzer's derived outputs so stale inference cannot be
        # mistaken for a newly validated complete cohort after an input changes.
        for name in ("per_seed.json", "per_seed.csv", "descriptive_cells.json", "descriptive_cells.csv",
                     "primary.json", "primary.csv", "results.md"):
            (output / name).unlink(missing_ok=True)
    runner.write_json(output / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = analyze(args.protocol.resolve(), args.results.resolve(),
                     args.selection.resolve() if args.selection else None, args.out.resolve())
    print(json.dumps(status, indent=2))
    return 0 if status["inference_enabled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
