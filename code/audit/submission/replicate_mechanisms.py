#!/usr/bin/env python3
"""Separate, descriptive Table 5/6 mechanism replication using immutable frozen source.

This supplement does not modify the 240-job training protocol. Freeze once, then run
each crew/replicate job in a separate process; global mechanism switches are process-local.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE = "cdaa1c031b4e2e6e690771c9a884ed70ed78653814863f99bb2fe0def367ae68"
CREWS = (3, 5, 7)
CONDITIONS = ("truthful", "lone_liar", "alibi", "framer")
RULES = ("mean", "soft_credibility", "sharp_credibility")
VARIANTS = ("baseline", "alibi_off", "catch_outside")
REPLICATES = tuple(range(10))
EPISODES = 500
EVIDENCE_BASE = 3987654321
WEIGHT_METRICS = (
    "n_speakers", "coalition_share", "coalition_weight_soft", "weight_minus_share",
    "cred_coalition", "cred_honest", "uninformative_share", "weight_on_uninformative",
    "coalition_caught", "argmax_differs", "mean_top_is_coalition", "soft_top_is_coalition",
    "mean_skips", "soft_skips",
)
OUTCOME_METRICS = (
    "false_ejection_rate", "coalition_favorable_rate", "creator_ejection_rate",
    "creator_survival", "no_ejection_rate", "incident_rate",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), separators=(",", ":"))
                             if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def engine(source):
    source = source.resolve()
    sys.path.insert(0, str(source / "src"))
    import numpy as np
    from social_collusion import runmeta
    if runmeta.REPO_ROOT.resolve() != source or runmeta.source_digest() != EXPECTED_SOURCE:
        raise ValueError("Engine import/source differs from the pinned frozen snapshot")
    from social_collusion.config import load_env_config
    from social_collusion.env import VecEnv
    from social_collusion.env.enums import ClaimType, Role
    from social_collusion.env.knowledge import ObserverKnowledge
    from social_collusion.metrics import outcome_metrics as om
    from social_collusion.policies import crew_aggregation as ca
    from social_collusion.policies.base import RoleRouter
    from social_collusion.policies.registry import make_coalition
    from social_collusion.policies.scripted_crew import TruthfulCrew
    for name, module in tuple(sys.modules.items()):
        if name.startswith("social_collusion") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source / "src"):
                raise ValueError(f"Non-frozen module imported: {name}")
    if ca.W_ALIBI_WITNESS != 2.0 or ca.CATCH_PENALTY_OUTSIDE_AGGREGATION is not False:
        raise ValueError("Unexpected baseline mechanism constants")
    return SimpleNamespace(**locals())


@contextmanager
def variant_switches(ca, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    old = (ca.W_ALIBI_WITNESS, ca.CATCH_PENALTY_OUTSIDE_AGGREGATION)
    try:
        ca.W_ALIBI_WITNESS = 0.0 if variant == "alibi_off" else 2.0
        ca.CATCH_PENALTY_OUTSIDE_AGGREGATION = variant == "catch_outside"
        yield
    finally:
        ca.W_ALIBI_WITNESS, ca.CATCH_PENALTY_OUTSIDE_AGGREGATION = old


def evidence_seed(crew, condition, replicate):
    return EVIDENCE_BASE + 10000 * replicate + 100 * crew + CONDITIONS.index(condition)


def evidence_hash(state):
    record = dict(roles=state.roles.tolist(), positions=state.positions.tolist(),
                  incident=[int(state.incident_creator), int(state.incident_victim),
                            int(state.incident_time), int(state.incident_room)],
                  report=[int(state.reporter), int(state.report_turn)],
                  speaking_order=state.speaking_order.tolist())
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def voter_records(e, state):
    """Every honest ballot participant, reconstructed before the ejection; baseline soft only."""
    np, ca = e.np, e.ca
    decision = state.copy()
    decision.alive[decision.votes >= 0] = True
    coalition = set(int(i) for i in decision.coalition)
    rows = []
    for voter in np.flatnonzero(decision.alive & (decision.roles == int(e.Role.CREW))):
        voter = int(voter)
        knowledge = e.ObserverKnowledge.build(decision, voter)
        _, spoke = ca.testimony_signals(decision, voter, knowledge)
        credibility = ca.speaker_credibility(decision, voter, knowledge)
        speakers = np.flatnonzero(spoke)
        if not len(speakers):
            raise ValueError("Baseline meeting lacks any testimony rows; voter record undefined")
        weights = ca._softmax(credibility[speakers] / ca.TAU_SOFT)
        is_coalition = np.array([int(i) in coalition for i in speakers])
        informative = np.zeros(decision.config.n_agents, dtype=bool)
        caught = np.zeros(decision.config.n_agents, dtype=bool)
        for claim in decision.claims:
            informative[claim.speaker] |= claim.claim_type != int(e.ClaimType.NO_INFORMATION)
            caught[claim.speaker] |= knowledge.caught_false(claim)
        scores = {rule: ca.suspicion_scores(decision, voter, rule) for rule in ("mean", "soft_credibility")}
        mean, soft = scores["mean"], scores["soft_credibility"]
        share, weight = float(is_coalition.mean()), float(weights[is_coalition].sum())
        rows.append(dict(
            voter=voter, actual_vote=int(decision.votes[voter]), voter_ejected=state.ejected == voter,
            n_speakers=len(speakers), coalition_share=share, coalition_weight_soft=weight,
            weight_minus_share=weight - share,
            cred_coalition=float(credibility[speakers][is_coalition].mean()),
            cred_honest=float(credibility[speakers][~is_coalition].mean()),
            uninformative_share=float((~informative[speakers]).mean()),
            weight_on_uninformative=float(weights[~informative[speakers]].sum()),
            coalition_caught=bool(caught[speakers][is_coalition].any()),
            argmax_differs=int(np.argmax(mean)) != int(np.argmax(soft)),
            mean_top_is_coalition=int(np.argmax(mean)) in coalition,
            soft_top_is_coalition=int(np.argmax(soft)) in coalition,
            mean_skips=float(np.max(mean)) < .5, soft_skips=float(np.max(soft)) < .5,
            speaker_ids=speakers.tolist(), speaker_is_coalition=is_coalition.tolist(),
            speaker_credibility=credibility[speakers].tolist(), speaker_weights=weights.tolist(),
            speaker_caught=caught[speakers].tolist(), speaker_informative=informative[speakers].tolist(),
        ))
    expected = int(np.sum((state.votes >= 0) & (state.roles == int(e.Role.CREW))))
    if len(rows) != expected:
        raise ValueError("Honest electorate count disagrees with recorded voter population")
    return rows


def run_cell(e, crew, condition, rule, variant, replicate, episodes):
    seed = evidence_seed(crew, condition, replicate)
    cfg = e.load_env_config("meeting_only").with_(n_agents=crew + 2, n_coalition=2,
                                                  coalition_objective="false_ejection")
    with variant_switches(e.ca, variant):
        policy = e.RoleRouter(e.TruthfulCrew(rule=rule), e.make_coalition(condition))
        states = e.VecEnv(cfg, 64, seed=seed).run_episodes(policy, episodes)
        games, voters = [], []
        for index, state in enumerate(states):
            identity = dict(crew=crew, condition=condition, rule=rule, variant=variant,
                            replicate=replicate, eval_seed=seed, episode_index=index,
                            evidence_sha256=evidence_hash(state))
            games.append({**identity, **e.om.episode_row(state)})
            if rule == "soft_credibility" and variant == "baseline":
                voters.extend({**identity, **row} for row in voter_records(e, state))
        summary = e.om.summarize(states)
    return games, voters, summary


def protocol_payload(e):
    return dict(study="scripted_mechanism_replication_20260909", created_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=EXPECTED_SOURCE, source_manifest=e.runmeta.source_manifest(), script_sha256=sha(__file__),
                crews=CREWS, conditions=CONDITIONS, rules=RULES, variants=VARIANTS,
                replicates=REPLICATES, episodes_per_cell=EPISODES, evidence_seed_base=EVIDENCE_BASE,
                total_jobs=len(CREWS) * len(REPLICATES), total_episodes=540000,
                analysis=dict(inference="descriptive only; no significance tests or outcome-dependent comparisons",
                              outcomes=OUTCOME_METRICS, voter_metrics=WEIGHT_METRICS,
                              cell_summary="equal-weight mean and sample SD across all ten replicate-level estimates",
                              voter_summary="pool all honest voters within each replicate/cell, then equal-weight replicates",
                              contrasts="alibi_off minus baseline and catch_outside minus baseline within each crew/condition/rule; soft/sharp minus mean within each crew/condition/variant",
                              intervals="no confidence intervals or p-values; report replicate differences, mean, SD and range",
                              intervention_scope="switches apply throughout crew responses and voting; effects include changed response transcripts",
                              pairing="identical environment seed and episode indices across all nine rule/variant cells; verify evidence hashes",
                              population="all honest pre-ejection voters for baseline soft-credibility diagnostics"))


def verified_protocol(e, path):
    payload = json.loads(path.read_text())
    if payload["source_sha256"] != EXPECTED_SOURCE or payload["script_sha256"] != sha(__file__):
        raise ValueError("Supplement source or script differs from its frozen protocol")
    expected = protocol_payload(e)
    if any(payload[k] != clean(expected[k]) for k in expected if k != "created_utc"):
        raise ValueError("Supplement design differs from frozen protocol")
    return payload


def run_job(e, args, episodes=EPISODES, smoke=False):
    if args.crew not in CREWS or args.replicate not in REPLICATES:
        raise ValueError("Select a planned --crew and --replicate")
    protocol = protocol_payload(e) if smoke else verified_protocol(e, args.protocol)
    out = args.out / (f"smoke_crew{args.crew}_s{args.replicate}" if smoke else f"mechanism_crew{args.crew}_s{args.replicate}")
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "complete.json"
    if marker.exists():
        raise ValueError(f"Refusing to overwrite completed job: {out}")
    lock = out / "running.lock"
    with lock.open("x") as stream:
        stream.write(str(__import__("os").getpid()))
    cells = []
    try:
        write_json(out / "runmeta.json", dict(e.runmeta.collect(), script_sha256=sha(__file__),
                   source_import_root=str(e.source), protocol_sha256=None if smoke else sha(args.protocol),
                   smoke=smoke, crew=args.crew, replicate=args.replicate, episodes_per_cell=episodes))
        for condition in CONDITIONS:
            matched = None
            for rule, variant in itertools.product(RULES, VARIANTS):
                games, voters, summary = run_cell(e, args.crew, condition, rule, variant, args.replicate, episodes)
                evidence = [row["evidence_sha256"] for row in games]
                if matched is None:
                    matched = evidence
                elif evidence != matched:
                    raise ValueError("Rule/variant cells did not retain matched environment draws")
                directory = out / condition / rule / variant
                games_path = directory / "episodes.csv.gz"
                write_csv(games_path, games)
                record = dict(crew=args.crew, replicate=args.replicate, condition=condition, rule=rule,
                              variant=variant, eval_seed=evidence_seed(args.crew, condition, args.replicate),
                              episodes=len(games), episode_file=str(games_path.relative_to(out)),
                              episode_sha256=sha(games_path), outcome_summary=summary)
                if voters:
                    voters_path = directory / "voters.csv.gz"
                    write_csv(voters_path, voters)
                    record.update(voters=len(voters), voter_file=str(voters_path.relative_to(out)),
                                  voter_sha256=sha(voters_path),
                                  voter_summary={key: float(e.np.mean([r[key] for r in voters])) for key in WEIGHT_METRICS})
                cells.append(record)
                write_json(out / "cells.json", cells)
                print(f"crew={args.crew} replicate={args.replicate} {condition}/{rule}/{variant}: {len(games)} episodes", flush=True)
        if e.runmeta.source_digest() != EXPECTED_SOURCE or sha(__file__) != protocol["script_sha256"]:
            raise ValueError("Source/script changed during this job")
        write_json(marker, dict(source_sha256=EXPECTED_SOURCE, script_sha256=sha(__file__),
                   protocol_sha256=None if smoke else sha(args.protocol), cells_sha256=sha(out / "cells.json"),
                   smoke=smoke, crew=args.crew, replicate=args.replicate,
                   cells=len(cells), episodes=sum(c["episodes"] for c in cells),
                   finished_utc=datetime.now(timezone.utc).isoformat()))
    finally:
        lock.unlink(missing_ok=True)
    return out


def descriptive(values):
    import numpy as np
    array = np.asarray(values, dtype=float)
    if len(array) != 10 or not np.isfinite(array).all():
        raise ValueError("Descriptive production summary requires every finite planned replicate")
    return dict(n_replicates=len(array), mean=float(array.mean()), sd=float(array.std(ddof=1)),
                minimum=float(array.min()), maximum=float(array.max()), replicate_values=array.tolist())


def analyze(e, args):
    verified_protocol(e, args.protocol)
    outcomes, weights, seed_rows = defaultdict(dict), defaultdict(dict), []
    for crew, replicate in itertools.product(CREWS, REPLICATES):
        folder = args.out / f"mechanism_crew{crew}_s{replicate}"
        marker = json.loads((folder / "complete.json").read_text())
        for key, expected in (("source_sha256", EXPECTED_SOURCE), ("script_sha256", sha(__file__)),
                              ("protocol_sha256", sha(args.protocol)), ("cells_sha256", sha(folder / "cells.json")),
                              ("smoke", False), ("crew", crew), ("replicate", replicate),
                              ("cells", 36), ("episodes", 36 * EPISODES)):
            if marker.get(key) != expected:
                raise ValueError(f"Invalid supplement completion marker: {folder} {key}")
        cells = json.loads((folder / "cells.json").read_text())
        expected_cells = set(itertools.product(CONDITIONS, RULES, VARIANTS))
        if len(cells) != 36 or {(c["condition"], c["rule"], c["variant"]) for c in cells} != expected_cells:
            raise ValueError("Incomplete supplement cell set")
        matched = {}
        for cell in cells:
            condition, rule, variant = cell["condition"], cell["rule"], cell["variant"]
            if (cell["crew"] != crew or cell["replicate"] != replicate or cell["episodes"] != EPISODES
                    or cell["eval_seed"] != evidence_seed(crew, condition, replicate)):
                raise ValueError("Supplement cell identity differs from its planned job")
            episode_path = folder / cell["episode_file"]
            if sha(episode_path) != cell["episode_sha256"]:
                raise ValueError(f"Episode checksum mismatch: {episode_path}")
            games = read_csv(episode_path)
            if len(games) != EPISODES or [int(g["episode_index"]) for g in games] != list(range(EPISODES)):
                raise ValueError("Incomplete/duplicate supplement episode indices")
            for game in games:
                if (int(game["crew"]) != crew or int(game["replicate"]) != replicate
                        or game["condition"] != condition or game["rule"] != rule or game["variant"] != variant
                        or int(game["eval_seed"]) != evidence_seed(crew, condition, replicate)):
                    raise ValueError("Episode identity disagrees with its indexed cell")
            evidence = [g["evidence_sha256"] for g in games]
            if evidence != matched.setdefault(condition, evidence):
                raise ValueError("Mismatched rule/variant evidence in retained records")
            fields = dict(false_ejection_rate="false_ejection", coalition_favorable_rate="coalition_favorable",
                          creator_ejection_rate="creator_ejected", creator_survival="creator_survived",
                          no_ejection_rate="no_ejection", incident_rate="had_incident")
            for metric, field in fields.items():
                value = float(e.np.mean([g[field] == "True" for g in games]))
                if not e.np.isclose(value, cell["outcome_summary"][metric], rtol=0, atol=1e-12):
                    raise ValueError("Stored episode outcomes disagree with advertised summary")
                outcomes[crew, condition, rule, variant, metric][replicate] = value
                seed_rows.append(dict(crew=crew, condition=condition, rule=rule, variant=variant,
                                      metric=metric, replicate=replicate, value=value, kind="outcome"))
            if rule == "soft_credibility" and variant == "baseline":
                voter_path = folder / cell["voter_file"]
                if sha(voter_path) != cell["voter_sha256"]:
                    raise ValueError("Voter record checksum mismatch")
                voters = read_csv(voter_path)
                if len(voters) != cell["voters"] or len({(v["episode_index"], v["voter"]) for v in voters}) != len(voters):
                    raise ValueError("Incomplete/duplicate voter records")
                for metric in WEIGHT_METRICS:
                    values = [float(v[metric]) if v[metric] not in ("True", "False") else float(v[metric] == "True") for v in voters]
                    value = float(e.np.mean(values))
                    if not e.np.isclose(value, cell["voter_summary"][metric], rtol=0, atol=1e-12):
                        raise ValueError("Voter records disagree with advertised summary")
                    weights[crew, condition, metric][replicate] = value
                    seed_rows.append(dict(crew=crew, condition=condition, rule=rule, variant=variant,
                                          metric=metric, replicate=replicate, value=value, kind="voter"))
    summaries, contrasts = [], []
    for (crew, condition, rule, variant, metric), values in sorted(outcomes.items()):
        summaries.append(dict(kind="outcome", crew=crew, condition=condition, rule=rule, variant=variant,
                              metric=metric, **descriptive([values[s] for s in REPLICATES])))
        if variant != "baseline":
            base = outcomes[crew, condition, rule, "baseline", metric]
            contrasts.append(dict(crew=crew, condition=condition, rule=rule, metric=metric,
                                  contrast=f"{variant} minus baseline", **descriptive([values[s] - base[s] for s in REPLICATES])))
        if rule != "mean":
            base = outcomes[crew, condition, "mean", variant, metric]
            contrasts.append(dict(crew=crew, condition=condition, variant=variant, metric=metric,
                                  contrast=f"{rule} minus mean", **descriptive([values[s] - base[s] for s in REPLICATES])))
    for (crew, condition, metric), values in sorted(weights.items()):
        summaries.append(dict(kind="voter", crew=crew, condition=condition, rule="soft_credibility", variant="baseline",
                              metric=metric, **descriptive([values[s] for s in REPLICATES])))
    directory = args.out / "analysis"
    for name, rows in (("per_replicate", seed_rows), ("cell_summary", summaries), ("paired_descriptive", contrasts)):
        write_csv(directory / f"{name}.csv", rows)
        write_json(directory / f"{name}.json", rows)
    write_json(directory / "complete.json", dict(source_sha256=EXPECTED_SOURCE, script_sha256=sha(__file__),
               protocol_sha256=sha(args.protocol), jobs=30, episodes=540000, inference="descriptive only"))


def self_test(e):
    original = (e.ca.W_ALIBI_WITNESS, e.ca.CATCH_PENALTY_OUTSIDE_AGGREGATION)
    for variant in VARIANTS:
        try:
            with variant_switches(e.ca, variant):
                assert e.ca.W_ALIBI_WITNESS == (0.0 if variant == "alibi_off" else 2.0)
                assert e.ca.CATCH_PENALTY_OUTSIDE_AGGREGATION == (variant == "catch_outside")
                raise RuntimeError("intentional switch-restoration probe")
        except RuntimeError:
            pass
        assert original == (e.ca.W_ALIBI_WITNESS, e.ca.CATCH_PENALTY_OUTSIDE_AGGREGATION)
    cfg = e.load_env_config("meeting_only").with_(n_agents=7)
    states = e.VecEnv(cfg, 16, seed=evidence_seed(5, "alibi", 0)).run_episodes(
        e.RoleRouter(e.TruthfulCrew(rule="soft_credibility"), e.make_coalition("alibi")), 16)
    caught_cases, changed_soft, ejected_voters = 0, 0, 0
    for state in states:
        records = voter_records(e, state)
        ejected_voters += sum(r["voter_ejected"] for r in records)
        for row in records:
            assert abs(sum(row["speaker_weights"]) - 1.0) < 1e-12
            restored = state.copy()
            restored.alive[restored.votes >= 0] = True
            voter = row["voter"]
            with variant_switches(e.ca, "baseline"):
                mean0 = e.ca.suspicion_scores(restored, voter, "mean")
                soft0 = e.ca.suspicion_scores(restored, voter, "soft_credibility")
            with variant_switches(e.ca, "catch_outside"):
                mean1 = e.ca.suspicion_scores(restored, voter, "mean")
                soft1 = e.ca.suspicion_scores(restored, voter, "soft_credibility")
            e.np.testing.assert_allclose(mean0, mean1, atol=1e-12, rtol=0)
            caught_cases += any(row["speaker_caught"])
            changed_soft += not e.np.allclose(soft0, soft1, atol=1e-12, rtol=0)
    assert caught_cases and changed_soft and ejected_voters
    assert original == (e.ca.W_ALIBI_WITNESS, e.ca.CATCH_PENALTY_OUTSIDE_AGGREGATION)
    print(f"Self-test passed: switch restoration; {caught_cases} caught-voter cases; {changed_soft} soft-score changes; {ejected_voters} ejected honest voters retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "smoke", "analyze", "self-test"))
    parser.add_argument("--source", type=Path, default=HERE.parents[1])
    parser.add_argument("--protocol", type=Path, default=HERE / "mechanism_protocol.json")
    parser.add_argument("--out", type=Path, default=HERE / "mechanism_results")
    parser.add_argument("--crew", type=int)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    e = engine(args.source)
    if args.action == "freeze":
        if args.protocol.exists():
            raise ValueError("Refusing to overwrite a frozen supplemental protocol")
        write_json(args.protocol, protocol_payload(e))
        print(f"Frozen 30 supplemental jobs / 540000 episodes -> {args.protocol}")
    elif args.action == "self-test":
        self_test(e)
    elif args.action == "analyze":
        analyze(e, args)
    else:
        run_job(e, args, episodes=3 if args.action == "smoke" else EPISODES, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
