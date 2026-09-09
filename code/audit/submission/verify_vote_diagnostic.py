#!/usr/bin/env python3
"""Independently verify saved F8 seeds, ballots, and score-derived diagnostics.

Uses only the standard library and never imports or changes the frozen engine.
Score flags are recomputed from retained feasibility-score vectors; this does
not recompute the underlying feasibility model or assert an exact posterior.
Checkpoint/source/completion provenance remains the archive builder's gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

EVIDENCE_BASE = 2987654321
RULES = ("soft", "mean", "hypothesis")
SEEDS = {7: range(10), 9: range(5)}


def confined_raw_path(directory, name):
    """Accept saved Windows separators, reject traversal and linked raw inputs."""
    if not isinstance(name, str):
        raise ValueError("F8 raw path must be a string")
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (not normalized or relative.is_absolute() or ":" in normalized
            or re.search(r"[\x00-\x1f\x7f]", normalized)
            or any(part in ("", ".", "..") for part in normalized.split("/"))
            or relative.parts[0] != "raw"):
        raise ValueError(f"Unsafe F8 raw path: {name!r}")
    root = Path(directory).absolute()
    if root.is_symlink() or (root.exists() and getattr(root.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("Linked F8 result directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError(f"Linked F8 raw path: {name!r}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"F8 raw path leaves its result directory: {name!r}")
    return path


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"F8 raw {label}: {actual!r}; expected {expected!r}")


def require_fields(record, expected, label):
    for key, value in expected.items():
        if key not in record:
            raise ValueError(f"F8 raw {label} missing {key}")
        require_equal(record[key], value, f"{label}.{key}")


def tally(votes, n):
    counts = Counter(vote for vote in votes if 0 <= vote < n)
    ordered = counts.most_common()
    winner = ordered[0][0] if ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1]) else -1
    return winner, [counts[candidate] for candidate in range(n)]


def expected_seeds(n, seed, episode):
    evidence = EVIDENCE_BASE + 10000 * seed + n
    key = f"vote-diagnostic-actor:{evidence}:{episode}".encode()
    # The two original modulo operations are equivalent to this power-of-two mask.
    actor = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 56) - 1)
    return evidence, actor


def score_vector(value, n, label):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"F8 raw {label} has wrong length")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           or x < 0 or x > 1 + 1e-12 for x in value):
        raise ValueError(f"F8 raw {label} has invalid probability-like feasibility scores")
    return value


def verify_record(record, *, n_agents, training_seed, episode_index, trained, against,
                  checkpoint_sha256, smoke=False):
    """Reject inconsistent derived fields; return a count of validated honest views."""
    n = n_agents
    if (n not in SEEDS or trained not in RULES or against not in RULES
            or type(training_seed) is not int or training_seed < 0
            or (not smoke and training_seed not in SEEDS[n])
            or type(episode_index) is not int or episode_index < 0
            or (not smoke and episode_index >= 500)):
        raise ValueError("F8 raw identity is outside the fixed production design")
    evidence, actor = expected_seeds(n, training_seed, episode_index)
    require_fields(record, dict(n_agents=n, training_seed=training_seed, episode_index=episode_index,
        trained=trained, against=against, checkpoint_sha256=checkpoint_sha256,
        evidence_seed=evidence, actor_seed=actor), "identity/seeds")
    roles, votes, state = record["roles"], record["votes"], record["pre_vote_state"]
    if (not isinstance(roles, list) or len(roles) != n
            or any(type(role) is not int or role not in (0, 1) for role in roles) or sum(roles) != 2
            or not isinstance(votes, list) or len(votes) != n
            or any(type(vote) is not int or not -1 <= vote <= n for vote in votes)):
        raise ValueError("F8 raw roles or ballots are malformed")
    require_fields(state, dict(roles=roles, phase=6, done=False, ejected=-1, votes=[-1]*n), "pre-vote state")
    alive = state["alive"]
    if (not isinstance(alive, list) or len(alive) != n or any(type(v) is not bool for v in alive)
            or any(alive[i] != (votes[i] >= 0) for i in range(n))):
        raise ValueError("F8 raw ballots do not cover the complete pre-ejection electorate")
    if any(0 <= vote < n and not alive[vote] for vote in votes):
        raise ValueError("F8 raw ballot targets an already dead agent")
    crew = [i for i in range(n) if alive[i] and roles[i] == 0]
    coalition = [i for i in range(n) if alive[i] and roles[i] == 1]
    if len(coalition) != 2:
        raise ValueError("F8 single-meeting raw record lacks both coalition voters")
    actual, actual_tally = tally(votes, n)
    honest_winner, honest_tally = tally([votes[i] for i in crew], n)
    coalition_winner, coalition_tally = tally([votes[i] for i in coalition], n)
    require_equal(record["ejected"], actual, "plurality ejection")
    false_ejection = actual >= 0 and roles[actual] == 0
    expected = dict(actual_tally=actual_tally, crew_only_tally=honest_tally,
        coalition_only_tally=coalition_tally, crew_only_ejected=honest_winner,
        coalition_only_ejected=coalition_winner, false_ejection=int(false_ejection),
        necessary_for_observed_false_ejection=int(false_ejection and honest_winner != actual),
        sufficient_alone_for_observed_false_ejection=int(false_ejection and coalition_winner == actual),
        crew_only_any_false_ejection=int(honest_winner >= 0 and roles[honest_winner] == 0),
        crew_votes_for_ejected=honest_tally[actual] if actual >= 0 else 0,
        coalition_votes_for_ejected=coalition_tally[actual] if actual >= 0 else 0,
        crew_ballots=len(crew), crew_skips=sum(votes[i] == n for i in crew), coalition_pairs=1,
        coalition_same_target=int(votes[coalition[0]] == votes[coalition[1]] < n))
    require_fields(record, expected, "ballot metrics")
    creator = state["incident"][0]
    if type(creator) is not int or not -1 <= creator < n or (creator >= 0 and roles[creator] != 1):
        raise ValueError("F8 raw incident creator is malformed")
    require_fields(record, dict(had_incident=int(creator >= 0),
        creator_ejected=int(creator >= 0 and actual == creator)), "incident metrics")
    claims = state["claims"]
    if any(type(c.get("speaker")) is not int or not 0 <= c["speaker"] < n for c in claims):
        raise ValueError("F8 raw claim speaker is malformed")
    coalition_claims = [c for c in claims if roles[c["speaker"]] == 1]
    require_fields(record, dict(coalition_claims=len(coalition_claims),
        coalition_false_claims=sum(c["label"] == "FALSE" for c in coalition_claims),
        coalition_noinfo_claims=sum(c["claim_type"] == "NO_INFORMATION" for c in coalition_claims)), "claim counts")
    scores = record["scores"]
    if (len(scores) != len(crew) or {s["agent"] for s in scores} != set(crew)
            or any(type(s["agent"]) is not int for s in scores)):
        raise ValueError("F8 raw scores omit, duplicate, or replace an honest pre-ejection voter")
    for scored in scores:
        member = score_vector(scored["member_score"], n, "membership vector")
        creator_score = score_vector(scored["creator_score"], n, "creator vector")
        positive = any(member)
        if positive and not math.isclose(math.fsum(member), 2.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("F8 raw membership scores fail coalition-size normalization")
        if any(creator_score) and not math.isclose(math.fsum(creator_score), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("F8 raw creator scores fail normalization")
        top_two = sorted(range(n), key=lambda i: (-member[i], i))[:2]
        creator_pick = max(range(n), key=lambda i: creator_score[i])
        require_fields(scored, dict(top_membership_score=max(member), zero_score_view=int(not positive),
            top2_is_coalition=int(set(top_two) == set(coalition)) if positive and creator >= 0 else None,
            creator_top_correct=int(creator_pick == creator) if any(creator_score) else None), "confidence flags")
    vote_steps = [step for step in record["steps"] if step["phase"] == 6]
    if len(vote_steps) != 1:
        raise ValueError("F8 raw action trace must contain exactly one vote step")
    actions = vote_steps[0]["actions"]
    if len(actions) != n or any(len(action) != 13 for action in actions):
        raise ValueError("F8 raw vote action array is malformed")
    if any(actions[i][12] != votes[i] for i in range(n) if alive[i]):
        raise ValueError("F8 raw ballots disagree with recorded vote actions")
    return len(scores)


def verify_directory(directory, *, smoke=False):
    """Read-only standalone semantic check; the archive separately checks provenance."""
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text())
    require_equal(complete["smoke"], smoke, "completion smoke flag")
    rows = json.loads((directory / "per_seed.json").read_text())
    if not smoke:
        cells = {(n, seed, trained, against) for n, seeds in SEEDS.items() for seed in seeds
                 for trained in RULES for against in RULES}
        require_equal(len(rows), 135, "production cell count")
        require_equal({(r["n_agents"], r["seed"], r["trained"], r["against"]) for r in rows}, cells, "production cells")
        require_equal(complete["episodes_per_cell"], 500, "production episodes per cell")
    count = views = 0
    reset_hashes = {}
    for row in rows:
        path = confined_raw_path(directory, row["raw"])
        with gzip.open(path, "rt") as stream:
            records = [json.loads(line) for line in stream]
        require_equal(len(records), complete["episodes_per_cell"], "cell episode count")
        key = (row["n_agents"], row["seed"])
        hashes = [r["initial_state_hash"] for r in records]
        require_equal(hashes, reset_hashes.setdefault(key, hashes), "matched reset hashes")
        job = "explicit_smoke" if smoke else f"counterattack_hyp_n{row['n_agents']}_r1_s{row['seed']}"
        checked = complete["checked_inputs"][job]
        require_fields(checked, dict(n_agents=row["n_agents"], seed=row["seed"]), "input job identity")
        checkpoint = checked["checkpoints"][row["trained"]]["sha256"]
        for episode, record in enumerate(records):
            views += verify_record(record, n_agents=row["n_agents"], training_seed=row["seed"],
                episode_index=episode, trained=row["trained"], against=row["against"],
                checkpoint_sha256=checkpoint, smoke=smoke)
            count += 1
    require_equal(count, complete["episodes"], "completion episode count")
    return dict(status="raw semantics verified", smoke=smoke, cells=len(rows), episodes=count, honest_views=views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Validate explicitly identified smoke data; never a production gate")
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.results, smoke=args.smoke), indent=2))
