"""Semantic corruption regressions use small synthetic raw records, not checkpoints."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("independent_vote_verifier", Path(__file__).with_name("verify_vote_diagnostic.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def valid_record():
    roles, votes = [1, 1, 0, 0, 0, 0, 0], [2, 3, 4, 2, 7, 7, -1]
    evidence, actor = verifier.expected_seeds(7, 0, 0)
    scored = dict(member_score=[1., 1., 0., 0., 0., 0., 0.], creator_score=[1., 0., 0., 0., 0., 0., 0.],
                  top_membership_score=1., zero_score_view=0, top2_is_coalition=1, creator_top_correct=1)
    actions = [[0] * 13 for _ in range(7)]
    for i, vote in enumerate(votes):
        actions[i][12] = vote if vote >= 0 else 0
    return dict(n_agents=7, training_seed=0, episode_index=0, trained="soft", against="mean",
        checkpoint_sha256="a"*64, evidence_seed=evidence, actor_seed=actor,
        roles=roles, votes=votes, ejected=2, actual_tally=[0,0,2,1,1,0,0],
        crew_only_tally=[0,0,1,0,1,0,0], coalition_only_tally=[0,0,1,1,0,0,0],
        crew_only_ejected=-1, coalition_only_ejected=-1, false_ejection=1,
        necessary_for_observed_false_ejection=1, sufficient_alone_for_observed_false_ejection=0,
        crew_only_any_false_ejection=0, crew_votes_for_ejected=1, coalition_votes_for_ejected=1,
        crew_ballots=4, crew_skips=2, coalition_pairs=1, coalition_same_target=0,
        had_incident=1, creator_ejected=0, coalition_claims=2, coalition_false_claims=1, coalition_noinfo_claims=1,
        pre_vote_state=dict(phase=6, done=False, ejected=-1, roles=list(roles), votes=[-1]*7,
            alive=[True]*6+[False], incident=[0,6,2,0], claims=[
                dict(speaker=0, claim_type="NO_INFORMATION", label="TRUE"),
                dict(speaker=1, claim_type="SELF_LOCATION", label="FALSE")]),
        scores=[dict(agent=i, **copy.deepcopy(scored)) for i in [2,3,4,5]],
        steps=[dict(phase=6, actions=actions)])


def check(row):
    return verifier.verify_record(row, n_agents=7, training_seed=0, episode_index=0,
        trained="soft", against="mean", checkpoint_sha256="a"*64)


def test_necessity_and_sufficiency_are_independently_recomputed():
    # Coalition splits its ballots: necessary for the innocent's plurality,
    # but its two votes alone would tie. Two honest voters abstain.
    assert check(valid_record()) == 4
    for key in ["necessary_for_observed_false_ejection", "sufficient_alone_for_observed_false_ejection",
                "crew_only_any_false_ejection", "coalition_same_target"]:
        row = valid_record()
        row[key] = 1 - row[key]
        with pytest.raises(ValueError, match=key):
            check(row)


@pytest.mark.parametrize("field", ["crew_skips", "crew_ballots", "false_ejection", "crew_votes_for_ejected",
                                   "coalition_votes_for_ejected", "had_incident", "creator_ejected",
                                   "coalition_claims", "coalition_false_claims", "coalition_noinfo_claims"])
def test_ballot_denominators_and_claim_incident_counts_are_not_trusted(field):
    row = valid_record()
    row[field] += 1
    with pytest.raises(ValueError, match=field):
        check(row)


@pytest.mark.parametrize("field", ["evidence_seed", "actor_seed"])
def test_common_but_wrong_random_seeds_are_rejected(field):
    row = valid_record()
    row[field] += 1
    with pytest.raises(ValueError, match=field):
        check(row)


@pytest.mark.parametrize("field", ["top_membership_score", "zero_score_view", "top2_is_coalition", "creator_top_correct"])
def test_confidence_flags_are_recomputed_from_vectors(field):
    row = valid_record()
    row["scores"][0][field] = 1 - row["scores"][0][field]
    with pytest.raises(ValueError, match=field):
        check(row)


def test_zero_score_views_keep_undefined_flags_and_stable_tie_order():
    row = valid_record()
    row["scores"][0].update(member_score=[0.]*7, creator_score=[0.]*7,
        top_membership_score=0., zero_score_view=1, top2_is_coalition=None, creator_top_correct=None)
    row["scores"][-1].update(member_score=[.5,.5,.5,.5,0.,0.,0.], top_membership_score=.5)
    assert check(row) == 4
    row["scores"][0]["top2_is_coalition"] = 0
    with pytest.raises(ValueError, match="top2_is_coalition"):
        check(row)


def test_ejected_voter_cannot_be_omitted_or_replaced():
    for replacement in [None, 3, 0]:
        row = valid_record()
        if replacement is None:
            row["scores"].pop(0)  # This voter is ejected after casting their ballot.
        else:
            row["scores"][0]["agent"] = replacement
        with pytest.raises(ValueError, match="honest pre-ejection voter"):
            check(row)


def test_score_vectors_require_finite_normalized_values():
    for value in [float("nan"), float("inf"), -.1, 1.1, .2]:
        row = valid_record()
        row["scores"][0]["member_score"][0] = value
        with pytest.raises(ValueError, match="scores"):
            check(row)


def test_ballots_electorate_and_action_trace_agree():
    row = valid_record()
    row["pre_vote_state"]["alive"][2] = False  # The old post-ejection sampling bug.
    with pytest.raises(ValueError, match="pre-ejection electorate"):
        check(row)
    row = valid_record()
    row["steps"][0]["actions"][2][12] = 3
    with pytest.raises(ValueError, match="vote actions"):
        check(row)
    row = valid_record()
    row["ejected"] = -1
    with pytest.raises(ValueError, match="plurality ejection"):
        check(row)


def test_smoke_seed_never_passes_the_default_production_gate():
    row = valid_record()
    evidence, actor = verifier.expected_seeds(7, 91, 0)
    row.update(training_seed=91, evidence_seed=evidence, actor_seed=actor)
    args = dict(n_agents=7, training_seed=91, episode_index=0, trained="soft", against="mean", checkpoint_sha256="a"*64)
    with pytest.raises(ValueError, match="fixed production design"):
        verifier.verify_record(row, **args)
    assert verifier.verify_record(row, smoke=True, **args) == 4


def test_independent_seed_formula_matches_a_retained_smoke_identity():
    assert verifier.expected_seeds(7, 91, 0) == (2988564328, 17703173249010507)


def test_saved_windows_raw_paths_resolve_on_linux(tmp_path):
    raw = tmp_path / "raw" / "one.jsonl.gz"
    raw.parent.mkdir()
    raw.write_bytes(b"test")
    assert verifier.confined_raw_path(tmp_path, "raw\\one.jsonl.gz") == raw


@pytest.mark.parametrize("name", ["../outside", "raw/../../outside", "raw\\..\\outside",
    "/tmp/outside", "C:\\outside", "\\\\server\\outside", "raw//outside", "raw/./outside", "raw/evil\x00.gz"])
def test_raw_references_cannot_escape_the_result_directory(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe F8 raw path"):
        verifier.confined_raw_path(tmp_path, name)


def test_linked_raw_files_are_refused(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "elsewhere"
    outside.write_text("unrelated")
    (raw / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="Linked F8 raw path"):
        verifier.confined_raw_path(tmp_path, "raw/linked")
