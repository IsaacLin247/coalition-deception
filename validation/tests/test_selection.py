"""Scientific selection invariants; these tests run no game simulation."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selector import rank_candidates


def cell(fe, capture, n=1000):
    return {"episodes": n, "false_ejections": fe, "creator_ejections": capture,
            "no_ejections": n - fe - capture, "incidents": n}


def fixture():
    metadata = {"candidate_order": ["legacy", "random", "skip"],
                "task_names": ["truthful", "adversarial"], "truthful_task": "truthful",
                "baseline_defense": "legacy", "fallback_defense": "legacy",
                "minimum_truthful_creator_capture": .20,
                "maximum_truthful_capture_loss": .05,
                "maximum_overall_capture_loss": .05}
    definitions = {name: {"name": name} for name in metadata["candidate_order"]}
    counts = {"legacy": {"truthful": cell(200, 500), "adversarial": cell(400, 400)},
              "random": {"truthful": cell(220, 500), "adversarial": cell(320, 400)},
              "skip": {"truthful": cell(0, 0), "adversarial": cell(0, 0)}}
    return metadata, definitions, counts


def test_abstain_everywhere_cannot_win_by_zero_false_ejection():
    metadata, definitions, counts = fixture()
    result = rank_candidates(metadata, counts, definitions)
    assert result["selected_candidate"] == "random"
    assert "skip" not in result["eligible_candidates"]


def test_minimax_rule_does_not_select_lower_average_with_worse_task():
    metadata, definitions, counts = fixture()
    counts["random"] = {"truthful": cell(0, 500), "adversarial": cell(450, 400)}
    result = rank_candidates(metadata, counts, definitions)
    assert result["selected_candidate"] == "legacy"


def test_exact_capture_boundary_is_eligible():
    metadata, definitions, counts = fixture()
    counts["random"] = {"truthful": cell(100, 450), "adversarial": cell(300, 350)}
    result = rank_candidates(metadata, counts, definitions)
    assert result["selected_candidate"] == "random"
    assert result["capture_floors_exact"] == ["9/20", "2/5"]


def test_failed_absolute_capture_gate_falls_back_without_relaxation():
    metadata, definitions, counts = fixture()
    for candidate in counts.values():
        candidate["truthful"] = cell(0, 100)
    result = rank_candidates(metadata, counts, definitions)
    assert result["eligible_candidates"] == []
    assert result["selected_candidate"] == "legacy"
    assert result["decision"] == "fallback_no_eligible_candidate"


def test_equal_rational_scores_follow_fixed_candidate_order():
    metadata, definitions, counts = fixture()
    counts["random"] = {"truthful": cell(100, 250, 500),
                        "adversarial": cell(200, 200, 500)}
    result = rank_candidates(metadata, counts, definitions)
    assert result["selected_candidate"] == "legacy"


@pytest.mark.parametrize("mutation", ["missing_candidate", "missing_task", "bad_count", "duplicate_order"])
def test_invalid_selection_inputs_fail_closed(mutation):
    metadata, definitions, counts = fixture()
    if mutation == "missing_candidate":
        counts.pop("skip")
    elif mutation == "missing_task":
        counts["random"].pop("adversarial")
    elif mutation == "bad_count":
        counts["random"]["truthful"]["creator_ejections"] = 1001
    else:
        metadata["candidate_order"].append("random")
    with pytest.raises(ValueError):
        rank_candidates(metadata, counts, definitions)
