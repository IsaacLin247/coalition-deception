"""Cross-host ingestion keeps exported CSVs confined to their completed job."""
from pathlib import Path

import pytest

from analysis import analyze_submission as analysis


def loop_fixture(folder, separator):
    """One-generation producer schema, including all seven raw evaluation cells."""
    job = dict(name="multigen_none_n7_r8_k1_s0", family="multigen",
               command=["experiments/multigen/run_multigen.py", "--eval-episodes", "2", "--crossplay-episodes", "2"])
    games = [dict(false_ejection=True, coalition_favorable=True, coalition_game_won=False,
                  total_false_ejections=1, total_meetings=1),
             dict(false_ejection=False, coalition_favorable=False, coalition_game_won=True,
                  total_false_ejections=2, total_meetings=4)]
    summary = analysis.episode_summary(games)
    stages = [("C", 0, "stage00_C0"), ("D", 1, "stage01_D1"), ("C", 1, "stage02_C1")]
    analysis.write_json(folder / "generations.json", {"rows": [
        dict(side=side, generation=generation, stage_index=index, stage_seed=index * 10000,
             n_agents=7, n_crew=5, n_coalition=2, max_rounds=8, eval_seed=1987654321, n_episodes=2,
             **summary) for index, (side, generation, _) in enumerate(stages)]})
    analysis.write_json(folder / "crossplay.json", {
        "matrices": {key: [[value, value], [value, value]] for key, value in summary.items()},
        "episodes_per_cell": 2})
    analysis.write_json(folder / "policy_matrices.json", {})
    evaluations = [("stage", stage_id) for _, _, stage_id in stages]
    evaluations += [("crossplay", f"C{i}_vs_D{j}") for i in range(2) for j in range(2)]
    records = []
    for scope, identifier in evaluations:
        if scope == "stage":
            side, generation, _ = next(stage for stage in stages if stage[2] == identifier)
            pair = f"C{generation if side == 'C' else generation - 1}_vs_D{generation}"
        else:
            pair = identifier
        episode_name = f"evaluation/{identifier}/episode_metrics.csv"
        round_name = f"evaluation/{identifier}/round_metrics.csv"
        analysis.write_csv(folder / episode_name, [dict(game, episode_index=i, pair=pair, eval_seed=1987654321)
                                                  for i, game in enumerate(games)])
        analysis.write_csv(folder / round_name, [dict(round=i, pair=pair, eval_seed=1987654321) for i in range(5)])
        records.append(dict(evaluation_id=identifier, scope=scope, pair=pair,
                            eval_seed=1987654321, n_episodes=2, n_meetings=5,
                            episode_metrics=episode_name.replace("/", separator),
                            round_metrics=round_name.replace("/", separator)))
    analysis.write_json(folder / "evaluation_records.json", {"evaluations": records})
    return job, records


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_ingest_complete_loop_from_posix_and_windows_exports(tmp_path, separator):
    job, _ = loop_fixture(tmp_path, separator)
    data = analysis.Dataset()
    analysis.ingest(job, tmp_path, data)
    group = "multigen_none_n7_r8_k1"
    assert data.cells[(group, "crossplay", "C1|D0", "false_ejection_rate")] == {0: .5}
    assert data.cells[(group, "crossplay", "C1|D0", "false_ejections_per_meeting")] == {0: .6}
    assert data.cells[(group, "stages", "D1", "mean_false_ejections")] == {0: 1.5}
    assert len(data.gaps) == 4


@pytest.mark.parametrize('field,value', [('stage_seed', 999), ('n_agents', 9), ('max_rounds', 1)])
def test_loop_rejects_relabelled_stage_identity(tmp_path, field, value):
    job, _ = loop_fixture(tmp_path, '/')
    path = tmp_path / 'generations.json'
    payload = analysis.read_json(path)
    payload['rows'][1][field] = value
    analysis.write_json(path, payload)
    with pytest.raises(ValueError, match='identity mismatch'):
        analysis.ingest(job, tmp_path, analysis.Dataset())


@pytest.mark.parametrize('field,value', [('eval_seed', 987654321), ('pair', 'C0_vs_D0'), ('n_episodes', 1)])
def test_loop_rejects_wrong_evaluation_stream_pair_or_count(tmp_path, field, value):
    job, records = loop_fixture(tmp_path, '/')
    records[1][field] = value
    analysis.write_json(tmp_path / 'evaluation_records.json', {'evaluations': records})
    with pytest.raises(ValueError, match='identity mismatch'):
        analysis.ingest(job, tmp_path, analysis.Dataset())


def test_loop_rejects_misidentified_or_duplicate_raw_episode(tmp_path):
    job, records = loop_fixture(tmp_path, '/')
    path = tmp_path / records[0]['episode_metrics']
    rows = analysis.read_csv(path)
    rows[1]['episode_index'] = 0
    analysis.write_csv(path, rows)
    with pytest.raises(ValueError, match='Raw episode identity mismatch'):
        analysis.ingest(job, tmp_path, analysis.Dataset())


@pytest.mark.parametrize("field", ["episode_metrics", "round_metrics"])
@pytest.mark.parametrize("unsafe", [
    "../outside.csv", "evaluation/../outside.csv", "..\\outside.csv",
    "evaluation\\..\\outside.csv", "/outside.csv", "\\outside.csv",
    "C:/outside.csv", "C:\\outside.csv", "C:outside.csv", "//server/share/outside.csv",
    "\\\\server\\share\\outside.csv", "evaluation/file.csv:stream", "", ".",
    "evaluation/./file.csv", "evaluation//file.csv", "file\x00.csv", None,
])
def test_ingest_rejects_unconfined_export_in_either_csv_field(tmp_path, field, unsafe):
    job, records = loop_fixture(tmp_path, "\\")
    records[0][field] = unsafe
    analysis.write_json(tmp_path / "evaluation_records.json", {"evaluations": records})
    with pytest.raises(ValueError, match="Unsafe relative export path"):
        analysis.ingest(job, tmp_path, analysis.Dataset())


def test_export_path_rejects_symlink_escape_and_accepts_mixed_separators(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    assert analysis.exported_artifact_path(root, "evaluation\\C0_vs_D0/episode_metrics.csv") == (
        root / "evaluation/C0_vs_D0/episode_metrics.csv")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "escaped").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Directory symlinks are unavailable on this host")
    with pytest.raises(ValueError, match="escapes job directory"):
        analysis.exported_artifact_path(root, "escaped\\episode_metrics.csv")
