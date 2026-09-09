"""Outcome rows must retain their planned seed, population and evaluation stream."""
import itertools
import pytest

from analysis import analyze_submission as analysis


@pytest.mark.parametrize('field,value', [('seed', 1), ('n_agents', 9), ('crew', 7),
                                        ('eval_seed', 987654321), ('episodes', 500)])
def test_f2_rejects_wrong_seed_population_holdout_or_denominator(tmp_path, field, value):
    job = dict(name='f2_tenseed_crew5_s0', family='f2')
    rows = [dict(seed=0, n_agents=7, crew=5, episodes=1000, eval_seed=1987654321,
                 condition=condition, rule=rule, false_ejection_rate=.25,
                 coalition_favorable_rate=.5, creator_ejection_rate=.1)
            for condition, rule in itertools.product(analysis.CONDITIONS, analysis.RULES)]
    path = tmp_path / 'f2_seed_results.csv'
    analysis.write_csv(path, rows)
    data = analysis.Dataset()
    analysis.ingest(job, tmp_path, data)
    assert len(data.rows) == 28 * 3
    rows[0][field] = value
    analysis.write_csv(path, rows)
    with pytest.raises(ValueError, match='identity mismatch'):
        analysis.ingest(job, tmp_path, analysis.Dataset())


def test_counterattack_and_ablation_reject_seed_relabeling_before_aggregation(tmp_path):
    job = dict(name='counterattack_hyp_n7_r1_s0', family='hypothesis')
    rows = [dict(seed=0, n_episodes=1000, coalition_trained_vs=trained, defense=against,
                 false_ejection_rate=.2) for trained, against in itertools.product(('soft','mean','hypothesis'), repeat=2)]
    analysis.write_csv(tmp_path / 'crossplay.csv', rows)
    analysis.ingest(job, tmp_path, analysis.Dataset())
    rows[0]['seed'] = 1
    analysis.write_csv(tmp_path / 'crossplay.csv', rows)
    with pytest.raises(ValueError, match='identity mismatch: seed'):
        analysis.ingest(job, tmp_path, analysis.Dataset())
    job = dict(name='ablation_n7_r1_s0', family='ablation')
    rows = [dict(seed=0, n_agents=7, max_rounds=1, n_episodes=1000, condition=condition,
                 false_ejection_rate=.2) for condition in analysis.ABLATIONS]
    analysis.write_json(tmp_path / 'result.json', {'rows': rows})
    analysis.ingest(job, tmp_path, analysis.Dataset())
    rows[0]['n_agents'] = 5
    analysis.write_json(tmp_path / 'result.json', {'rows': rows})
    with pytest.raises(ValueError, match='identity mismatch: n_agents'):
        analysis.ingest(job, tmp_path, analysis.Dataset())
