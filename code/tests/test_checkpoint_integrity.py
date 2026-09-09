"""Corrupt or misidentified checkpoints must not qualify as reproducible runs."""
import pytest
import torch

from analysis import checkpoint_integrity as integrity
from analysis.checkpoint_integrity import read_checkpoint, validate_checkpoints
from analysis.checkpoint_plan import expected_stage_specs, frozen_inputs
from social_collusion.config import load_env_config
from social_collusion.rl.torch_policy import TorchPolicy


@pytest.fixture
def saved(tmp_path):
    policy = TorchPolicy(load_env_config('meeting_only').with_(n_agents=5),
                         hidden_dim=8, gru_dim=8)
    path = tmp_path / 'checkpoint_20.pt'
    policy.save(path, extra={'updates': 20})
    return policy, path


def test_checkpoint_hash_and_update_validation(saved):
    _, path = saved
    result = read_checkpoint(path, 20)
    assert result['bytes'] > 0 and len(result['sha256']) == 64
    assert read_checkpoint(path, 20) == result
    with pytest.raises(ValueError, match='update metadata'):
        read_checkpoint(path, 40)


@pytest.mark.parametrize('defect', ['empty', 'nan', 'shape', 'heads'])
def test_unusable_checkpoint_rejected(saved, defect):
    _, path = saved
    if defect == 'empty':
        path.write_bytes(b'')
    else:
        blob = torch.load(path, weights_only=True)
        name = next(iter(blob['state_dict']))
        if defect == 'nan':
            blob['state_dict'][name].fill_(float('nan'))
        elif defect == 'shape':
            blob['state_dict'][name] = torch.zeros(1)
        else:
            blob['head_sizes'][0] += 1
        torch.save(blob, path)
    with pytest.raises(ValueError, match='Invalid retained checkpoint'):
        read_checkpoint(path, 20)


def test_final_checkpoint_must_match_last_budget_parameters(saved, monkeypatch):
    policy, path = saved
    job = dict(name='test_s0', family='f3', expected_stages={'': 20}, checkpoint_interval=20)
    # Isolate tensor equality here; real frozen metadata is tested below.
    monkeypatch.setattr(integrity, 'expected_stage_specs', lambda job: {'': None})
    policy.save(path.parent / 'checkpoint_final.pt', extra={'updates': 20})
    records = validate_checkpoints(job, path.parent)
    assert len(records) == 2
    assert records[0]['parameter_sha256'] == records[1]['parameter_sha256']
    with torch.no_grad():
        next(policy.model.parameters()).add_(1)
    policy.save(path.parent / 'checkpoint_final.pt', extra={'updates': 20})
    with pytest.raises(ValueError, match='differs from last'):
        validate_checkpoints(job, path.parent)


def planned(name):
    return next(job for job in frozen_inputs()[0]['jobs'] if job['name'] == name)


def test_all_frozen_stage_specs_include_exact_seed_offsets_rewards_and_information_conditions():
    jobs = frozen_inputs()[0]['jobs']
    assert sum(len(expected_stage_specs(job)) for job in jobs) == 1255
    specs = expected_stage_specs(planned('f3_tenseed_crew5_s3'))
    assert [record['seed'] for record in specs.values()] == [3, 10003, 20003]
    assert [record['learner_role'] for record in specs.values()] == ['coalition', 'crew', 'coalition']
    assert all(record['env_config']['actor_private_history'] for record in specs.values())
    loop = expected_stage_specs(planned('multigen_none_n7_r1_k10_s2'))
    assert loop['stage01_D1']['seed'] == 10002
    assert loop['stage01_D1']['env_config']['coalition_objective'] == 'meeting_ejection'
    assert loop['stage02_C1']['env_config']['coalition_objective'] == 'balanced_ejection'
    reward = expected_stage_specs(planned('multigen_rwbal_n7_r1_k4_s2'))
    assert reward['stage01_D1']['env_config']['coalition_objective'] == 'balanced_ejection'
    ablation = expected_stage_specs(planned('ablation_n7_r1_s0'))
    assert ablation['coalition_default']['env_config']['partner_known'] is True
    assert ablation['coalition_no_partner']['env_config']['partner_known'] is False
    assert ablation['coalition_no_channel']['env_config']['enable_symbol_channel'] is False
    attack = expected_stage_specs(planned('counterattack_n7_r1_s0'))
    assert attack['coalition_vs_both']['env_config']['vote_aggregation'] == 'dependence_weighted'
    assert attack['coalition_vs_rule']['env_config']['vote_aggregation'] == 'majority'
    fake = dict(planned('f3_tenseed_crew5_s3'), checkpoint_interval=100)
    with pytest.raises(ValueError, match='frozen protocol'):
        expected_stage_specs(fake)


@pytest.fixture
def planned_checkpoint(tmp_path):
    spec = expected_stage_specs(planned('ablation_n7_r1_s0'))['coalition_no_partner']
    from social_collusion.config import EnvConfig
    config = dict(spec['env_config'])
    config['rooms'] = tuple(config['rooms'])
    config['edges'] = tuple(tuple(edge) for edge in config['edges'])
    policy = TorchPolicy(EnvConfig(**config), hidden_dim=128, gru_dim=128)
    path = tmp_path / 'checkpoint_50.pt'
    policy.save(path, extra=dict(updates=50, seed=spec['seed'], algo=spec['algo'],
                                episodes=50 * spec['algo']['rollout_episodes'], env_steps=12345))
    return path, spec


def test_checkpoint_matches_frozen_spec_without_trusting_copied_runmeta(planned_checkpoint):
    path, spec = planned_checkpoint
    record = read_checkpoint(path, 50, spec)
    assert record['seed'] == 0 and record['learner_role'] == 'coalition'
    assert 'not embedded' in record['learner_role_provenance']
    assert len(record['expected_stage_sha256']) == 64


@pytest.mark.parametrize('defect,match', [
    ('seed', 'planned seed'), ('partner', 'planned condition'), ('history', 'planned condition'),
    ('reward', 'planned condition'), ('algorithm', 'algorithm/budget'), ('architecture', 'architecture'),
    ('episodes', 'rollout budget')])
def test_internally_valid_wrong_seed_or_condition_is_rejected(planned_checkpoint, defect, match):
    path, spec = planned_checkpoint
    blob = torch.load(path, weights_only=True)
    if defect == 'seed':
        blob['extra']['seed'] += 1
    elif defect == 'partner':
        # Changing partner availability leaves tensor shapes internally valid.
        blob['env_config']['partner_known'] = True
    elif defect == 'history':
        blob['env_config']['actor_private_history'] = False
    elif defect == 'reward':
        blob['env_config']['coalition_objective'] = 'survive'
    elif defect == 'algorithm':
        blob['extra']['algo']['lr'] *= 2
    elif defect == 'architecture':
        blob['centralized'] = True
    else:
        blob['extra']['episodes'] -= 32
    torch.save(blob, path)
    with pytest.raises(ValueError, match=match):
        read_checkpoint(path, 50, spec)


def test_same_seed_opponent_identity_is_explicitly_not_embedded():
    specs = expected_stage_specs(planned('counterattack_hyp_n7_r1_s0'))
    # The frozen producer does not serialize the opponent. Do not manufacture a
    # condition guarantee unsupported by these identical metadata expectations.
    assert specs['coalition_vs_soft'] == specs['coalition_vs_hypothesis']
