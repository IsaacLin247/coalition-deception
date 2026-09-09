"""Read-only checks of retained checkpoints; never mutate a frozen training run."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

try:
    from .checkpoint_plan import expected_stage_specs, digest
except ImportError:
    from checkpoint_plan import expected_stage_specs, digest


def validation_code_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("checkpoint_integrity.py", "checkpoint_plan.py")}


@lru_cache(maxsize=64)
def expected_schema(config_json, hidden, gru, centralized, recurrent):
    from social_collusion.config import EnvConfig
    from social_collusion.rl.torch_policy import TorchPolicy

    config = json.loads(config_json)
    config['rooms'] = tuple(config['rooms'])
    config['edges'] = tuple(tuple(edge) for edge in config['edges'])
    config.pop('adjacency', None)
    policy = TorchPolicy(EnvConfig(**config), hidden_dim=hidden, gru_dim=gru,
                         centralized_critic=centralized, use_gru=recurrent, device='cpu')
    return ({key: (tuple(value.shape), str(value.dtype))
             for key, value in policy.model.state_dict().items()}, list(policy.head_sizes))


def read_checkpoint(path, updates, expected=None):
    import torch

    path = Path(path)
    try:
        raw = path.read_bytes()
        blob = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if blob['extra']['updates'] != updates:
            raise ValueError(f"update metadata is {blob['extra']['updates']}, expected {updates}")
        if expected is not None:
            if blob['extra'].get('seed') != expected['seed']:
                raise ValueError(f"planned seed is {expected['seed']}, checkpoint has {blob['extra'].get('seed')}")
            if blob['env_config'] != expected['env_config']:
                changed = sorted(key for key in set(blob['env_config']) | set(expected['env_config'])
                                 if blob['env_config'].get(key) != expected['env_config'].get(key))
                raise ValueError(f"environment differs from planned condition: {changed}")
            if blob['extra'].get('algo') != expected['algo']:
                raise ValueError('algorithm/budget differs from the planned stage')
            for key, value in expected['architecture'].items():
                if blob.get(key) != value:
                    raise ValueError(f'architecture {key} differs from planned stage')
            rollout = int(expected['algo']['rollout_episodes'])
            if blob['extra'].get('episodes') != updates * rollout:
                raise ValueError('episode count differs from planned rollout budget')
            if (not isinstance(blob['extra'].get('env_steps'), int) or blob['extra']['env_steps'] < 0
                    or (updates > 0 and blob['extra']['env_steps'] <= 0)):
                raise ValueError('missing or invalid environment-step metadata')
        config = json.dumps(blob['env_config'], sort_keys=True)
        schema, heads = expected_schema(config, int(blob['hidden_dim']), int(blob['gru_dim']),
                                        bool(blob['centralized']), bool(blob['use_gru']))
        state = blob['state_dict']
        if set(state) != set(schema) or list(blob['head_sizes']) != heads:
            raise ValueError('parameter names or action heads disagree with saved architecture')
        parameters = hashlib.sha256()
        for name in sorted(state):
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'{name} is not a tensor')
            if (tuple(value.shape), str(value.dtype)) != schema[name]:
                raise ValueError(f'{name} shape/dtype disagrees with saved architecture')
            if not torch.isfinite(value).all():
                raise ValueError(f'{name} contains nonfinite parameters')
            parameters.update(name.encode())
            parameters.update(value.detach().contiguous().numpy().tobytes())
        record = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    parameter_sha256=parameters.hexdigest(), updates=updates,
                    config_sha256=hashlib.sha256(config.encode()).hexdigest())
        if expected is not None:
            record.update(seed=blob['extra']['seed'], episodes=blob['extra']['episodes'],
                          env_steps=blob['extra']['env_steps'], algo_sha256=digest(expected['algo']),
                          expected_stage_sha256=digest(expected), learner_role=expected['learner_role'],
                          learner_role_provenance=expected['learner_role_provenance'])
        return record
    except Exception as error:
        raise ValueError(f'Invalid retained checkpoint {path}: {error}') from error


def validate_checkpoints(job, folder):
    records = []
    expected = expected_stage_specs(job)
    for stage, budget in job.get('expected_stages', {}).items():
        directory = Path(folder) / stage
        updates = list(range(job['checkpoint_interval'], budget + 1, job['checkpoint_interval']))
        if job['family'] == 'f1':
            updates.insert(0, 0)
        stage_records = [read_checkpoint(directory / f'checkpoint_{update}.pt', update, expected[stage])
                         for update in updates]
        final = read_checkpoint(directory / 'checkpoint_final.pt', budget, expected[stage])
        if len({r['config_sha256'] for r in [*stage_records, final]}) != 1:
            raise ValueError(f'Checkpoint configurations changed within stage {stage}')
        if updates and updates[-1] == budget and stage_records[-1]['parameter_sha256'] != final['parameter_sha256']:
            raise ValueError(f'Final checkpoint differs from last completed update in stage {stage}')
        if expected[stage] is not None:
            history = json.loads((directory / 'history.json').read_text())['history']
            for record in [*stage_records, final]:
                if record['updates'] == 0:
                    continue
                row = history[record['updates'] - 1]
                if (row.get('updates_completed') != record['updates']
                        or row.get('episodes') != record['episodes'] or row.get('env_steps') != record['env_steps']):
                    raise ValueError(f'Checkpoint counters disagree with retained update history in {stage}')
        records.extend(dict(job=job['name'], stage=stage, **record)
                       for record in [*stage_records, final])
    return records
