"""Evaluation must sample episode indices, not fast completions or training RNG state."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from social_collusion.config import load_env_config
from social_collusion.env.enums import Phase, N_HEADS
from social_collusion.env.vec_env import VecEnv
from social_collusion.policies.base import BasePolicy
from social_collusion.policies.scripted_crew import TruthfulCrew
from social_collusion.rl.train import Trainer
from social_collusion.rl.torch_policy import TorchPolicy


class NoopPolicy(BasePolicy):
    def act(self, states, obs, masks, active, rng):
        return np.zeros((*active.shape, N_HEADS), dtype=int)


@pytest.mark.parametrize('count', [2, 3])
def test_fixed_episode_indices_include_slow_initial_episode(count):
    cfg = load_env_config('full_short_game')
    env = VecEnv(cfg, 2, seed=11)
    elapsed = {}

    def controlled_step(actions, validate=False):
        # Episode 0 takes five ticks; all later episodes take one. The old
        # completion-count stopping rule selected episodes 1, 2, ... and omitted 0.
        for b, state in enumerate(env.states):
            episode = env.episode_ids[b]
            state.seed = episode  # observable identity of the chosen environment draw
            elapsed[episode] = elapsed.get(episode, 0) + 1
            if elapsed[episode] >= (5 if episode == 0 else 1):
                state.phase = int(Phase.TERMINAL)
        return np.zeros((2, cfg.n_agents)), env.done(), [{}, {}]

    env.step = controlled_step
    states = env.run_episodes(NoopPolicy(), count)
    assert [state.seed for state in states] == list(range(count))
    assert all(state.phase == int(Phase.TERMINAL) for state in states)


def test_evaluation_is_repeatable_and_does_not_advance_training_rng():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tr = Trainer(load_env_config('meeting_only'), {'n_envs': 2}, TruthfulCrew(), seed=12)
        before = tr.actors[0].generator.get_state().clone()
        first = tr.evaluate(4, eval_seed=222)
        second = tr.evaluate(4, eval_seed=222)
        assert torch.equal(before, tr.actors[0].generator.get_state())
        for key in first:
            assert first[key] == pytest.approx(second[key], nan_ok=True)
    finally:
        torch.set_num_threads(old_threads)


def test_untrained_checkpoint_and_completed_update_metadata(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tr = Trainer(load_env_config('meeting_only'), {
            'n_envs': 2, 'rollout_episodes': 2, 'total_updates': 1,
            'eval_episodes': 2, 'eval_every': 1, 'checkpoint_every': 1,
            'evaluate_initial': True, 'ppo_epochs': 1, 'n_minibatches': 1,
            'hidden_dim': 16,
        }, TruthfulCrew(), seed=1, run_dir=tmp_path)
        result = tr.train(log_every=1)
        initial = torch.load(tmp_path / 'checkpoint_0.pt', weights_only=False)
        final = torch.load(tmp_path / 'checkpoint_1.pt', weights_only=False)
        assert initial['extra']['updates'] == 0
        assert final['extra']['updates'] == 1
        assert [r['updates_completed'] for r in result.eval_curve] == [0, 1]
        assert any(not torch.equal(initial['state_dict'][key], final['state_dict'][key])
                   for key in initial['state_dict'])
        loaded = TorchPolicy.load(tmp_path / 'checkpoint_1.pt')
        assert loaded.model.actor.torso[0].out_features == 16
    finally:
        torch.set_num_threads(old_threads)
