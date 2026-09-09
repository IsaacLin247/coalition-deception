"""PPO's episodic targets must include true terminal rewards across unequal game lengths."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from social_collusion.config import load_env_config
from social_collusion.env.enums import Phase, Role
from social_collusion.env.rewards import terminal_rewards
from social_collusion.policies.random_policy import RandomPolicy
from social_collusion.policies.scripted_crew import TruthfulCrew
from social_collusion.rl.train import Trainer


@pytest.mark.parametrize("rounds,learner_role", [(1, Role.COALITION), (3, Role.COALITION), (3, Role.CREW)])
def test_collect_preserves_complete_episodes_and_terminal_credit(rounds, learner_role):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = load_env_config("full_short_game").with_(n_agents=7, max_rounds=rounds)
        trainer = Trainer(
            cfg,
            {"n_envs": 4, "rollout_episodes": 5, "total_updates": 1},
            TruthfulCrew(),
            seed=13,
            learner_role=learner_role,
            opponent_policy=RandomPolicy() if learner_role == Role.CREW else None,
        )
        completed = {}
        autoreset = trainer.vec.autoreset

        def record_completed():
            completed.update({
                episode: state
                for episode, state in zip(trainer.vec.episode_ids, trainer.vec.states)
                if state.phase == int(Phase.TERMINAL)
            })
            return autoreset()

        trainer.vec.autoreset = record_completed
        previous_ids = set()
        for _ in range(2):
            completed.clear()
            buffers, finished = trainer.collect()
            buffer = buffers[0]
            assert set(buffer.ep_id) == set(completed)
            assert len(finished) == 8  # enough complete batches to cover the five-episode target
            assert not previous_ids.intersection(buffer.ep_id)
            previous_ids.update(buffer.ep_id)
            assert all(s.phase == int(Phase.FREE_PLAY) and s.turn == 0 for s in trainer.vec.states)

            # With gamma=lambda=1, every stored action must receive that agent's exact full
            # episode return, including agents eliminated before their teammates finish.
            batch = buffer.finalize(gamma=1.0, gae_lambda=1.0)
            expected = np.array([
                terminal_rewards(completed[episode])[agent]
                for episode, agent in zip(buffer.ep_id, buffer.agent)
            ])
            np.testing.assert_allclose(batch["ret"], expected, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)


def test_default_evaluation_uses_separate_evidence_seed_and_respects_override():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = load_env_config("meeting_only")
        trainer = Trainer(cfg, {"n_envs": 2}, TruthfulCrew(), seed=13)
        training_positions = [s.positions.copy() for s in trainer.vec.states]
        states = trainer.eval_states(2)
        assert all(s.seed == 987654334 for s in states)
        assert all(s.seed != trainer.seed for s in states)
        assert any(not np.array_equal(s.positions, p) for s, p in zip(states, training_positions))
        explicit = trainer.eval_states(2, eval_seed=456)
        assert all(s.seed == 456 for s in explicit)
    finally:
        torch.set_num_threads(previous_threads)
