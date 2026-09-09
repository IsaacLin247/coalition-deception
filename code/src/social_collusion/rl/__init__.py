"""PPO / IPPO / MAPPO learners. Importing this module requires torch."""

from social_collusion.rl.networks import (
    ActorCritic,
    MultiHeadActor,
    masked_log_probs,
    sample_actions,
)
from social_collusion.rl.ppo import PPOConfig, PPOLearner
from social_collusion.rl.rollout_buffer import Rollout, phase_stratified_batches
from social_collusion.rl.torch_policy import FrozenCopy, TorchPolicy
from social_collusion.rl.train import CoalitionPolicy, Trainer, TrainResult

__all__ = [
    "ActorCritic",
    "MultiHeadActor",
    "masked_log_probs",
    "sample_actions",
    "PPOConfig",
    "PPOLearner",
    "Rollout",
    "phase_stratified_batches",
    "TorchPolicy",
    "FrozenCopy",
    "Trainer",
    "TrainResult",
    "CoalitionPolicy",
]
