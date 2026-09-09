"""Pure game engine: numpy only, no torch and no PettingZoo below `pettingzoo_env`."""

from social_collusion.env.action_masks import compute_masks, head_sizes, sample_masked
from social_collusion.env.enums import (
    ClaimType,
    Head,
    Phase,
    ResponseType,
    Role,
    RoomAction,
    TruthLabel,
)
from social_collusion.env.observations import build_actor_obs, build_all_actor_obs, obs_dim
from social_collusion.env.state import Claim, Event, GameState, Response, TransitionResult
from social_collusion.env.transition import IllegalActionError, reset, transition
from social_collusion.env.vec_env import VecEnv, play_episode

__all__ = [
    "Phase",
    "Role",
    "RoomAction",
    "ClaimType",
    "ResponseType",
    "TruthLabel",
    "Head",
    "GameState",
    "Claim",
    "Response",
    "Event",
    "TransitionResult",
    "reset",
    "transition",
    "IllegalActionError",
    "compute_masks",
    "head_sizes",
    "sample_masked",
    "build_actor_obs",
    "build_all_actor_obs",
    "obs_dim",
    "VecEnv",
    "play_episode",
]
