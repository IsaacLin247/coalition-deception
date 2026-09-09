"""Social-Collusion Benchmark.

A compact, deterministic, partially observable social-deduction environment built so that
hidden-coalition *collusion* can be identified causally rather than inferred from win rate.

Layering (deliberate, see docs/research_contract.md):

    env/          pure game engine - numpy only, no torch, no PettingZoo
    policies/     scripted baselines + the supervised belief listener
    rl/           PPO / IPPO / MAPPO learners (torch, imported lazily)
    metrics/      outcome, coordination, deception, communication, causal metrics
    replay/       event-sourced episode records and their validator
    visualization/ renderers that *consume* replays and never mutate state
"""

__version__ = "0.1.0"

from social_collusion.config import EnvConfig, load_algo_config, load_env_config  # noqa: E402

__all__ = ["EnvConfig", "load_env_config", "load_algo_config", "__version__"]
