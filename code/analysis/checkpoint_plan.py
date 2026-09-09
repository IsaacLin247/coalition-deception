"""Planned checkpoint metadata for the immutable September 2026 study.

Derive expectations from the frozen job command, its pinned driver semantics and
YAML defaults, never from a result's copied runmeta/checkpoint. The frozen save
schema does not embed learner role or opponent identity. Role is reported as
driver-derived; no claim of checkpoint-only opponent identity verification is made.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "cdaa1c031b4e2e6e690771c9a884ed70ed78653814863f99bb2fe0def367ae68"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def frozen_inputs():
    import yaml
    from social_collusion.config import EnvConfig

    directory = REPO
    protocol = json.loads((directory / "submission_protocol.json").read_text())
    source = REPO
    if protocol["source_sha256"] != EXPECTED_SOURCE or digest(protocol["sources"]) != EXPECTED_SOURCE:
        raise ValueError("Checkpoint plan requires the exact frozen September 2026 source")
    for name, expected in protocol["sources"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint plan frozen source changed: {name}")
    # EnvConfig supplies defaults omitted from YAML. Its imported implementation
    # must match the frozen source, whether installed from the snapshot or repo.
    config_source = Path(sys.modules[EnvConfig.__module__].__file__)
    if hashlib.sha256(config_source.read_bytes()).hexdigest() != protocol["sources"]["src/social_collusion/config.py"]:
        raise ValueError("Imported EnvConfig does not match the frozen checkpoint plan")
    read_yaml = lambda name: yaml.safe_load((source / name).read_text())
    return (protocol, read_yaml("configs/algo/ippo.yaml"),
            {name: read_yaml(f"configs/env/{name}.yaml") for name in ("meeting_only", "full_short_game")},
            read_yaml("experiments/multigen/config.yaml"))


def expected_stage_specs(job):
    """Exact per-stage seed, complete environment, algorithm and architecture."""
    if not job.get("expected_stages"):
        return {}
    from social_collusion.config import EnvConfig

    protocol, base_algo, environments, loop_defaults = frozen_inputs()
    planned = [candidate for candidate in protocol["jobs"] if candidate["name"] == job["name"]]
    stripped = lambda value: {key: item for key, item in value.items() if key != "worker"}
    if len(planned) != 1 or stripped(job) != stripped(planned[0]):
        raise ValueError(f"Checkpoint job differs from its frozen protocol: {job['name']}")
    command = job["command"]
    if len(command[1:]) % 2 or any(not flag.startswith("--") for flag in command[1::2]):
        raise ValueError("Unexpected frozen checkpoint command syntax")
    options = dict(zip(command[1::2], command[2::2]))
    if len(options) != len(command[1::2]):
        raise ValueError("Duplicate frozen checkpoint options")
    integer = lambda key, default: int(options.get("--" + key, default))
    seed, n, rounds = integer("seed", 0), integer("n-agents", 7), integer("max-rounds", 1)
    if not job["name"].endswith(f"_s{seed}"):
        raise ValueError("Frozen command seed differs from job ID")
    family = job["family"]

    def env(name, **changes):
        raw = dict(environments[name])
        raw["rooms"] = tuple(raw["rooms"])
        raw["edges"] = tuple(tuple(edge) for edge in raw["edges"])
        return EnvConfig(**raw).with_(**changes).to_dict()

    def spec(config, algo, stage_seed, role):
        return dict(seed=stage_seed, env_config=config, algo=algo,
                    architecture=dict(hidden_dim=int(algo["hidden_dim"]),
                        gru_dim=int(algo["gru_dim"] if algo["use_gru"] else algo["hidden_dim"]),
                        use_gru=bool(algo["use_gru"]), centralized=bool(algo["centralized_critic"])),
                    learner_role=role, learner_role_provenance="derived from frozen driver; not embedded in checkpoint")

    def algo(**changes):
        return dict(base_algo, **changes)

    expected = {}
    if family == "f1":
        config = env("meeting_only", n_agents=n, n_coalition=2, coalition_objective="false_ejection")
        settings = algo(total_updates=integer("updates", 400), eval_every=integer("eval-every", 20),
            eval_episodes=integer("eval-episodes", 400), checkpoint_every=integer("checkpoint-every", 20),
            evaluate_initial=True, n_envs=integer("n-envs", 128), rollout_episodes=integer("rollout-episodes", 128))
        expected[""] = spec(config, settings, seed, "coalition")
    elif family in ("f3", "f4", "budget"):
        config = env("full_short_game", n_agents=n, n_coalition=integer("coalition", 2), max_rounds=rounds,
            vote_aggregation="majority", coalition_objective=options.get("--single-round-objective", "false_ejection")
            if rounds == 1 else "survive", reputation_accuracy=False, reveal_role_on_eject=False)
        common = dict(n_envs=integer("n-envs", 32), rollout_episodes=integer("rollout-episodes", 32),
                      eval_every=integer("eval-every", 50), checkpoint_every=integer("checkpoint-every", 50),
                      eval_episodes=integer("eval-episodes", 200))
        stages = (("coalition0_vs_soft_credibility", "initial-updates", 0, "coalition"),
                  ("crew1_vs_learned_coalition0", "defender-updates", 10000, "crew"),
                  ("coalition1_vs_learned_crew1", "adapted-updates", 20000, "coalition"))
        for name, budget, offset, role in stages:
            expected[name] = spec(config, algo(**common, total_updates=integer(budget, 300)), seed + offset, role)
    elif family in ("counterattack", "hypothesis", "ablation"):
        changes = dict(n_agents=n, n_coalition=2, max_rounds=rounds, vote_aggregation="majority",
            coalition_objective="survive" if rounds > 1 else "balanced_ejection",
            reputation_accuracy=False, reveal_role_on_eject=False)
        is_ablation = family == "ablation"
        if not is_ablation:
            changes.update(dependence_penalty=float(options.get("--dependence-penalty", 4.0)),
                           dependence_window=integer("dependence-window", 3))
        settings = algo(n_envs=integer("n-envs", 32), rollout_episodes=integer("rollout-episodes", 32),
            eval_every=0 if is_ablation else integer("eval-every", 25),
            eval_episodes=integer("eval-episodes", 1000 if is_ablation else 400),
            checkpoint_every=50, total_updates=integer("updates", 400))
        conditions = ({"default": {}, "no_channel": dict(enable_symbol_channel=False, symbol_mode="none"),
            "no_partner": dict(partner_known=False), "no_channel_no_partner": dict(enable_symbol_channel=False,
            symbol_mode="none", partner_known=False)} if is_ablation else
            {name: dict(vote_aggregation="dependence_weighted") if name == "both" else {}
             for name in options["--defenses"].split(",")})
        for name, overrides in conditions.items():
            expected[("coalition_" if is_ablation else "coalition_vs_") + name] = spec(
                env("full_short_game", **dict(changes, **overrides)), settings, seed, "coalition")
    elif family in ("multigen", "dependence", "reward"):
        settings = dict(loop_defaults)
        for key in ("generations", "n_agents", "max_rounds", "dependence_window", "updates_per_stage",
                    "eval_episodes", "crossplay_episodes"):
            settings[key] = integer(key.replace("_", "-"), settings[key])
        for key in ("defense", "multi_round_objective"):
            settings[key] = options.get("--" + key.replace("_", "-"), settings[key])
        settings["dependence_penalty"] = float(options.get("--dependence-penalty", settings["dependence_penalty"]))
        objectives = dict(settings["single_round_objectives"])
        if "--single-round-objectives" in options:
            objectives = dict(zip(("coalition", "crew"), options["--single-round-objectives"].split(",")))
        algorithm = algo(**{key: int(settings[key]) for key in
            ("n_envs", "rollout_episodes", "eval_every", "checkpoint_every", "eval_episodes")},
            total_updates=int(settings["updates_per_stage"]))
        for index in range(2 * settings["generations"] + 1):
            side, generation = ("C", index // 2) if index % 2 == 0 else ("D", (index + 1) // 2)
            role = "coalition" if side == "C" else "crew"
            config = env(settings["env"], n_agents=settings["n_agents"], n_coalition=int(settings["n_coalition"]),
                max_rounds=settings["max_rounds"], vote_aggregation="dependence_weighted" if settings["defense"] == "dependence" else "majority",
                reputation_accuracy=False, reveal_role_on_eject=False,
                dependence_penalty=settings["dependence_penalty"], dependence_window=settings["dependence_window"],
                dependence_top_j=int(settings["dependence_top_j"]), coalition_objective=settings["multi_round_objective"]
                if settings["max_rounds"] > 1 else objectives[role])
            expected[f"stage{index:02d}_{side}{generation}"] = spec(config, algorithm, seed + 10000 * index, role)
    else:
        raise ValueError(f"Unsupported frozen checkpoint family: {family}")
    if {stage: value["algo"]["total_updates"] for stage, value in expected.items()} != job["expected_stages"]:
        raise ValueError("Derived stage budgets differ from the frozen protocol")
    if any(value["algo"]["checkpoint_every"] != job["checkpoint_interval"] for value in expected.values()):
        raise ValueError("Derived retention interval differs from the frozen protocol")
    return expected
