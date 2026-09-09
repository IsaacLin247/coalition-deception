# Research code

This directory contains the audited environment, policies, PPO training, complete 240-job study runner, scientific supplements, analysis, and regression tests. The accompanying paper is a working manuscript; the corrected replication is still in progress. This public package contains code and protocols, not completed research data or checkpoints.

## Install and verify

Use Python 3.10 or newer in a dedicated environment. Run these commands from this `code/` directory:

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -e '.[rl,env,analysis,dev]'
python -m pytest tests
python -m pytest --import-mode=importlib audit/submission/test_vote_diagnostic.py audit/submission/test_verify_vote_diagnostic.py
python experiments/submission/run_study.py status
```

The editable install is required for this repository workflow because configuration files live alongside the source tree. Dependencies are NumPy, PyTorch, PyYAML, pandas, matplotlib, PettingZoo, Gymnasium, pytest, and Hypothesis; the installation command includes all of them. Production experiments use CPU execution and one computation thread per job. Results can differ across library versions and platforms even with identical seeds.

## Run the complete study

`submission_protocol.json` is already frozen for this exact portable source. It has the same 240 job definitions, seeds, training budgets, evaluations, and comparison design as the original post-audit study. A different source fingerprint records the cleaned inventory and updated replay presentation utility. Do not use this protocol to relabel results produced by the original frozen source.

```sh
python experiments/submission/run_study.py list
python experiments/submission/run_study.py run --worker all --slots 4 --results results/reproduction
python experiments/submission/run_study.py status --worker all --results results/reproduction
```

Four slots is an example; select a concurrency level your machine can support. The complete study is substantial: 1,255 training stages and 512,000 PPO updates, plus scripted evaluation. The runner checks dependencies, source identity, required artifacts, and completion markers. It resumes by skipping completed jobs. Interrupted individual jobs do not resume an intermediate checkpoint automatically. Worker names are scheduling labels; `--worker all` runs the entire plan on one machine. A `--families` subset is useful for development but cannot satisfy the full analysis gate.

Do not edit hashed source files during a reproduction. Changes require a separately documented new protocol and matching supplement/analysis pins; they must not overwrite or masquerade as this package's frozen design. The original ongoing computation is separate and remains unchanged.

## Analyze completed outputs

```sh
python analysis/analyze_submission.py --protocol submission_protocol.json --runs results/reproduction --out results/analysis
python analysis/render_submission.py --analysis results/analysis --out results/presentation
```

The analyzer verifies job identity, checkpoints, raw outcomes, all planned seeds, and the complete comparison set. Final inference is blocked while required evidence is missing. To inspect an unfinished reproduction, add `--interim` to both commands; this explicitly suppresses inferential p-values and confidence intervals. The scientific definitions and statistical assumptions are in [the analysis plan](audit/submission/ANALYSIS_PLAN.md).

The current `analyze_ablation_budget.py` and `analyze_revision.py` helpers are also retained because the regression suite checks their repaired statistical estimands and matched reference cohorts. The complete reproduction workflow uses the gated commands above.

These commands regenerate tables and plots from a new reproduction. They do not silently replace the accompanying manuscript's figures or convert provisional manuscript claims into final findings.

## Mechanism supplement

The independently frozen mechanism design has 30 jobs and 540,000 episodes. It holds the environment evidence fixed across three aggregation rules and three mechanism variants. The results are descriptive.

```sh
python audit/submission/replicate_mechanisms.py self-test
python audit/submission/run_mechanism_queue.py
```

The queue runs two processes and writes `audit/submission/mechanism_results/`; it performs the independently checked final analysis after all jobs succeed. To run a selected planned job or choose a different output directory:

```sh
python audit/submission/replicate_mechanisms.py run --crew 3 --replicate 0 --out results/mechanisms
python audit/submission/replicate_mechanisms.py analyze --out results/mechanisms
```

The last command requires all 30 planned jobs at that destination. Valid crew counts are 3, 5, and 7; each uses replicate IDs 0 through 9.

## F8 ballot diagnostic

The frozen diagnostic uses the 15 prescribed hypothesis-training jobs from the new main reproduction. It re-evaluates three trained coalitions against three rules over 135 cells and 67,500 episodes, retaining raw ballot and feasibility records. All summaries are descriptive.

```sh
python audit/submission/replicate_vote_diagnostic_frozen.py check --results results/reproduction
python audit/submission/replicate_vote_diagnostic_frozen.py run --results results/reproduction --out results/vote_diagnostic
python audit/submission/verify_vote_diagnostic.py --results results/vote_diagnostic
```

`check` succeeds only once every required training input passes validation. Add `--wait` to `run` to wait for the prescribed completion markers. The run refuses to overwrite a nonempty diagnostic output directory. The `.sha256` sidecar and script, main protocol, input checkpoint, and raw-record checks retain their original integrity requirements.

## Theory and replay presentation

```sh
python theory/simulate_model.py --help
python experiments/sensor_fusion/sensor_fusion_demo.py --help
python scripts/plot_f4_full_game.py --help
```

The theory model is a restricted mathematical example, separate from the learned environment. The replay renderer requires an actual retained replay; its role-hidden view remains omniscient because it shows the full trajectory. Hiding role labels does not make it a crew-observation view.

## Provenance of this clean package

[provenance/packaging.json](provenance/packaging.json) records both source fingerprints, the omitted historical/operational files, and the one retained source file updated since the original freeze: `scripts/plot_f4_full_game.py`. All shared engine, policy, training-driver, and configuration bytes match the original audited study. The original protocol is retained solely as [provenance/original_study_protocol.json](provenance/original_study_protocol.json).

The new main job list is exactly equal to that original list. Portable checkpoint validation now reads the source at this code root. Mechanism source pins and all supplemental protocol hashes were regenerated for the package, preserving every scientific design field. These fresh packaging timestamps do not imply prospective preregistration or newly completed experiments.

See the repository-level [cahnges.md](../cahnges.md) for the publication audit's corrections, new findings, and remaining work. The root paper build instructions describe how to rebuild the bundled manuscript.
