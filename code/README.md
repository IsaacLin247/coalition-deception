# Research code

This directory contains the audited environment, policies, PPO training, complete 240-job study runner, scientific supplements, analysis, and regression tests. The corrected study has completed all 240 jobs and final numerical validation, plus both descriptive supplements. Main-study tables and statistical results are in [../data/final_analysis/](../data/final_analysis/); this code directory contains no raw research data or checkpoints. The original study and a new reproduction have different source fingerprints, as explained below.

## Install and verify

Use Python 3.10 or newer in a dedicated environment. Run these commands from this `code/` directory:

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -e '.[rl,env,analysis,dev]'
python -m pytest tests
python -m pytest --import-mode=importlib audit/submission/test_vote_diagnostic.py audit/submission/test_verify_vote_diagnostic.py
python analysis/verify_packaging.py
python experiments/submission/run_study.py status
```

The editable install is required for this repository workflow because configuration files live alongside the source tree. Dependencies are NumPy, PyTorch, PyYAML, pandas, matplotlib, PettingZoo, Gymnasium, pytest, and Hypothesis; the installation command includes all of them. Production experiments use CPU execution and one computation thread per job. Results can differ across library versions and platforms even with identical seeds.

The runner's default status describes a new local reproduction, so a fresh checkout correctly reports zero completed jobs. It does not query the completed original study or download its results.

## Run the complete study

`submission_protocol.json` is already frozen for this exact portable source. It has the same 240 job definitions, seeds, training budgets, evaluations, and comparison design as the original post-audit study. A different source fingerprint records the cleaned inventory and updated replay presentation utility. Do not use this protocol to relabel results produced by the original frozen source.

```sh
python experiments/submission/run_study.py list
python experiments/submission/run_study.py run --worker all --slots 4 --results results/reproduction
python experiments/submission/run_study.py status --worker all --results results/reproduction
```

Four slots is an example; select a concurrency level your machine can support. The complete study is substantial: 1,255 training stages and 512,000 PPO updates, plus scripted evaluation. The runner checks dependencies, source identity, required artifacts, and completion markers. It resumes by skipping completed jobs. Interrupted individual jobs do not resume an intermediate checkpoint automatically. Worker names are scheduling labels; `--worker all` runs the entire plan on one machine. A `--families` subset is useful for development but cannot satisfy the full analysis gate.

Do not edit hashed source files during a reproduction. Changes require a separately documented new protocol and matching supplement/analysis pins; they must not overwrite or masquerade as this package's frozen design. The completed original computation remains separately identified and unchanged.

## Analyze completed outputs

```sh
python analysis/analyze_submission.py --protocol submission_protocol.json --runs results/reproduction --out results/analysis
python analysis/render_submission.py --analysis results/analysis --out results/presentation
```

The analyzer verifies job identity, checkpoints, raw outcomes, all planned seeds, and the complete comparison set. Final inference is blocked while required evidence is missing. To inspect an unfinished reproduction, add `--interim` to both commands; this explicitly suppresses inferential p-values and confidence intervals. The scientific definitions and statistical assumptions are in [the analysis plan](audit/submission/ANALYSIS_PLAN.md).

The current `analyze_ablation_budget.py` and `analyze_revision.py` helpers are also retained because the regression suite checks their repaired statistical estimands and matched reference cohorts. The complete reproduction workflow uses the gated commands above.

These commands regenerate tables and plots from a new reproduction. Generated outputs retain that reproduction's source identity; incorporate new outcomes into a manuscript only with the matching analysis and provenance.

### Reanalyze the completed original study

Use the retained **original** protocol when analyzing the completed study's raw run directory, including a run directory extracted from its reproducibility archive. Replace `PATH_TO_ORIGINAL_RUNS` with the directory containing the 240 job folders and their `_control` completion markers:

```sh
python analysis/verify_packaging.py
python analysis/analyze_submission.py --protocol provenance/original_study_protocol.json --runs PATH_TO_ORIGINAL_RUNS --out results/original_analysis
python analysis/render_submission.py --analysis results/original_analysis --out results/original_presentation
```

The packaging verifier checks both protocol hashes, all 240 identical job definitions, the complete portable source inventory, and the documented differences from the original manifest. All 71 shared files other than the replay presentation script are byte-identical to the frozen originals. Checkpoint expectations therefore come from the same scientific driver and configuration bytes; the analyzer still requires the original source hash in every original job's completion and run metadata. It does not relabel those runs with the portable hash. A full verification against all 240 original jobs reproduced every numerical field in the five final analysis tables; [the verification receipt](provenance/final_analysis_verification.json) records the comparison and canonical table hashes.

The published per-seed table also supports an independent numerical check without checkpoints or training. This verifier does not import the production analyzer; it reconstructs all 303 planned signed contrasts, exact sign-flip p-values, 20,000-resample bootstrap intervals, and the 15 family-wise Holm corrections:

```sh
python audit/submission/verify_final_inference.py --analysis ../data/final_analysis
```

The final analyzer retains undefined terminal coordination when raw meeting histories prove that an earlier coalition ejection left fewer than two coalition ballots in every terminal meeting. Two cells in one nine-agent multigeneration run meet that condition. Undefined values remain undefined; missing primary outcomes, unexplained missing coordination, infinities, incomplete histories, or missing seeds still block final inference.

This command is for main-study analysis. Original mechanism/F8 result validation and reproduction use the original pinned supplement scripts and protocols retained with the original reproducibility archive. The supplement commands below use this distribution's portable pins for a new reproduction; do not apply those pins to original supplemental outputs.

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

[provenance/packaging.json](provenance/packaging.json) records both source fingerprints, the omitted historical/operational files, and the one retained source file updated since the original freeze: `scripts/plot_f4_full_game.py`. All shared engine, policy, training-driver, and configuration bytes match the original audited study. The original protocol is retained as [provenance/original_study_protocol.json](provenance/original_study_protocol.json) for provenance and reanalysis of original results.

The new main job list is exactly equal to that original list. Portable checkpoint validation now reads the source at this code root. Mechanism source pins and all supplemental protocol hashes were regenerated for the package, preserving every scientific design field. These fresh packaging timestamps do not imply prospective preregistration or newly completed experiments.

See the repository-level [cahnges.md](../cahnges.md) for the publication audit's corrections, new findings, and remaining work. The root paper build instructions describe how to rebuild the bundled manuscript.
