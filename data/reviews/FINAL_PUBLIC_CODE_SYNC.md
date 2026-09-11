# Final public-code synchronization

Completed on 11 September 2026 UTC (10 September in America/Chicago). This handoff covers `coalition-deception/code/` only. The parent task owns the paper, public root documentation/data, archive upload, commits, and push.

## Changes

- `analysis/analyze_submission.py` and `tests/test_submission_analysis.py` now exactly match the final original-workspace versions. The analyzer accepts the two undefined terminal-coordination cells only after complete raw meeting histories prove that an earlier coalition ejection removed the second terminal ballot. No zero imputation, seed exclusion, or relaxation of primary-outcome completeness was introduced.
- `analysis/verify_packaging.py` and `tests/test_packaging_provenance.py` verify the original and portable protocol hashes, all 240 identical job designs, the exact portable source inventory, and the documented source differences. Regression checks reject changed source bytes, added source files, and changed original protocol contents.
- `audit/submission/verify_final_inference.py` is an unchanged copy of the independently implemented original verifier. It reconstructs all 303 contrasts from the retained per-seed table without importing the production analyzer.
- `README.md` now describes the completed study, independent inference checking, original-run reanalysis, the separate source fingerprints, and the distinction between original and portable supplemental protocol pins. It explains that a fresh checkout reports zero new reproduction jobs. Commands contain no machine-specific paths.
- `provenance/packaging.json` records the analysis-only additions. `provenance/final_analysis_verification.json` records the successful complete original-run ingestion through the public analyzer and exact semantic table comparisons.

## Preserved provenance

The original 82-file source fingerprint remains `691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`. The portable 72-file fingerprint remains `cdaa1c031b4e2e6e690771c9a884ed70ed78653814863f99bb2fe0def367ae68`. Both main protocols and every frozen scientific input remain unchanged. The two packages have exactly equal 240-job designs; all 71 shared files apart from the pre-existing replay presentation difference are byte-identical to the original frozen source. The ten omitted files remain historical/operational packaging omissions.

The public checkpoint plan intentionally retains its portable source pin. Its scientific driver/configuration bytes and all expected job/stage specifications equal the original study. Original-run reanalysis uses `provenance/original_study_protocol.json`, so original completion markers and run metadata must retain the original fingerprint. New reproductions use `submission_protocol.json`. No result was relabeled with another source fingerprint.

## Verification

- Full public suite plus both supplemental suites: **366 tests passed in 60.86 seconds**, using importlib collection to handle the two vote-diagnostic test filenames.
- Wheel build passed: `social_collusion-0.1.0-py3-none-any.whl`, 156,242 bytes, SHA-256 `9fab49852435b916c457d3685f1324757479332d04430a218fa4132a56c02e03`. The wheel covers the installable library; documented research commands require the source checkout and editable installation.
- Packaging verification passed after all additions; the frozen source fingerprints are unchanged.
- Public runner status lists the complete 240-job portable design; zero completed jobs in a fresh local reproduction is expected.
- Full public-analyzer ingestion of the original raw records and retained checkpoints passed **240/240 jobs** with final inference enabled and no missing comparisons or validation errors.
- Exact semantic equality with the original completed analysis passed for **108,133 per-seed records, 12,873 descriptive summary rows, 303 paired-effect rows, 56 matched-reference summaries, and 770 restricted-pool gaps**. Only machine-dependent artifact path strings were excluded from the per-seed comparison; all numerical and categorical scientific fields match exactly.
- The copied independent inference verifier passed every one of the **303 contrasts in 15 families**, including complete sign enumeration, 20,000-resample percentile intervals, and Holm adjustments, at absolute tolerance `1e-12`.
- Analyzer and regression-test byte identity with the final original versions was checked. The analyzer SHA-256 is `b76f5ec4345a174d101d456ab7efe3ffc98c755ffdfc2d5c90b1b8542a61f92b`.
- `git diff --check` passed.

## Data and archive handoff

No raw data, checkpoints, private operational scripts, or original machine paths were added to `code/`. The parent task should publish completed analysis tables under root `data/final_analysis/`, as linked by the code README. The independent verifier accepts those files through `--analysis ../data/final_analysis` when run from `code/`.

Original mechanism/F8 outputs require their original pinned scripts, source snapshot, and protocols retained in the code-and-data archive. The portable supplement protocols remain separately pinned to reproduce the same designs from new portable runs. Original archive validation receipts hash original bytes; sanitized public copies should document their transformation and new hashes rather than claiming byte identity.

The full original code-and-data archive remains the archival source of raw records and retained checkpoints. No training, evaluation resampling, frozen-source changes, or result-data edits occurred during this synchronization. No git commit or push was performed by this subtask.
