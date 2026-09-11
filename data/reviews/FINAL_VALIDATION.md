# Final corrected-study validation

Validated on 11 September 2026 UTC (10 September in America/Chicago). This report covers the completed numerical study and release-data eligibility. It does not assert that a journal has accepted the paper, that authors have completed their submission declarations, or that an archive has already been uploaded. The separately generated archive completion receipt identifies the exact packaged bytes.

## Completed study

The final analyzer includes **all 240 planned jobs** across 30 experiment groups, with no missing jobs, analysis errors, or missing comparisons. All **303 signed, paired contrasts in the 15 declared comparison families** have their complete planned seed cohorts. Final inference is enabled. The renderer contains 2,487 table cells and 39 figures and identifies the current analyzer, renderer, source fingerprint, and exact input-table hashes.

The frozen engine has 82 executable/configuration inputs and fingerprint `691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`. The study contains 1,255 planned training stages. The final checkpoint manifest covers **11,885 retained checkpoints**, including the genuine untrained F1 endpoints. Architecture, finite tensors, planned seed/environment/algorithm/budget, retained history counters, and final-versus-last-numbered parameters were checked by the final analyzer; the archive check confirmed their current byte hashes and the exact required checkpoint set. Learner roles are derived from the frozen driver, and opponent identity is not embedded in checkpoints. Consequently, this is strong operational provenance, not a cryptographic proof of the complete training process.

The **mechanism supplement** contains all 30 jobs, 540,000 episodes, and 240,000 retained honest-voter records. The archive check reran its pinned read-only analysis with output writers replaced by comparators. Episode identity, common evidence, vote denominators, outcomes, and all retained JSON/CSV aggregate and paired descriptive summaries matched.

The **F8 ballot diagnostic** contains all 15 required training-job inputs, 135 cells, and 67,500 episodes. Every recorded episode passed the independent standard-library semantic verifier: actual, crew-only, and coalition-only ballot tallies; necessity and sufficiency flags; full pre-ejection electorates; score-vector summaries; seed formulas; identities; and denominators. All 18 across-seed descriptive cell summaries match the frozen summarizer. Recorded checkpoint/history hashes match the main-study files. Removing ballots holds recorded behavior fixed; it does not establish how players would react to actual voter removal. The verifier checks retained score-vector consistency, not a fresh replay of the hypothesis solver.

## Independent inference recomputation

`verify_final_inference.py` reconstructs every declared signed difference from the retained per-seed table without importing the production analyzer. It separately recomputes paired means and sample SDs, complete exact sign enumeration, all 20,000-resample percentile bootstrap intervals, and full-family Holm adjustments. All 303 comparisons matched within absolute tolerance `1e-12`. Thirty contrasts have Holm-adjusted p-values below 0.05 within their respective prespecified families; this is not a single correction across all 303 tests. Five-seed comparisons cannot attain an unadjusted two-sided sign-flip p-value below 0.0625. A nonsignificant result is not evidence of equivalence, stationarity, or convergence. The mechanism and F8 supplements remain descriptive.

The two undefined terminal-coordination entries in `multigen_none_n9_r8_k3_s1` were retained as undefined only after the raw meeting histories proved that a coalition member had already been ejected before the terminal meeting in every affected episode. No zeros were imputed, no seed was removed, and all primary matrix outcomes remain complete. See `UNDEFINED_COORDINATION_REVIEW.md` for the bounded explanation and evidence.

## Commands and outcomes

Commands were run from the original repository root with its existing `.venv`. No training, evaluation resampling, source-snapshot changes, checkpoint edits, or outcome selection was performed.

```sh
.venv/bin/python -B audit/submission/build_reproducibility_archive.py check
```

Passed with `READY`. Before this report and its verification receipts were added to the documentation allowlist, the validated payload contained **27,587 files and 12,680,720,655 bytes** (about 11.81 GiB). All main-study, mechanism, and F8 gates passed. `final_archive_check.log` retains the command output. The builder repeats these checks before archive creation and checks every selected file again while streaming it; its final manifest is authoritative for the complete packaged file count and sizes.

```sh
.venv/bin/python -B audit/submission/verify_final_inference.py \
  > audit/submission/final_inference_verification.json

.venv/bin/python -B -m pytest --import-mode=importlib -q \
  tests/test_submission_analysis.py tests/test_submission_identity.py \
  tests/test_submission_export_paths.py tests/test_submission_rendering.py \
  tests/test_checkpoint_integrity.py tests/test_vote_diagnostic.py \
  audit/submission/test_verify_vote_diagnostic.py \
  audit/submission/test_reproducibility_archive.py \
  audit/submission/test_sync_vote_diagnostic.py \
  audit/submission/test_sync_desktop.py audit/submission/test_vote_diagnostic.py
```

Independent inference recomputation passed. **All 171 targeted tests passed.** Importlib collection is required when the original and audit test directories are combined because both contain a file named `test_vote_diagnostic.py`; the first default-import collection attempt stopped on that name collision before running tests. `final_validation_tests.log` retains the successful run. The archive allowlist was subsequently extended only to include this report, its receipts, and the independent inference verifier; scientific inputs and outputs were unchanged.

## Publication artifact scope

The complete data can be published as GitHub Release assets without adding roughly 12.7 GB of raw records and checkpoints to the source repository. GitHub currently requires each release asset to be smaller than 2 GiB; a compressed archive larger than that must be split into smaller chunks with a reconstruction guide and per-part plus reconstructed-archive SHA-256 checksums. See [GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). Local free disk space was approximately 766 GiB before creation, sufficient for the validated payload and its compressed archive.

The authorized archive is a **code-and-data archive**, with no manuscript-review assertion or reviewed-paper inclusion flags. The revised manuscript is published separately in the source repository. Full archival engine provenance remains the original 82-file fingerprint; the smaller portable repository has its own fingerprint and must not relabel the original runs.

`final_validation_receipt.json` records the completed checks and exact analysis, protocol, verifier, and checkpoint-manifest hashes. The archive's `.complete.json`, `.manifest.json`, and `.sha256` sidecars record the built artifact independently of this report.
