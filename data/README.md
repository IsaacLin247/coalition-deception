# Completed corrected-study data

This directory publishes the complete seed-level analysis of **all 240 planned main-study jobs**, together with the completed mechanism and ballot-diagnostic supplements. These are the corrected September 2026 results used for the revised paper. No result is selected or omitted because of its direction or significance.

## Included data

| Directory | Contents |
| --- | --- |
| [final_analysis/](final_analysis/) | All 108,133 per-seed values, 12,873 descriptive summary cells, 303 planned paired contrasts in 15 families, 56 matched-reference summaries, and 770 finite-policy-pool gap records; JSON and CSV versions, the complete comparison plan, original study protocol, completion status, and checkpoint hash manifest. |
| [final_analysis/presentation/](final_analysis/presentation/) | All 39 final analysis figures in PDF and PNG, their captions and input data, and the 2,487-cell presentation tables. This is the full analysis presentation, including material beyond the paper's selected figures. |
| [mechanisms/](mechanisms/) | Complete per-replicate, cell-summary, and paired descriptive tables from 30 jobs and 540,000 episodes, with the original protocol. |
| [vote_diagnostic/](vote_diagnostic/) | Per-seed and across-seed summaries for all 135 cells and 67,500 episodes, the 15 training-job input records, completion metadata, and original protocol. |
| [reviews/](reviews/) | Final validation, code-publication verification, completed mechanism findings, bounded hypothesis-score review, and supporting evidence/receipts. |

Ten seeds are retained for the core cohorts and five for the designated supplementary cohorts. The exact cohort for every comparison is explicit in `planned_comparisons.json`, `paired_effects.json`, and the summary seed fields. A seed is the unit of inference; episode observations within a seed are not treated as independent training replicates. Bootstrap intervals are pointwise 95% intervals, and Holm correction uses each of the 15 declared families. The mechanism and ballot-diagnostic supplements are descriptive.

Undefined descriptive values remain undefined. In particular, the two terminal-coordination cells explained in [the completed review](reviews/UNDEFINED_COORDINATION_REVIEW.md) have a proven zero denominator. No zeros were imputed and no primary-outcome seed was removed. [The analysis plan](final_analysis/ANALYSIS_PLAN.md) defines estimands, comparisons, uncertainty, and completeness gates.

## Raw records and checkpoints

The source repository contains the full seed-level tables and the hash manifest for **11,885 retained checkpoints**, including the genuine untrained F1 endpoints. It does not contain the raw episode/meeting/voter files or checkpoint tensors. Those larger artifacts are distributed separately through the repository's [GitHub Releases](https://github.com/IsaacLin247/coalition-deception/releases), together with the exact original frozen source and reproduction/verification tools. Follow the release's reconstruction and checksum instructions before using split archive assets.

The original archive preserves `results/submission_20260909/`, `audit/submission/mechanism_results/`, and `audit/submission/vote_diagnostic_results/`. Paths in these published tables are relative to that archive root, not necessarily files present in this Git checkout. The checkpoint manifest records architecture/content and operational provenance checks; the original checkpoints do not embed opponent identity, so it is not proof of the entire training process.

## Publication identity and path transformations

The original study source fingerprint remains **`691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`**. The smaller portable `code/` distribution has a distinct frozen fingerprint. Original data must not be relabeled with the portable fingerprint; use the original protocol for original-run reanalysis, as explained in [code/README.md](../code/README.md).

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) records each copied file's original repository path, original SHA-256, published SHA-256, byte sizes, and any transformation. Files without machine-specific paths are byte-identical copies. In the affected JSON/CSV records, the original POSIX repository prefix was removed; Windows main-run paths were normalized and mapped to `results/submission_20260909/`. The manifest records the mapping rules, affected fields, counts, and hashes of the removed prefixes without publishing private machine locations. Changed JSON files use consistent indentation/key ordering.

**No numerical values, seeds, row identities, event counts, checkpoint hashes, statistical results, or original source fingerprints were changed.** These transformations only change filesystem-location strings and JSON formatting. Comparisons against original inputs checked every other JSON value and every other CSV field. Original validation receipts and presentation input hashes intentionally continue to identify the original bytes; the manifest connects them to the published bytes. The separately generated publication inference receipt identifies the public inputs. The short completed coordination review and this guide are newly written publication documentation, identified separately in the manifest.

## Verify the published inference

From the repository root, with NumPy installed:

```sh
python code/audit/submission/verify_final_inference.py --analysis data/final_analysis
```

This independent verifier imports no production analysis implementation. It reconstructs every signed seed-level contrast, paired mean and sample SD, exact sign-flip p-value, 20,000-resample percentile interval, and family-wise Holm adjustment. It passed all **303 comparisons** on these published inputs within absolute tolerance `1e-12`; the receipt is [publication_inference_verification.json](final_analysis/publication_inference_verification.json). This numerical check complements the raw-record/checkpoint validation documented in [FINAL_VALIDATION.md](reviews/FINAL_VALIDATION.md); it does not substitute for checking raw data in the full archive.
