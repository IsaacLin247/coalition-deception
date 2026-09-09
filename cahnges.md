# Audit changes and new findings

**9 September 2026 — corrected replication in progress.** This public edition contains the repaired research code, reproduction protocols and current working paper. It does not contain a completed experimental data release and is not submission-ready. The filename `cahnges.md` is intentional.

All 30 F1 and 30 F3 jobs and the separate 540,000-episode mechanism study are complete. The local controller has also finished all 30 F2 evaluations without reported failures. Remaining experimental computation runs on the desktop. Final analysis of the entire 240-job study is pending.

## Research and implementation corrections

The audit reproduced archived summaries but found defects that require new experiments. Agreement with an archived CSV does not validate how its policies were trained or evaluated.

- **Training and evaluation:** the spatial PPO collector could finalize unfinished trajectories with incorrect terminal targets. It now collects complete vector batches. The evaluator also favored short games by retaining the first completions; it now completes predetermined episode indices. Evaluation uses copied policies and independent sampling streams, so monitoring no longer advances training randomness. F1 now retains a genuinely untrained checkpoint, distinct monitoring and final evaluation streams, and intermediate checkpoints with actual update counts.
- **Information access:** learned spatial actors now retain the same within-round private positional history available to scripted crews. Scripted movement no longer inspects unseen neighboring occupancy. Hidden-partner action masks use role-blind attempts with silent resolution; message presence no longer directly reveals a missing partner. These repairs remove demonstrated information channels, without preventing ordinary inference from observed outcomes. The hidden-partner condition changes information and communication together.
- **Feasibility and hypothesis scoring:** joint compatibility checks no longer silently skip larger constraint sets or omit joint temporal constraints. Exactness applies to the constraints actually emitted; documented relaxations remain. Publicly impossible coalition members are excluded, private evidence is enforced, and impossible hypotheses receive zero weight. The resulting hypothesis score is explicitly a heuristic, not an exact posterior or a guarantee against deception.
- **Outcome measurement:** dependence statistics now use the complete electorate before ejection and count historical meetings once. Records distinguish terminal false ejection from any innocent ejection during a game, counts per game, and pooled counts per meeting, including incident-free meetings. Creator survival requires an actual incident; it is not the complement of creator ejection when no incident occurred. Raw game and meeting records support independent recomputation.
- **Reproduction checks:** source inventories reject empty manifests. Completion checks validate planned jobs, stages, configurations, checkpoints and raw outcomes. Windows-exported relative result paths resolve on Linux while rejecting absolute paths and directory traversal. A partial training history alone does not mark a job complete.

Regression tests, smoke experiments and independent raw-record checks cover these repairs. Passing tests establishes implementation properties, not the paper's empirical conclusions.

## Mathematical and statistical corrections

The Gaussian analysis now distinguishes exact pairwise probabilities from a deterministic mean-field surrogate. Applying softmax to expected counts is not equivalent to averaging softmax weights from realized counts; the predicted amplification direction can differ. Median contamination-bias approximations are separated from finite-sample mean-squared-error bounds, with variance terms and valid order-statistic arguments restored. The dependence penalty uses the correct maximum of within- and cross-group agreement, and its attenuation claim is restricted to the parameter region where the derivation holds. Numerical edge cases and heatmap/contour alignment were repaired.

The paper no longer treats finite PPO training as an exact game-theoretic best response or observed alternation as proof of asymptotic nonconvergence. Mathematical minority assumptions count actual reporters: a 3+2 game becomes 2+2 after an honest victim is removed.

Some archived tests labeled “sign-flip” actually counted signs while discarding effect magnitudes; these were corrected to paired mean sign-flip calculations. Control comparisons now use matched seed cohorts. Undefined conditional rates remain undefined. The new analysis specifies 303 comparisons in 15 families, paired seed comparisons and multiplicity correction. Seeds are the replicate unit; episodes are not independent training replicates. Interim outputs suppress p-values and confidence intervals, including for completed groups. The supplemental studies are post-audit descriptive work, not retrospective preregistrations.

## Findings from completed corrected components

The following values are descriptive means; “±” denotes sample SD across ten seeds. They must not be pooled with archived runs.

**F1: meeting-only coalition learning.** Each population completed 400 updates, with 400 held-out episodes per seed at both untrained and final checkpoints.

| Initial crew + coalition | False ejection, untrained → final | Same-target voting, untrained → final |
| --- | --- | --- |
| 3+2 | 0.045 ± 0.011 → 0.271 ± 0.046 | 0.126 ± 0.023 → 0.613 ± 0.062 |
| 5+2 | 0.073 ± 0.008 → 0.212 ± 0.018 | 0.108 ± 0.023 → 0.316 ± 0.058 |
| 7+2 | 0.091 ± 0.017 → 0.194 ± 0.027 | 0.092 ± 0.016 → 0.244 ± 0.041 |

Creator survival also rises, while the mean false-claim rate falls, at all three sizes. Same-target voting excludes shared abstention. These joint changes do not identify a causal mechanism.

**F3: spatial adaptation and transfer.** All three populations completed ten seeds, with 1,000 final games per seed in each of four policy pairings. At 5+2, false ejection is 0.194 ± 0.051 for the initial coalition against scripted crew, 0.403 ± 0.040 against the learned defender, 0.538 ± 0.053 after coalition adaptation, and 0.076 ± 0.014 when that adapted coalition returns to scripted crew. The pattern depends on the opponent.

The old universal adaptation claim fails: against the learned defender, coalition retraining lowers false ejection at 3+2 from 0.463 to 0.415, while raising it at 7+2 from 0.475 to 0.587. Initial-coalition incident probabilities against scripted crew are now 0.986, 0.963 and 0.943, contradicting the archived rare-incident account. Multiple repairs changed together, preventing attribution to one fix. Conditional rates exclude undefined strata explicitly; unconditional rates retain every seed.

**Mechanism replication.** Ten independent replicates at each population cover four testimony conditions, three aggregation rules and three intervention variants, using matched environment evidence. The 540,000 outcomes and 240,000 pre-ejection honest-voter records replace one-seed diagnostics and supply the previously missing catch-penalty intervention implementation.

Under coordinated alibis, soft credibility exceeds mean false ejection by 0.0112, 0.0680 and 0.1586 across increasing populations, despite coalition weights below their uniform shares. Removing alibi support helps against alibis but worsens lone-liar and framer errors. Moving the catch penalty outside aggregation raises 3+2 soft lone-liar error from 0.1734 to 0.3274. There is no uniformly beneficial repair. Lower false ejection can also mean abstention: truthful 3+2 has zero false ejections but a 0.9496 ± 0.0094 no-ejection rate. These are measured properties of these scripted conditions, not general guarantees.

**F2: a truthful-testimony cost.** Reviewed 3+2 and 5+2 results show hypothesis-rule false ejection of 0.2447 ± 0.0175 and 0.3700 ± 0.0083 under truthful testimony, versus zero for mean and soft credibility. At 5+2 it improves on the mean against alibis and framers, showing a tradeoff rather than universal accuracy. A bounded replay of ten false ejections retained the true coalition in all 40 honest views and demonstrated score ties and coalition-only ejections after honest abstention. It does not explain the full error rate. “Truthful” concerns supported claim contents; coalition voting remains adversarial.

## Paper, packaging and remaining work

The neutral title is now *Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark*. The paper incorporates corrected F1/F3 and mechanism measurements; other empirical figures and tables, including F2 Tables 3–4, remain explicitly archived or provisional pending integration. Citations, mathematical qualifications and novelty claims were corrected. The synchronized Markdown, TeX and PDF use 12-point text with attached captions and readable multipage tables. A replay exposing full trajectories is correctly labeled omniscient.

The clean repository separates `code/` and `paper/` and removes historical operational clutter. Its portable protocol preserves the original 240 jobs, seeds and budgets, but has a distinct source fingerprint; [packaging provenance](code/provenance/packaging.json) records the relationship. The original ongoing runs remain unchanged. Moving remaining computation to the desktop changes execution placement, not experimental design; software/platform differences remain relevant to reproduction. See the [code guide](code/README.md) for validation and execution commands.

The clean package passed **360 tests**. Its wheel includes the replay viewer asset; the mechanism self-test retained ejected honest voters and detected the intended score changes. All 240 job definitions and 72 packaged source hashes were rechecked. The rebuilt paper has 59 pages, 19 figures and 18 tables. These checks validate the package, not unfinished scientific results.

Remaining work includes completing and validating the main study and ballot diagnostic, integrating their results, reconsidering conclusions, releasing the final data/checkpoints, and obtaining author review and submission declarations. This code-and-paper release does not replace those steps.
