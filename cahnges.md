# Audit changes and final findings

**Completed corrected revision — 11 September 2026 UTC.** The requested filename cahnges.md is retained. All 240 main jobs, 1,255 learner stages, 303 planned comparisons, and both supplemental studies are complete. The manuscript replaces its remaining archived empirical sections with the final corrected evidence. The [final results review](data/reviews/FINAL_RESULTS_REVIEW.md), complete machine-readable data, and validation receipts provide further detail.

## Research and implementation corrections

The audit reproduced 437 archived summary rows but found defects requiring new experiments. Agreement with a historical CSV did not validate how its policies were trained or evaluated.

- **Training and evaluation:** spatial PPO completes episode batches before finalizing terminal targets. Evaluation completes predetermined episode IDs rather than retaining the fastest games. Copied policies and independent random streams prevent monitoring from changing training randomness. F1 retains a genuinely untrained checkpoint and independent initial/final holdouts.
- **Information access:** learned actors retain within-round private history comparable to scripted crews. Scripted movement no longer inspects unseen adjacent rooms. Hidden-partner controls remove demonstrated legality-mask and message-presence identity channels, while permitting inference from observed events. This is a combined information/communication intervention.
- **Feasibility and hypothesis scoring:** joint checks no longer omit larger or temporally coupled constraint sets. Publicly impossible coalitions are excluded, private facts are enforced, and impossible hypotheses receive zero weight. Exactness concerns emitted constraints, with documented relaxations; normalized feasibility scores are not exact generative posteriors.
- **Whole-game measurement:** all meetings and the complete electorate before ejection are retained. Historical meetings are counted once. Outcomes distinguish terminal false ejection, any innocent ejection per game, counts per game, and pooled counts per meeting, including incident-free meetings. Creator survival requires an incident.
- **Reproduction:** source inventories reject empty manifests. Completion checks validate jobs, stages, configurations, checkpoint tensors/update counts, hashes, and raw outcomes. Windows-exported paths resolve on Linux. The portable package preserves the study design with an explicit distinct source identity.

The final postprocessing correction concerned two undefined terminal-coordination cells. Complete meeting histories proved that an earlier coalition ejection left fewer than two terminal ballots in every episode. The analyzer accepts only that proven zero-denominator case; it preserves nulls, all seeds, primary outcomes, missing-data guards, and planned comparisons.

## Theory and statistics

The Gaussian analysis distinguishes softmax of expected agreement counts from expected softmax of realized counts; their amplification directions can differ. Median bias approximations are separated from finite-sample MSE bounds and variance. The dependence penalty uses the actual maximum of within- and cross-group agreement, with a restricted attenuation result. Figures and references were corrected. A 3+2 game has 2+2 reporters after its victim is removed, outside the strict-minority theorem.

The final analysis uses paired seed differences, 20,000-resample pointwise bootstrap intervals, exact mean sign-flip tests, and Holm correction within 15 fixed families. All 303 contrasts remain included. Episodes are not independent training replicates. Five-seed tests cannot reject at raw 0.05; the 27-contrast F4 family cannot reject at Holm 0.05 with its ten-seed exact-test minimum. Families were not shrunk to obtain significance. Supplements remain descriptive.

References and novelty claims were checked, including repair of a DOI pointing to an unrelated paper. Finite PPO responses and alternating outcomes are not described as exact best responses, convergence, or a permanent attracting cycle.

## Final findings

Values below are seed/replicate means. Full SDs, intervals, signed comparisons, and adjusted p-values accompany the data.

| Component | Completed finding |
| --- | --- |
| F1 — meeting-only learning | False ejection rises from 0.045 to approximately 0.271 at 3+2, 0.073 to 0.2125 at 5+2, and 0.091 to 0.194 at 7+2. All three specified FE effects have Holm p=0.0117. Aligned voting and creator survival rise descriptively while false claims fall. |
| F2 — scripted rules | Hypothesis-rule truthful-testimony FE is 0.2447/0.3700/0.3926 across increasing sizes, versus zero for mean and soft credibility. It improves some alibi/framer outcomes but has a substantial accuracy tradeoff. Truthful claims do not mean cooperative coalition voting. |
| F3 — spatial adaptation | At 5+2, FE is 0.1940 for C0–D0, 0.4031 for C0–D1, 0.5385 for C1–D1 and 0.0759 for C1–D0. At 3+2 coalition adaptation lowers the mean from 0.4635 to 0.4153, with an interval crossing zero; a uniform adaptation claim fails. |
| F4 — multi-round play | At 5+2 victory falls 0.9670→0.3603 after defender training and returns to 0.8514 after coalition retraining. During retraining, any innocent ejection changes only 0.8392→0.8467 while mean FE count falls by 0.4249. Victory and accumulated harm diverge. No F4 contrast passes the retained Holm family. |
| F5 — repeated responses | At generation ten, 5+2 multi-round victory is 0.3068 for C9–D10 and 0.7968 for C10–D10, but only 0.1804 for C10–D0. Ten-seed matrix-gain tests pass Holm p=0.0488; parity and late/early tests do not. This is finite evidence, not an asymptotic result. |
| F7 — dependence defense | At 5+2, dependence-specific counterattack effects are uncertain. In the matched four-generation loop, the defense increases any-FE by 4.24 percentage points after C stages and 10.44 after D stages (Holm p=0.0117), while lowering C-stage wins by 2.88 points (Holm p=0.0410). |
| F8 — hypothesis attack | At 5+2, targeted training raises hypothesis-defender FE from 0.1255 to 0.4217: effect 0.2962, pointwise 95% CI [0.2521,0.3418], Holm p=0.0273. The coalition transfers poorly to mean and soft rules. |
| Controls | No declared ablation, budget or reward contrast passes Holm at 0.05. Limited five-seed resolution prevents interpreting this as equivalence or proof that the controls do not matter. |

The 540,000-episode mechanism study and 240,000 honest-voter records show soft-minus-mean alibi FE differences of +1.12/+6.80/+15.86 percentage points despite below-uniform coalition weights. Removing alibi support or moving caught-lying penalties changes errors differently across attacks. No tested score repair is uniformly beneficial.

The 67,500-episode ballot supplement contains 135 cells and 317,670 honest score views. For hypothesis-trained attacks against the hypothesis rule, FE is 0.4240 at 5+2 and 0.6048 at 7+2, with honest skip rates about 66.1% and 89.9%. Coalition ballots are necessary for about 97.3% and 100% of observed false ejections under fixed recorded votes. This supports an abstention-related route, not a claim about behavior after changing the voting rules.

## Manuscript and validation

The title remains **Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark**. The abstract, methods, all results, interpretation, limitations, conclusion, data availability and reproduction appendix describe the completed study. All 19 figures and 18 tables are embedded in synchronized Markdown/TeX/PDF; the value manifest records selected final rows and input hashes. The corrected historical omniscient replay is no longer used as a final empirical result.

Validation includes:

- All **240 jobs and 11,885 checkpoints** accepted by raw/source/checkpoint gates.
- Independent agreement for **all 303 effects, SDs, exact p-values, deterministic bootstrap intervals and Holm adjustments**.
- Public-code reanalysis reproducing every semantic value in the five main tables, including 108,133 per-seed records.
- **366 public-package tests** and wheel build passed; both source fingerprints preserved.
- Independent supplemental raw-semantic and summary checks.
- Manuscript source-table, citation, caption, build, and PDF layout checks.

The repository contains complete analysis tables and diagnostics, with documented archive-relative paths. Raw records, checkpoints, original source, and checksums are supplied through the [corrected-study release](https://github.com/IsaacLin247/coalition-deception/releases/tag/corrected-study-2026-09-11). Its seven numbered parts reconstruct the complete archive; verify the supplied hashes before extraction.

Computation and manuscript integration for that corrected study are complete. Further scientific validation is described below. Author affiliations/contact, funding, competing interests, contributions, peer-review model and approval remain author-supplied submission facts. No journal submission or acceptance is claimed.

## Additional validation authorized after review — 11 September 2026

An external critique identified two remaining design concerns: the repaired hypothesis rule is not established as a strong defense, and nine of the fifteen earlier statistical families cannot reject under their specified exact-test resolution. Those nine families contain 231 of 303 contrasts. The old results and family definitions remain intact; the new study does not reinterpret their nonsignificance as equivalence.

The [new validation protocol](validation/README.md) specifies twelve tie/threshold candidates on separate development evidence, then a fixed selected rule for untouched evaluation and fresh attack training. Canonical random and skip variants are retained in final evaluation even when they are not selected. The feasibility solver, information access, game rules, and PPO collector remain unchanged. Development selection requires declared creator-capture floors so that lower false ejection alone cannot reward a practically useless abstention policy.

The new primary analysis fixes 72 independent replicate IDs and six contrasts covering truthful-condition error, adaptation against the selected defense, matched adaptive defense comparison, and the distinction between eight-round coalition victory and innocent-ejection outcomes. All other new comparisons are descriptive. A conditional power analysis and its full calculation are retained; no result-dependent seed expansion or statistical family shrinking is permitted.

All new research simulations and training run on `ssh desktop` in a separate workspace. Complete episode and meeting records, full ballots, decision-time score/tie/skip records, checkpoints, frozen source hashes, and selection provenance are retained. Results will be added to the manuscript after the entire new cohort passes verification. No new empirical finding is asserted merely from preparing this follow-up.
