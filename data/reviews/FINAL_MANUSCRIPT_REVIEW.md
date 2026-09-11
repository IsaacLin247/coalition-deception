# Independent review of the final integrated manuscript

Reviewed `paper/when_credibility_collapses.md` and its generator against the completed analysis, selected-value manifest, mechanism summaries, vote supplement, and retained theory/sensor outputs. This is a content and numerical review; PDF layout and publishing are checked separately. No manuscript or research-code edits were made by this reviewer.

Snapshot reviewed: manuscript SHA-256 `c9521705bdbd5bd99cc298d13ea4a2e75c2051e1b4c34790565b920e002b3fdd`. Findings below refer to that snapshot and may already be resolved in the parent's subsequent edits.

## Required correction

**F3 prose uses four incorrect decimal values even though Table 7 is correct.** The paragraph beginning “Coalition adaptation against D1 has different signs across populations” says the 3+2 mean falls from `0.4630` to `0.4150`, with effect `−0.0480`, and gives the 7+2 effect as `0.1119`. The final source values are:

| Quantity | Correct value | Source |
| --- | --- | --- |
| 3+2 C0–D1 false-ejection probability | 0.4635 | `analysis/cell_summary.json`, `f3_tenseed_crew3`, `crossplay`, `coalition0_vs_learned_crew1`, `false_ejection_rate` |
| 3+2 C1–D1 false-ejection probability | 0.4153 | Same group/table/metric, `coalition1_vs_learned_crew1` |
| 3+2 adaptation difference | −0.0482 | `analysis/paired_effects.json`, `f3_tenseed_crew3|adaptation|false_ejection_rate` |
| 7+2 adaptation difference | +0.1124 | Same file, `f3_tenseed_crew7|adaptation|false_ejection_rate` |

Suggested replacement: “At 3+2 the mean falls from 0.4635 to 0.4153, an effect of −0.0482 that does not reject; at 5+2 and 7+2 the increases are 0.1354 and 0.1124.” The 3+2 pointwise interval is [−0.0989025, 0.0065], Holm p=0.130859375; the two larger-population increases both have Holm p=0.017578125. This is a transcription issue, not a revised statistical conclusion.

## Minor corrections and reproducibility improvements

- **F6 dangling reference:** “condition (i) has Δc…” does not refer to a currently labeled condition. Appendix A uses numbered proposition items. Replace it with “the surrogate has Δc…” or explicitly cite the amplification condition in Appendix A. The reported approximate values −0.6 and +0.12 are correct.
- **Manifest coverage:** `paper/data/final_table_sources.json` hashes four main-analysis inputs but does not hash `vote_diagnostic_results/summary.json`, even though Table 17 and the ballot plot consume it. Add that input hash. For a complete source inventory, also bind retained mechanism summaries, theory summaries/curve data and the sensor summary; those inputs are currently described in the manifest note rather than bound by digest. The numerical values checked from these sources are correct.
- **Control interpretation:** the inherited related-work contribution says controls “test whether the dynamics are properties of the interaction.” The generator's intended replacement does not match the existing text, so this phrase survives. “Test sensitivity to the selected information, budget and reward settings” states what these limited controls establish more precisely and matches the qualified Results/Limitations sections.
- **Final availability check:** the manuscript describes the public `corrected-study-2026-09-11` release as already supplying source, checkpoints and raw records. That is appropriate only after the parent's authorized publication completes and the named release assets are accessible. This review does not claim to have verified a completed upload.

## Checks that passed

**Source manifest and generated tables.** All **422 selected rows** in the manifest exactly equal their appropriate full-cohort or matched-reference source row. All **four recorded main-input SHA-256 hashes** match current inputs. All **14 generated table Markdown strings** occur verbatim in the manuscript. This verifies Tables 2–4 and 7–17 against the strings produced by the generator; the selected rows were independently compared to their originating JSON files. Table 17 values were separately checked against the vote summary.

**Retained mechanism tables.** The existing transcription validator was evaluated without overwriting its prior output artifact. All **108 false-ejection mean/SD cells, 60 voter-metric mean/SD cells and 12 uniform shares** in Tables 5 and 6 match their ten-replicate source summary. The retained descriptive mechanism prose is consistent with the checked records: soft-minus-mean alibi errors +0.0112/+0.0680/+0.1586; below-uniform coalition weights; high abstention in truthful 3+2; and attack-dependent intervention costs. No unjustified p-values were added to the supplement.

**F1 inference.** The reported Holm p=0.0117 rounds the actual 0.01171875 correctly for all three final-minus-initial false-ejection contrasts in the six-test F1 family. Aligned voting, creator survival and false-claim changes are correctly described as descriptive. The table distinguishes actual untrained holdout endpoints from monitored curves.

**Inference scope.** The paper correctly states 240 jobs, 303 contrasts and 15 complete families; paired seed-level inference, sample SD, 20,000-resample pointwise bootstrap intervals, exact mean sign-flips and within-family Holm adjustment. The symmetry/exchangeability assumption, lack of paper-wide multiplicity control, five-seed p-value floor and 27-test F4 Holm floor are explicit. F4, F2, ablation, budget and reward outcomes are not misrepresented as passing their adjusted tests. The prior independent arithmetic review recomputed all 303 effects/SDs/CIs/raw and adjusted p-values with zero discrepancies.

**Stage summaries and controls.** Tables 11, 12 and 15 correctly exclude C0. The analysis uses stage evaluations C1…Ck and D1…Dk; the matrix adaptation gain separately averages M[g,g]−M[g−1,g] from final crossplay. Values in Table 12 are full-horizon means rather than incorrectly labeled late-horizon estimates. Budget and reward references use the exact five-seed cohorts and matched horizons. Defended-loop comparisons use ten matching seeds and four generations. The reported stage means and matrix gains need not subtract identically because the evaluations differ.

**F4 and F7 claims.** The paper correctly separates coalition victory, any innocent ejection, mean count per game and terminal-meeting FE. At 5+2, F4 values 0.9670→0.3603→0.8514 and any-FE 0.8392→0.8467 match the final sources. The dependence loop's increased any-FE at both C/D stages and modest decrease in C-stage coalition victory are reported with their correct signs, intervals and Holm p-values. The 5+2 dependence-specific adaptive differences −0.0123 and −0.0148 are reported as uncertain, not as successful attacks or immunity.

**F5 narrative.** All six hard-coded 5+2 multigeneration matrix values match final crossplay: C0–D0 0.9658, C0–D1 0.3524, C1–D1 0.8562, C9–D10 0.3068, C10–D10 0.7968, C10–D0 0.1804. The seven ten-seed matrix-gain tests do pass the 25-test dynamics-family Holm threshold at p=0.048828125. No parity/late-versus-early claim is promoted into evidence of stationarity or a period-two orbit. Finite-pool gaps are explicitly retrospective and not global exploitability.

**F8 and ballot interpretation.** Main adaptive hypothesis endpoints, matched comparisons and Holm p=0.0273 are correct. Table 17 and its prose use the separate diagnostic stream (0.4240/0.6048) rather than replacing main endpoints (0.4217/0.6036). Necessity and sufficiency are distinguished and conditioned on observed false ejections. The percentage statements are means of within-seed conditional rates, as the caption states. The prose appropriately holds other votes fixed, avoids claims of behavioral causation or exact Bayesian inference, and states the limits of the score interpretation.

**Theory and sensor retention.** The scalar model remains separate from the corrected game replication. Retained MSE values agree with `theory/figures/summary.json`; pairwise Gaussian probabilities, surrogate/realized-weight distinction, explicit variance terms, strict-minority median bound and parameter-restricted dependence result remain correctly qualified. The 5+2 theoretical condition uses actual report counts rather than initial game players. Sensor means and sample SDs agree with `experiments/sensor_fusion/figures/sensor_fusion_summary.json`, including credibility weight 0.254±0.006, coordinated RMSE 0.99±0.03, and dependence RMSE 0.56±0.01 over 20 runs. The analogy is not claimed as a general guarantee or deployed-system validation.

**Crossreferences.** Section 3.3/3.4 point to scripted policies/aggregation rules; Section 4.4 points to controls; Section 5 is evaluation; Section 6.2 contains the mechanism subsection; Section 6.5 is the controls results. Figure and numbered table references are consistent with the new sequence. The one dangling “condition (i)” reference is noted above.

## Review conclusion

The integrated manuscript substantively reflects the completed corrected results and their limits. Fix the F3 transcription paragraph, resolve the minor reference/provenance issues, complete the promised release publication, and perform the separate PDF/build checks. No additional numerical or substantive-claim blocker was identified within this review's scope.

## Resolution addendum — corrected manuscript

Current reviewed manuscript SHA-256: **`1eb6a0c08019cf61ca55e076b62d33dbe22919316c29157d2e2c30121252b6b1`**. This addendum preserves the original findings above and records their resolution after the parent's edits.

- The F3 prose now correctly gives 0.4635→0.4153, −0.0482, and +0.1124.
- The dangling “condition (i)” wording is replaced by “the surrogate has”; the contribution now states that controls test sensitivity to the selected settings.
- The table-source manifest now contains ten input hashes. Every hash matches its original source file, including the newly bound ballot, mechanism, theory, curve, sensor and pre-integration manuscript inputs.
- The added F4 false-ejection count differences match `paired_effects.json`: **−0.4249/game at 5+2**, pointwise 95% CI [−0.5236, −0.3239975], and **−1.0066/game at 7+2**, CI [−1.0559025, −0.9592]. Both have Holm p=0.052734375. They are correctly presented as descriptive changes alongside increased victory, without claiming adjusted significance or substituting count for the probability of any false ejection.
- The rebuilt PDF reports **53 pages**. Its 19 figures and 18 tables retain the checked structure. Detailed rendering/bounds verification remains the parent's separate build check.

All content corrections raised by this reviewer are resolved for this snapshot. The parent reports that release assets are fully uploaded and hash-verified in a draft release; public availability still requires the planned publication step. This addendum does not claim independent verification that the draft has been made public.
