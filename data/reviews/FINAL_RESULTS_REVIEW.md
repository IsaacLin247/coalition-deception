# Final results review for manuscript integration

Reviewed against the complete final analysis dated 2026-09-11 UTC (September 10 in America/Chicago). This review reads the frozen post-audit replication and its two completed supplements. Historical results in the working manuscript are superseded wherever a corrected replication exists. No training outputs or main manuscript files were changed by this review.

## Completeness and independent numerical checks

`analysis/status.json` reports **240/240 jobs**, `inference_enabled: true`, no problems, and no missing comparisons. There are **303 planned contrasts in 15 complete families**. Independently recomputing all 303 paired means, sample SDs, exact signed-mean p-values, deterministic 20,000-resample paired bootstrap intervals, and familywise Holm adjustments from the recorded per-seed differences found **zero discrepancies**; see `FINAL_RESULTS_NUMERIC_CHECK.json`. This check verifies the final analysis arithmetic and does not replace the existing raw episode/checkpoint validation.

The mechanism supplement reports 30 jobs and 540,000 episodes; its outputs explicitly permit descriptive inference only. The vote supplement reports 135 cells and 67,500 episodes, comprising 15 training seeds with nine defense-crossplay cells each; it also permits descriptive means and sample SDs only. The main study and both supplements identify frozen scientific source `691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`.

All ± values below are **between-seed sample SDs**. Effect intervals are **pointwise paired seed-bootstrap 95% intervals**, not simultaneous or familywise intervals. Exact sign-flip tests assume sign symmetry/exchangeability under the null. Five seeds have minimum two-sided raw p = 0.0625; ten have minimum p = 0.001953125. The plan is a dated post-audit analysis specification, not a preregistration of the original study. A nonsignificant result does not establish absence, equivalence, or robustness.

| Family | Planned tests | Holm p < .05 | Smallest Holm p |

| --- | --- | --- | --- |

| ablation | 9 | 0 | 0.56250000 |

| ablation_behavior | 18 | 0 | 1.00000000 |

| budget | 8 | 0 | 0.50000000 |

| dependence_counterattack | 12 | 0 | 0.75000000 |

| dependence_loop | 6 | 5 | 0.01171875 |

| f1_learning | 6 | 6 | 0.01171875 |

| f2_condition_effects | 63 | 0 | 0.12304688 |

| f2_rule_effects | 36 | 0 | 0.07031250 |

| f3_cycle | 9 | 8 | 0.01757812 |

| f4_cycle | 27 | 0 | 0.05273438 |

| hypothesis_adaptation | 14 | 4 | 0.02734375 |

| loop_dynamics | 25 | 7 | 0.04882812 |

| reward_controls | 18 | 0 | 1.00000000 |

| static_learned | 28 | 0 | 1.00000000 |

| static_rules | 24 | 0 | 1.00000000 |



Thirty comparisons pass 0.05 **within their own declared families**. This does not control familywise error jointly over all 303 comparisons. In particular, no F2 or F4 comparison can reject at 0.05 under the chosen exact test/family sizes: F2 condition family 63, F2 rule family 36, F4 family 27. Do not shrink families after observing these results.

## Main findings that change the paper

1. Learned fixed-defense attacks remain effective (F1); the corrected untrained-to-final increases now have complete paired uncertainty and Holm results.
2. The hypothesis rule has a much larger truthful-testimony cost than archived: FE 0.2447, 0.3700 and 0.3926 at 3+2, 5+2 and 7+2. It is particularly effective against scripted alibi/framer at larger sizes, rather than uniformly superior.
3. F3 trained defenders increase false ejection relative to scripted D0; coalition re-adaptation further increases it at 5+2 and 7+2, while the 3+2 difference is negative and uncertain.
4. F4 coalition adaptation recovers game wins, but at 5+2 and 7+2 barely changes the probability of any false ejection and reduces the number of false ejections per game. Game wins and harm must remain separate.
5. Repeated response gains persist over the finite observed horizon; parity and late/early contrasts do not survive Holm. This is no evidence for an attracting period-2 orbit, stationarity or asymptotic non-convergence.
6. The dependence loop lowers C-stage coalition wins modestly while increasing whole-game false-ejection probability at both C and D stages. A smaller adaptation gap is not a reduction in absolute harm.
7. At 5+2, adaptive attacks on the dependence defenses do not reproduce the archived defense-specific increase; the 3+2 combined rule/vote condition remains strongly vulnerable descriptively.
8. The adaptive attack on hypothesis scoring is stronger than archived, and at 5+2 exceeds the soft-credibility and mean matched adaptive outcomes. A separate fixed-ballot diagnostic supports an abstention/vote vulnerability, not a claim that honest evidence is absent or Bayesian inference is exact.

## F1: fixed scripted defense

| Crew + coalition | Initial FE | Final FE |

| --- | --- | --- |

| 3+2 | 0.0450 ± 0.0105 | 0.2712 ± 0.0456 |

| 5+2 | 0.0730 ± 0.0084 | 0.2125 ± 0.0184 |

| 7+2 | 0.0910 ± 0.0168 | 0.1940 ± 0.0266 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| f1_tenseed_crew3 / final minus initial / false_ejection_rate | 10 | +0.226250 | [+0.197250, +0.255000] | 0.00195312 | 0.01171875 |

| f1_tenseed_crew5 / final minus initial / false_ejection_rate | 10 | +0.139500 | [+0.129500, +0.150000] | 0.00195312 | 0.01171875 |

| f1_tenseed_crew7 / final minus initial / false_ejection_rate | 10 | +0.103000 | [+0.087250, +0.120750] | 0.00195312 | 0.01171875 |



The F1 false-claim mean falls at all three sizes, so remove any statement that explicit lying necessarily increases during this training. Same-target voting rises. Endpoints use the retained true untrained actor and independent final evaluation; monitored curves are separate descriptive measurements.

## F2: seven-rule meeting-only comparison

| Size | Condition | mean | soft_credibility | sharp_credibility | dependence_aware | median | trimmed | hypothesis |

| --- | --- | --- | --- | --- | --- | --- | --- | --- |

| 3+2 | truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1372 ± 0.0173 | 0.1372 ± 0.0173 | 0.2447 ± 0.0175 |

| 3+2 | lone_liar | 0.3282 ± 0.0144 | 0.1782 ± 0.0124 | 0.2036 ± 0.0110 | 0.2099 ± 0.0124 | 0.2451 ± 0.0136 | 0.2451 ± 0.0136 | 0.4118 ± 0.0111 |

| 3+2 | alibi | 0.5119 ± 0.0143 | 0.5237 ± 0.0124 | 0.5039 ± 0.0152 | 0.4509 ± 0.0127 | 0.4386 ± 0.0157 | 0.4386 ± 0.0157 | 0.4817 ± 0.0136 |

| 3+2 | framer | 0.5416 ± 0.0150 | 0.5352 ± 0.0158 | 0.5190 ± 0.0171 | 0.5352 ± 0.0158 | 0.5316 ± 0.0172 | 0.5316 ± 0.0172 | 0.3517 ± 0.0145 |

| 5+2 | truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3889 ± 0.0178 | 0.3118 ± 0.0172 | 0.3700 ± 0.0083 |

| 5+2 | lone_liar | 0.1311 ± 0.0113 | 0.1586 ± 0.0119 | 0.2777 ± 0.0145 | 0.1939 ± 0.0137 | 0.4303 ± 0.0181 | 0.3590 ± 0.0173 | 0.1907 ± 0.0117 |

| 5+2 | alibi | 0.3273 ± 0.0148 | 0.3919 ± 0.0097 | 0.4279 ± 0.0118 | 0.3262 ± 0.0096 | 0.4223 ± 0.0236 | 0.4195 ± 0.0204 | 0.1469 ± 0.0156 |

| 5+2 | framer | 0.4106 ± 0.0097 | 0.4030 ± 0.0147 | 0.4004 ± 0.0164 | 0.4795 ± 0.0181 | 0.4226 ± 0.0211 | 0.5309 ± 0.0164 | 0.1542 ± 0.0107 |

| 7+2 | truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5039 ± 0.0185 | 0.3796 ± 0.0175 | 0.3926 ± 0.0130 |

| 7+2 | lone_liar | 0.0998 ± 0.0088 | 0.1507 ± 0.0091 | 0.2998 ± 0.0094 | 0.1783 ± 0.0124 | 0.5529 ± 0.0166 | 0.4172 ± 0.0164 | 0.0699 ± 0.0066 |

| 7+2 | alibi | 0.2121 ± 0.0161 | 0.3778 ± 0.0141 | 0.4511 ± 0.0127 | 0.3401 ± 0.0157 | 0.5572 ± 0.0161 | 0.4863 ± 0.0207 | 0.0650 ± 0.0096 |

| 7+2 | framer | 0.2604 ± 0.0099 | 0.2551 ± 0.0104 | 0.3145 ± 0.0140 | 0.3546 ± 0.0118 | 0.5217 ± 0.0155 | 0.5349 ± 0.0183 | 0.0876 ± 0.0105 |



The hypothesis rule is no longer a near-zero-cost or exact Bayesian baseline. At 3+2 it is worse than soft under lone liar and worse than median/trimmed under alibi. At 5+2/7+2 it substantially lowers scripted alibi/framer FE but has high truthful-testimony FE. “Truthful” concerns testimony only: the coalition still selects reports and casts protective ballots. The bounded truthful-hypothesis audit retained the true coalition and actual timeline in all inspected cases; ties and protective voting explain inspected failures without implying a population-level mechanism share. Do not conflate this sweep with the spatial F7 sweep or the separate mechanism replication.

| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| f2_tenseed_crew5 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 10 | -0.245000 | [-0.254400, -0.234800] | 0.00195312 | 0.07031250 |

| f2_tenseed_crew5 / framer / dependence_aware minus framer / soft_credibility / false_ejection_rate | 10 | +0.076500 | [+0.071200, +0.082500] | 0.00195312 | 0.07031250 |

| f2_tenseed_crew5 / framer / hypothesis minus framer / soft_credibility / false_ejection_rate | 10 | -0.248800 | [-0.260100, -0.239400] | 0.00195312 | 0.07031250 |

| f2_tenseed_crew7 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 10 | -0.312800 | [-0.320800, -0.305600] | 0.00195312 | 0.07031250 |



## F3: single-meeting staged crossplay

| Size | C0–D0 FE | C0–D1 FE | C1–D1 FE | C1–D0 FE |

| --- | --- | --- | --- | --- |

| 3+2 | 0.1883 ± 0.0454 | 0.4635 ± 0.0728 | 0.4153 ± 0.0888 | 0.0676 ± 0.0337 |

| 5+2 | 0.1940 ± 0.0511 | 0.4031 ± 0.0400 | 0.5385 ± 0.0534 | 0.0759 ± 0.0140 |

| 7+2 | 0.2109 ± 0.0215 | 0.4747 ± 0.0374 | 0.5871 ± 0.0397 | 0.0832 ± 0.0197 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| f3_tenseed_crew3 / defender effect / false_ejection_rate | 10 | -0.275200 | [-0.315003, -0.238498] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew3 / adaptation / false_ejection_rate | 10 | -0.048200 | [-0.098903, +0.006500] | 0.13085938 | 0.13085938 |

| f3_tenseed_crew3 / net change / false_ejection_rate | 10 | +0.227000 | [+0.171000, +0.283200] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew5 / defender effect / false_ejection_rate | 10 | -0.209100 | [-0.250203, -0.168398] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew5 / adaptation / false_ejection_rate | 10 | +0.135400 | [+0.109700, +0.161000] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew5 / net change / false_ejection_rate | 10 | +0.344500 | [+0.302800, +0.392100] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew7 / defender effect / false_ejection_rate | 10 | -0.263800 | [-0.296000, -0.231300] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew7 / adaptation / false_ejection_rate | 10 | +0.112400 | [+0.077300, +0.148702] | 0.00195312 | 0.01757812 |

| f3_tenseed_crew7 / net change / false_ejection_rate | 10 | +0.376200 | [+0.348700, +0.404400] | 0.00195312 | 0.01757812 |



The planned “defender effect” is **C0–D0 minus C0–D1**; negative values mean the learned defender concedes more false ejection. Coalition “adaptation” is C1–D1 minus C0–D1. The archived general incident-avoidance account fails: C0–D0 incident rates are 0.986, 0.963 and 0.943. Conditional incident-free metrics may be undefined for some seeds; retain every seed for unconditional outcomes and explicitly state defined denominators for conditional rows.

## F4: whole-game outcomes must be separated

| Size | Pair | Coalition wins | Any false ejection | False ejections/game | Terminal-meeting FE |

| --- | --- | --- | --- | --- | --- |

| 3+2 | C0–D0 | 0.9967 ± 0.0037 | 0.5617 ± 0.3120 | 0.6256 ± 0.4096 | 0.5610 ± 0.3123 |

| 3+2 | C0–D1 | 0.2821 ± 0.2545 | 0.2863 ± 0.1969 | 0.3239 ± 0.2510 | 0.1816 ± 0.1675 |

| 3+2 | C1–D1 | 0.9408 ± 0.0583 | 0.5206 ± 0.0757 | 0.5393 ± 0.0932 | 0.5061 ± 0.0695 |

| 3+2 | C1–D0 | 0.9882 ± 0.0211 | 0.0879 ± 0.0139 | 0.0879 ± 0.0139 | 0.0879 ± 0.0139 |

| 5+2 | C0–D0 | 0.9670 ± 0.0123 | 0.9740 ± 0.0120 | 2.9063 ± 0.0588 | 0.9640 ± 0.0136 |

| 5+2 | C0–D1 | 0.3603 ± 0.1060 | 0.8392 ± 0.0426 | 1.6336 ± 0.1629 | 0.3537 ± 0.1029 |

| 5+2 | C1–D1 | 0.8514 ± 0.0259 | 0.8467 ± 0.0187 | 1.2087 ± 0.0595 | 0.5075 ± 0.0378 |

| 5+2 | C1–D0 | 0.2182 ± 0.0584 | 0.2889 ± 0.0835 | 0.3324 ± 0.1172 | 0.0371 ± 0.0230 |

| 7+2 | C0–D0 | 0.9634 ± 0.0069 | 0.9782 ± 0.0092 | 4.8620 ± 0.0420 | 0.9703 ± 0.0072 |

| 7+2 | C0–D1 | 0.3040 ± 0.0263 | 0.9469 ± 0.0063 | 2.9957 ± 0.0947 | 0.3666 ± 0.0261 |

| 7+2 | C1–D1 | 0.8144 ± 0.0161 | 0.9470 ± 0.0050 | 1.9891 ± 0.0529 | 0.5371 ± 0.0290 |

| 7+2 | C1–D0 | 0.0630 ± 0.0258 | 0.4086 ± 0.0538 | 0.5580 ± 0.1118 | 0.0149 ± 0.0113 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| f4_tenseed_crew3 / adaptation / any_false_ejection_rate | 10 | +0.234300 | [+0.086900, +0.372600] | 0.02539062 | 0.15234375 |

| f4_tenseed_crew3 / adaptation / coalition_game_win_rate | 10 | +0.658700 | [+0.518200, +0.792302] | 0.00195312 | 0.05273438 |

| f4_tenseed_crew3 / adaptation / mean_false_ejections | 10 | +0.215400 | [+0.033798, +0.386902] | 0.04101562 | 0.20507812 |

| f4_tenseed_crew5 / adaptation / any_false_ejection_rate | 10 | +0.007500 | [-0.018800, +0.034900] | 0.61914062 | 1.00000000 |

| f4_tenseed_crew5 / adaptation / coalition_game_win_rate | 10 | +0.491100 | [+0.431900, +0.558602] | 0.00195312 | 0.05273438 |

| f4_tenseed_crew5 / adaptation / mean_false_ejections | 10 | -0.424900 | [-0.523600, -0.323998] | 0.00195312 | 0.05273438 |

| f4_tenseed_crew7 / adaptation / any_false_ejection_rate | 10 | +0.000100 | [-0.003800, +0.004500] | 1.00000000 | 1.00000000 |

| f4_tenseed_crew7 / adaptation / coalition_game_win_rate | 10 | +0.510400 | [+0.499600, +0.522202] | 0.00195312 | 0.05273438 |

| f4_tenseed_crew7 / adaptation / mean_false_ejections | 10 | -1.006600 | [-1.055903, -0.959200] | 0.00195312 | 0.05273438 |



At 5+2, re-adaptation raises coalition wins by 0.4911 while changing any-FE by only +0.0075 and reducing false-ejection count by 0.4249/game. At 7+2, corresponding changes are +0.5104, +0.0001 and −1.0066. These are different estimands and cannot be summarized as a single recovered “harm” outcome. None of the 27 F4 tests passes Holm at 0.05; the smallest Holm p is 0.052734375 despite large, consistent pointwise effects. Avoid a significance claim based only on a bootstrap interval excluding zero.

## F5: repeated adaptation and finite observed pools

| Configuration | Metric | C-stage mean | D-stage mean | Matrix adaptation gain |

| --- | --- | --- | --- | --- |

| 3+2 r=8 k=10 | any_false_ejection_rate | 0.5088 ± 0.0311 | 0.2480 ± 0.0511 | 0.2548 ± 0.0623 |

| 3+2 r=8 k=10 | coalition_game_win_rate | 0.9177 ± 0.0208 | 0.3415 ± 0.0586 | 0.5756 ± 0.0496 |

| 5+2 r=1 k=10 | false_ejection_rate | 0.5574 ± 0.0155 | 0.2569 ± 0.0226 | 0.3004 ± 0.0322 |

| 5+2 r=8 k=10 | any_false_ejection_rate | 0.8461 ± 0.0067 | 0.6310 ± 0.0294 | 0.2131 ± 0.0329 |

| 5+2 r=8 k=10 | coalition_game_win_rate | 0.8045 ± 0.0245 | 0.3029 ± 0.0347 | 0.5006 ± 0.0498 |

| 7+2 r=8 k=3 | any_false_ejection_rate | 0.9437 ± 0.0086 | 0.9114 ± 0.0056 | 0.0324 ± 0.0184 |

| 7+2 r=8 k=3 | coalition_game_win_rate | 0.7515 ± 0.0485 | 0.3304 ± 0.0435 | 0.4177 ± 0.0881 |

| 5+2 defended r=8 k=4 | any_false_ejection_rate | 0.8897 ± 0.0070 | 0.7626 ± 0.0474 | 0.1290 ± 0.0462 |

| 5+2 defended r=8 k=4 | coalition_game_win_rate | 0.7886 ± 0.0259 | 0.3481 ± 0.0542 | 0.4453 ± 0.0697 |



Stage means summarize C1…Ck and D1…Dk at their 1,000-episode stage evaluations. Matrix adaptation gains use the separate final 500-episode crossplay matrix: mean_g[M(g,g)−M(g−1,g)]. Therefore the two stage means need not subtract exactly to the matrix gain. These full-horizon stage means must not be mislabeled as late-horizon means from the archived table.

| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain / any_false_ejection_rate | 10 | +0.129000 | [+0.103400, +0.156650] | 0.00195312 | 0.04882812 |

| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain / coalition_game_win_rate | 10 | +0.445300 | [+0.405149, +0.486251] | 0.00195312 | 0.04882812 |

| multigen_none_n5_r8_k10 / matrix_adaptation_gain / any_false_ejection_rate | 10 | +0.254820 | [+0.219259, +0.291580] | 0.00195312 | 0.04882812 |

| multigen_none_n5_r8_k10 / matrix_adaptation_gain / coalition_game_win_rate | 10 | +0.575600 | [+0.548760, +0.607020] | 0.00195312 | 0.04882812 |

| multigen_none_n7_r1_k10 / matrix_adaptation_gain / false_ejection_rate | 10 | +0.300360 | [+0.281599, +0.319420] | 0.00195312 | 0.04882812 |

| multigen_none_n7_r8_k10 / matrix_adaptation_gain / any_false_ejection_rate | 10 | +0.213080 | [+0.194120, +0.232620] | 0.00195312 | 0.04882812 |

| multigen_none_n7_r8_k10 / late_minus_early / any_false_ejection_rate | 10 | -0.000300 | [-0.011533, +0.010800] | 0.94531250 | 1.00000000 |

| multigen_none_n7_r8_k10 / parity_gap / any_false_ejection_rate | 10 | +0.052514 | [+0.016075, +0.087752] | 0.02929688 | 0.43945312 |

| multigen_none_n7_r8_k10 / matrix_adaptation_gain / coalition_game_win_rate | 10 | +0.500580 | [+0.472020, +0.529821] | 0.00195312 | 0.04882812 |

| multigen_none_n7_r8_k10 / late_minus_early / coalition_game_win_rate | 10 | -0.027167 | [-0.065900, +0.007133] | 0.20507812 | 1.00000000 |

| multigen_none_n7_r8_k10 / parity_gap / coalition_game_win_rate | 10 | +0.116436 | [+0.036485, +0.196016] | 0.03125000 | 0.43945312 |

| multigen_none_n9_r8_k3 / matrix_adaptation_gain / any_false_ejection_rate | 5 | +0.032400 | [+0.018267, +0.046533] | 0.06250000 | 0.75000000 |

| multigen_none_n9_r8_k3 / matrix_adaptation_gain / coalition_game_win_rate | 5 | +0.417733 | [+0.349467, +0.488400] | 0.06250000 | 0.75000000 |



All seven ten-seed matrix-gain tests pass Holm p=0.048828125; five-seed 7+2 gains are descriptive (Holm .75). No parity or late-minus-early test passes Holm. At 5+2 multi-round, parity gap is +0.116436 for game wins and +0.052514 for any-FE, both Holm .439453125; game-win late-minus-early is −0.027167 with Holm 1. Fresh initialization at every stage and fixed scripted probes prohibit interpreting policy differences as convergence diagnostics for one continuously trained policy. For example, mean final three successive-policy KL values are about 3.637 nats for coalition and 6.144 for crew; corresponding JS values are 0.770 and 0.949 bits summed over heads, not a single distribution’s divergence bounded by one bit.

| Final-generation configuration/metric | Gap mean ± SD | Range of per-seed MC bands | Seeds with lower band > 0 |

| --- | --- | --- | --- |

| multigen_none_n7_r8_k10 / any_false_ejection_rate | 0.6110 ± 0.0582 | [0.3058, 0.8582] | 10/10 |

| multigen_none_n7_r8_k10 / coalition_game_win_rate | 0.6164 ± 0.0500 | [0.3598, 0.8822] | 10/10 |

| multigen_none_n7_r1_k10 / false_ejection_rate | 0.4432 ± 0.0506 | [0.1458, 0.6422] | 10/10 |

| multigen_none_n5_r8_k10 / any_false_ejection_rate | 0.4280 ± 0.1493 | [0.0918, 0.9541] | 10/10 |

| multigen_none_n5_r8_k10 / coalition_game_win_rate | 0.6762 ± 0.2135 | [0.0838, 1.0000] | 10/10 |

| multigen_none_n9_r8_k3 / any_false_ejection_rate | 0.4952 ± 0.0925 | [0.2612, 0.7964] | 5/5 |

| multigen_none_n9_r8_k3 / coalition_game_win_rate | 0.6700 ± 0.0326 | [0.4612, 0.8524] | 5/5 |

| multigen_dependence_n7_r8_k4 / any_false_ejection_rate | 0.6510 ± 0.0687 | [0.3338, 0.8902] | 10/10 |

| multigen_dependence_n7_r8_k4 / coalition_game_win_rate | 0.5564 ± 0.0937 | [0.2038, 0.8922] | 10/10 |



These are retrospective gaps within each finite observed policy pool, including future generations for earlier k. Hoeffding bands account for selecting extrema within each observed matrix and assume independent episode draws within a cell. Across-seed SD and these within-matrix Monte Carlo bands answer different uncertainty questions. The displayed range is the envelope of separate seed bands, not a confidence interval for the mean or a simultaneous guarantee across all seeds/matrices. The nonzero observed-pool gaps support remaining opportunities within the tested pools, not estimates of global exploitability, equilibrium distance or an attracting orbit.

## F7: dependence-aware defenses

### Scripted and frozen-policy spatial tests

| Spatial condition (5+2) | Soft | Dependence rule | Mean | Hypothesis |

| --- | --- | --- | --- | --- |

| truthful | 0.0066 ± 0.0030 | 0.0066 ± 0.0030 | 0.0066 ± 0.0030 | 0.2702 ± 0.0103 |

| lone_liar | 0.1512 ± 0.0110 | 0.2040 ± 0.0115 | 0.1320 ± 0.0076 | 0.2238 ± 0.0138 |

| alibi | 0.4212 ± 0.0154 | 0.3280 ± 0.0124 | 0.3520 ± 0.0125 | 0.1682 ± 0.0066 |

| framer | 0.3994 ± 0.0135 | 0.5214 ± 0.0140 | 0.3878 ± 0.0114 | 0.2224 ± 0.0155 |



The dependence rule lowers scripted alibi FE from .4212 to .3280, but raises framer FE from .3994 to .5214 and lone-liar FE from .1512 to .2040. These are five-seed descriptive comparisons (all static-rule Holm p=1). The strength sweep must keep all strengths: crew-rule-only alibi FE declines .4188→.2904 from λ=0 to 8, while weighted-vote-only is non-monotone and is worse at λ=1 (.4782). Combined and vote-only interventions produce truthful-condition FE .1078 at λ=8 versus .0092 at λ=0; crew-rule-only remains .0092. Do not claim all forms or strengths improve defense. Frozen later C1–C3 policies transfer poorly to simple mean/soft/hypothesis rules; report these as crossplay of observed policies, not adaptive robustness.

### Fresh adaptive attacks

| Trained against (5+2) | vs soft | vs dependence rule | vs combined rule + weighted vote |

| --- | --- | --- | --- |

| soft | 0.1942 ± 0.0546 | 0.1819 ± 0.0515 | 0.2044 ± 0.0522 |

| rule | 0.1736 ± 0.0305 | 0.1696 ± 0.0377 | 0.1927 ± 0.0342 |

| both | 0.1688 ± 0.0383 | 0.1640 ± 0.0421 | 0.1896 ± 0.0402 |



| Trained against (3+2) | vs soft | vs dependence rule | vs combined rule + weighted vote |

| --- | --- | --- | --- |

| soft | 0.2098 ± 0.0475 | 0.2058 ± 0.0382 | 0.4132 ± 0.0402 |

| rule | 0.1936 ± 0.0301 | 0.2256 ± 0.0359 | 0.4388 ± 0.0164 |

| both | 0.0248 ± 0.0064 | 0.0388 ± 0.0168 | 0.8722 ± 0.0104 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| counterattack_n5_r1 / rule adaptation / false_ejection_rate | 5 | +0.019800 | [-0.005000, +0.039800] | 0.18750000 | 1.00000000 |

| counterattack_n5_r1 / both adaptation / false_ejection_rate | 5 | +0.459000 | [+0.427600, +0.491800] | 0.06250000 | 0.75000000 |

| counterattack_n7_r1 / rule adaptation / false_ejection_rate | 10 | -0.012300 | [-0.051602, +0.037400] | 0.63085938 | 1.00000000 |

| counterattack_n7_r1 / both adaptation / false_ejection_rate | 10 | -0.014800 | [-0.051303, +0.021500] | 0.46484375 | 1.00000000 |



The archived claim that defense-trained coalitions increase FE against each dependence defense at 5+2 does not hold in the corrected point estimates. These nonsignificant negative differences do not establish immunity. At 3+2, adaptation against the combined rule/tally raises FE .4132→.8722 and transfers weakly to soft (.0248); retain this substantial but five-seed descriptive vulnerability.

### Matched four-generation defense comparison

| Metric | Statistic | Undefended h=4 | Defended h=4 |

| --- | --- | --- | --- |

| any_false_ejection_rate | c_stage_mean | 0.8473 ± 0.0086 | 0.8897 ± 0.0070 |

| any_false_ejection_rate | d_stage_mean | 0.6583 ± 0.0514 | 0.7626 ± 0.0474 |

| any_false_ejection_rate | matrix_adaptation_gain | 0.1913 ± 0.0506 | 0.1290 ± 0.0462 |

| coalition_game_win_rate | c_stage_mean | 0.8174 ± 0.0245 | 0.7886 ± 0.0259 |

| coalition_game_win_rate | d_stage_mean | 0.3085 ± 0.0653 | 0.3481 ± 0.0542 |

| coalition_game_win_rate | matrix_adaptation_gain | 0.5090 ± 0.0864 | 0.4453 ± 0.0697 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| multigen_dependence_n7_r8_k4 / c_stage_mean minus reference h4 / any_false_ejection_rate | 10 | +0.042425 | [+0.034775, +0.049400] | 0.00195312 | 0.01171875 |

| multigen_dependence_n7_r8_k4 / d_stage_mean minus reference h4 / any_false_ejection_rate | 10 | +0.104375 | [+0.073274, +0.135025] | 0.00195312 | 0.01171875 |

| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain minus reference h4 / any_false_ejection_rate | 10 | -0.062350 | [-0.090000, -0.034050] | 0.00585938 | 0.02343750 |

| multigen_dependence_n7_r8_k4 / c_stage_mean minus reference h4 / coalition_game_win_rate | 10 | -0.028775 | [-0.045625, -0.012050] | 0.01367188 | 0.04101562 |

| multigen_dependence_n7_r8_k4 / d_stage_mean minus reference h4 / coalition_game_win_rate | 10 | +0.039525 | [-0.000975, +0.089300] | 0.13867188 | 0.13867188 |

| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain minus reference h4 / coalition_game_win_rate | 10 | -0.063700 | [-0.123400, -0.015850] | 0.02343750 | 0.04687500 |



The defense increases C-stage and D-stage any-FE while lowering their difference. It modestly lowers C-stage game wins; the D-stage game-win increase is uncertain. The intervention changes D0 and the later voting environment, so it is a protocol-level comparison, not an isolated causal estimate of one credibility penalty. Avoid “reduced harm,” “robust defense” or “breaks the cycle.” Do not reuse the archived post-ejection channel table as mechanistic evidence.

## Controls: information, training budget and reward

The paired control reference uses **exactly seeds 0–4** and a matched generation horizon, not a ten-seed main mean. No ablation, budget or reward contrast passes Holm; five-seed resolution and wide intervals prevent equivalence or “this factor cannot explain the behavior” claims. Information ablations also change the feasible communication/coordination protocol and do not guarantee permanent partner ignorance.

| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| ablation_n7_r1 / no_channel minus default / false_ejection_rate | 5 | +0.033000 | [-0.072600, +0.141200] | 0.62500000 | 1.00000000 |

| ablation_n7_r1 / no_partner minus default / false_ejection_rate | 5 | +0.022600 | [-0.058200, +0.103400] | 0.68750000 | 1.00000000 |

| ablation_n7_r1 / no_channel_no_partner minus default / false_ejection_rate | 5 | +0.066400 | [+0.003000, +0.129800] | 0.18750000 | 1.00000000 |

| ablation_n7_r8 / no_channel minus default / any_false_ejection_rate | 5 | +0.010600 | [+0.002800, +0.018800] | 0.06250000 | 0.56250000 |

| ablation_n7_r8 / no_partner minus default / any_false_ejection_rate | 5 | +0.000600 | [-0.009400, +0.010600] | 0.93750000 | 1.00000000 |

| ablation_n7_r8 / no_channel_no_partner minus default / any_false_ejection_rate | 5 | -0.002400 | [-0.006200, +0.001400] | 0.37500000 | 1.00000000 |

| ablation_n7_r8 / no_channel minus default / coalition_game_win_rate | 5 | +0.010400 | [+0.003600, +0.018600] | 0.12500000 | 1.00000000 |

| ablation_n7_r8 / no_partner minus default / coalition_game_win_rate | 5 | -0.004800 | [-0.017000, +0.007400] | 0.50000000 | 1.00000000 |

| ablation_n7_r8 / no_channel_no_partner minus default / coalition_game_win_rate | 5 | -0.004000 | [-0.009000, +0.002000] | 0.31250000 | 1.00000000 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| f3_budget_d1600_crew5 / coalition0_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | -0.005400 | [-0.058800, +0.055200] | 0.81250000 | 1.00000000 |

| f3_budget_d1600_crew5 / coalition1_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | -0.009200 | [-0.063400, +0.037600] | 0.87500000 | 1.00000000 |

| f3_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | +0.052000 | [-0.007400, +0.123600] | 0.25000000 | 1.00000000 |

| f3_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | +0.042000 | [-0.042400, +0.118800] | 0.43750000 | 1.00000000 |

| f4_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / any_false_ejection_rate | 5 | -0.094800 | [-0.152800, -0.040000] | 0.12500000 | 0.87500000 |

| f4_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / any_false_ejection_rate | 5 | +0.005800 | [-0.025000, +0.036600] | 0.75000000 | 1.00000000 |

| f4_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / coalition_game_win_rate | 5 | -0.189200 | [-0.298200, -0.088800] | 0.06250000 | 0.50000000 |

| f4_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / coalition_game_win_rate | 5 | -0.006000 | [-0.045200, +0.027600] | 0.81250000 | 1.00000000 |



For the single-meeting budget reference, C0–D1/C1–D1 FE is .3874/.5252 on the matched five seeds, versus .4394/.5672 at 800 and .3820/.5160 at 1,600. Multi-round 800-update defenders lower C0 wins .3540→.1648 and any-FE .8400→.7452, while re-adapted C1 outcomes remain .8482 wins/.8520 any-FE. This is evidence of observed budget sensitivity of initial defense, not proof that larger budgets cannot matter.

| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| multigen_rwbal_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / false_ejection_rate | 5 | -0.122300 | [-0.204000, -0.048500] | 0.06250000 | 1.00000000 |

| multigen_rwbal_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / coalition_favorable_rate | 5 | -0.185800 | [-0.254200, -0.125500] | 0.06250000 | 1.00000000 |

| multigen_rwfe_n7_r8_k2 / matrix_adaptation_gain minus reference h2 / any_false_ejection_rate | 5 | -0.010600 | [-0.070600, +0.049400] | 0.68750000 | 1.00000000 |

| multigen_rwfe_n7_r8_k2 / matrix_adaptation_gain minus reference h2 / coalition_game_win_rate | 5 | -0.107800 | [-0.210400, -0.018800] | 0.12500000 | 1.00000000 |

| multigen_rwmeet_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / false_ejection_rate | 5 | -0.066600 | [-0.134600, +0.015300] | 0.18750000 | 1.00000000 |

| multigen_rwmeet_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / coalition_favorable_rate | 5 | -0.090800 | [-0.170000, -0.011600] | 0.25000000 | 1.00000000 |



Reward variants retain positive matrix response gains descriptively but alter their magnitudes. Balanced-ejection single-meeting FE gain is .1642, meeting-ejection gain .2199, versus matched default .2865; coalition-favorable gains are .0481 and .1431 versus .2339. Multi-round framing-count game-win gain is .3326 versus matched default .4404. Thus controls do not justify saying response amplitude is unchanged, invariant to reward, or completely explained by the intervention.

## F8: hypothesis adaptation and separate fixed-ballot diagnostic

| Trained against (5+2) | vs soft | vs mean | vs hypothesis |

| --- | --- | --- | --- |

| soft | 0.1942 ± 0.0546 | 0.1427 ± 0.0530 | 0.1255 ± 0.0584 |

| mean | 0.1484 ± 0.0399 | 0.1043 ± 0.0371 | 0.0769 ± 0.0307 |

| hypothesis | 0.0976 ± 0.0679 | 0.0962 ± 0.0627 | 0.4217 ± 0.0665 |



| Trained against (7+2) | vs soft | vs mean | vs hypothesis |

| --- | --- | --- | --- |

| soft | 0.2018 ± 0.0129 | 0.1360 ± 0.0167 | 0.1102 ± 0.0326 |

| mean | 0.1886 ± 0.0556 | 0.1386 ± 0.0513 | 0.1238 ± 0.0739 |

| hypothesis | 0.0298 ± 0.0073 | 0.0368 ± 0.0075 | 0.6036 ± 0.0184 |



| Contrast (signed as named) | n | Difference | Pointwise 95% CI | Exact p | Holm p |

| --- | --- | --- | --- | --- | --- |

| counterattack_hyp_n7_r1 / hypothesis adaptation / false_ejection_rate | 10 | +0.296200 | [+0.252100, +0.341800] | 0.00195312 | 0.02734375 |

| counterattack_hyp_n7_r1 / hypothesis matched adaptive difference / false_ejection_rate | 10 | +0.227500 | [+0.180898, +0.277700] | 0.00195312 | 0.02734375 |

| counterattack_hyp_n7_r1 / hypothesis / hypothesis minus mean / mean / false_ejection_rate | 10 | +0.317400 | [+0.271400, +0.365000] | 0.00195312 | 0.02734375 |

| counterattack_hyp_n9_r1 / hypothesis adaptation / false_ejection_rate | 5 | +0.493400 | [+0.461400, +0.523200] | 0.06250000 | 0.56250000 |

| counterattack_hyp_n9_r1 / hypothesis matched adaptive difference / false_ejection_rate | 5 | +0.401800 | [+0.383400, +0.420800] | 0.06250000 | 0.56250000 |

| counterattack_hyp_n9_r1 / hypothesis / hypothesis minus mean / mean / false_ejection_rate | 5 | +0.465000 | [+0.407000, +0.512200] | 0.06250000 | 0.56250000 |



At 5+2, direct adaptation increases hypothesis FE .1255→.4217; its matched adaptive outcome exceeds soft (.1942) and mean (.1043), with all three relevant Holm p=.02734375. This supersedes the archived claim that hypothesis and soft matched adaptive rates are similar. At 7+2, .1102→.6036 is a large five-seed descriptive increase (Holm .5625). Hypothesis-trained transfer is weak: .0976/.0962 to soft/mean at 5+2 and .0298/.0368 at 7+2; avoid asserting formal specificity beyond the planned transfer uncertainty.

| Diagnostic metric | 5+2 soft→hypothesis | 5+2 hypothesis→hypothesis | 7+2 hypothesis→hypothesis |

| --- | --- | --- | --- |

| false_ejection_rate | 0.1174 ± 0.0527 | 0.4240 ± 0.0660 | 0.6048 ± 0.0121 |

| coalition_false_claim_rate | 0.3750 ± 0.0912 | 0.0303 ± 0.0328 | 0.0016 ± 0.0019 |

| coalition_noinfo_claim_rate | 0.0124 ± 0.0114 | 0.5405 ± 0.3257 | 0.9962 ± 0.0053 |

| coalition_same_target_rate | 0.2436 ± 0.0670 | 0.6420 ± 0.0789 | 0.6636 ± 0.0293 |

| crew_skip_rate | 0.2711 ± 0.1061 | 0.6608 ± 0.0480 | 0.8988 ± 0.0110 |

| crew_top_membership_score | 0.7464 ± 0.0755 | 0.5072 ± 0.0158 | 0.4097 ± 0.0063 |

| crew_creator_score_top_correct | 0.6172 ± 0.0632 | 0.6570 ± 0.0518 | 0.6228 ± 0.0363 |

| coalition_ballots_necessary_share_of_fe | 0.6838 ± 0.1049 | 0.9726 ± 0.0310 | 1.0000 ± 0.0000 |

| coalition_ballots_sufficient_alone_share_of_fe | 0.6618 ± 0.1304 | 0.9775 ± 0.0247 | 1.0000 ± 0.0000 |

| crew_only_any_false_ejection_rate | 0.0402 ± 0.0164 | 0.0142 ± 0.0149 | 0.0000 ± 0.0000 |



This 500-episode/cell supplement uses its own evidence stream and **does not replace** the main crossplay endpoint (.4240 vs .4217 at 5+2). It includes the full pre-vote electorate and applies the hypothesis model to each honest voter’s evidence. Coalition ballots are necessary for about 97.3% of observed FE at 5+2 and 100% at 7+2 under fixed recorded-ballot deletion; they are sufficient alone for about 97.8%/100%. Crew skip rates are .6608/.8988. Necessary and sufficient are distinct counterfactual tallies conditional on FE, not percentages of all games; neither is a behavioral causal intervention that lets agents react. Zero-score views are zero in these cells, and creator top-score correctness remains about .657/.623, so “no usable honest evidence” or “the crew has no information” overstates the diagnostic. The skip threshold, uncertain membership scores and coordinated ballots are consistent with an abstention-related vulnerability. These summaries have no newly authorized p-values or bootstrap CIs.

## Mechanism supplement and analytical bridge

The 540,000-meeting replication supersedes old one-seed mechanism summaries and remains descriptive. Paired soft-minus-mean alibi FE differences are +.0112, +.0680, +.1586 at 3+2, 5+2, 7+2. Mean coalition softmax weight is .26296, .14247 and .09756, below the corresponding uniform speaker share .5, .3333 and .25. The independent 240,000-voter check found no weight above share beyond 1e−12. Consequently, benchmark failures cannot be described as observed above-share credibility capture. They can arise despite relative downweighting through report selection, truth checking, ballot thresholds and vote aggregation. Removing alibi credit helps alibi conditions but worsens lone-liar/framer cases; moving the catch penalty outside credibility can worsen 3+2 soft lone-liar FE .1734→.3274. Truthful 3+2 FE is zero while no-ejection rate is .9496±.0094, illustrating why low FE alone is insufficient.

Retain the corrected analytical proposition as a surrogate result under explicitly stated report-generation, group independence and strict-minority assumptions. It does not establish the mechanism of the game measurements. After an incident, 3+2 becomes two honest and two coalition speakers, outside the theorem’s strict-minority regime. Bias thresholds, realized estimator MSE, and finite-sample median comparisons are different objects. The sensor-fusion illustration and corrected theory Monte Carlo are separate analytical experiments and do not acquire inference from the 240-job replication.

## Manuscript replacement checklist

- Replace all archived result tables/figures in F2, F4, F5, F7, F8 and the three controls with complete corrected outputs; do not keep archived numbers as primary evidence. Revise the title/abstract/introduction/conclusion around mixed defense tradeoffs, finite repeated adaptation and the stronger adaptive hypothesis vulnerability.
- Remove pending-replication/provisional language for the completed numerical study. State remaining methodological limits directly. Keep history in audit documentation, not interleaved as the primary scientific narrative.
- Replace old raw p-values, ratios, counts, slopes, policy distances and narrative values wherever unsupported by the corrected outputs. Every claim of statistical evidence must identify the actual Holm family/result.
- Use one-round FE, terminal FE, any-FE, false-ejection count/game, incident-gated creator survival and coalition wins with their proper denominators. Recompute all outcome tables from the final source tables rather than changing captions alone.
- Explain finite-pool gaps and their Monte Carlo uncertainty, full-horizon vs early/late stage summaries, fresh actor initialization, and descriptive policy distances.
- Correct Appendix B to the repaired full-episode PPO collection, full within-round actor history, no hypothesis pseudoweight, exact compatibility constraints and independent final evidence stream. The former archived defaults must not describe the final experiments.
- Preserve the actual source and protocol hashes with the completed results. Code cleanup creates a new fingerprint, not a new identity for old runs.
- Ensure the final manuscript and package contain no unresolved draft placeholders, archived replay presented as current evidence, stale figure references, or unwarranted global “best defense,” convergence, exact posterior or captured-weight claims. Author declarations and any repository/data-archive statement must match what has actually been published.

## Evidence files

- `analysis/status.json`, `analysis/cell_summary.json`, `analysis/paired_effects.json`, `analysis/matched_reference_summary.json`, `analysis/restricted_pool_gaps.json`: final completeness, estimates and all 303 fixed contrasts.
- `ANALYSIS_PLAN.md`, `analysis/planned_comparisons.json`: fixed estimands, uncertainty and multiplicity families.
- `FINAL_RESULTS_NUMERIC_CHECK.json`: independent arithmetic check performed for this review.
- `mechanism_results/analysis/complete.json`, `cell_summary.json`, `paired_descriptive.json`; `mechanism_weight_audit.json`: complete descriptive mechanism replication.
- `vote_diagnostic_results/complete.json`, `summary.json`: complete descriptive fixed-ballot supplement.
- `F2_TRUTHFUL_HYPOTHESIS_AUDIT.md`, `HYPOTHESIS_CALIBRATION_REVIEW.md`: bounded failure probes; these are not prevalence estimates.
- `UNDEFINED_COORDINATION_REVIEW.md`: raw-evidence justification for two undefined terminal coordination cells in one nine-player crossplay job; no primary outcomes or seeds were omitted.
