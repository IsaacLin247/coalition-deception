# Completed mechanism replication: independent findings

These are NEW completed results from the repaired, frozen implementation, not the archived F2 sweep or its former single-seed mechanism diagnostic. Completion records identify 30 jobs and 540,000 meetings: 3 initial crew sizes × 10 evaluation replicates × 4 scripted conditions × 3 aggregation rules × 3 variants × 500 meetings. All results below are means ± sample SD across ten replicate estimates. Paired changes use the SD of the ten within-replicate differences. They are descriptive; no significance, confidence interval, multiplicity, or population-guarantee claim is made.

## Design and denominator checks

- Each crew/condition/replicate shares its evidence seed and episode identities across all nine rule/variant cells. Conditions use different streams. Responses and votes are regenerated under the changed rule; these are policy interventions on matched evidence, not fixed-transcript vote interventions or an identification of a unique causal channel.
- `alibi_off` sets `W_ALIBI_WITNESS=0`, removing both the testimony-away-from-scene alibi credit and the positive `WAS_WITH` subject credit (normally 0.75 times that coefficient). `catch_outside` applies caught-lying suspicion as the voter's own evidence rather than within the speaker's aggregated row. Switches remain active during crew responses and voting.
- An incident occurs in every recorded meeting and removes one honest player before testimony. Initial 3+2, 5+2, and 7+2 games therefore have 2+2, 4+2, and 6+2 actual reporters. The coalition's uniform shares are 1/2, 1/3, and 1/4. Appendix A's n and m count reporters, so the 3+2 game has n=m and falls outside its strict-minority assumption m<n.
- Table 5 averages over all honest voters in each replicate: respectively 1,000, 2,000, or 3,000 decisions per condition. “Caught” means at least one coalition claim was privately refuted by that voter; it is not the fraction of meetings or claims caught. “Top pick differs” compares the mean and soft rule on the same soft-baseline public record and eligible candidates; it is not an actual cross-run vote discrepancy. Credibility scores include private refutation and self-contradiction penalties as well as pairwise compatibility, so they are not pure corroboration counts.
- All scripted speakers were informative in these recordings: uninformative share and its allocated weight were both zero. This experiment does not test NO_INFORMATION behavior.

## Voter weights (all twelve conditions)

| Game, condition | Credibility C | Credibility H | Coalition weight (uniform) | Coalition caught | Top pick differs |
|:--|--:|--:|--:|--:|--:|
| 3+2, Truthful | 3.000 ± 0.000 | 3.000 ± 0.000 | 0.5000 ± 0.0000 (0.5000) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 3+2, Lone liar | 1.581 ± 0.037 | 2.982 ± 0.006 | 0.4225 ± 0.0022 (0.5000) | 0.4649 ± 0.0129 | 0.1264 ± 0.0076 |
| 3+2, Alibi | 0.164 ± 0.075 | 2.960 ± 0.008 | 0.2630 ± 0.0059 (0.5000) | 0.5583 ± 0.0147 | 0.1674 ± 0.0161 |
| 3+2, Framer | -1.547 ± 0.052 | 2.593 ± 0.040 | 0.1479 ± 0.0041 (0.5000) | 0.7787 ± 0.0080 | 0.0029 ± 0.0014 |
| 5+2, Truthful | 5.000 ± 0.000 | 5.000 ± 0.000 | 0.3333 ± 0.0000 (0.3333) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 5+2, Lone liar | 2.836 ± 0.054 | 4.987 ± 0.003 | 0.2606 ± 0.0017 (0.3333) | 0.5353 ± 0.0138 | 0.0319 ± 0.0054 |
| 5+2, Alibi | 0.778 ± 0.078 | 4.970 ± 0.004 | 0.1425 ± 0.0030 (0.3333) | 0.6317 ± 0.0093 | 0.0769 ± 0.0085 |
| 5+2, Framer | -0.335 ± 0.081 | 4.476 ± 0.025 | 0.1062 ± 0.0035 (0.3333) | 0.6921 ± 0.0125 | 0.0085 ± 0.0021 |
| 7+2, Truthful | 7.000 ± 0.000 | 7.000 ± 0.000 | 0.2500 ± 0.0000 (0.2500) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 7+2, Lone liar | 4.068 ± 0.036 | 6.989 ± 0.002 | 0.1863 ± 0.0010 (0.2500) | 0.5819 ± 0.0064 | 0.0174 ± 0.0036 |
| 7+2, Alibi | 1.520 ± 0.124 | 6.973 ± 0.003 | 0.0976 ± 0.0027 (0.2500) | 0.6535 ± 0.0104 | 0.0672 ± 0.0069 |
| 7+2, Framer | 0.361 ± 0.057 | 6.402 ± 0.013 | 0.0630 ± 0.0018 (0.2500) | 0.6830 ± 0.0080 | 0.0086 ± 0.0032 |

No attacked condition has average coalition weight above its uniform share. This alone would not rule out individual meetings with weight capture. A separate raw-record audit recomputed softmax weights from credibility scores for all 240,000 honest voter records, agreeing within 1.67×10⁻¹⁶, and found zero voters or meetings with coalition weight above the relevant uniform share plus 10⁻¹². This is an observed absence in these scripted records, not a guarantee for arbitrary attacks, rules, or environments. Truthful scores are uniform, with zero recorded same-record mean-versus-soft top-pick differences.

## False-ejection results (all conditions, rules, and variants)

M = mean, S = soft credibility, H = sharp credibility; unqualified columns are baseline. Each entry contains 5,000 meetings across ten replicates.

| Game, condition | M | S | H | M: alibi off | S: alibi off | H: alibi off | M: catch outside | S: catch outside | H: catch outside |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 3+2, Truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 3+2, Lone liar | 0.3290 ± 0.0179 | 0.1734 ± 0.0205 | 0.2010 ± 0.0230 | 0.3662 ± 0.0184 | 0.2106 ± 0.0237 | 0.2188 ± 0.0277 | 0.3290 ± 0.0179 | 0.3274 ± 0.0178 | 0.3230 ± 0.0175 |
| 3+2, Alibi | 0.5176 ± 0.0249 | 0.5288 ± 0.0238 | 0.5144 ± 0.0252 | 0.4030 ± 0.0219 | 0.3728 ± 0.0193 | 0.4406 ± 0.0222 | 0.5180 ± 0.0250 | 0.5046 ± 0.0223 | 0.4528 ± 0.0201 |
| 3+2, Framer | 0.5354 ± 0.0195 | 0.5294 ± 0.0200 | 0.5124 ± 0.0202 | 0.5370 ± 0.0200 | 0.5306 ± 0.0198 | 0.5172 ± 0.0209 | 0.5354 ± 0.0195 | 0.5294 ± 0.0200 | 0.5110 ± 0.0204 |
| 5+2, Truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 5+2, Lone liar | 0.1326 ± 0.0181 | 0.1598 ± 0.0184 | 0.2792 ± 0.0187 | 0.1876 ± 0.0171 | 0.2064 ± 0.0199 | 0.3244 ± 0.0154 | 0.1326 ± 0.0181 | 0.1316 ± 0.0189 | 0.1316 ± 0.0189 |
| 5+2, Alibi | 0.3122 ± 0.0198 | 0.3802 ± 0.0242 | 0.4198 ± 0.0282 | 0.2312 ± 0.0224 | 0.2310 ± 0.0215 | 0.3414 ± 0.0191 | 0.3122 ± 0.0198 | 0.2890 ± 0.0183 | 0.2676 ± 0.0210 |
| 5+2, Framer | 0.4060 ± 0.0248 | 0.4014 ± 0.0275 | 0.4054 ± 0.0214 | 0.5244 ± 0.0236 | 0.4890 ± 0.0259 | 0.4706 ± 0.0229 | 0.4060 ± 0.0248 | 0.4010 ± 0.0276 | 0.3790 ± 0.0261 |
| 7+2, Truthful | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 7+2, Lone liar | 0.0908 ± 0.0179 | 0.1438 ± 0.0231 | 0.3012 ± 0.0237 | 0.1416 ± 0.0139 | 0.1834 ± 0.0169 | 0.3672 ± 0.0205 | 0.0908 ± 0.0179 | 0.0902 ± 0.0179 | 0.0902 ± 0.0179 |
| 7+2, Alibi | 0.2080 ± 0.0190 | 0.3666 ± 0.0250 | 0.4362 ± 0.0256 | 0.1632 ± 0.0191 | 0.2090 ± 0.0157 | 0.4012 ± 0.0254 | 0.2080 ± 0.0190 | 0.1916 ± 0.0193 | 0.1786 ± 0.0199 |
| 7+2, Framer | 0.2654 ± 0.0176 | 0.2574 ± 0.0180 | 0.3162 ± 0.0163 | 0.4670 ± 0.0212 | 0.3906 ± 0.0212 | 0.4192 ± 0.0231 | 0.2654 ± 0.0176 | 0.2568 ± 0.0176 | 0.2412 ± 0.0172 |

## Paired descriptive changes

| Game, condition | S − M | H − M | S: alibi off − baseline | S: catch outside − baseline |
|:--|--:|--:|--:|--:|
| 3+2, Truthful | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 3+2, Lone liar | -0.1556 ± 0.0195 | -0.1280 ± 0.0245 | +0.0372 ± 0.0102 | +0.1540 ± 0.0188 |
| 3+2, Alibi | +0.0112 ± 0.0080 | -0.0032 ± 0.0082 | -0.1560 ± 0.0130 | -0.0242 ± 0.0066 |
| 3+2, Framer | -0.0060 ± 0.0031 | -0.0230 ± 0.0049 | +0.0012 ± 0.0010 | +0.0000 ± 0.0000 |
| 5+2, Truthful | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 5+2, Lone liar | +0.0272 ± 0.0082 | +0.1466 ± 0.0159 | +0.0466 ± 0.0109 | -0.0282 ± 0.0087 |
| 5+2, Alibi | +0.0680 ± 0.0136 | +0.1076 ± 0.0219 | -0.1492 ± 0.0173 | -0.0912 ± 0.0150 |
| 5+2, Framer | -0.0046 ± 0.0104 | -0.0006 ± 0.0108 | +0.0876 ± 0.0143 | -0.0004 ± 0.0008 |
| 7+2, Truthful | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 7+2, Lone liar | +0.0530 ± 0.0103 | +0.2104 ± 0.0139 | +0.0396 ± 0.0159 | -0.0536 ± 0.0104 |
| 7+2, Alibi | +0.1586 ± 0.0112 | +0.2282 ± 0.0172 | -0.1576 ± 0.0196 | -0.1750 ± 0.0084 |
| 7+2, Framer | -0.0080 ± 0.0060 | +0.0508 ± 0.0124 | +0.1332 ± 0.0222 | -0.0006 ± 0.0010 |

The soft rule's baseline false-ejection disadvantage under alibi increases with game size: +0.0112 ± 0.0080, +0.0680 ± 0.0136, and +0.1586 ± 0.0112 relative to the mean. Moving the caught penalty outside reduces soft alibi false ejections by 0.0242 ± 0.0066, 0.0912 ± 0.0150, and 0.1750 ± 0.0084. It also improves sharp alibi outcomes at all three sizes. At 5+2 and 7+2, the resulting soft alibi rates are below the baseline mean; the evidence does not support the old blanket claim that catch-outside always leaves credibility worse than mean.

Neither intervention is uniformly beneficial. Removing alibi support lowers alibi false ejections for all three rules and all three sizes, but raises them for the lone liar and framer (including soft framer increases of 0.0876 ± 0.0143 at 5+2 and 0.1332 ± 0.0222 at 7+2). Catch-outside sharply worsens soft lone-liar false ejections at 3+2: +0.1540 ± 0.0188, from 0.1734 to 0.3274. It improves the lone-liar rates at the larger sizes, while soft framer rates change by at most 0.0006. Baseline soft framer rates are slightly lower than the mean at all three sizes; sharp is worse than the mean at 7+2 by 0.0508 ± 0.0124. The data support an interaction among testimony, penalties, and decisions, not a single monotone ranking of credibility rules.

False-ejection reductions need not represent more successful identification. All truthful cells have zero observed false ejections, yet truthful 3+2 baseline and catch-outside meetings have no-ejection rate 0.9496 ± 0.0094 and creator-ejection rate 0.0504 ± 0.0094; alibi-off changes these to 0.9062 ± 0.0115 and 0.0938 ± 0.0115. At truthful 5+2 and 7+2 every meeting ejects a coalition member, with creator-ejection rates 0.9030 ± 0.0116 and 0.9204 ± 0.0113 (the remainder eject the partner). All 3+2 coordinated alibi/framer cells have coalition-favorable rate 1 and creator-ejection rate 0: lower false ejection there means more no-ejection outcomes. For the 5+2 lone liar, soft catch-outside lowers false ejection but raises no-ejection from 0.0402 ± 0.0049 to 0.1100 ± 0.0105 and coalition-favorable outcomes from 0.2000 ± 0.0207 to 0.2416 ± 0.0257. Thus the broader coalition-favorable objective must not be conflated with false ejection.

## Numerical negative-control caveat

Mean aggregation makes catch-inside and catch-outside scores algebraically equivalent, but floating-point addition order prevents bitwise invariance. Exactly two of the 5,000 paired 3+2 alibi episodes change false-ejection status: replicate 5 / episode 476 and replicate 7 / episode 20. Independent replays reproduce the original evidence hashes and the same responses. One honest voter's score changes by 2.220446049250313×10⁻¹⁶, changing a coalition-target argmax tie from the higher index to the lower index. Splitting those honest ballots changes no-ejection into crew ejection. Consequently the mean rate is 0.5176 ± 0.0249 at baseline and 0.5180 ± 0.0250 outside; this is a numerical tie effect, not evidence of a substantive mean-rule mechanism. No frozen engine or production output was changed.

## Independent checks and retained artifacts

- Recomputed all 816 cell summaries from 8,160 per-replicate values and all 864 paired descriptive summaries: no discrepancy at absolute tolerance 10⁻¹⁴ / relative tolerance 10⁻¹² (`mechanism_summary_audit.json`).
- `mechanism_weight_audit.json` records the full raw softmax check and per-replicate above-share indicators. The negative-control script, JSON, and log are `mechanism_negative_control_probe.{py,json,log}`; the script runs four bounded batches to recreate the two cases, with no new sweep.
- `mechanism_paper_replacement.md` supplies the exact subsection replacement. Adjacent archived F2 ranking/hypothesis paragraphs, the later F6 correspondence paragraph, discussion, and limitations still require the root's coordinated rewrite. In particular replace “caught in most meetings” by the actual voter-based denominator, remove the obsolete soft-framer 0.472 example, remove the one-seed limitation for Tables 5/6, and avoid saying that the new results establish a single causal channel.

Provenance:
- `source_sha256`: `691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35`
- `protocol_sha256`: `a0adaef10149da078388efcb642bd24236addd35ef4803d88aecb44203ce1610`
- `script_sha256`: `2cce07aa5277a0feb723aa2121eb8b0dc80a6fa277c293124f3c8882a6a15495`
