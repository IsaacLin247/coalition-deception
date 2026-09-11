# Verified replication tables and figures

240/240 analyzed jobs; 30 complete seed groups displayed. All planned jobs are complete.

Cells report mean ± sample SD across independent seeds. SD is between-seed variability, not a confidence interval. Terminal-meeting outcomes and whole-game outcomes are labeled separately. No archived figures or captions are used.

## F1: untrained and trained holdout endpoints

Both endpoints use the independent final evidence stream. Initial means the untrained actor; final means the fixed-budget checkpoint.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `f1_tenseed_crew3` / initial | 10 | 0.045 ± 0.011 | 0.201 ± 0.019 |
| `f1_tenseed_crew3` / final | 10 | 0.271 ± 0.046 | 0.691 ± 0.052 |
| `f1_tenseed_crew5` / initial | 10 | 0.073 ± 0.008 | 0.099 ± 0.014 |
| `f1_tenseed_crew5` / final | 10 | 0.212 ± 0.018 | 0.259 ± 0.025 |
| `f1_tenseed_crew7` / initial | 10 | 0.091 ± 0.017 | 0.105 ± 0.017 |
| `f1_tenseed_crew7` / final | 10 | 0.194 ± 0.027 | 0.215 ± 0.029 |

[f1_learning_and_holdout.pdf](f1_learning_and_holdout.pdf) — F1 monitored curves and independent holdout endpoints. Curve axes use the analyzer's completed-update values; monitored evaluations are separate from both holdout endpoints.

## Scripted rule evaluation: depstatic_n7_r1



| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `depstatic_n7_r1` / alibi / dependence_aware | 5 | 0.328 ± 0.012 |
| `depstatic_n7_r1` / alibi / hypothesis | 5 | 0.168 ± 0.007 |
| `depstatic_n7_r1` / alibi / mean | 5 | 0.352 ± 0.013 |
| `depstatic_n7_r1` / alibi / median | 5 | 0.443 ± 0.013 |
| `depstatic_n7_r1` / alibi / sharp_credibility | 5 | 0.412 ± 0.008 |
| `depstatic_n7_r1` / alibi / soft_credibility | 5 | 0.421 ± 0.015 |
| `depstatic_n7_r1` / alibi / trimmed | 5 | 0.474 ± 0.015 |
| `depstatic_n7_r1` / framer / dependence_aware | 5 | 0.521 ± 0.014 |
| `depstatic_n7_r1` / framer / hypothesis | 5 | 0.222 ± 0.015 |
| `depstatic_n7_r1` / framer / mean | 5 | 0.388 ± 0.011 |
| `depstatic_n7_r1` / framer / median | 5 | 0.438 ± 0.010 |
| `depstatic_n7_r1` / framer / sharp_credibility | 5 | 0.383 ± 0.016 |
| `depstatic_n7_r1` / framer / soft_credibility | 5 | 0.399 ± 0.014 |
| `depstatic_n7_r1` / framer / trimmed | 5 | 0.569 ± 0.012 |
| `depstatic_n7_r1` / lone_liar / dependence_aware | 5 | 0.204 ± 0.011 |
| `depstatic_n7_r1` / lone_liar / hypothesis | 5 | 0.224 ± 0.014 |
| `depstatic_n7_r1` / lone_liar / mean | 5 | 0.132 ± 0.008 |
| `depstatic_n7_r1` / lone_liar / median | 5 | 0.439 ± 0.014 |
| `depstatic_n7_r1` / lone_liar / sharp_credibility | 5 | 0.210 ± 0.007 |
| `depstatic_n7_r1` / lone_liar / soft_credibility | 5 | 0.151 ± 0.011 |
| `depstatic_n7_r1` / lone_liar / trimmed | 5 | 0.351 ± 0.010 |
| `depstatic_n7_r1` / truthful / dependence_aware | 5 | 0.007 ± 0.003 |
| `depstatic_n7_r1` / truthful / hypothesis | 5 | 0.270 ± 0.010 |
| `depstatic_n7_r1` / truthful / mean | 5 | 0.007 ± 0.003 |
| `depstatic_n7_r1` / truthful / median | 5 | 0.398 ± 0.016 |
| `depstatic_n7_r1` / truthful / sharp_credibility | 5 | 0.007 ± 0.003 |
| `depstatic_n7_r1` / truthful / soft_credibility | 5 | 0.007 ± 0.003 |
| `depstatic_n7_r1` / truthful / trimmed | 5 | 0.273 ± 0.011 |

[depstatic_n7_r1_rules.pdf](depstatic_n7_r1_rules.pdf) — Scripted coalition and crew-rule evaluation (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Static learned-policy evaluation: depstatic_n7_r1



| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `depstatic_n7_r1` / C0 / dependence_aware / both | 5 | 0.197 ± 0.073 |
| `depstatic_n7_r1` / C0 / dependence_aware / crew_rule | 5 | 0.185 ± 0.072 |
| `depstatic_n7_r1` / C0 / hypothesis / crew_rule | 5 | 0.111 ± 0.061 |
| `depstatic_n7_r1` / C0 / mean / crew_rule | 5 | 0.145 ± 0.068 |
| `depstatic_n7_r1` / C0 / median / crew_rule | 5 | 0.361 ± 0.031 |
| `depstatic_n7_r1` / C0 / sharp_credibility / crew_rule | 5 | 0.250 ± 0.057 |
| `depstatic_n7_r1` / C0 / soft_credibility / crew_rule | 5 | 0.193 ± 0.067 |
| `depstatic_n7_r1` / C0 / trimmed / crew_rule | 5 | 0.325 ± 0.041 |
| `depstatic_n7_r1` / C1 / dependence_aware / both | 5 | 0.100 ± 0.019 |
| `depstatic_n7_r1` / C1 / dependence_aware / crew_rule | 5 | 0.062 ± 0.020 |
| `depstatic_n7_r1` / C1 / hypothesis / crew_rule | 5 | 0.057 ± 0.028 |
| `depstatic_n7_r1` / C1 / mean / crew_rule | 5 | 0.038 ± 0.014 |
| `depstatic_n7_r1` / C1 / median / crew_rule | 5 | 0.246 ± 0.088 |
| `depstatic_n7_r1` / C1 / sharp_credibility / crew_rule | 5 | 0.126 ± 0.044 |
| `depstatic_n7_r1` / C1 / soft_credibility / crew_rule | 5 | 0.074 ± 0.021 |
| `depstatic_n7_r1` / C1 / trimmed / crew_rule | 5 | 0.205 ± 0.066 |
| `depstatic_n7_r1` / C2 / dependence_aware / both | 5 | 0.112 ± 0.029 |
| `depstatic_n7_r1` / C2 / dependence_aware / crew_rule | 5 | 0.086 ± 0.022 |
| `depstatic_n7_r1` / C2 / hypothesis / crew_rule | 5 | 0.061 ± 0.021 |
| `depstatic_n7_r1` / C2 / mean / crew_rule | 5 | 0.058 ± 0.018 |
| `depstatic_n7_r1` / C2 / median / crew_rule | 5 | 0.276 ± 0.072 |
| `depstatic_n7_r1` / C2 / sharp_credibility / crew_rule | 5 | 0.154 ± 0.034 |
| `depstatic_n7_r1` / C2 / soft_credibility / crew_rule | 5 | 0.092 ± 0.021 |
| `depstatic_n7_r1` / C2 / trimmed / crew_rule | 5 | 0.245 ± 0.046 |
| `depstatic_n7_r1` / C3 / dependence_aware / both | 5 | 0.114 ± 0.019 |
| `depstatic_n7_r1` / C3 / dependence_aware / crew_rule | 5 | 0.076 ± 0.023 |
| `depstatic_n7_r1` / C3 / hypothesis / crew_rule | 5 | 0.050 ± 0.022 |
| `depstatic_n7_r1` / C3 / mean / crew_rule | 5 | 0.045 ± 0.016 |
| `depstatic_n7_r1` / C3 / median / crew_rule | 5 | 0.265 ± 0.060 |
| `depstatic_n7_r1` / C3 / sharp_credibility / crew_rule | 5 | 0.148 ± 0.027 |
| `depstatic_n7_r1` / C3 / soft_credibility / crew_rule | 5 | 0.080 ± 0.020 |
| `depstatic_n7_r1` / C3 / trimmed / crew_rule | 5 | 0.228 ± 0.037 |

## Dependence strength sweep: depstatic_n7_r1

Every recorded strength is shown; no best strength is selected.

| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `depstatic_n7_r1` / alibi / both / 0.0 | 5 | 0.419 ± 0.004 |
| `depstatic_n7_r1` / alibi / both / 1.0 | 5 | 0.430 ± 0.011 |
| `depstatic_n7_r1` / alibi / both / 2.0 | 5 | 0.424 ± 0.007 |
| `depstatic_n7_r1` / alibi / both / 4.0 | 5 | 0.386 ± 0.021 |
| `depstatic_n7_r1` / alibi / both / 6.0 | 5 | 0.362 ± 0.019 |
| `depstatic_n7_r1` / alibi / both / 8.0 | 5 | 0.340 ± 0.018 |
| `depstatic_n7_r1` / alibi / crew_rule / 0.0 | 5 | 0.419 ± 0.004 |
| `depstatic_n7_r1` / alibi / crew_rule / 1.0 | 5 | 0.375 ± 0.009 |
| `depstatic_n7_r1` / alibi / crew_rule / 2.0 | 5 | 0.367 ± 0.007 |
| `depstatic_n7_r1` / alibi / crew_rule / 4.0 | 5 | 0.323 ± 0.022 |
| `depstatic_n7_r1` / alibi / crew_rule / 6.0 | 5 | 0.296 ± 0.017 |
| `depstatic_n7_r1` / alibi / crew_rule / 8.0 | 5 | 0.290 ± 0.017 |
| `depstatic_n7_r1` / alibi / vote_tally / 0.0 | 5 | 0.419 ± 0.004 |
| `depstatic_n7_r1` / alibi / vote_tally / 1.0 | 5 | 0.478 ± 0.010 |
| `depstatic_n7_r1` / alibi / vote_tally / 2.0 | 5 | 0.475 ± 0.010 |
| `depstatic_n7_r1` / alibi / vote_tally / 4.0 | 5 | 0.442 ± 0.013 |
| `depstatic_n7_r1` / alibi / vote_tally / 6.0 | 5 | 0.443 ± 0.013 |
| `depstatic_n7_r1` / alibi / vote_tally / 8.0 | 5 | 0.407 ± 0.016 |
| `depstatic_n7_r1` / truthful / both / 0.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / both / 1.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / both / 2.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / both / 4.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / both / 6.0 | 5 | 0.021 ± 0.003 |
| `depstatic_n7_r1` / truthful / both / 8.0 | 5 | 0.108 ± 0.006 |
| `depstatic_n7_r1` / truthful / crew_rule / 0.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / crew_rule / 1.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / crew_rule / 2.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / crew_rule / 4.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / crew_rule / 6.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / crew_rule / 8.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 0.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 1.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 2.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 4.0 | 5 | 0.009 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 6.0 | 5 | 0.021 ± 0.003 |
| `depstatic_n7_r1` / truthful / vote_tally / 8.0 | 5 | 0.108 ± 0.006 |

## Scripted rule evaluation: f2_tenseed_crew3



| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `f2_tenseed_crew3` / alibi / dependence_aware | 10 | 0.451 ± 0.013 |
| `f2_tenseed_crew3` / alibi / hypothesis | 10 | 0.482 ± 0.014 |
| `f2_tenseed_crew3` / alibi / mean | 10 | 0.512 ± 0.014 |
| `f2_tenseed_crew3` / alibi / median | 10 | 0.439 ± 0.016 |
| `f2_tenseed_crew3` / alibi / sharp_credibility | 10 | 0.504 ± 0.015 |
| `f2_tenseed_crew3` / alibi / soft_credibility | 10 | 0.524 ± 0.012 |
| `f2_tenseed_crew3` / alibi / trimmed | 10 | 0.439 ± 0.016 |
| `f2_tenseed_crew3` / framer / dependence_aware | 10 | 0.535 ± 0.016 |
| `f2_tenseed_crew3` / framer / hypothesis | 10 | 0.352 ± 0.014 |
| `f2_tenseed_crew3` / framer / mean | 10 | 0.542 ± 0.015 |
| `f2_tenseed_crew3` / framer / median | 10 | 0.532 ± 0.017 |
| `f2_tenseed_crew3` / framer / sharp_credibility | 10 | 0.519 ± 0.017 |
| `f2_tenseed_crew3` / framer / soft_credibility | 10 | 0.535 ± 0.016 |
| `f2_tenseed_crew3` / framer / trimmed | 10 | 0.532 ± 0.017 |
| `f2_tenseed_crew3` / lone_liar / dependence_aware | 10 | 0.210 ± 0.012 |
| `f2_tenseed_crew3` / lone_liar / hypothesis | 10 | 0.412 ± 0.011 |
| `f2_tenseed_crew3` / lone_liar / mean | 10 | 0.328 ± 0.014 |
| `f2_tenseed_crew3` / lone_liar / median | 10 | 0.245 ± 0.014 |
| `f2_tenseed_crew3` / lone_liar / sharp_credibility | 10 | 0.204 ± 0.011 |
| `f2_tenseed_crew3` / lone_liar / soft_credibility | 10 | 0.178 ± 0.012 |
| `f2_tenseed_crew3` / lone_liar / trimmed | 10 | 0.245 ± 0.014 |
| `f2_tenseed_crew3` / truthful / dependence_aware | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew3` / truthful / hypothesis | 10 | 0.245 ± 0.018 |
| `f2_tenseed_crew3` / truthful / mean | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew3` / truthful / median | 10 | 0.137 ± 0.017 |
| `f2_tenseed_crew3` / truthful / sharp_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew3` / truthful / soft_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew3` / truthful / trimmed | 10 | 0.137 ± 0.017 |

[f2_tenseed_crew3_rules.pdf](f2_tenseed_crew3_rules.pdf) — Scripted coalition and crew-rule evaluation (3+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Scripted rule evaluation: f2_tenseed_crew5



| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `f2_tenseed_crew5` / alibi / dependence_aware | 10 | 0.326 ± 0.010 |
| `f2_tenseed_crew5` / alibi / hypothesis | 10 | 0.147 ± 0.016 |
| `f2_tenseed_crew5` / alibi / mean | 10 | 0.327 ± 0.015 |
| `f2_tenseed_crew5` / alibi / median | 10 | 0.422 ± 0.024 |
| `f2_tenseed_crew5` / alibi / sharp_credibility | 10 | 0.428 ± 0.012 |
| `f2_tenseed_crew5` / alibi / soft_credibility | 10 | 0.392 ± 0.010 |
| `f2_tenseed_crew5` / alibi / trimmed | 10 | 0.419 ± 0.020 |
| `f2_tenseed_crew5` / framer / dependence_aware | 10 | 0.479 ± 0.018 |
| `f2_tenseed_crew5` / framer / hypothesis | 10 | 0.154 ± 0.011 |
| `f2_tenseed_crew5` / framer / mean | 10 | 0.411 ± 0.010 |
| `f2_tenseed_crew5` / framer / median | 10 | 0.423 ± 0.021 |
| `f2_tenseed_crew5` / framer / sharp_credibility | 10 | 0.400 ± 0.016 |
| `f2_tenseed_crew5` / framer / soft_credibility | 10 | 0.403 ± 0.015 |
| `f2_tenseed_crew5` / framer / trimmed | 10 | 0.531 ± 0.016 |
| `f2_tenseed_crew5` / lone_liar / dependence_aware | 10 | 0.194 ± 0.014 |
| `f2_tenseed_crew5` / lone_liar / hypothesis | 10 | 0.191 ± 0.012 |
| `f2_tenseed_crew5` / lone_liar / mean | 10 | 0.131 ± 0.011 |
| `f2_tenseed_crew5` / lone_liar / median | 10 | 0.430 ± 0.018 |
| `f2_tenseed_crew5` / lone_liar / sharp_credibility | 10 | 0.278 ± 0.014 |
| `f2_tenseed_crew5` / lone_liar / soft_credibility | 10 | 0.159 ± 0.012 |
| `f2_tenseed_crew5` / lone_liar / trimmed | 10 | 0.359 ± 0.017 |
| `f2_tenseed_crew5` / truthful / dependence_aware | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew5` / truthful / hypothesis | 10 | 0.370 ± 0.008 |
| `f2_tenseed_crew5` / truthful / mean | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew5` / truthful / median | 10 | 0.389 ± 0.018 |
| `f2_tenseed_crew5` / truthful / sharp_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew5` / truthful / soft_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew5` / truthful / trimmed | 10 | 0.312 ± 0.017 |

[f2_tenseed_crew5_rules.pdf](f2_tenseed_crew5_rules.pdf) — Scripted coalition and crew-rule evaluation (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Scripted rule evaluation: f2_tenseed_crew7



| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `f2_tenseed_crew7` / alibi / dependence_aware | 10 | 0.340 ± 0.016 |
| `f2_tenseed_crew7` / alibi / hypothesis | 10 | 0.065 ± 0.010 |
| `f2_tenseed_crew7` / alibi / mean | 10 | 0.212 ± 0.016 |
| `f2_tenseed_crew7` / alibi / median | 10 | 0.557 ± 0.016 |
| `f2_tenseed_crew7` / alibi / sharp_credibility | 10 | 0.451 ± 0.013 |
| `f2_tenseed_crew7` / alibi / soft_credibility | 10 | 0.378 ± 0.014 |
| `f2_tenseed_crew7` / alibi / trimmed | 10 | 0.486 ± 0.021 |
| `f2_tenseed_crew7` / framer / dependence_aware | 10 | 0.355 ± 0.012 |
| `f2_tenseed_crew7` / framer / hypothesis | 10 | 0.088 ± 0.011 |
| `f2_tenseed_crew7` / framer / mean | 10 | 0.260 ± 0.010 |
| `f2_tenseed_crew7` / framer / median | 10 | 0.522 ± 0.016 |
| `f2_tenseed_crew7` / framer / sharp_credibility | 10 | 0.314 ± 0.014 |
| `f2_tenseed_crew7` / framer / soft_credibility | 10 | 0.255 ± 0.010 |
| `f2_tenseed_crew7` / framer / trimmed | 10 | 0.535 ± 0.018 |
| `f2_tenseed_crew7` / lone_liar / dependence_aware | 10 | 0.178 ± 0.012 |
| `f2_tenseed_crew7` / lone_liar / hypothesis | 10 | 0.070 ± 0.007 |
| `f2_tenseed_crew7` / lone_liar / mean | 10 | 0.100 ± 0.009 |
| `f2_tenseed_crew7` / lone_liar / median | 10 | 0.553 ± 0.017 |
| `f2_tenseed_crew7` / lone_liar / sharp_credibility | 10 | 0.300 ± 0.009 |
| `f2_tenseed_crew7` / lone_liar / soft_credibility | 10 | 0.151 ± 0.009 |
| `f2_tenseed_crew7` / lone_liar / trimmed | 10 | 0.417 ± 0.016 |
| `f2_tenseed_crew7` / truthful / dependence_aware | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew7` / truthful / hypothesis | 10 | 0.393 ± 0.013 |
| `f2_tenseed_crew7` / truthful / mean | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew7` / truthful / median | 10 | 0.504 ± 0.018 |
| `f2_tenseed_crew7` / truthful / sharp_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew7` / truthful / soft_credibility | 10 | 0.000 ± 0.000 |
| `f2_tenseed_crew7` / truthful / trimmed | 10 | 0.380 ± 0.017 |

[f2_tenseed_crew7_rules.pdf](f2_tenseed_crew7_rules.pdf) — Scripted coalition and crew-rule evaluation (7+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f3_tenseed_crew3

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Each game contains one meeting.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `f3_tenseed_crew3` / coalition0_vs_soft_credibility | 10 | 0.188 ± 0.045 | 0.816 ± 0.067 |
| `f3_tenseed_crew3` / coalition0_vs_learned_crew1 | 10 | 0.463 ± 0.073 | 0.791 ± 0.081 |
| `f3_tenseed_crew3` / coalition1_vs_learned_crew1 | 10 | 0.415 ± 0.089 | 0.750 ± 0.095 |
| `f3_tenseed_crew3` / coalition1_vs_soft_credibility | 10 | 0.068 ± 0.034 | 0.492 ± 0.153 |

[f3_tenseed_crew3_crossplay.pdf](f3_tenseed_crew3_crossplay.pdf) — Matched three-stage crossplay (3+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f3_tenseed_crew5

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Each game contains one meeting.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `f3_tenseed_crew5` / coalition0_vs_soft_credibility | 10 | 0.194 ± 0.051 | 0.244 ± 0.056 |
| `f3_tenseed_crew5` / coalition0_vs_learned_crew1 | 10 | 0.403 ± 0.040 | 0.727 ± 0.057 |
| `f3_tenseed_crew5` / coalition1_vs_learned_crew1 | 10 | 0.538 ± 0.053 | 0.754 ± 0.063 |
| `f3_tenseed_crew5` / coalition1_vs_soft_credibility | 10 | 0.076 ± 0.014 | 0.105 ± 0.019 |

[f3_tenseed_crew5_crossplay.pdf](f3_tenseed_crew5_crossplay.pdf) — Matched three-stage crossplay (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f3_tenseed_crew7

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Each game contains one meeting.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `f3_tenseed_crew7` / coalition0_vs_soft_credibility | 10 | 0.211 ± 0.022 | 0.252 ± 0.032 |
| `f3_tenseed_crew7` / coalition0_vs_learned_crew1 | 10 | 0.475 ± 0.037 | 0.824 ± 0.033 |
| `f3_tenseed_crew7` / coalition1_vs_learned_crew1 | 10 | 0.587 ± 0.040 | 0.832 ± 0.027 |
| `f3_tenseed_crew7` / coalition1_vs_soft_credibility | 10 | 0.083 ± 0.020 | 0.111 ± 0.022 |

[f3_tenseed_crew7_crossplay.pdf](f3_tenseed_crew7_crossplay.pdf) — Matched three-stage crossplay (7+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f4_tenseed_crew3

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Whole-game harm counts every completed meeting, including incident-free meetings. Terminal-meeting FE is a separate outcome.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `f4_tenseed_crew3` / coalition0_vs_soft_credibility | 10 | 0.562 ± 0.312 | 0.626 ± 0.410 | 0.997 ± 0.004 | 0.561 ± 0.312 |
| `f4_tenseed_crew3` / coalition0_vs_learned_crew1 | 10 | 0.286 ± 0.197 | 0.324 ± 0.251 | 0.282 ± 0.255 | 0.182 ± 0.168 |
| `f4_tenseed_crew3` / coalition1_vs_learned_crew1 | 10 | 0.521 ± 0.076 | 0.539 ± 0.093 | 0.941 ± 0.058 | 0.506 ± 0.070 |
| `f4_tenseed_crew3` / coalition1_vs_soft_credibility | 10 | 0.088 ± 0.014 | 0.088 ± 0.014 | 0.988 ± 0.021 | 0.088 ± 0.014 |

[f4_tenseed_crew3_crossplay.pdf](f4_tenseed_crew3_crossplay.pdf) — Matched three-stage crossplay (3+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f4_tenseed_crew5

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Whole-game harm counts every completed meeting, including incident-free meetings. Terminal-meeting FE is a separate outcome.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `f4_tenseed_crew5` / coalition0_vs_soft_credibility | 10 | 0.974 ± 0.012 | 2.906 ± 0.059 | 0.967 ± 0.012 | 0.964 ± 0.014 |
| `f4_tenseed_crew5` / coalition0_vs_learned_crew1 | 10 | 0.839 ± 0.043 | 1.634 ± 0.163 | 0.360 ± 0.106 | 0.354 ± 0.103 |
| `f4_tenseed_crew5` / coalition1_vs_learned_crew1 | 10 | 0.847 ± 0.019 | 1.209 ± 0.060 | 0.851 ± 0.026 | 0.508 ± 0.038 |
| `f4_tenseed_crew5` / coalition1_vs_soft_credibility | 10 | 0.289 ± 0.084 | 0.332 ± 0.117 | 0.218 ± 0.058 | 0.037 ± 0.023 |

[f4_tenseed_crew5_crossplay.pdf](f4_tenseed_crew5_crossplay.pdf) — Matched three-stage crossplay (5+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Matched three-stage cycle: f4_tenseed_crew7

C0 and C1 are the initial/adapted coalition; D0 is scripted soft credibility and D1 is the learned defender. Whole-game harm counts every completed meeting, including incident-free meetings. Terminal-meeting FE is a separate outcome.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `f4_tenseed_crew7` / coalition0_vs_soft_credibility | 10 | 0.978 ± 0.009 | 4.862 ± 0.042 | 0.963 ± 0.007 | 0.970 ± 0.007 |
| `f4_tenseed_crew7` / coalition0_vs_learned_crew1 | 10 | 0.947 ± 0.006 | 2.996 ± 0.095 | 0.304 ± 0.026 | 0.367 ± 0.026 |
| `f4_tenseed_crew7` / coalition1_vs_learned_crew1 | 10 | 0.947 ± 0.005 | 1.989 ± 0.053 | 0.814 ± 0.016 | 0.537 ± 0.029 |
| `f4_tenseed_crew7` / coalition1_vs_soft_credibility | 10 | 0.409 ± 0.054 | 0.558 ± 0.112 | 0.063 ± 0.026 | 0.015 ± 0.011 |

[f4_tenseed_crew7_crossplay.pdf](f4_tenseed_crew7_crossplay.pdf) — Matched three-stage crossplay (7+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Adaptive-defense crossplay: counterattack_hyp_n7_r1

Cell names give training defense / evaluation defense; effects are defined separately with explicit signs.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `counterattack_hyp_n7_r1` / hypothesis / hypothesis | 10 | 0.422 ± 0.066 | 0.693 ± 0.058 |
| `counterattack_hyp_n7_r1` / hypothesis / mean | 10 | 0.096 ± 0.063 | 0.129 ± 0.080 |
| `counterattack_hyp_n7_r1` / hypothesis / soft | 10 | 0.098 ± 0.068 | 0.132 ± 0.087 |
| `counterattack_hyp_n7_r1` / mean / hypothesis | 10 | 0.077 ± 0.031 | 0.259 ± 0.053 |
| `counterattack_hyp_n7_r1` / mean / mean | 10 | 0.104 ± 0.037 | 0.171 ± 0.040 |
| `counterattack_hyp_n7_r1` / mean / soft | 10 | 0.148 ± 0.040 | 0.189 ± 0.046 |
| `counterattack_hyp_n7_r1` / soft / hypothesis | 10 | 0.126 ± 0.058 | 0.330 ± 0.089 |
| `counterattack_hyp_n7_r1` / soft / mean | 10 | 0.143 ± 0.053 | 0.217 ± 0.053 |
| `counterattack_hyp_n7_r1` / soft / soft | 10 | 0.194 ± 0.055 | 0.248 ± 0.065 |

[counterattack_hyp_n7_r1_crossplay.pdf](counterattack_hyp_n7_r1_crossplay.pdf) — Coalition training-defense × evaluation-defense crossplay (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Adaptive-defense crossplay: counterattack_hyp_n9_r1

Cell names give training defense / evaluation defense; effects are defined separately with explicit signs.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `counterattack_hyp_n9_r1` / hypothesis / hypothesis | 5 | 0.604 ± 0.018 | 0.863 ± 0.013 |
| `counterattack_hyp_n9_r1` / hypothesis / mean | 5 | 0.037 ± 0.007 | 0.068 ± 0.013 |
| `counterattack_hyp_n9_r1` / hypothesis / soft | 5 | 0.030 ± 0.007 | 0.074 ± 0.014 |
| `counterattack_hyp_n9_r1` / mean / hypothesis | 5 | 0.124 ± 0.074 | 0.332 ± 0.127 |
| `counterattack_hyp_n9_r1` / mean / mean | 5 | 0.139 ± 0.051 | 0.182 ± 0.053 |
| `counterattack_hyp_n9_r1` / mean / soft | 5 | 0.189 ± 0.056 | 0.230 ± 0.059 |
| `counterattack_hyp_n9_r1` / soft / hypothesis | 5 | 0.110 ± 0.033 | 0.304 ± 0.032 |
| `counterattack_hyp_n9_r1` / soft / mean | 5 | 0.136 ± 0.017 | 0.178 ± 0.014 |
| `counterattack_hyp_n9_r1` / soft / soft | 5 | 0.202 ± 0.013 | 0.251 ± 0.030 |

[counterattack_hyp_n9_r1_crossplay.pdf](counterattack_hyp_n9_r1_crossplay.pdf) — Coalition training-defense × evaluation-defense crossplay (7+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Adaptive-defense crossplay: counterattack_n5_r1

Cell names give training defense / evaluation defense; effects are defined separately with explicit signs.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `counterattack_n5_r1` / both / both | 5 | 0.872 ± 0.010 | 0.946 ± 0.022 |
| `counterattack_n5_r1` / both / rule | 5 | 0.039 ± 0.017 | 0.797 ± 0.014 |
| `counterattack_n5_r1` / both / soft | 5 | 0.025 ± 0.006 | 0.795 ± 0.016 |
| `counterattack_n5_r1` / rule / both | 5 | 0.439 ± 0.016 | 0.836 ± 0.020 |
| `counterattack_n5_r1` / rule / rule | 5 | 0.226 ± 0.036 | 0.841 ± 0.018 |
| `counterattack_n5_r1` / rule / soft | 5 | 0.194 ± 0.030 | 0.836 ± 0.017 |
| `counterattack_n5_r1` / soft / both | 5 | 0.413 ± 0.040 | 0.822 ± 0.008 |
| `counterattack_n5_r1` / soft / rule | 5 | 0.206 ± 0.038 | 0.838 ± 0.034 |
| `counterattack_n5_r1` / soft / soft | 5 | 0.210 ± 0.048 | 0.839 ± 0.036 |

[counterattack_n5_r1_crossplay.pdf](counterattack_n5_r1_crossplay.pdf) — Coalition training-defense × evaluation-defense crossplay (3+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Adaptive-defense crossplay: counterattack_n7_r1

Cell names give training defense / evaluation defense; effects are defined separately with explicit signs.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `counterattack_n7_r1` / both / both | 10 | 0.190 ± 0.040 | 0.226 ± 0.049 |
| `counterattack_n7_r1` / both / rule | 10 | 0.164 ± 0.042 | 0.228 ± 0.051 |
| `counterattack_n7_r1` / both / soft | 10 | 0.169 ± 0.038 | 0.217 ± 0.046 |
| `counterattack_n7_r1` / rule / both | 10 | 0.193 ± 0.034 | 0.228 ± 0.044 |
| `counterattack_n7_r1` / rule / rule | 10 | 0.170 ± 0.038 | 0.235 ± 0.049 |
| `counterattack_n7_r1` / rule / soft | 10 | 0.174 ± 0.031 | 0.223 ± 0.034 |
| `counterattack_n7_r1` / soft / both | 10 | 0.204 ± 0.052 | 0.235 ± 0.060 |
| `counterattack_n7_r1` / soft / rule | 10 | 0.182 ± 0.051 | 0.243 ± 0.060 |
| `counterattack_n7_r1` / soft / soft | 10 | 0.194 ± 0.055 | 0.248 ± 0.065 |

[counterattack_n7_r1_crossplay.pdf](counterattack_n7_r1_crossplay.pdf) — Coalition training-defense × evaluation-defense crossplay (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Generation stages: multigen_dependence_n7_r8_k4

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_dependence_n7_r8_k4` / C0 | 10 | 1.000 ± 0.000 | 3.498 ± 0.048 | 0.993 ± 0.004 | 0.996 ± 0.003 |
| `multigen_dependence_n7_r8_k4` / D1 | 10 | 0.877 ± 0.048 | 2.081 ± 0.245 | 0.253 ± 0.071 | 0.253 ± 0.071 |
| `multigen_dependence_n7_r8_k4` / C1 | 10 | 0.905 ± 0.017 | 1.401 ± 0.055 | 0.864 ± 0.029 | 0.543 ± 0.040 |
| `multigen_dependence_n7_r8_k4` / D2 | 10 | 0.719 ± 0.114 | 1.138 ± 0.368 | 0.408 ± 0.147 | 0.294 ± 0.111 |
| `multigen_dependence_n7_r8_k4` / C2 | 10 | 0.885 ± 0.018 | 1.448 ± 0.177 | 0.778 ± 0.048 | 0.523 ± 0.058 |
| `multigen_dependence_n7_r8_k4` / D3 | 10 | 0.719 ± 0.112 | 1.162 ± 0.331 | 0.372 ± 0.141 | 0.271 ± 0.111 |
| `multigen_dependence_n7_r8_k4` / C3 | 10 | 0.879 ± 0.025 | 1.474 ± 0.142 | 0.753 ± 0.060 | 0.521 ± 0.067 |
| `multigen_dependence_n7_r8_k4` / D4 | 10 | 0.736 ± 0.067 | 1.163 ± 0.241 | 0.359 ± 0.098 | 0.281 ± 0.071 |
| `multigen_dependence_n7_r8_k4` / C4 | 10 | 0.890 ± 0.020 | 1.528 ± 0.204 | 0.760 ± 0.091 | 0.528 ± 0.082 |

[multigen_dependence_n7_r8_k4_stages.pdf](multigen_dependence_n7_r8_k4_stages.pdf) — Stage evaluations, 5+2 players, up to 8 rounds. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_dependence_n7_r8_k4_matrix.pdf](multigen_dependence_n7_r8_k4_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_dependence_n7_r8_k4

**At least one false ejection / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 1.000 ± 0.000 | 0.872 ± 0.047 | 0.887 ± 0.048 | 0.837 ± 0.071 | 0.869 ± 0.050 |
| C1 | 0.269 ± 0.058 | 0.905 ± 0.020 | 0.727 ± 0.119 | 0.803 ± 0.043 | 0.800 ± 0.047 |
| C2 | 0.200 ± 0.031 | 0.863 ± 0.041 | 0.893 ± 0.016 | 0.715 ± 0.118 | 0.808 ± 0.085 |
| C3 | 0.210 ± 0.050 | 0.893 ± 0.051 | 0.760 ± 0.105 | 0.881 ± 0.023 | 0.740 ± 0.069 |
| C4 | 0.248 ± 0.076 | 0.879 ± 0.040 | 0.822 ± 0.091 | 0.735 ± 0.115 | 0.891 ± 0.028 |

**Mean false ejections / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 3.502 ± 0.051 | 2.063 ± 0.243 | 2.146 ± 0.314 | 1.899 ± 0.399 | 2.085 ± 0.318 |
| C1 | 0.349 ± 0.072 | 1.398 ± 0.061 | 1.151 ± 0.378 | 1.295 ± 0.126 | 1.328 ± 0.239 |
| C2 | 0.259 ± 0.041 | 1.327 ± 0.088 | 1.463 ± 0.170 | 1.163 ± 0.337 | 1.312 ± 0.238 |
| C3 | 0.270 ± 0.064 | 1.376 ± 0.127 | 1.246 ± 0.377 | 1.484 ± 0.135 | 1.175 ± 0.249 |
| C4 | 0.317 ± 0.108 | 1.351 ± 0.099 | 1.299 ± 0.272 | 1.197 ± 0.298 | 1.531 ± 0.197 |

**Coalition game wins**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.993 ± 0.004 | 0.242 ± 0.070 | 0.282 ± 0.105 | 0.228 ± 0.120 | 0.269 ± 0.100 |
| C1 | 0.227 ± 0.059 | 0.863 ± 0.035 | 0.412 ± 0.154 | 0.569 ± 0.155 | 0.492 ± 0.083 |
| C2 | 0.173 ± 0.025 | 0.763 ± 0.111 | 0.785 ± 0.052 | 0.369 ± 0.146 | 0.596 ± 0.187 |
| C3 | 0.182 ± 0.034 | 0.837 ± 0.070 | 0.488 ± 0.161 | 0.758 ± 0.062 | 0.362 ± 0.096 |
| C4 | 0.205 ± 0.048 | 0.804 ± 0.082 | 0.618 ± 0.162 | 0.433 ± 0.205 | 0.761 ± 0.091 |

**Terminal-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.996 ± 0.004 | 0.242 ± 0.070 | 0.282 ± 0.105 | 0.228 ± 0.120 | 0.269 ± 0.100 |
| C1 | 0.085 ± 0.021 | 0.542 ± 0.041 | 0.296 ± 0.115 | 0.391 ± 0.107 | 0.366 ± 0.063 |
| C2 | 0.070 ± 0.020 | 0.484 ± 0.063 | 0.528 ± 0.063 | 0.273 ± 0.113 | 0.407 ± 0.117 |
| C3 | 0.068 ± 0.017 | 0.523 ± 0.091 | 0.350 ± 0.106 | 0.530 ± 0.070 | 0.282 ± 0.070 |
| C4 | 0.072 ± 0.026 | 0.514 ± 0.061 | 0.394 ± 0.108 | 0.307 ± 0.138 | 0.531 ± 0.086 |

## Loop descriptors: multigen_dependence_n7_r8_k4



| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_dependence_n7_r8_k4` / c_stage_mean | 10 | 0.890 ± 0.007 | 1.463 ± 0.081 | 0.789 ± 0.026 | 0.529 ± 0.028 |
| `multigen_dependence_n7_r8_k4` / d_stage_mean | 10 | 0.763 ± 0.047 | 1.386 ± 0.158 | 0.348 ± 0.054 | 0.275 ± 0.050 |
| `multigen_dependence_n7_r8_k4` / late_minus_early | 10 | -0.011 ± 0.018 | 0.077 ± 0.139 | -0.064 ± 0.047 | -0.009 ± 0.046 |
| `multigen_dependence_n7_r8_k4` / matrix_adaptation_gain | 10 | 0.129 ± 0.046 | 0.081 ± 0.099 | 0.445 ± 0.070 | 0.260 ± 0.044 |
| `multigen_dependence_n7_r8_k4` / parity_gap | 10 | -0.008 ± 0.040 | -0.019 ± 0.116 | 0.007 ± 0.112 | -0.011 ± 0.051 |

## Finite observed-policy response gaps: multigen_dependence_n7_r8_k4

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_dependence_n7_r8_k4` / 0 | 10 | 0.210 ± 0.050 | 0.846 ± 0.047 |
| `multigen_dependence_n7_r8_k4` / 1 | 10 | 0.649 ± 0.065 | 0.658 ± 0.071 |
| `multigen_dependence_n7_r8_k4` / 2 | 10 | 0.708 ± 0.038 | 0.615 ± 0.061 |
| `multigen_dependence_n7_r8_k4` / 3 | 10 | 0.674 ± 0.050 | 0.576 ± 0.070 |
| `multigen_dependence_n7_r8_k4` / 4 | 10 | 0.651 ± 0.069 | 0.556 ± 0.094 |

[multigen_dependence_n7_r8_k4_response_gaps.pdf](multigen_dependence_n7_r8_k4_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_none_n5_r8_k10

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n5_r8_k10` / C0 | 10 | 0.567 ± 0.312 | 0.630 ± 0.407 | 0.997 ± 0.003 | 0.567 ± 0.311 |
| `multigen_none_n5_r8_k10` / D1 | 10 | 0.291 ± 0.200 | 0.330 ± 0.257 | 0.279 ± 0.252 | 0.178 ± 0.164 |
| `multigen_none_n5_r8_k10` / C1 | 10 | 0.508 ± 0.072 | 0.526 ± 0.090 | 0.943 ± 0.051 | 0.493 ± 0.063 |
| `multigen_none_n5_r8_k10` / D2 | 10 | 0.222 ± 0.084 | 0.228 ± 0.089 | 0.342 ± 0.158 | 0.152 ± 0.070 |
| `multigen_none_n5_r8_k10` / C2 | 10 | 0.528 ± 0.161 | 0.558 ± 0.213 | 0.894 ± 0.052 | 0.491 ± 0.131 |
| `multigen_none_n5_r8_k10` / D3 | 10 | 0.285 ± 0.193 | 0.311 ± 0.224 | 0.305 ± 0.206 | 0.193 ± 0.162 |
| `multigen_none_n5_r8_k10` / C3 | 10 | 0.594 ± 0.117 | 0.652 ± 0.217 | 0.905 ± 0.058 | 0.566 ± 0.115 |
| `multigen_none_n5_r8_k10` / D4 | 10 | 0.198 ± 0.075 | 0.202 ± 0.074 | 0.326 ± 0.174 | 0.151 ± 0.090 |
| `multigen_none_n5_r8_k10` / C4 | 10 | 0.462 ± 0.111 | 0.480 ± 0.150 | 0.925 ± 0.068 | 0.443 ± 0.081 |
| `multigen_none_n5_r8_k10` / D5 | 10 | 0.175 ± 0.128 | 0.185 ± 0.142 | 0.168 ± 0.149 | 0.088 ± 0.101 |
| `multigen_none_n5_r8_k10` / C5 | 10 | 0.603 ± 0.144 | 0.670 ± 0.205 | 0.888 ± 0.049 | 0.551 ± 0.112 |
| `multigen_none_n5_r8_k10` / D6 | 10 | 0.269 ± 0.102 | 0.275 ± 0.106 | 0.437 ± 0.170 | 0.204 ± 0.093 |
| `multigen_none_n5_r8_k10` / C6 | 10 | 0.412 ± 0.096 | 0.421 ± 0.105 | 0.933 ± 0.047 | 0.402 ± 0.084 |
| `multigen_none_n5_r8_k10` / D7 | 10 | 0.214 ± 0.103 | 0.226 ± 0.112 | 0.263 ± 0.134 | 0.132 ± 0.086 |
| `multigen_none_n5_r8_k10` / C7 | 10 | 0.533 ± 0.162 | 0.571 ± 0.242 | 0.942 ± 0.036 | 0.519 ± 0.143 |
| `multigen_none_n5_r8_k10` / D8 | 10 | 0.305 ± 0.175 | 0.318 ± 0.187 | 0.471 ± 0.228 | 0.259 ± 0.183 |
| `multigen_none_n5_r8_k10` / C8 | 10 | 0.491 ± 0.146 | 0.514 ± 0.195 | 0.940 ± 0.049 | 0.472 ± 0.127 |
| `multigen_none_n5_r8_k10` / D9 | 10 | 0.262 ± 0.143 | 0.284 ± 0.179 | 0.399 ± 0.220 | 0.210 ± 0.147 |
| `multigen_none_n5_r8_k10` / C9 | 10 | 0.500 ± 0.114 | 0.525 ± 0.141 | 0.898 ± 0.073 | 0.465 ± 0.087 |
| `multigen_none_n5_r8_k10` / D10 | 10 | 0.259 ± 0.102 | 0.265 ± 0.104 | 0.426 ± 0.203 | 0.208 ± 0.123 |
| `multigen_none_n5_r8_k10` / C10 | 10 | 0.457 ± 0.078 | 0.475 ± 0.091 | 0.910 ± 0.083 | 0.436 ± 0.067 |

[multigen_none_n5_r8_k10_stages.pdf](multigen_none_n5_r8_k10_stages.pdf) — Stage evaluations, 3+2 players, up to 8 rounds. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_none_n5_r8_k10_matrix.pdf](multigen_none_n5_r8_k10_matrix.pdf) — Cross-generation evaluation matrix (3+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_none_n5_r8_k10

**At least one false ejection / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.561 ± 0.315 | 0.279 ± 0.194 | 0.463 ± 0.157 | 0.422 ± 0.253 | 0.454 ± 0.203 | 0.314 ± 0.220 | 0.454 ± 0.226 | 0.338 ± 0.174 | 0.560 ± 0.158 | 0.467 ± 0.189 | 0.437 ± 0.183 |
| C1 | 0.098 ± 0.023 | 0.499 ± 0.063 | 0.224 ± 0.094 | 0.373 ± 0.172 | 0.320 ± 0.160 | 0.431 ± 0.124 | 0.297 ± 0.119 | 0.384 ± 0.124 | 0.388 ± 0.125 | 0.354 ± 0.122 | 0.343 ± 0.053 |
| C2 | 0.176 ± 0.290 | 0.384 ± 0.200 | 0.520 ± 0.168 | 0.284 ± 0.189 | 0.422 ± 0.037 | 0.318 ± 0.223 | 0.400 ± 0.164 | 0.298 ± 0.156 | 0.400 ± 0.179 | 0.397 ± 0.114 | 0.332 ± 0.076 |
| C3 | 0.096 ± 0.041 | 0.435 ± 0.088 | 0.283 ± 0.079 | 0.588 ± 0.120 | 0.201 ± 0.074 | 0.423 ± 0.136 | 0.332 ± 0.143 | 0.407 ± 0.149 | 0.406 ± 0.169 | 0.375 ± 0.121 | 0.356 ± 0.071 |
| C4 | 0.096 ± 0.017 | 0.349 ± 0.139 | 0.345 ± 0.083 | 0.330 ± 0.215 | 0.463 ± 0.109 | 0.175 ± 0.125 | 0.352 ± 0.089 | 0.280 ± 0.120 | 0.386 ± 0.129 | 0.321 ± 0.146 | 0.377 ± 0.099 |
| C5 | 0.086 ± 0.020 | 0.436 ± 0.056 | 0.288 ± 0.088 | 0.441 ± 0.167 | 0.250 ± 0.082 | 0.598 ± 0.145 | 0.267 ± 0.100 | 0.431 ± 0.159 | 0.330 ± 0.108 | 0.382 ± 0.123 | 0.350 ± 0.096 |
| C6 | 0.084 ± 0.031 | 0.366 ± 0.143 | 0.357 ± 0.122 | 0.313 ± 0.177 | 0.376 ± 0.080 | 0.264 ± 0.208 | 0.403 ± 0.096 | 0.213 ± 0.108 | 0.430 ± 0.080 | 0.340 ± 0.166 | 0.361 ± 0.114 |
| C7 | 0.082 ± 0.022 | 0.457 ± 0.070 | 0.294 ± 0.089 | 0.470 ± 0.161 | 0.283 ± 0.131 | 0.391 ± 0.166 | 0.349 ± 0.177 | 0.533 ± 0.157 | 0.306 ± 0.170 | 0.386 ± 0.083 | 0.362 ± 0.097 |
| C8 | 0.091 ± 0.021 | 0.419 ± 0.130 | 0.320 ± 0.097 | 0.363 ± 0.167 | 0.316 ± 0.099 | 0.322 ± 0.188 | 0.369 ± 0.147 | 0.312 ± 0.164 | 0.488 ± 0.146 | 0.264 ± 0.143 | 0.381 ± 0.110 |
| C9 | 0.100 ± 0.012 | 0.397 ± 0.089 | 0.392 ± 0.077 | 0.388 ± 0.166 | 0.320 ± 0.135 | 0.349 ± 0.147 | 0.338 ± 0.147 | 0.351 ± 0.113 | 0.377 ± 0.162 | 0.489 ± 0.121 | 0.266 ± 0.097 |
| C10 | 0.103 ± 0.047 | 0.371 ± 0.146 | 0.334 ± 0.116 | 0.357 ± 0.183 | 0.341 ± 0.056 | 0.321 ± 0.183 | 0.378 ± 0.114 | 0.307 ± 0.096 | 0.429 ± 0.101 | 0.342 ± 0.160 | 0.450 ± 0.072 |

**Mean false ejections / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.627 ± 0.417 | 0.312 ± 0.245 | 0.514 ± 0.190 | 0.492 ± 0.311 | 0.480 ± 0.210 | 0.348 ± 0.268 | 0.497 ± 0.267 | 0.371 ± 0.215 | 0.645 ± 0.218 | 0.530 ± 0.236 | 0.491 ± 0.230 |
| C1 | 0.098 ± 0.023 | 0.515 ± 0.082 | 0.231 ± 0.099 | 0.397 ± 0.196 | 0.328 ± 0.159 | 0.474 ± 0.171 | 0.304 ± 0.125 | 0.405 ± 0.140 | 0.410 ± 0.152 | 0.366 ± 0.131 | 0.360 ± 0.068 |
| C2 | 0.200 ± 0.366 | 0.410 ± 0.234 | 0.549 ± 0.219 | 0.311 ± 0.218 | 0.433 ± 0.046 | 0.341 ± 0.245 | 0.419 ± 0.199 | 0.316 ± 0.183 | 0.434 ± 0.255 | 0.435 ± 0.161 | 0.342 ± 0.079 |
| C3 | 0.096 ± 0.041 | 0.447 ± 0.087 | 0.296 ± 0.086 | 0.642 ± 0.206 | 0.205 ± 0.073 | 0.460 ± 0.170 | 0.340 ± 0.151 | 0.425 ± 0.175 | 0.427 ± 0.187 | 0.396 ± 0.138 | 0.367 ± 0.077 |
| C4 | 0.096 ± 0.017 | 0.353 ± 0.139 | 0.355 ± 0.088 | 0.352 ± 0.232 | 0.483 ± 0.151 | 0.185 ± 0.135 | 0.361 ± 0.095 | 0.299 ± 0.141 | 0.398 ± 0.142 | 0.344 ± 0.176 | 0.386 ± 0.098 |
| C5 | 0.086 ± 0.020 | 0.448 ± 0.056 | 0.301 ± 0.096 | 0.471 ± 0.178 | 0.254 ± 0.082 | 0.662 ± 0.206 | 0.272 ± 0.103 | 0.461 ± 0.198 | 0.340 ± 0.114 | 0.398 ± 0.130 | 0.366 ± 0.116 |
| C6 | 0.084 ± 0.031 | 0.372 ± 0.145 | 0.368 ± 0.128 | 0.332 ± 0.187 | 0.383 ± 0.084 | 0.298 ± 0.289 | 0.411 ± 0.105 | 0.224 ± 0.119 | 0.443 ± 0.079 | 0.373 ± 0.217 | 0.372 ± 0.118 |
| C7 | 0.082 ± 0.022 | 0.473 ± 0.074 | 0.305 ± 0.092 | 0.531 ± 0.234 | 0.287 ± 0.131 | 0.436 ± 0.231 | 0.364 ± 0.195 | 0.569 ± 0.232 | 0.319 ± 0.182 | 0.404 ± 0.090 | 0.379 ± 0.106 |
| C8 | 0.091 ± 0.021 | 0.437 ± 0.137 | 0.331 ± 0.106 | 0.391 ± 0.189 | 0.320 ± 0.097 | 0.355 ± 0.265 | 0.379 ± 0.161 | 0.321 ± 0.166 | 0.512 ± 0.199 | 0.284 ± 0.177 | 0.391 ± 0.118 |
| C9 | 0.100 ± 0.012 | 0.407 ± 0.089 | 0.413 ± 0.086 | 0.438 ± 0.224 | 0.326 ± 0.134 | 0.374 ± 0.164 | 0.349 ± 0.159 | 0.366 ± 0.133 | 0.394 ± 0.181 | 0.513 ± 0.148 | 0.271 ± 0.099 |
| C10 | 0.103 ± 0.047 | 0.379 ± 0.152 | 0.343 ± 0.119 | 0.382 ± 0.201 | 0.348 ± 0.057 | 0.356 ± 0.234 | 0.391 ± 0.125 | 0.317 ± 0.101 | 0.447 ± 0.114 | 0.361 ± 0.184 | 0.467 ± 0.089 |

**Coalition game wins**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.996 ± 0.005 | 0.273 ± 0.245 | 0.524 ± 0.155 | 0.420 ± 0.262 | 0.544 ± 0.229 | 0.257 ± 0.195 | 0.585 ± 0.281 | 0.374 ± 0.206 | 0.627 ± 0.167 | 0.517 ± 0.245 | 0.516 ± 0.246 |
| C1 | 0.989 ± 0.020 | 0.944 ± 0.050 | 0.345 ± 0.161 | 0.649 ± 0.266 | 0.541 ± 0.263 | 0.677 ± 0.168 | 0.619 ± 0.217 | 0.670 ± 0.129 | 0.672 ± 0.227 | 0.669 ± 0.222 | 0.637 ± 0.189 |
| C2 | 0.994 ± 0.007 | 0.613 ± 0.261 | 0.894 ± 0.051 | 0.305 ± 0.197 | 0.797 ± 0.239 | 0.336 ± 0.258 | 0.746 ± 0.211 | 0.477 ± 0.256 | 0.742 ± 0.189 | 0.624 ± 0.281 | 0.614 ± 0.306 |
| C3 | 0.988 ± 0.013 | 0.798 ± 0.137 | 0.463 ± 0.134 | 0.905 ± 0.060 | 0.335 ± 0.178 | 0.696 ± 0.113 | 0.616 ± 0.250 | 0.714 ± 0.187 | 0.656 ± 0.188 | 0.611 ± 0.155 | 0.642 ± 0.162 |
| C4 | 0.996 ± 0.008 | 0.666 ± 0.317 | 0.706 ± 0.275 | 0.451 ± 0.332 | 0.923 ± 0.067 | 0.167 ± 0.147 | 0.813 ± 0.165 | 0.379 ± 0.160 | 0.794 ± 0.189 | 0.543 ± 0.319 | 0.708 ± 0.281 |
| C5 | 0.997 ± 0.004 | 0.824 ± 0.133 | 0.535 ± 0.152 | 0.699 ± 0.226 | 0.503 ± 0.268 | 0.891 ± 0.049 | 0.431 ± 0.168 | 0.760 ± 0.139 | 0.635 ± 0.246 | 0.680 ± 0.120 | 0.663 ± 0.169 |
| C6 | 0.992 ± 0.013 | 0.652 ± 0.299 | 0.704 ± 0.232 | 0.506 ± 0.365 | 0.825 ± 0.144 | 0.321 ± 0.233 | 0.932 ± 0.048 | 0.263 ± 0.135 | 0.831 ± 0.168 | 0.556 ± 0.302 | 0.694 ± 0.283 |
| C7 | 0.985 ± 0.038 | 0.805 ± 0.146 | 0.523 ± 0.155 | 0.696 ± 0.139 | 0.499 ± 0.264 | 0.674 ± 0.186 | 0.578 ± 0.257 | 0.940 ± 0.039 | 0.469 ± 0.230 | 0.723 ± 0.166 | 0.635 ± 0.188 |
| C8 | 0.989 ± 0.018 | 0.747 ± 0.247 | 0.629 ± 0.259 | 0.596 ± 0.319 | 0.696 ± 0.279 | 0.462 ± 0.255 | 0.738 ± 0.220 | 0.462 ± 0.234 | 0.935 ± 0.058 | 0.400 ± 0.223 | 0.746 ± 0.228 |
| C9 | 0.993 ± 0.006 | 0.747 ± 0.210 | 0.653 ± 0.200 | 0.595 ± 0.194 | 0.620 ± 0.319 | 0.512 ± 0.291 | 0.658 ± 0.260 | 0.677 ± 0.233 | 0.652 ± 0.210 | 0.898 ± 0.073 | 0.430 ± 0.209 |
| C10 | 0.992 ± 0.014 | 0.712 ± 0.293 | 0.628 ± 0.279 | 0.509 ± 0.318 | 0.721 ± 0.235 | 0.457 ± 0.291 | 0.731 ± 0.202 | 0.527 ± 0.219 | 0.773 ± 0.214 | 0.558 ± 0.255 | 0.912 ± 0.078 |

**Terminal-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.560 ± 0.315 | 0.168 ± 0.152 | 0.351 ± 0.116 | 0.324 ± 0.237 | 0.357 ± 0.158 | 0.176 ± 0.142 | 0.370 ± 0.215 | 0.236 ± 0.141 | 0.463 ± 0.131 | 0.371 ± 0.176 | 0.329 ± 0.166 |
| C1 | 0.098 ± 0.023 | 0.484 ± 0.055 | 0.151 ± 0.074 | 0.341 ± 0.187 | 0.274 ± 0.176 | 0.347 ± 0.108 | 0.259 ± 0.128 | 0.336 ± 0.112 | 0.344 ± 0.133 | 0.312 ± 0.124 | 0.295 ± 0.055 |
| C2 | 0.176 ± 0.288 | 0.338 ± 0.179 | 0.486 ± 0.141 | 0.192 ± 0.155 | 0.382 ± 0.073 | 0.224 ± 0.206 | 0.357 ± 0.130 | 0.221 ± 0.127 | 0.359 ± 0.142 | 0.319 ± 0.109 | 0.276 ± 0.114 |
| C3 | 0.096 ± 0.040 | 0.412 ± 0.101 | 0.223 ± 0.073 | 0.558 ± 0.115 | 0.156 ± 0.091 | 0.356 ± 0.102 | 0.278 ± 0.152 | 0.355 ± 0.134 | 0.363 ± 0.166 | 0.317 ± 0.099 | 0.306 ± 0.074 |
| C4 | 0.096 ± 0.017 | 0.325 ± 0.164 | 0.302 ± 0.100 | 0.276 ± 0.233 | 0.443 ± 0.081 | 0.086 ± 0.093 | 0.330 ± 0.085 | 0.184 ± 0.092 | 0.365 ± 0.126 | 0.268 ± 0.155 | 0.334 ± 0.126 |
| C5 | 0.086 ± 0.020 | 0.411 ± 0.067 | 0.241 ± 0.086 | 0.396 ± 0.174 | 0.204 ± 0.103 | 0.545 ± 0.117 | 0.200 ± 0.092 | 0.378 ± 0.132 | 0.295 ± 0.124 | 0.335 ± 0.113 | 0.303 ± 0.066 |
| C6 | 0.084 ± 0.031 | 0.338 ± 0.167 | 0.311 ± 0.126 | 0.260 ± 0.199 | 0.352 ± 0.088 | 0.165 ± 0.163 | 0.391 ± 0.085 | 0.130 ± 0.088 | 0.404 ± 0.096 | 0.286 ± 0.161 | 0.311 ± 0.134 |
| C7 | 0.082 ± 0.022 | 0.429 ± 0.087 | 0.238 ± 0.087 | 0.415 ± 0.141 | 0.241 ± 0.149 | 0.326 ± 0.120 | 0.289 ± 0.178 | 0.518 ± 0.136 | 0.260 ± 0.179 | 0.346 ± 0.089 | 0.319 ± 0.094 |
| C8 | 0.091 ± 0.020 | 0.389 ± 0.148 | 0.275 ± 0.098 | 0.314 ± 0.182 | 0.288 ± 0.125 | 0.237 ± 0.160 | 0.329 ± 0.146 | 0.242 ± 0.172 | 0.472 ± 0.132 | 0.210 ± 0.146 | 0.338 ± 0.113 |
| C9 | 0.100 ± 0.012 | 0.372 ± 0.105 | 0.328 ± 0.089 | 0.345 ± 0.171 | 0.285 ± 0.158 | 0.258 ± 0.163 | 0.296 ± 0.148 | 0.296 ± 0.099 | 0.344 ± 0.158 | 0.453 ± 0.094 | 0.214 ± 0.125 |
| C10 | 0.102 ± 0.047 | 0.342 ± 0.163 | 0.293 ± 0.125 | 0.303 ± 0.206 | 0.305 ± 0.081 | 0.233 ± 0.175 | 0.323 ± 0.106 | 0.244 ± 0.106 | 0.396 ± 0.108 | 0.287 ± 0.162 | 0.430 ± 0.060 |

## Loop descriptors: multigen_none_n5_r8_k10



| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n5_r8_k10` / c_stage_mean | 10 | 0.509 ± 0.031 | 0.539 ± 0.044 | 0.918 ± 0.021 | 0.484 ± 0.025 |
| `multigen_none_n5_r8_k10` / d_stage_mean | 10 | 0.248 ± 0.051 | 0.262 ± 0.060 | 0.341 ± 0.059 | 0.177 ± 0.044 |
| `multigen_none_n5_r8_k10` / late_minus_early | 10 | -0.061 ± 0.102 | -0.075 ± 0.133 | 0.002 ± 0.042 | -0.059 ± 0.096 |
| `multigen_none_n5_r8_k10` / matrix_adaptation_gain | 10 | 0.255 ± 0.062 | 0.271 ± 0.079 | 0.576 ± 0.050 | 0.301 ± 0.053 |
| `multigen_none_n5_r8_k10` / parity_gap | 10 | 0.048 ± 0.066 | 0.050 ± 0.070 | 0.133 ± 0.163 | 0.057 ± 0.077 |

## Finite observed-policy response gaps: multigen_none_n5_r8_k10

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_none_n5_r8_k10` / 0 | 10 | 0.453 ± 0.315 | 0.887 ± 0.107 |
| `multigen_none_n5_r8_k10` / 1 | 10 | 0.489 ± 0.102 | 0.665 ± 0.173 |
| `multigen_none_n5_r8_k10` / 2 | 10 | 0.475 ± 0.119 | 0.697 ± 0.188 |
| `multigen_none_n5_r8_k10` / 3 | 10 | 0.569 ± 0.124 | 0.632 ± 0.107 |
| `multigen_none_n5_r8_k10` / 4 | 10 | 0.483 ± 0.138 | 0.808 ± 0.163 |
| `multigen_none_n5_r8_k10` / 5 | 10 | 0.571 ± 0.134 | 0.607 ± 0.157 |
| `multigen_none_n5_r8_k10` / 6 | 10 | 0.477 ± 0.193 | 0.783 ± 0.136 |
| `multigen_none_n5_r8_k10` / 7 | 10 | 0.535 ± 0.134 | 0.598 ± 0.189 |
| `multigen_none_n5_r8_k10` / 8 | 10 | 0.560 ± 0.149 | 0.720 ± 0.166 |
| `multigen_none_n5_r8_k10` / 9 | 10 | 0.506 ± 0.082 | 0.648 ± 0.188 |
| `multigen_none_n5_r8_k10` / 10 | 10 | 0.428 ± 0.149 | 0.676 ± 0.214 |

[multigen_none_n5_r8_k10_response_gaps.pdf](multigen_none_n5_r8_k10_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_none_n7_r1_k10

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_none_n7_r1_k10` / C0 | 10 | 0.194 ± 0.055 | 0.248 ± 0.065 |
| `multigen_none_n7_r1_k10` / D1 | 10 | 0.298 ± 0.088 | 0.527 ± 0.115 |
| `multigen_none_n7_r1_k10` / C1 | 10 | 0.564 ± 0.057 | 0.751 ± 0.038 |
| `multigen_none_n7_r1_k10` / D2 | 10 | 0.217 ± 0.054 | 0.446 ± 0.093 |
| `multigen_none_n7_r1_k10` / C2 | 10 | 0.547 ± 0.063 | 0.741 ± 0.023 |
| `multigen_none_n7_r1_k10` / D3 | 10 | 0.255 ± 0.065 | 0.520 ± 0.105 |
| `multigen_none_n7_r1_k10` / C3 | 10 | 0.579 ± 0.058 | 0.752 ± 0.029 |
| `multigen_none_n7_r1_k10` / D4 | 10 | 0.264 ± 0.056 | 0.514 ± 0.069 |
| `multigen_none_n7_r1_k10` / C4 | 10 | 0.546 ± 0.030 | 0.738 ± 0.027 |
| `multigen_none_n7_r1_k10` / D5 | 10 | 0.272 ± 0.076 | 0.513 ± 0.122 |
| `multigen_none_n7_r1_k10` / C5 | 10 | 0.552 ± 0.041 | 0.736 ± 0.026 |
| `multigen_none_n7_r1_k10` / D6 | 10 | 0.256 ± 0.037 | 0.494 ± 0.065 |
| `multigen_none_n7_r1_k10` / C6 | 10 | 0.579 ± 0.041 | 0.762 ± 0.040 |
| `multigen_none_n7_r1_k10` / D7 | 10 | 0.270 ± 0.062 | 0.537 ± 0.109 |
| `multigen_none_n7_r1_k10` / C7 | 10 | 0.555 ± 0.030 | 0.738 ± 0.019 |
| `multigen_none_n7_r1_k10` / D8 | 10 | 0.259 ± 0.030 | 0.499 ± 0.043 |
| `multigen_none_n7_r1_k10` / C8 | 10 | 0.554 ± 0.049 | 0.752 ± 0.034 |
| `multigen_none_n7_r1_k10` / D9 | 10 | 0.223 ± 0.082 | 0.434 ± 0.103 |
| `multigen_none_n7_r1_k10` / C9 | 10 | 0.564 ± 0.062 | 0.754 ± 0.031 |
| `multigen_none_n7_r1_k10` / D10 | 10 | 0.255 ± 0.087 | 0.484 ± 0.128 |
| `multigen_none_n7_r1_k10` / C10 | 10 | 0.534 ± 0.039 | 0.718 ± 0.041 |

[multigen_none_n7_r1_k10_stages.pdf](multigen_none_n7_r1_k10_stages.pdf) — Stage evaluations, 5+2 players, single meeting. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_none_n7_r1_k10_matrix.pdf](multigen_none_n7_r1_k10_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_none_n7_r1_k10

**Single-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.195 ± 0.053 | 0.299 ± 0.089 | 0.415 ± 0.044 | 0.379 ± 0.049 | 0.371 ± 0.049 | 0.379 ± 0.070 | 0.393 ± 0.040 | 0.381 ± 0.065 | 0.391 ± 0.048 | 0.360 ± 0.060 | 0.377 ± 0.057 |
| C1 | 0.069 ± 0.021 | 0.563 ± 0.062 | 0.214 ± 0.062 | 0.418 ± 0.063 | 0.369 ± 0.061 | 0.392 ± 0.061 | 0.349 ± 0.054 | 0.379 ± 0.073 | 0.356 ± 0.056 | 0.336 ± 0.068 | 0.339 ± 0.077 |
| C2 | 0.081 ± 0.034 | 0.349 ± 0.049 | 0.548 ± 0.069 | 0.251 ± 0.065 | 0.430 ± 0.059 | 0.359 ± 0.070 | 0.383 ± 0.055 | 0.409 ± 0.047 | 0.402 ± 0.050 | 0.357 ± 0.069 | 0.374 ± 0.052 |
| C3 | 0.082 ± 0.019 | 0.367 ± 0.115 | 0.319 ± 0.066 | 0.577 ± 0.054 | 0.258 ± 0.062 | 0.415 ± 0.071 | 0.347 ± 0.073 | 0.386 ± 0.071 | 0.345 ± 0.049 | 0.378 ± 0.079 | 0.337 ± 0.067 |
| C4 | 0.079 ± 0.024 | 0.370 ± 0.043 | 0.410 ± 0.065 | 0.342 ± 0.059 | 0.550 ± 0.035 | 0.274 ± 0.075 | 0.418 ± 0.034 | 0.363 ± 0.070 | 0.374 ± 0.046 | 0.338 ± 0.081 | 0.372 ± 0.052 |
| C5 | 0.067 ± 0.020 | 0.372 ± 0.093 | 0.343 ± 0.054 | 0.406 ± 0.058 | 0.332 ± 0.051 | 0.550 ± 0.040 | 0.259 ± 0.047 | 0.420 ± 0.067 | 0.348 ± 0.045 | 0.367 ± 0.053 | 0.332 ± 0.078 |
| C6 | 0.084 ± 0.031 | 0.379 ± 0.080 | 0.366 ± 0.067 | 0.366 ± 0.085 | 0.413 ± 0.047 | 0.354 ± 0.058 | 0.572 ± 0.050 | 0.274 ± 0.064 | 0.408 ± 0.078 | 0.318 ± 0.066 | 0.389 ± 0.068 |
| C7 | 0.101 ± 0.029 | 0.374 ± 0.100 | 0.352 ± 0.063 | 0.402 ± 0.055 | 0.357 ± 0.061 | 0.416 ± 0.076 | 0.304 ± 0.064 | 0.563 ± 0.039 | 0.259 ± 0.029 | 0.385 ± 0.055 | 0.325 ± 0.075 |
| C8 | 0.074 ± 0.019 | 0.405 ± 0.063 | 0.383 ± 0.065 | 0.380 ± 0.068 | 0.372 ± 0.058 | 0.371 ± 0.079 | 0.386 ± 0.046 | 0.352 ± 0.077 | 0.552 ± 0.050 | 0.231 ± 0.085 | 0.416 ± 0.071 |
| C9 | 0.079 ± 0.027 | 0.368 ± 0.085 | 0.340 ± 0.070 | 0.399 ± 0.073 | 0.355 ± 0.053 | 0.378 ± 0.083 | 0.354 ± 0.041 | 0.426 ± 0.052 | 0.338 ± 0.034 | 0.566 ± 0.067 | 0.253 ± 0.084 |
| C10 | 0.089 ± 0.038 | 0.372 ± 0.071 | 0.370 ± 0.061 | 0.372 ± 0.055 | 0.393 ± 0.065 | 0.372 ± 0.067 | 0.361 ± 0.054 | 0.364 ± 0.066 | 0.411 ± 0.062 | 0.298 ± 0.071 | 0.533 ± 0.043 |

**No coalition member ejected**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.250 ± 0.063 | 0.526 ± 0.117 | 0.648 ± 0.052 | 0.608 ± 0.064 | 0.601 ± 0.086 | 0.595 ± 0.073 | 0.626 ± 0.035 | 0.595 ± 0.088 | 0.610 ± 0.048 | 0.555 ± 0.064 | 0.585 ± 0.067 |
| C1 | 0.097 ± 0.024 | 0.748 ± 0.034 | 0.440 ± 0.104 | 0.643 ± 0.063 | 0.601 ± 0.080 | 0.626 ± 0.066 | 0.571 ± 0.059 | 0.619 ± 0.072 | 0.609 ± 0.058 | 0.552 ± 0.093 | 0.570 ± 0.109 |
| C2 | 0.114 ± 0.039 | 0.610 ± 0.082 | 0.740 ± 0.020 | 0.523 ± 0.104 | 0.670 ± 0.072 | 0.620 ± 0.097 | 0.635 ± 0.041 | 0.640 ± 0.064 | 0.662 ± 0.031 | 0.593 ± 0.087 | 0.619 ± 0.067 |
| C3 | 0.115 ± 0.023 | 0.611 ± 0.133 | 0.556 ± 0.096 | 0.749 ± 0.029 | 0.509 ± 0.075 | 0.654 ± 0.052 | 0.582 ± 0.081 | 0.644 ± 0.074 | 0.580 ± 0.064 | 0.607 ± 0.111 | 0.572 ± 0.085 |
| C4 | 0.110 ± 0.026 | 0.626 ± 0.070 | 0.628 ± 0.052 | 0.592 ± 0.067 | 0.746 ± 0.037 | 0.510 ± 0.123 | 0.681 ± 0.034 | 0.594 ± 0.104 | 0.646 ± 0.055 | 0.555 ± 0.122 | 0.632 ± 0.068 |
| C5 | 0.095 ± 0.022 | 0.622 ± 0.107 | 0.578 ± 0.068 | 0.633 ± 0.069 | 0.572 ± 0.074 | 0.728 ± 0.028 | 0.501 ± 0.077 | 0.663 ± 0.066 | 0.569 ± 0.055 | 0.616 ± 0.074 | 0.562 ± 0.112 |
| C6 | 0.119 ± 0.034 | 0.631 ± 0.077 | 0.618 ± 0.062 | 0.613 ± 0.090 | 0.673 ± 0.064 | 0.590 ± 0.078 | 0.750 ± 0.049 | 0.540 ± 0.104 | 0.679 ± 0.062 | 0.549 ± 0.089 | 0.646 ± 0.081 |
| C7 | 0.134 ± 0.036 | 0.606 ± 0.126 | 0.594 ± 0.048 | 0.655 ± 0.058 | 0.590 ± 0.064 | 0.655 ± 0.058 | 0.542 ± 0.080 | 0.749 ± 0.019 | 0.501 ± 0.046 | 0.615 ± 0.076 | 0.579 ± 0.087 |
| C8 | 0.104 ± 0.025 | 0.646 ± 0.065 | 0.632 ± 0.061 | 0.622 ± 0.073 | 0.652 ± 0.071 | 0.600 ± 0.075 | 0.663 ± 0.024 | 0.586 ± 0.108 | 0.752 ± 0.037 | 0.438 ± 0.103 | 0.644 ± 0.080 |
| C9 | 0.114 ± 0.037 | 0.612 ± 0.105 | 0.592 ± 0.076 | 0.634 ± 0.085 | 0.592 ± 0.089 | 0.633 ± 0.092 | 0.601 ± 0.054 | 0.660 ± 0.062 | 0.588 ± 0.069 | 0.759 ± 0.040 | 0.483 ± 0.126 |
| C10 | 0.118 ± 0.046 | 0.619 ± 0.084 | 0.626 ± 0.045 | 0.631 ± 0.065 | 0.644 ± 0.089 | 0.616 ± 0.067 | 0.635 ± 0.041 | 0.611 ± 0.087 | 0.654 ± 0.063 | 0.527 ± 0.080 | 0.715 ± 0.039 |

## Loop descriptors: multigen_none_n7_r1_k10



| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_none_n7_r1_k10` / c_stage_mean | 10 | 0.557 ± 0.016 | 0.744 ± 0.010 |
| `multigen_none_n7_r1_k10` / d_stage_mean | 10 | 0.257 ± 0.023 | 0.497 ± 0.031 |
| `multigen_none_n7_r1_k10` / late_minus_early | 10 | -0.013 ± 0.068 | -0.007 ± 0.036 |
| `multigen_none_n7_r1_k10` / matrix_adaptation_gain | 10 | 0.300 ± 0.032 | 0.246 ± 0.036 |
| `multigen_none_n7_r1_k10` / parity_gap | 10 | 0.033 ± 0.035 | 0.040 ± 0.036 |

## Finite observed-policy response gaps: multigen_none_n7_r1_k10

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `multigen_none_n7_r1_k10` / 0 | 10 | 0.001 ± 0.002 |
| `multigen_none_n7_r1_k10` / 1 | 10 | 0.494 ± 0.057 |
| `multigen_none_n7_r1_k10` / 2 | 10 | 0.467 ± 0.083 |
| `multigen_none_n7_r1_k10` / 3 | 10 | 0.495 ± 0.057 |
| `multigen_none_n7_r1_k10` / 4 | 10 | 0.471 ± 0.042 |
| `multigen_none_n7_r1_k10` / 5 | 10 | 0.484 ± 0.044 |
| `multigen_none_n7_r1_k10` / 6 | 10 | 0.488 ± 0.060 |
| `multigen_none_n7_r1_k10` / 7 | 10 | 0.464 ± 0.052 |
| `multigen_none_n7_r1_k10` / 8 | 10 | 0.485 ± 0.042 |
| `multigen_none_n7_r1_k10` / 9 | 10 | 0.487 ± 0.076 |
| `multigen_none_n7_r1_k10` / 10 | 10 | 0.443 ± 0.051 |

[multigen_none_n7_r1_k10_response_gaps.pdf](multigen_none_n7_r1_k10_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_none_n7_r8_k10

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n7_r8_k10` / C0 | 10 | 0.973 ± 0.009 | 2.900 ± 0.048 | 0.967 ± 0.009 | 0.965 ± 0.010 |
| `multigen_none_n7_r8_k10` / D1 | 10 | 0.836 ± 0.039 | 1.641 ± 0.156 | 0.356 ± 0.105 | 0.350 ± 0.100 |
| `multigen_none_n7_r8_k10` / C1 | 10 | 0.846 ± 0.015 | 1.208 ± 0.046 | 0.854 ± 0.027 | 0.512 ± 0.038 |
| `multigen_none_n7_r8_k10` / D2 | 10 | 0.625 ± 0.134 | 0.882 ± 0.347 | 0.356 ± 0.111 | 0.199 ± 0.100 |
| `multigen_none_n7_r8_k10` / C2 | 10 | 0.850 ± 0.030 | 1.264 ± 0.185 | 0.780 ± 0.052 | 0.458 ± 0.046 |
| `multigen_none_n7_r8_k10` / D3 | 10 | 0.575 ± 0.115 | 0.750 ± 0.226 | 0.276 ± 0.103 | 0.140 ± 0.057 |
| `multigen_none_n7_r8_k10` / C3 | 10 | 0.844 ± 0.027 | 1.189 ± 0.088 | 0.815 ± 0.031 | 0.467 ± 0.063 |
| `multigen_none_n7_r8_k10` / D4 | 10 | 0.597 ± 0.051 | 0.772 ± 0.114 | 0.246 ± 0.073 | 0.135 ± 0.043 |
| `multigen_none_n7_r8_k10` / C4 | 10 | 0.850 ± 0.033 | 1.198 ± 0.090 | 0.820 ± 0.034 | 0.464 ± 0.038 |
| `multigen_none_n7_r8_k10` / D5 | 10 | 0.628 ± 0.113 | 0.850 ± 0.226 | 0.316 ± 0.079 | 0.169 ± 0.050 |
| `multigen_none_n7_r8_k10` / C5 | 10 | 0.837 ± 0.020 | 1.201 ± 0.078 | 0.812 ± 0.046 | 0.475 ± 0.036 |
| `multigen_none_n7_r8_k10` / D6 | 10 | 0.581 ± 0.126 | 0.749 ± 0.217 | 0.243 ± 0.108 | 0.141 ± 0.078 |
| `multigen_none_n7_r8_k10` / C6 | 10 | 0.861 ± 0.030 | 1.299 ± 0.123 | 0.787 ± 0.078 | 0.477 ± 0.071 |
| `multigen_none_n7_r8_k10` / D7 | 10 | 0.621 ± 0.100 | 0.837 ± 0.196 | 0.328 ± 0.179 | 0.196 ± 0.128 |
| `multigen_none_n7_r8_k10` / C7 | 10 | 0.835 ± 0.025 | 1.182 ± 0.084 | 0.808 ± 0.045 | 0.459 ± 0.053 |
| `multigen_none_n7_r8_k10` / D8 | 10 | 0.625 ± 0.129 | 0.891 ± 0.369 | 0.296 ± 0.115 | 0.171 ± 0.088 |
| `multigen_none_n7_r8_k10` / C8 | 10 | 0.859 ± 0.030 | 1.310 ± 0.177 | 0.791 ± 0.083 | 0.496 ± 0.042 |
| `multigen_none_n7_r8_k10` / D9 | 10 | 0.603 ± 0.091 | 0.795 ± 0.186 | 0.301 ± 0.080 | 0.158 ± 0.055 |
| `multigen_none_n7_r8_k10` / C9 | 10 | 0.839 ± 0.032 | 1.208 ± 0.105 | 0.774 ± 0.073 | 0.448 ± 0.048 |
| `multigen_none_n7_r8_k10` / D10 | 10 | 0.620 ± 0.110 | 0.895 ± 0.322 | 0.311 ± 0.144 | 0.190 ± 0.114 |
| `multigen_none_n7_r8_k10` / C10 | 10 | 0.841 ± 0.035 | 1.227 ± 0.129 | 0.803 ± 0.030 | 0.500 ± 0.065 |

[multigen_none_n7_r8_k10_stages.pdf](multigen_none_n7_r8_k10_stages.pdf) — Stage evaluations, 5+2 players, up to 8 rounds. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_none_n7_r8_k10_matrix.pdf](multigen_none_n7_r8_k10_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_none_n7_r8_k10

**At least one false ejection / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.971 ± 0.011 | 0.834 ± 0.035 | 0.797 ± 0.070 | 0.800 ± 0.050 | 0.804 ± 0.048 | 0.827 ± 0.061 | 0.800 ± 0.076 | 0.820 ± 0.061 | 0.808 ± 0.068 | 0.796 ± 0.053 | 0.824 ± 0.052 |
| C1 | 0.278 ± 0.085 | 0.848 ± 0.017 | 0.617 ± 0.138 | 0.719 ± 0.068 | 0.680 ± 0.068 | 0.745 ± 0.068 | 0.689 ± 0.106 | 0.753 ± 0.072 | 0.726 ± 0.118 | 0.720 ± 0.067 | 0.712 ± 0.084 |
| C2 | 0.228 ± 0.046 | 0.786 ± 0.064 | 0.850 ± 0.032 | 0.576 ± 0.118 | 0.779 ± 0.063 | 0.671 ± 0.116 | 0.750 ± 0.099 | 0.717 ± 0.106 | 0.739 ± 0.103 | 0.718 ± 0.109 | 0.732 ± 0.088 |
| C3 | 0.256 ± 0.023 | 0.813 ± 0.035 | 0.651 ± 0.133 | 0.843 ± 0.028 | 0.602 ± 0.052 | 0.768 ± 0.050 | 0.643 ± 0.126 | 0.767 ± 0.050 | 0.683 ± 0.125 | 0.700 ± 0.088 | 0.713 ± 0.105 |
| C4 | 0.248 ± 0.050 | 0.801 ± 0.052 | 0.768 ± 0.054 | 0.630 ± 0.106 | 0.853 ± 0.027 | 0.632 ± 0.111 | 0.766 ± 0.086 | 0.691 ± 0.129 | 0.766 ± 0.107 | 0.687 ± 0.101 | 0.776 ± 0.106 |
| C5 | 0.249 ± 0.056 | 0.799 ± 0.046 | 0.669 ± 0.133 | 0.747 ± 0.035 | 0.638 ± 0.057 | 0.833 ± 0.023 | 0.588 ± 0.124 | 0.773 ± 0.067 | 0.690 ± 0.127 | 0.728 ± 0.065 | 0.703 ± 0.087 |
| C6 | 0.296 ± 0.034 | 0.797 ± 0.053 | 0.758 ± 0.096 | 0.655 ± 0.074 | 0.776 ± 0.046 | 0.682 ± 0.100 | 0.863 ± 0.034 | 0.622 ± 0.102 | 0.769 ± 0.067 | 0.666 ± 0.097 | 0.782 ± 0.077 |
| C7 | 0.234 ± 0.062 | 0.807 ± 0.034 | 0.684 ± 0.128 | 0.679 ± 0.102 | 0.699 ± 0.073 | 0.726 ± 0.095 | 0.646 ± 0.131 | 0.834 ± 0.041 | 0.624 ± 0.132 | 0.777 ± 0.041 | 0.685 ± 0.098 |
| C8 | 0.238 ± 0.061 | 0.790 ± 0.046 | 0.727 ± 0.130 | 0.678 ± 0.071 | 0.751 ± 0.105 | 0.695 ± 0.082 | 0.764 ± 0.064 | 0.671 ± 0.101 | 0.859 ± 0.036 | 0.605 ± 0.086 | 0.778 ± 0.056 |
| C9 | 0.236 ± 0.052 | 0.812 ± 0.038 | 0.696 ± 0.138 | 0.674 ± 0.115 | 0.693 ± 0.114 | 0.734 ± 0.088 | 0.645 ± 0.137 | 0.795 ± 0.057 | 0.672 ± 0.141 | 0.834 ± 0.034 | 0.623 ± 0.115 |
| C10 | 0.235 ± 0.055 | 0.792 ± 0.038 | 0.715 ± 0.107 | 0.684 ± 0.069 | 0.729 ± 0.086 | 0.702 ± 0.096 | 0.748 ± 0.085 | 0.699 ± 0.098 | 0.758 ± 0.089 | 0.670 ± 0.072 | 0.838 ± 0.043 |

**Mean false ejections / game**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2.891 ± 0.048 | 1.625 ± 0.150 | 1.569 ± 0.329 | 1.496 ± 0.190 | 1.464 ± 0.179 | 1.591 ± 0.187 | 1.470 ± 0.329 | 1.580 ± 0.289 | 1.580 ± 0.373 | 1.424 ± 0.176 | 1.592 ± 0.304 |
| C1 | 0.324 ± 0.118 | 1.205 ± 0.057 | 0.864 ± 0.348 | 0.974 ± 0.141 | 0.905 ± 0.144 | 1.003 ± 0.109 | 0.946 ± 0.183 | 1.058 ± 0.178 | 1.069 ± 0.381 | 0.988 ± 0.147 | 1.040 ± 0.256 |
| C2 | 0.256 ± 0.057 | 1.096 ± 0.127 | 1.265 ± 0.179 | 0.760 ± 0.239 | 1.060 ± 0.122 | 0.901 ± 0.230 | 1.044 ± 0.196 | 0.989 ± 0.211 | 1.091 ± 0.326 | 1.012 ± 0.251 | 1.049 ± 0.193 |
| C3 | 0.287 ± 0.037 | 1.123 ± 0.061 | 0.950 ± 0.366 | 1.187 ± 0.088 | 0.773 ± 0.111 | 1.052 ± 0.088 | 0.878 ± 0.258 | 1.073 ± 0.191 | 0.985 ± 0.366 | 0.946 ± 0.155 | 1.041 ± 0.375 |
| C4 | 0.285 ± 0.075 | 1.124 ± 0.093 | 1.111 ± 0.203 | 0.880 ± 0.227 | 1.208 ± 0.090 | 0.858 ± 0.222 | 1.065 ± 0.163 | 0.979 ± 0.241 | 1.144 ± 0.332 | 0.959 ± 0.210 | 1.137 ± 0.236 |
| C5 | 0.273 ± 0.069 | 1.107 ± 0.081 | 0.961 ± 0.344 | 1.012 ± 0.098 | 0.850 ± 0.131 | 1.197 ± 0.087 | 0.756 ± 0.217 | 1.076 ± 0.182 | 1.003 ± 0.366 | 0.987 ± 0.102 | 0.984 ± 0.216 |
| C6 | 0.334 ± 0.050 | 1.108 ± 0.093 | 1.147 ± 0.341 | 0.880 ± 0.158 | 1.056 ± 0.127 | 0.957 ± 0.261 | 1.299 ± 0.127 | 0.845 ± 0.202 | 1.117 ± 0.285 | 0.925 ± 0.212 | 1.107 ± 0.174 |
| C7 | 0.261 ± 0.078 | 1.144 ± 0.065 | 0.982 ± 0.319 | 0.923 ± 0.204 | 0.942 ± 0.145 | 0.982 ± 0.194 | 0.880 ± 0.229 | 1.193 ± 0.110 | 0.895 ± 0.361 | 1.068 ± 0.126 | 0.975 ± 0.261 |
| C8 | 0.260 ± 0.076 | 1.080 ± 0.094 | 1.049 ± 0.328 | 0.923 ± 0.197 | 1.019 ± 0.219 | 0.952 ± 0.175 | 1.050 ± 0.149 | 0.930 ± 0.202 | 1.311 ± 0.191 | 0.797 ± 0.183 | 1.113 ± 0.147 |
| C9 | 0.256 ± 0.058 | 1.138 ± 0.073 | 0.980 ± 0.302 | 0.906 ± 0.194 | 0.968 ± 0.194 | 0.990 ± 0.182 | 0.883 ± 0.245 | 1.089 ± 0.126 | 1.000 ± 0.423 | 1.207 ± 0.114 | 0.900 ± 0.329 |
| C10 | 0.261 ± 0.062 | 1.089 ± 0.078 | 1.026 ± 0.305 | 0.912 ± 0.163 | 0.986 ± 0.176 | 0.972 ± 0.213 | 1.035 ± 0.174 | 0.988 ± 0.262 | 1.101 ± 0.318 | 0.917 ± 0.158 | 1.222 ± 0.146 |

**Coalition game wins**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.966 ± 0.011 | 0.352 ± 0.107 | 0.291 ± 0.086 | 0.292 ± 0.076 | 0.272 ± 0.083 | 0.337 ± 0.115 | 0.262 ± 0.108 | 0.370 ± 0.138 | 0.287 ± 0.092 | 0.253 ± 0.105 | 0.350 ± 0.098 |
| C1 | 0.205 ± 0.063 | 0.856 ± 0.031 | 0.351 ± 0.115 | 0.588 ± 0.086 | 0.481 ± 0.134 | 0.618 ± 0.115 | 0.454 ± 0.124 | 0.626 ± 0.146 | 0.522 ± 0.150 | 0.525 ± 0.174 | 0.557 ± 0.102 |
| C2 | 0.181 ± 0.038 | 0.747 ± 0.121 | 0.780 ± 0.057 | 0.281 ± 0.100 | 0.700 ± 0.133 | 0.467 ± 0.178 | 0.562 ± 0.204 | 0.542 ± 0.210 | 0.547 ± 0.142 | 0.504 ± 0.169 | 0.567 ± 0.177 |
| C3 | 0.203 ± 0.034 | 0.812 ± 0.024 | 0.345 ± 0.156 | 0.813 ± 0.031 | 0.250 ± 0.074 | 0.667 ± 0.147 | 0.382 ± 0.153 | 0.649 ± 0.120 | 0.467 ± 0.162 | 0.507 ± 0.199 | 0.531 ± 0.128 |
| C4 | 0.181 ± 0.038 | 0.742 ± 0.124 | 0.620 ± 0.118 | 0.358 ± 0.139 | 0.822 ± 0.031 | 0.313 ± 0.086 | 0.590 ± 0.172 | 0.474 ± 0.237 | 0.606 ± 0.169 | 0.418 ± 0.143 | 0.604 ± 0.161 |
| C5 | 0.197 ± 0.044 | 0.796 ± 0.065 | 0.395 ± 0.167 | 0.651 ± 0.082 | 0.364 ± 0.126 | 0.808 ± 0.051 | 0.246 ± 0.110 | 0.671 ± 0.133 | 0.431 ± 0.168 | 0.562 ± 0.208 | 0.508 ± 0.159 |
| C6 | 0.242 ± 0.019 | 0.758 ± 0.138 | 0.576 ± 0.133 | 0.464 ± 0.129 | 0.682 ± 0.058 | 0.452 ± 0.161 | 0.791 ± 0.080 | 0.331 ± 0.183 | 0.641 ± 0.144 | 0.377 ± 0.142 | 0.701 ± 0.052 |
| C7 | 0.181 ± 0.045 | 0.782 ± 0.086 | 0.475 ± 0.140 | 0.504 ± 0.155 | 0.486 ± 0.198 | 0.607 ± 0.142 | 0.364 ± 0.186 | 0.812 ± 0.047 | 0.296 ± 0.113 | 0.647 ± 0.097 | 0.436 ± 0.144 |
| C8 | 0.195 ± 0.054 | 0.775 ± 0.097 | 0.537 ± 0.163 | 0.514 ± 0.142 | 0.626 ± 0.154 | 0.501 ± 0.164 | 0.624 ± 0.143 | 0.457 ± 0.143 | 0.792 ± 0.087 | 0.306 ± 0.086 | 0.705 ± 0.079 |
| C9 | 0.197 ± 0.048 | 0.783 ± 0.100 | 0.487 ± 0.182 | 0.525 ± 0.210 | 0.495 ± 0.254 | 0.635 ± 0.129 | 0.351 ± 0.178 | 0.744 ± 0.068 | 0.378 ± 0.129 | 0.769 ± 0.075 | 0.307 ± 0.151 |
| C10 | 0.180 ± 0.047 | 0.789 ± 0.074 | 0.519 ± 0.160 | 0.540 ± 0.119 | 0.589 ± 0.136 | 0.540 ± 0.132 | 0.598 ± 0.140 | 0.507 ± 0.145 | 0.629 ± 0.121 | 0.428 ± 0.113 | 0.797 ± 0.037 |

**Terminal-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.964 ± 0.011 | 0.345 ± 0.103 | 0.287 ± 0.088 | 0.288 ± 0.076 | 0.268 ± 0.078 | 0.335 ± 0.112 | 0.259 ± 0.109 | 0.358 ± 0.132 | 0.282 ± 0.092 | 0.248 ± 0.101 | 0.345 ± 0.097 |
| C1 | 0.041 ± 0.027 | 0.510 ± 0.049 | 0.196 ± 0.105 | 0.322 ± 0.067 | 0.253 ± 0.081 | 0.333 ± 0.062 | 0.257 ± 0.079 | 0.353 ± 0.100 | 0.299 ± 0.113 | 0.289 ± 0.087 | 0.334 ± 0.097 |
| C2 | 0.026 ± 0.011 | 0.433 ± 0.091 | 0.459 ± 0.050 | 0.143 ± 0.059 | 0.370 ± 0.079 | 0.246 ± 0.098 | 0.302 ± 0.109 | 0.297 ± 0.120 | 0.309 ± 0.072 | 0.278 ± 0.103 | 0.353 ± 0.135 |
| C3 | 0.029 ± 0.009 | 0.460 ± 0.028 | 0.210 ± 0.117 | 0.469 ± 0.058 | 0.137 ± 0.042 | 0.366 ± 0.072 | 0.215 ± 0.093 | 0.365 ± 0.080 | 0.266 ± 0.115 | 0.277 ± 0.079 | 0.317 ± 0.108 |
| C4 | 0.031 ± 0.017 | 0.435 ± 0.086 | 0.336 ± 0.059 | 0.210 ± 0.082 | 0.470 ± 0.040 | 0.173 ± 0.052 | 0.336 ± 0.098 | 0.277 ± 0.147 | 0.344 ± 0.108 | 0.236 ± 0.083 | 0.384 ± 0.134 |
| C5 | 0.024 ± 0.009 | 0.460 ± 0.042 | 0.235 ± 0.110 | 0.360 ± 0.044 | 0.195 ± 0.067 | 0.470 ± 0.032 | 0.142 ± 0.076 | 0.378 ± 0.075 | 0.261 ± 0.125 | 0.303 ± 0.089 | 0.317 ± 0.126 |
| C6 | 0.038 ± 0.016 | 0.437 ± 0.080 | 0.332 ± 0.094 | 0.237 ± 0.055 | 0.358 ± 0.046 | 0.236 ± 0.055 | 0.478 ± 0.082 | 0.203 ± 0.128 | 0.349 ± 0.091 | 0.207 ± 0.078 | 0.416 ± 0.074 |
| C7 | 0.024 ± 0.013 | 0.464 ± 0.057 | 0.261 ± 0.098 | 0.271 ± 0.100 | 0.253 ± 0.106 | 0.326 ± 0.098 | 0.198 ± 0.102 | 0.469 ± 0.064 | 0.174 ± 0.087 | 0.338 ± 0.038 | 0.277 ± 0.128 |
| C8 | 0.024 ± 0.016 | 0.433 ± 0.064 | 0.298 ± 0.103 | 0.270 ± 0.063 | 0.344 ± 0.111 | 0.270 ± 0.076 | 0.355 ± 0.091 | 0.264 ± 0.113 | 0.500 ± 0.045 | 0.161 ± 0.065 | 0.429 ± 0.088 |
| C9 | 0.023 ± 0.011 | 0.459 ± 0.065 | 0.268 ± 0.114 | 0.273 ± 0.117 | 0.282 ± 0.146 | 0.332 ± 0.092 | 0.205 ± 0.102 | 0.426 ± 0.066 | 0.225 ± 0.098 | 0.451 ± 0.058 | 0.187 ± 0.122 |
| C10 | 0.026 ± 0.010 | 0.449 ± 0.051 | 0.287 ± 0.099 | 0.282 ± 0.075 | 0.313 ± 0.098 | 0.291 ± 0.081 | 0.333 ± 0.097 | 0.278 ± 0.103 | 0.348 ± 0.075 | 0.220 ± 0.054 | 0.494 ± 0.078 |

## Loop descriptors: multigen_none_n7_r8_k10



| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n7_r8_k10` / c_stage_mean | 10 | 0.846 ± 0.007 | 1.229 ± 0.025 | 0.804 ± 0.025 | 0.476 ± 0.007 |
| `multigen_none_n7_r8_k10` / d_stage_mean | 10 | 0.631 ± 0.029 | 0.906 ± 0.069 | 0.303 ± 0.035 | 0.185 ± 0.029 |
| `multigen_none_n7_r8_k10` / late_minus_early | 10 | -0.000 ± 0.019 | 0.028 ± 0.138 | -0.027 ± 0.062 | 0.002 ± 0.054 |
| `multigen_none_n7_r8_k10` / matrix_adaptation_gain | 10 | 0.213 ± 0.033 | 0.322 ± 0.055 | 0.501 ± 0.050 | 0.291 ± 0.033 |
| `multigen_none_n7_r8_k10` / parity_gap | 10 | 0.053 ± 0.061 | 0.074 ± 0.088 | 0.116 ± 0.137 | 0.063 ± 0.074 |

## Finite observed-policy response gaps: multigen_none_n7_r8_k10

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_none_n7_r8_k10` / 0 | 10 | 0.247 ± 0.029 | 0.806 ± 0.036 |
| `multigen_none_n7_r8_k10` / 1 | 10 | 0.582 ± 0.080 | 0.663 ± 0.057 |
| `multigen_none_n7_r8_k10` / 2 | 10 | 0.629 ± 0.047 | 0.607 ± 0.075 |
| `multigen_none_n7_r8_k10` / 3 | 10 | 0.596 ± 0.030 | 0.627 ± 0.057 |
| `multigen_none_n7_r8_k10` / 4 | 10 | 0.610 ± 0.046 | 0.643 ± 0.041 |
| `multigen_none_n7_r8_k10` / 5 | 10 | 0.606 ± 0.069 | 0.644 ± 0.067 |
| `multigen_none_n7_r8_k10` / 6 | 10 | 0.578 ± 0.051 | 0.582 ± 0.113 |
| `multigen_none_n7_r8_k10` / 7 | 10 | 0.620 ± 0.075 | 0.636 ± 0.071 |
| `multigen_none_n7_r8_k10` / 8 | 10 | 0.629 ± 0.075 | 0.607 ± 0.099 |
| `multigen_none_n7_r8_k10` / 9 | 10 | 0.609 ± 0.073 | 0.597 ± 0.118 |
| `multigen_none_n7_r8_k10` / 10 | 10 | 0.611 ± 0.058 | 0.616 ± 0.050 |

[multigen_none_n7_r8_k10_response_gaps.pdf](multigen_none_n7_r8_k10_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_none_n9_r8_k3

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n9_r8_k3` / C0 | 5 | 0.969 ± 0.011 | 4.810 ± 0.065 | 0.956 ± 0.013 | 0.961 ± 0.013 |
| `multigen_none_n9_r8_k3` / D1 | 5 | 0.947 ± 0.006 | 2.954 ± 0.112 | 0.306 ± 0.025 | 0.357 ± 0.021 |
| `multigen_none_n9_r8_k3` / C1 | 5 | 0.943 ± 0.010 | 1.997 ± 0.074 | 0.812 ± 0.019 | 0.549 ± 0.023 |
| `multigen_none_n9_r8_k3` / D2 | 5 | 0.907 ± 0.022 | 1.930 ± 0.239 | 0.394 ± 0.119 | 0.237 ± 0.085 |
| `multigen_none_n9_r8_k3` / C2 | 5 | 0.944 ± 0.014 | 2.051 ± 0.105 | 0.700 ± 0.101 | 0.414 ± 0.077 |
| `multigen_none_n9_r8_k3` / D3 | 5 | 0.881 ± 0.029 | 1.576 ± 0.293 | 0.291 ± 0.134 | 0.161 ± 0.074 |
| `multigen_none_n9_r8_k3` / C3 | 5 | 0.944 ± 0.016 | 2.075 ± 0.146 | 0.742 ± 0.042 | 0.444 ± 0.059 |

[multigen_none_n9_r8_k3_stages.pdf](multigen_none_n9_r8_k3_stages.pdf) — Stage evaluations, 7+2 players, up to 8 rounds. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_none_n9_r8_k3_matrix.pdf](multigen_none_n9_r8_k3_matrix.pdf) — Cross-generation evaluation matrix (7+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_none_n9_r8_k3

**At least one false ejection / game**

| Coalition / defender | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| C0 | 0.973 ± 0.009 | 0.951 ± 0.010 | 0.936 ± 0.008 | 0.923 ± 0.020 |
| C1 | 0.412 ± 0.054 | 0.939 ± 0.011 | 0.907 ± 0.021 | 0.916 ± 0.028 |
| C2 | 0.330 ± 0.076 | 0.928 ± 0.014 | 0.946 ± 0.013 | 0.870 ± 0.034 |
| C3 | 0.453 ± 0.097 | 0.934 ± 0.011 | 0.908 ± 0.008 | 0.939 ± 0.023 |

**Mean false ejections / game**

| Coalition / defender | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| C0 | 4.827 ± 0.069 | 2.973 ± 0.109 | 2.840 ± 0.135 | 2.509 ± 0.414 |
| C1 | 0.568 ± 0.114 | 1.994 ± 0.097 | 1.942 ± 0.234 | 1.826 ± 0.238 |
| C2 | 0.434 ± 0.137 | 1.897 ± 0.077 | 2.059 ± 0.113 | 1.558 ± 0.310 |
| C3 | 0.632 ± 0.174 | 1.994 ± 0.127 | 2.114 ± 0.244 | 2.054 ± 0.142 |

**Coalition game wins**

| Coalition / defender | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| C0 | 0.959 ± 0.015 | 0.308 ± 0.024 | 0.240 ± 0.061 | 0.173 ± 0.077 |
| C1 | 0.072 ± 0.020 | 0.811 ± 0.023 | 0.398 ± 0.110 | 0.478 ± 0.130 |
| C2 | 0.042 ± 0.010 | 0.785 ± 0.044 | 0.707 ± 0.104 | 0.289 ± 0.133 |
| C3 | 0.061 ± 0.016 | 0.768 ± 0.040 | 0.371 ± 0.114 | 0.731 ± 0.043 |

**Terminal-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| C0 | 0.964 ± 0.015 | 0.361 ± 0.031 | 0.280 ± 0.068 | 0.204 ± 0.083 |
| C1 | 0.018 ± 0.012 | 0.543 ± 0.043 | 0.235 ± 0.082 | 0.288 ± 0.092 |
| C2 | 0.008 ± 0.005 | 0.508 ± 0.033 | 0.412 ± 0.091 | 0.152 ± 0.072 |
| C3 | 0.015 ± 0.009 | 0.507 ± 0.062 | 0.249 ± 0.091 | 0.434 ± 0.065 |

## Loop descriptors: multigen_none_n9_r8_k3



| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_none_n9_r8_k3` / c_stage_mean | 5 | 0.944 ± 0.009 | 2.041 ± 0.038 | 0.752 ± 0.048 | 0.469 ± 0.035 |
| `multigen_none_n9_r8_k3` / d_stage_mean | 5 | 0.911 ± 0.006 | 2.154 ± 0.071 | 0.330 ± 0.044 | 0.252 ± 0.032 |
| `multigen_none_n9_r8_k3` / late_minus_early | 5 | 0.001 ± 0.015 | 0.078 ± 0.165 | -0.070 ± 0.030 | -0.105 ± 0.050 |
| `multigen_none_n9_r8_k3` / matrix_adaptation_gain | 5 | 0.032 ± 0.018 | -0.122 ± 0.111 | 0.418 ± 0.088 | 0.213 ± 0.066 |

## Finite observed-policy response gaps: multigen_none_n9_r8_k3

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_none_n9_r8_k3` / 0 | 5 | 0.050 ± 0.021 | 0.798 ± 0.079 |
| `multigen_none_n9_r8_k3` / 1 | 5 | 0.542 ± 0.056 | 0.744 ± 0.031 |
| `multigen_none_n9_r8_k3` / 2 | 5 | 0.617 ± 0.067 | 0.664 ± 0.101 |
| `multigen_none_n9_r8_k3` / 3 | 5 | 0.495 ± 0.092 | 0.670 ± 0.033 |

[multigen_none_n9_r8_k3_response_gaps.pdf](multigen_none_n9_r8_k3_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_rwbal_n7_r1_k4

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_rwbal_n7_r1_k4` / C0 | 5 | 0.185 ± 0.071 | 0.234 ± 0.078 |
| `multigen_rwbal_n7_r1_k4` / D1 | 5 | 0.395 ± 0.045 | 0.691 ± 0.065 |
| `multigen_rwbal_n7_r1_k4` / C1 | 5 | 0.526 ± 0.049 | 0.713 ± 0.029 |
| `multigen_rwbal_n7_r1_k4` / D2 | 5 | 0.393 ± 0.046 | 0.733 ± 0.068 |
| `multigen_rwbal_n7_r1_k4` / C2 | 5 | 0.604 ± 0.062 | 0.798 ± 0.035 |
| `multigen_rwbal_n7_r1_k4` / D3 | 5 | 0.359 ± 0.082 | 0.676 ± 0.146 |
| `multigen_rwbal_n7_r1_k4` / C3 | 5 | 0.557 ± 0.073 | 0.731 ± 0.062 |
| `multigen_rwbal_n7_r1_k4` / D4 | 5 | 0.371 ± 0.101 | 0.679 ± 0.113 |
| `multigen_rwbal_n7_r1_k4` / C4 | 5 | 0.504 ± 0.088 | 0.736 ± 0.060 |

[multigen_rwbal_n7_r1_k4_stages.pdf](multigen_rwbal_n7_r1_k4_stages.pdf) — Stage evaluations, 5+2 players, single meeting. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_rwbal_n7_r1_k4_matrix.pdf](multigen_rwbal_n7_r1_k4_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_rwbal_n7_r1_k4

**Single-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.185 ± 0.069 | 0.386 ± 0.052 | 0.439 ± 0.021 | 0.451 ± 0.041 | 0.410 ± 0.091 |
| C1 | 0.078 ± 0.005 | 0.524 ± 0.055 | 0.402 ± 0.055 | 0.456 ± 0.082 | 0.440 ± 0.106 |
| C2 | 0.052 ± 0.013 | 0.396 ± 0.039 | 0.598 ± 0.069 | 0.355 ± 0.083 | 0.406 ± 0.097 |
| C3 | 0.067 ± 0.012 | 0.443 ± 0.068 | 0.451 ± 0.035 | 0.546 ± 0.080 | 0.377 ± 0.122 |
| C4 | 0.064 ± 0.024 | 0.445 ± 0.053 | 0.415 ± 0.022 | 0.420 ± 0.057 | 0.509 ± 0.104 |

**No coalition member ejected**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.236 ± 0.079 | 0.675 ± 0.080 | 0.750 ± 0.033 | 0.697 ± 0.065 | 0.698 ± 0.082 |
| C1 | 0.111 ± 0.006 | 0.702 ± 0.038 | 0.731 ± 0.071 | 0.679 ± 0.096 | 0.730 ± 0.074 |
| C2 | 0.076 ± 0.016 | 0.701 ± 0.054 | 0.796 ± 0.031 | 0.676 ± 0.153 | 0.690 ± 0.089 |
| C3 | 0.092 ± 0.014 | 0.709 ± 0.061 | 0.762 ± 0.031 | 0.722 ± 0.066 | 0.684 ± 0.126 |
| C4 | 0.085 ± 0.030 | 0.723 ± 0.051 | 0.719 ± 0.048 | 0.691 ± 0.096 | 0.739 ± 0.070 |

## Loop descriptors: multigen_rwbal_n7_r1_k4



| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_rwbal_n7_r1_k4` / c_stage_mean | 5 | 0.548 ± 0.015 | 0.745 ± 0.021 |
| `multigen_rwbal_n7_r1_k4` / d_stage_mean | 5 | 0.380 ± 0.018 | 0.695 ± 0.028 |
| `multigen_rwbal_n7_r1_k4` / late_minus_early | 5 | -0.035 ± 0.059 | -0.022 ± 0.034 |
| `multigen_rwbal_n7_r1_k4` / matrix_adaptation_gain | 5 | 0.164 ± 0.040 | 0.048 ± 0.023 |
| `multigen_rwbal_n7_r1_k4` / parity_gap | 5 | -0.012 ± 0.059 | -0.027 ± 0.043 |

## Finite observed-policy response gaps: multigen_rwbal_n7_r1_k4

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `multigen_rwbal_n7_r1_k4` / 0 | 5 | 0.000 ± 0.000 |
| `multigen_rwbal_n7_r1_k4` / 1 | 5 | 0.446 ± 0.055 |
| `multigen_rwbal_n7_r1_k4` / 2 | 5 | 0.546 ± 0.067 |
| `multigen_rwbal_n7_r1_k4` / 3 | 5 | 0.480 ± 0.078 |
| `multigen_rwbal_n7_r1_k4` / 4 | 5 | 0.471 ± 0.105 |

[multigen_rwbal_n7_r1_k4_response_gaps.pdf](multigen_rwbal_n7_r1_k4_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_rwfe_n7_r8_k2

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_rwfe_n7_r8_k2` / C0 | 5 | 0.852 ± 0.020 | 1.464 ± 0.088 | 0.472 ± 0.070 | 0.328 ± 0.045 |
| `multigen_rwfe_n7_r8_k2` / D1 | 5 | 0.934 ± 0.016 | 2.202 ± 0.176 | 0.489 ± 0.084 | 0.436 ± 0.078 |
| `multigen_rwfe_n7_r8_k2` / C1 | 5 | 0.901 ± 0.013 | 1.283 ± 0.057 | 0.895 ± 0.016 | 0.538 ± 0.041 |
| `multigen_rwfe_n7_r8_k2` / D2 | 5 | 0.667 ± 0.069 | 0.837 ± 0.152 | 0.619 ± 0.096 | 0.245 ± 0.054 |
| `multigen_rwfe_n7_r8_k2` / C2 | 5 | 0.875 ± 0.031 | 1.185 ± 0.088 | 0.871 ± 0.015 | 0.484 ± 0.039 |

[multigen_rwfe_n7_r8_k2_stages.pdf](multigen_rwfe_n7_r8_k2_stages.pdf) — Stage evaluations, 5+2 players, up to 8 rounds. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_rwfe_n7_r8_k2_matrix.pdf](multigen_rwfe_n7_r8_k2_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, up to 8 rounds). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_rwfe_n7_r8_k2

**At least one false ejection / game**

| Coalition / defender | D0 | D1 | D2 |
|---|---:|---:|---:|
| C0 | 0.855 ± 0.024 | 0.933 ± 0.016 | 0.892 ± 0.038 |
| C1 | 0.186 ± 0.017 | 0.896 ± 0.016 | 0.666 ± 0.069 |
| C2 | 0.224 ± 0.029 | 0.824 ± 0.030 | 0.884 ± 0.032 |

**Mean false ejections / game**

| Coalition / defender | D0 | D1 | D2 |
|---|---:|---:|---:|
| C0 | 1.474 ± 0.081 | 2.192 ± 0.173 | 1.686 ± 0.209 |
| C1 | 0.201 ± 0.020 | 1.282 ± 0.049 | 0.831 ± 0.157 |
| C2 | 0.247 ± 0.036 | 1.128 ± 0.070 | 1.198 ± 0.093 |

**Coalition game wins**

| Coalition / defender | D0 | D1 | D2 |
|---|---:|---:|---:|
| C0 | 0.480 ± 0.078 | 0.484 ± 0.084 | 0.452 ± 0.081 |
| C1 | 0.144 ± 0.025 | 0.886 ± 0.015 | 0.618 ± 0.108 |
| C2 | 0.197 ± 0.023 | 0.873 ± 0.020 | 0.880 ± 0.020 |

**Terminal-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 |
|---|---:|---:|---:|
| C0 | 0.337 ± 0.048 | 0.427 ± 0.082 | 0.348 ± 0.098 |
| C1 | 0.017 ± 0.004 | 0.529 ± 0.043 | 0.244 ± 0.062 |
| C2 | 0.030 ± 0.011 | 0.487 ± 0.040 | 0.496 ± 0.049 |

## Loop descriptors: multigen_rwfe_n7_r8_k2



| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection |
|---|---:|---:|---:|---:|---:|
| `multigen_rwfe_n7_r8_k2` / c_stage_mean | 5 | 0.888 ± 0.020 | 1.234 ± 0.068 | 0.883 ± 0.014 | 0.511 ± 0.038 |
| `multigen_rwfe_n7_r8_k2` / d_stage_mean | 5 | 0.800 ± 0.029 | 1.520 ± 0.039 | 0.554 ± 0.051 | 0.341 ± 0.036 |
| `multigen_rwfe_n7_r8_k2` / late_minus_early | 5 | -0.026 ± 0.025 | -0.098 ± 0.058 | -0.024 ± 0.014 | -0.053 ± 0.025 |
| `multigen_rwfe_n7_r8_k2` / matrix_adaptation_gain | 5 | 0.090 ± 0.032 | -0.272 ± 0.097 | 0.333 ± 0.060 | 0.177 ± 0.059 |

## Finite observed-policy response gaps: multigen_rwfe_n7_r8_k2

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_rwfe_n7_r8_k2` / 0 | 5 | 0.004 ± 0.008 | 0.072 ± 0.088 |
| `multigen_rwfe_n7_r8_k2` / 1 | 5 | 0.748 ± 0.033 | 0.748 ± 0.030 |
| `multigen_rwfe_n7_r8_k2` / 2 | 5 | 0.674 ± 0.029 | 0.683 ± 0.029 |

[multigen_rwfe_n7_r8_k2_response_gaps.pdf](multigen_rwfe_n7_r8_k2_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Generation stages: multigen_rwmeet_n7_r1_k4

Stages use the fixed training sequence C0, D1, C1, … . Values are descriptive across the complete seed group.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_rwmeet_n7_r1_k4` / C0 | 5 | 0.476 ± 0.060 | 0.940 ± 0.015 |
| `multigen_rwmeet_n7_r1_k4` / D1 | 5 | 0.496 ± 0.038 | 0.781 ± 0.007 |
| `multigen_rwmeet_n7_r1_k4` / C1 | 5 | 0.538 ± 0.040 | 0.798 ± 0.024 |
| `multigen_rwmeet_n7_r1_k4` / D2 | 5 | 0.302 ± 0.037 | 0.557 ± 0.049 |
| `multigen_rwmeet_n7_r1_k4` / C2 | 5 | 0.584 ± 0.104 | 0.753 ± 0.068 |
| `multigen_rwmeet_n7_r1_k4` / D3 | 5 | 0.289 ± 0.037 | 0.558 ± 0.047 |
| `multigen_rwmeet_n7_r1_k4` / C3 | 5 | 0.588 ± 0.072 | 0.763 ± 0.058 |
| `multigen_rwmeet_n7_r1_k4` / D4 | 5 | 0.302 ± 0.036 | 0.591 ± 0.057 |
| `multigen_rwmeet_n7_r1_k4` / C4 | 5 | 0.570 ± 0.030 | 0.775 ± 0.034 |

[multigen_rwmeet_n7_r1_k4_stages.pdf](multigen_rwmeet_n7_r1_k4_stages.pdf) — Stage evaluations, 5+2 players, single meeting. The band is between-seed SD; similar levels do not establish stationarity or convergence.

[multigen_rwmeet_n7_r1_k4_matrix.pdf](multigen_rwmeet_n7_r1_k4_matrix.pdf) — Cross-generation evaluation matrix (5+2 players, single meeting). Columns identify the evaluated defender/condition; rows identify the coalition or crew rule. Each cell is mean ± SD across the complete seed group.

## Cross-generation matrices: multigen_rwmeet_n7_r1_k4

**Single-meeting false ejection**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.472 ± 0.052 | 0.499 ± 0.050 | 0.515 ± 0.054 | 0.460 ± 0.019 | 0.478 ± 0.061 |
| C1 | 0.077 ± 0.030 | 0.546 ± 0.050 | 0.291 ± 0.034 | 0.449 ± 0.068 | 0.354 ± 0.049 |
| C2 | 0.162 ± 0.069 | 0.485 ± 0.057 | 0.568 ± 0.101 | 0.302 ± 0.045 | 0.504 ± 0.091 |
| C3 | 0.142 ± 0.025 | 0.509 ± 0.029 | 0.340 ± 0.034 | 0.596 ± 0.080 | 0.294 ± 0.045 |
| C4 | 0.103 ± 0.028 | 0.446 ± 0.015 | 0.433 ± 0.032 | 0.343 ± 0.055 | 0.555 ± 0.049 |

**No coalition member ejected**

| Coalition / defender | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.936 ± 0.016 | 0.780 ± 0.015 | 0.700 ± 0.034 | 0.714 ± 0.079 | 0.738 ± 0.025 |
| C1 | 0.106 ± 0.038 | 0.806 ± 0.033 | 0.554 ± 0.048 | 0.662 ± 0.078 | 0.616 ± 0.043 |
| C2 | 0.274 ± 0.116 | 0.797 ± 0.029 | 0.736 ± 0.066 | 0.578 ± 0.050 | 0.764 ± 0.052 |
| C3 | 0.221 ± 0.072 | 0.816 ± 0.017 | 0.590 ± 0.069 | 0.771 ± 0.051 | 0.591 ± 0.062 |
| C4 | 0.140 ± 0.041 | 0.786 ± 0.020 | 0.659 ± 0.047 | 0.596 ± 0.046 | 0.762 ± 0.037 |

## Loop descriptors: multigen_rwmeet_n7_r1_k4



| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_rwmeet_n7_r1_k4` / c_stage_mean | 5 | 0.570 ± 0.051 | 0.772 ± 0.038 |
| `multigen_rwmeet_n7_r1_k4` / d_stage_mean | 5 | 0.347 ± 0.006 | 0.622 ± 0.029 |
| `multigen_rwmeet_n7_r1_k4` / late_minus_early | 5 | 0.018 ± 0.043 | -0.006 ± 0.022 |
| `multigen_rwmeet_n7_r1_k4` / matrix_adaptation_gain | 5 | 0.220 ± 0.049 | 0.143 ± 0.038 |
| `multigen_rwmeet_n7_r1_k4` / parity_gap | 5 | 0.073 ± 0.057 | 0.024 ± 0.046 |

## Finite observed-policy response gaps: multigen_rwmeet_n7_r1_k4

Gap=max_i M[i,k]−min_j M[k,j] over the complete observed policy pool, including later policies. This is not true exploitability or a convergence certificate. These SDs do not replace the analyzer's separate Monte Carlo bounds.

| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `multigen_rwmeet_n7_r1_k4` / 0 | 5 | 0.048 ± 0.066 |
| `multigen_rwmeet_n7_r1_k4` / 1 | 5 | 0.477 ± 0.070 |
| `multigen_rwmeet_n7_r1_k4` / 2 | 5 | 0.440 ± 0.024 |
| `multigen_rwmeet_n7_r1_k4` / 3 | 5 | 0.454 ± 0.093 |
| `multigen_rwmeet_n7_r1_k4` / 4 | 5 | 0.453 ± 0.062 |

[multigen_rwmeet_n7_r1_k4_response_gaps.pdf](multigen_rwmeet_n7_r1_k4_response_gaps.pdf) — Finite observed-policy response gaps. All observed policy generations define the pool; shading shows seed SD, not Monte Carlo confidence bounds or uncertainty over unobserved policies.

## Coordination controls: ablation_n7_r1

Five complete training seeds per condition. Hidden-partner removes explicit identity and legality/message-availability oracles; ordinary content/outcome inference remains possible. Creator survival is incident-gated per episode.

| Setting / cell | Seeds | Single-meeting false ejection | Coalition false claims | Incident and creator survives | Same-target votes |
|---|---:|---:|---:|---:|---:|
| `ablation_n7_r1` / default | 5 | 0.185 ± 0.071 | 0.374 ± 0.116 | 0.401 ± 0.051 | 0.240 ± 0.096 |
| `ablation_n7_r1` / no_channel | 5 | 0.218 ± 0.071 | 0.339 ± 0.099 | 0.433 ± 0.045 | 0.339 ± 0.120 |
| `ablation_n7_r1` / no_channel_no_partner | 5 | 0.252 ± 0.038 | 0.328 ± 0.065 | 0.388 ± 0.039 | 0.263 ± 0.068 |
| `ablation_n7_r1` / no_partner | 5 | 0.208 ± 0.059 | 0.371 ± 0.088 | 0.374 ± 0.026 | 0.210 ± 0.046 |

## Coordination controls: ablation_n7_r8

Five complete training seeds per condition. Hidden-partner removes explicit identity and legality/message-availability oracles; ordinary content/outcome inference remains possible. Creator survival is incident-gated per episode.

| Setting / cell | Seeds | At least one false ejection / game | Mean false ejections / game | Coalition game wins | Terminal-meeting false ejection | Coalition false claims (terminal meeting) | Incident and creator survives (terminal meeting) | Same-target votes (terminal meeting) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ablation_n7_r8` / default | 5 | 0.973 ± 0.011 | 2.896 ± 0.049 | 0.967 ± 0.010 | 0.966 ± 0.012 | 0.498 ± 0.084 | 0.004 ± 0.003 | 0.735 ± 0.106 |
| `ablation_n7_r8` / no_channel | 5 | 0.983 ± 0.003 | 2.948 ± 0.024 | 0.977 ± 0.004 | 0.977 ± 0.001 | 0.484 ± 0.095 | 0.003 ± 0.004 | 0.626 ± 0.119 |
| `ablation_n7_r8` / no_channel_no_partner | 5 | 0.970 ± 0.014 | 2.908 ± 0.094 | 0.963 ± 0.014 | 0.962 ± 0.014 | 0.447 ± 0.107 | 0.007 ± 0.006 | 0.580 ± 0.127 |
| `ablation_n7_r8` / no_partner | 5 | 0.973 ± 0.007 | 2.860 ± 0.064 | 0.962 ± 0.008 | 0.961 ± 0.012 | 0.554 ± 0.139 | 0.008 ± 0.010 | 0.690 ± 0.134 |

## Defender budgets: r=1, matched five-seed cohort

Every budget, including the 400-update reference, uses the same five seed IDs. Only the defender budget changes.

| Setting / cell | Seeds | Single-meeting false ejection |
|---|---:|---:|
| `f3_budget_d1600_crew5` / coalition0_vs_learned_crew1 | 5 | 0.382 ± 0.055 |
| `f3_budget_d1600_crew5` / coalition1_vs_learned_crew1 | 5 | 0.516 ± 0.039 |
| `f3_budget_d800_crew5` / coalition0_vs_learned_crew1 | 5 | 0.439 ± 0.069 |
| `f3_budget_d800_crew5` / coalition1_vs_learned_crew1 | 5 | 0.567 ± 0.055 |
| `f3_tenseed_crew5` / coalition0_vs_learned_crew1 | 5 | 0.387 ± 0.040 |
| `f3_tenseed_crew5` / coalition1_vs_learned_crew1 | 5 | 0.525 ± 0.048 |

## Reward objectives: r=1, horizon=4, matched five-seed cohort

Reference and reward controls use seeds 0–4 and the same fixed generation horizon. The full ten-seed reference mean is not substituted here.

| Setting / cell | Seeds | Single-meeting false ejection | No coalition member ejected |
|---|---:|---:|---:|
| `multigen_none_n7_r1_k10` / c_stage_mean | 5 | 0.551 ± 0.040 | 0.743 ± 0.018 |
| `multigen_none_n7_r1_k10` / d_stage_mean | 5 | 0.267 ± 0.049 | 0.514 ± 0.058 |
| `multigen_none_n7_r1_k10` / matrix_adaptation_gain | 5 | 0.287 ± 0.075 | 0.234 ± 0.074 |
| `multigen_rwbal_n7_r1_k4` / c_stage_mean | 5 | 0.548 ± 0.015 | 0.745 ± 0.021 |
| `multigen_rwbal_n7_r1_k4` / d_stage_mean | 5 | 0.380 ± 0.018 | 0.695 ± 0.028 |
| `multigen_rwbal_n7_r1_k4` / matrix_adaptation_gain | 5 | 0.164 ± 0.040 | 0.048 ± 0.023 |
| `multigen_rwmeet_n7_r1_k4` / c_stage_mean | 5 | 0.570 ± 0.051 | 0.772 ± 0.038 |
| `multigen_rwmeet_n7_r1_k4` / d_stage_mean | 5 | 0.347 ± 0.006 | 0.622 ± 0.029 |
| `multigen_rwmeet_n7_r1_k4` / matrix_adaptation_gain | 5 | 0.220 ± 0.049 | 0.143 ± 0.038 |

## Defender budgets: r=8, matched five-seed cohort

Every budget, including the 400-update reference, uses the same five seed IDs. Only the defender budget changes.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `f4_budget_d800_crew5` / coalition0_vs_learned_crew1 | 5 | 0.745 ± 0.048 | 0.165 ± 0.078 |
| `f4_budget_d800_crew5` / coalition1_vs_learned_crew1 | 5 | 0.852 ± 0.034 | 0.848 ± 0.029 |
| `f4_tenseed_crew5` / coalition0_vs_learned_crew1 | 5 | 0.840 ± 0.050 | 0.354 ± 0.115 |
| `f4_tenseed_crew5` / coalition1_vs_learned_crew1 | 5 | 0.846 ± 0.012 | 0.854 ± 0.029 |

## Reward objectives: r=8, horizon=2, matched five-seed cohort

Reference and reward controls use seeds 0–4 and the same fixed generation horizon. The full ten-seed reference mean is not substituted here.

| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_none_n7_r8_k10` / c_stage_mean | 5 | 0.854 ± 0.020 | 0.810 ± 0.043 |
| `multigen_none_n7_r8_k10` / d_stage_mean | 5 | 0.757 ± 0.060 | 0.374 ± 0.076 |
| `multigen_none_n7_r8_k10` / matrix_adaptation_gain | 5 | 0.101 ± 0.055 | 0.440 ± 0.095 |
| `multigen_rwfe_n7_r8_k2` / c_stage_mean | 5 | 0.888 ± 0.020 | 0.883 ± 0.014 |
| `multigen_rwfe_n7_r8_k2` / d_stage_mean | 5 | 0.800 ± 0.029 | 0.554 ± 0.051 |
| `multigen_rwfe_n7_r8_k2` / matrix_adaptation_gain | 5 | 0.090 ± 0.032 | 0.333 ± 0.060 |

## Dependence-aware loop: matched ten-seed cohort, horizon=4



| Setting / cell | Seeds | At least one false ejection / game | Coalition game wins |
|---|---:|---:|---:|
| `multigen_dependence_n7_r8_k4` / c_stage_mean | 10 | 0.890 ± 0.007 | 0.789 ± 0.026 |
| `multigen_dependence_n7_r8_k4` / d_stage_mean | 10 | 0.763 ± 0.047 | 0.348 ± 0.054 |
| `multigen_dependence_n7_r8_k4` / matrix_adaptation_gain | 10 | 0.129 ± 0.046 | 0.445 ± 0.070 |
| `multigen_none_n7_r8_k10` / c_stage_mean | 10 | 0.847 ± 0.009 | 0.817 ± 0.025 |
| `multigen_none_n7_r8_k10` / d_stage_mean | 10 | 0.658 ± 0.051 | 0.309 ± 0.065 |
| `multigen_none_n7_r8_k10` / matrix_adaptation_gain | 10 | 0.191 ± 0.051 | 0.509 ± 0.086 |

## Paired effects

Signed differences are defined by the analyzer's recorded contrasts. Intervals are pointwise 95% seed-bootstrap intervals. Exact mean sign-flip p-values retain difference magnitudes; Holm correction uses the complete declared family. Small seed counts limit attainable p-values, so nonsignificance does not establish absence or equivalence.

### ablation

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| ablation_n7_r1 / no_channel minus default / false_ejection_rate | 5 | 0.033 ± 0.139 | [-0.073, 0.141] | 0.625 | 1 |
| ablation_n7_r1 / no_partner minus default / false_ejection_rate | 5 | 0.023 ± 0.106 | [-0.058, 0.103] | 0.6875 | 1 |
| ablation_n7_r1 / no_channel_no_partner minus default / false_ejection_rate | 5 | 0.066 ± 0.081 | [0.003, 0.130] | 0.1875 | 1 |
| ablation_n7_r8 / no_channel minus default / any_false_ejection_rate | 5 | 0.011 ± 0.010 | [0.003, 0.019] | 0.0625 | 0.5625 |
| ablation_n7_r8 / no_partner minus default / any_false_ejection_rate | 5 | 0.001 ± 0.013 | [-0.009, 0.011] | 0.9375 | 1 |
| ablation_n7_r8 / no_channel_no_partner minus default / any_false_ejection_rate | 5 | -0.002 ± 0.005 | [-0.006, 0.001] | 0.375 | 1 |
| ablation_n7_r8 / no_channel minus default / coalition_game_win_rate | 5 | 0.010 ± 0.010 | [0.004, 0.019] | 0.125 | 1 |
| ablation_n7_r8 / no_partner minus default / coalition_game_win_rate | 5 | -0.005 ± 0.016 | [-0.017, 0.007] | 0.5 | 1 |
| ablation_n7_r8 / no_channel_no_partner minus default / coalition_game_win_rate | 5 | -0.004 ± 0.008 | [-0.009, 0.002] | 0.3125 | 1 |

### ablation_behavior

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| ablation_n7_r1 / no_channel minus default / same_target_vote_rate | 5 | 0.099 ± 0.203 | [-0.085, 0.222] | 0.3125 | 1 |
| ablation_n7_r1 / no_partner minus default / same_target_vote_rate | 5 | -0.031 ± 0.088 | [-0.101, 0.039] | 0.5 | 1 |
| ablation_n7_r1 / no_channel_no_partner minus default / same_target_vote_rate | 5 | 0.023 ± 0.138 | [-0.085, 0.130] | 0.625 | 1 |
| ablation_n7_r1 / no_channel minus default / coalition_false_claim_rate | 5 | -0.035 ± 0.191 | [-0.184, 0.114] | 0.6875 | 1 |
| ablation_n7_r1 / no_partner minus default / coalition_false_claim_rate | 5 | -0.003 ± 0.180 | [-0.132, 0.148] | 1 | 1 |
| ablation_n7_r1 / no_channel_no_partner minus default / coalition_false_claim_rate | 5 | -0.046 ± 0.063 | [-0.102, -0.010] | 0.125 | 1 |
| ablation_n7_r1 / no_channel minus default / creator_survival | 5 | 0.032 ± 0.092 | [-0.040, 0.103] | 0.5 | 1 |
| ablation_n7_r1 / no_partner minus default / creator_survival | 5 | -0.027 ± 0.044 | [-0.060, 0.009] | 0.25 | 1 |
| ablation_n7_r1 / no_channel_no_partner minus default / creator_survival | 5 | -0.013 ± 0.074 | [-0.070, 0.043] | 0.6875 | 1 |
| ablation_n7_r8 / no_channel minus default / same_target_vote_rate | 5 | -0.110 ± 0.162 | [-0.229, 0.027] | 0.25 | 1 |
| ablation_n7_r8 / no_partner minus default / same_target_vote_rate | 5 | -0.045 ± 0.202 | [-0.211, 0.089] | 0.8125 | 1 |
| ablation_n7_r8 / no_channel_no_partner minus default / same_target_vote_rate | 5 | -0.155 ± 0.108 | [-0.236, -0.060] | 0.125 | 1 |
| ablation_n7_r8 / no_channel minus default / coalition_false_claim_rate | 5 | -0.014 ± 0.171 | [-0.149, 0.121] | 0.75 | 1 |
| ablation_n7_r8 / no_partner minus default / coalition_false_claim_rate | 5 | 0.056 ± 0.067 | [0.001, 0.108] | 0.125 | 1 |
| ablation_n7_r8 / no_channel_no_partner minus default / coalition_false_claim_rate | 5 | -0.051 ± 0.113 | [-0.138, 0.033] | 0.5 | 1 |
| ablation_n7_r8 / no_channel minus default / creator_survival | 5 | -0.001 ± 0.006 | [-0.005, 0.004] | 0.875 | 1 |
| ablation_n7_r8 / no_partner minus default / creator_survival | 5 | 0.004 ± 0.009 | [-0.003, 0.012] | 0.4375 | 1 |
| ablation_n7_r8 / no_channel_no_partner minus default / creator_survival | 5 | 0.003 ± 0.006 | [-0.002, 0.007] | 0.4375 | 1 |

### budget

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f3_budget_d1600_crew5 / coalition0_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | -0.005 ± 0.073 | [-0.059, 0.055] | 0.8125 | 1 |
| f3_budget_d1600_crew5 / coalition1_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | -0.009 ± 0.064 | [-0.063, 0.038] | 0.875 | 1 |
| f3_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | 0.052 ± 0.087 | [-0.007, 0.124] | 0.25 | 1 |
| f3_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / false_ejection_rate | 5 | 0.042 ± 0.096 | [-0.042, 0.119] | 0.4375 | 1 |
| f4_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / any_false_ejection_rate | 5 | -0.095 ± 0.075 | [-0.153, -0.040] | 0.125 | 0.875 |
| f4_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / any_false_ejection_rate | 5 | 0.006 ± 0.041 | [-0.025, 0.037] | 0.75 | 1 |
| f4_budget_d800_crew5 / coalition0_vs_learned_crew1 extended minus 400 / coalition_game_win_rate | 5 | -0.189 ± 0.138 | [-0.298, -0.089] | 0.0625 | 0.5 |
| f4_budget_d800_crew5 / coalition1_vs_learned_crew1 extended minus 400 / coalition_game_win_rate | 5 | -0.006 ± 0.046 | [-0.045, 0.028] | 0.8125 | 1 |

### dependence_counterattack

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| counterattack_n5_r1 / rule adaptation / false_ejection_rate | 5 | 0.020 ± 0.030 | [-0.005, 0.040] | 0.1875 | 1 |
| counterattack_n5_r1 / rule matched adaptive difference / false_ejection_rate | 5 | 0.016 ± 0.042 | [-0.021, 0.043] | 0.5 | 1 |
| counterattack_n5_r1 / rule transfer / false_ejection_rate | 5 | -0.016 ± 0.043 | [-0.050, 0.016] | 0.4375 | 1 |
| counterattack_n5_r1 / both adaptation / false_ejection_rate | 5 | 0.459 ± 0.041 | [0.428, 0.492] | 0.0625 | 0.75 |
| counterattack_n5_r1 / both matched adaptive difference / false_ejection_rate | 5 | 0.662 ± 0.055 | [0.619, 0.704] | 0.0625 | 0.75 |
| counterattack_n5_r1 / both transfer / false_ejection_rate | 5 | -0.185 ± 0.043 | [-0.220, -0.152] | 0.0625 | 0.75 |
| counterattack_n7_r1 / rule adaptation / false_ejection_rate | 10 | -0.012 ± 0.076 | [-0.052, 0.037] | 0.6309 | 1 |
| counterattack_n7_r1 / rule matched adaptive difference / false_ejection_rate | 10 | -0.025 ± 0.078 | [-0.066, 0.025] | 0.3516 | 1 |
| counterattack_n7_r1 / rule transfer / false_ejection_rate | 10 | -0.021 ± 0.072 | [-0.059, 0.025] | 0.3945 | 1 |
| counterattack_n7_r1 / both adaptation / false_ejection_rate | 10 | -0.015 ± 0.062 | [-0.051, 0.021] | 0.4648 | 1 |
| counterattack_n7_r1 / both matched adaptive difference / false_ejection_rate | 10 | -0.005 ± 0.063 | [-0.042, 0.032] | 0.8203 | 1 |
| counterattack_n7_r1 / both transfer / false_ejection_rate | 10 | -0.025 ± 0.061 | [-0.062, 0.009] | 0.2168 | 1 |

### dependence_loop

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| multigen_dependence_n7_r8_k4 / c_stage_mean minus reference h4 / any_false_ejection_rate | 10 | 0.042 ± 0.013 | [0.035, 0.049] | 0.001953 | 0.01172 |
| multigen_dependence_n7_r8_k4 / d_stage_mean minus reference h4 / any_false_ejection_rate | 10 | 0.104 ± 0.052 | [0.073, 0.135] | 0.001953 | 0.01172 |
| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain minus reference h4 / any_false_ejection_rate | 10 | -0.062 ± 0.047 | [-0.090, -0.034] | 0.005859 | 0.02344 |
| multigen_dependence_n7_r8_k4 / c_stage_mean minus reference h4 / coalition_game_win_rate | 10 | -0.029 ± 0.029 | [-0.046, -0.012] | 0.01367 | 0.04102 |
| multigen_dependence_n7_r8_k4 / d_stage_mean minus reference h4 / coalition_game_win_rate | 10 | 0.040 ± 0.077 | [-0.001, 0.089] | 0.1387 | 0.1387 |
| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain minus reference h4 / coalition_game_win_rate | 10 | -0.064 ± 0.094 | [-0.123, -0.016] | 0.02344 | 0.04688 |

### f1_learning

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f1_tenseed_crew3 / final minus initial / false_ejection_rate | 10 | 0.226 ± 0.050 | [0.197, 0.255] | 0.001953 | 0.01172 |
| f1_tenseed_crew3 / final minus initial / coalition_favorable_rate | 10 | 0.490 ± 0.059 | [0.454, 0.523] | 0.001953 | 0.01172 |
| f1_tenseed_crew5 / final minus initial / false_ejection_rate | 10 | 0.140 ± 0.017 | [0.130, 0.150] | 0.001953 | 0.01172 |
| f1_tenseed_crew5 / final minus initial / coalition_favorable_rate | 10 | 0.160 ± 0.020 | [0.149, 0.173] | 0.001953 | 0.01172 |
| f1_tenseed_crew7 / final minus initial / false_ejection_rate | 10 | 0.103 ± 0.028 | [0.087, 0.121] | 0.001953 | 0.01172 |
| f1_tenseed_crew7 / final minus initial / coalition_favorable_rate | 10 | 0.110 ± 0.029 | [0.095, 0.129] | 0.001953 | 0.01172 |

### f2_condition_effects

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f2_tenseed_crew3 / lone_liar / mean minus truthful / mean / false_ejection_rate | 10 | 0.328 ± 0.014 | [0.320, 0.337] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / median minus truthful / median / false_ejection_rate | 10 | 0.108 ± 0.013 | [0.100, 0.116] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.108 ± 0.013 | [0.100, 0.116] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.204 ± 0.011 | [0.197, 0.210] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.178 ± 0.012 | [0.171, 0.186] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.210 ± 0.012 | [0.202, 0.217] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / lone_liar / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | 0.167 ± 0.019 | [0.155, 0.177] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / mean minus truthful / mean / false_ejection_rate | 10 | 0.512 ± 0.014 | [0.503, 0.520] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / median minus truthful / median / false_ejection_rate | 10 | 0.301 ± 0.016 | [0.291, 0.311] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.301 ± 0.016 | [0.291, 0.311] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.504 ± 0.015 | [0.495, 0.513] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.524 ± 0.012 | [0.516, 0.531] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.451 ± 0.013 | [0.443, 0.458] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / alibi / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | 0.237 ± 0.025 | [0.221, 0.250] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / mean minus truthful / mean / false_ejection_rate | 10 | 0.542 ± 0.015 | [0.533, 0.550] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / median minus truthful / median / false_ejection_rate | 10 | 0.394 ± 0.021 | [0.382, 0.405] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.394 ± 0.021 | [0.381, 0.406] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.519 ± 0.017 | [0.509, 0.529] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.535 ± 0.016 | [0.526, 0.544] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.535 ± 0.016 | [0.526, 0.544] | 0.001953 | 0.123 |
| f2_tenseed_crew3 / framer / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | 0.107 ± 0.016 | [0.097, 0.115] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / mean minus truthful / mean / false_ejection_rate | 10 | 0.131 ± 0.011 | [0.124, 0.138] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / median minus truthful / median / false_ejection_rate | 10 | 0.041 ± 0.011 | [0.034, 0.047] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.047 ± 0.013 | [0.039, 0.054] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.278 ± 0.014 | [0.269, 0.286] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.159 ± 0.012 | [0.151, 0.165] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.194 ± 0.014 | [0.185, 0.201] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / lone_liar / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.179 ± 0.015 | [-0.187, -0.170] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / mean minus truthful / mean / false_ejection_rate | 10 | 0.327 ± 0.015 | [0.319, 0.336] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / median minus truthful / median / false_ejection_rate | 10 | 0.033 ± 0.014 | [0.025, 0.042] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.108 ± 0.019 | [0.096, 0.118] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.428 ± 0.012 | [0.421, 0.435] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.392 ± 0.010 | [0.386, 0.397] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.326 ± 0.010 | [0.321, 0.332] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / alibi / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.223 ± 0.019 | [-0.234, -0.212] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / mean minus truthful / mean / false_ejection_rate | 10 | 0.411 ± 0.010 | [0.405, 0.416] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / median minus truthful / median / false_ejection_rate | 10 | 0.034 ± 0.013 | [0.026, 0.042] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.219 ± 0.022 | [0.206, 0.231] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.400 ± 0.016 | [0.390, 0.409] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.403 ± 0.015 | [0.395, 0.412] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.479 ± 0.018 | [0.470, 0.491] | 0.001953 | 0.123 |
| f2_tenseed_crew5 / framer / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.216 ± 0.014 | [-0.224, -0.207] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / mean minus truthful / mean / false_ejection_rate | 10 | 0.100 ± 0.009 | [0.095, 0.105] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / median minus truthful / median / false_ejection_rate | 10 | 0.049 ± 0.011 | [0.043, 0.056] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.038 ± 0.008 | [0.033, 0.042] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.300 ± 0.009 | [0.295, 0.306] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.151 ± 0.009 | [0.145, 0.156] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.178 ± 0.012 | [0.171, 0.186] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / lone_liar / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.323 ± 0.015 | [-0.331, -0.314] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / mean minus truthful / mean / false_ejection_rate | 10 | 0.212 ± 0.016 | [0.202, 0.221] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / median minus truthful / median / false_ejection_rate | 10 | 0.053 ± 0.014 | [0.045, 0.062] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.107 ± 0.013 | [0.099, 0.115] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.451 ± 0.013 | [0.444, 0.459] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.378 ± 0.014 | [0.370, 0.386] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.340 ± 0.016 | [0.332, 0.350] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / alibi / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.328 ± 0.016 | [-0.337, -0.318] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / mean minus truthful / mean / false_ejection_rate | 10 | 0.260 ± 0.010 | [0.255, 0.266] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / median minus truthful / median / false_ejection_rate | 10 | 0.018 ± 0.015 | [0.009, 0.027] | 0.007812 | 0.123 |
| f2_tenseed_crew7 / framer / trimmed minus truthful / trimmed / false_ejection_rate | 10 | 0.155 ± 0.014 | [0.148, 0.164] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / sharp_credibility minus truthful / sharp_credibility / false_ejection_rate | 10 | 0.314 ± 0.014 | [0.307, 0.323] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / soft_credibility minus truthful / soft_credibility / false_ejection_rate | 10 | 0.255 ± 0.010 | [0.249, 0.261] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / dependence_aware minus truthful / dependence_aware / false_ejection_rate | 10 | 0.355 ± 0.012 | [0.348, 0.362] | 0.001953 | 0.123 |
| f2_tenseed_crew7 / framer / hypothesis minus truthful / hypothesis / false_ejection_rate | 10 | -0.305 ± 0.015 | [-0.313, -0.296] | 0.001953 | 0.123 |

### f2_rule_effects

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f2_tenseed_crew3 / alibi / mean minus alibi / soft_credibility / false_ejection_rate | 10 | -0.012 ± 0.004 | [-0.014, -0.009] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / alibi / median minus alibi / soft_credibility / false_ejection_rate | 10 | -0.085 ± 0.009 | [-0.090, -0.080] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / alibi / trimmed minus alibi / soft_credibility / false_ejection_rate | 10 | -0.085 ± 0.009 | [-0.090, -0.080] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / alibi / sharp_credibility minus alibi / soft_credibility / false_ejection_rate | 10 | -0.020 ± 0.006 | [-0.023, -0.016] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / alibi / dependence_aware minus alibi / soft_credibility / false_ejection_rate | 10 | -0.073 ± 0.008 | [-0.078, -0.068] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 10 | -0.042 ± 0.013 | [-0.050, -0.035] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / framer / mean minus framer / soft_credibility / false_ejection_rate | 10 | 0.006 ± 0.003 | [0.005, 0.008] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / framer / median minus framer / soft_credibility / false_ejection_rate | 10 | -0.004 ± 0.005 | [-0.006, -0.001] | 0.04688 | 0.1875 |
| f2_tenseed_crew3 / framer / trimmed minus framer / soft_credibility / false_ejection_rate | 10 | -0.004 ± 0.005 | [-0.006, -0.001] | 0.04688 | 0.1875 |
| f2_tenseed_crew3 / framer / sharp_credibility minus framer / soft_credibility / false_ejection_rate | 10 | -0.016 ± 0.004 | [-0.019, -0.014] | 0.001953 | 0.07031 |
| f2_tenseed_crew3 / framer / dependence_aware minus framer / soft_credibility / false_ejection_rate | 10 | 0.000 ± 0.000 | [0.000, 0.000] | 1 | 1 |
| f2_tenseed_crew3 / framer / hypothesis minus framer / soft_credibility / false_ejection_rate | 10 | -0.184 ± 0.017 | [-0.193, -0.173] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / alibi / mean minus alibi / soft_credibility / false_ejection_rate | 10 | -0.065 ± 0.014 | [-0.072, -0.056] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / alibi / median minus alibi / soft_credibility / false_ejection_rate | 10 | 0.030 ± 0.020 | [0.018, 0.042] | 0.003906 | 0.07031 |
| f2_tenseed_crew5 / alibi / trimmed minus alibi / soft_credibility / false_ejection_rate | 10 | 0.028 ± 0.017 | [0.018, 0.038] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / alibi / sharp_credibility minus alibi / soft_credibility / false_ejection_rate | 10 | 0.036 ± 0.011 | [0.029, 0.042] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / alibi / dependence_aware minus alibi / soft_credibility / false_ejection_rate | 10 | -0.066 ± 0.009 | [-0.071, -0.060] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 10 | -0.245 ± 0.017 | [-0.254, -0.235] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / framer / mean minus framer / soft_credibility / false_ejection_rate | 10 | 0.008 ± 0.009 | [0.002, 0.012] | 0.02734 | 0.1641 |
| f2_tenseed_crew5 / framer / median minus framer / soft_credibility / false_ejection_rate | 10 | 0.020 ± 0.019 | [0.007, 0.030] | 0.02148 | 0.1504 |
| f2_tenseed_crew5 / framer / trimmed minus framer / soft_credibility / false_ejection_rate | 10 | 0.128 ± 0.013 | [0.120, 0.136] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / framer / sharp_credibility minus framer / soft_credibility / false_ejection_rate | 10 | -0.003 ± 0.007 | [-0.007, 0.002] | 0.2969 | 0.5938 |
| f2_tenseed_crew5 / framer / dependence_aware minus framer / soft_credibility / false_ejection_rate | 10 | 0.076 ± 0.010 | [0.071, 0.082] | 0.001953 | 0.07031 |
| f2_tenseed_crew5 / framer / hypothesis minus framer / soft_credibility / false_ejection_rate | 10 | -0.249 ± 0.018 | [-0.260, -0.239] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / mean minus alibi / soft_credibility / false_ejection_rate | 10 | -0.166 ± 0.011 | [-0.172, -0.159] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / median minus alibi / soft_credibility / false_ejection_rate | 10 | 0.179 ± 0.018 | [0.169, 0.190] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / trimmed minus alibi / soft_credibility / false_ejection_rate | 10 | 0.108 ± 0.019 | [0.097, 0.120] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / sharp_credibility minus alibi / soft_credibility / false_ejection_rate | 10 | 0.073 ± 0.009 | [0.069, 0.079] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / dependence_aware minus alibi / soft_credibility / false_ejection_rate | 10 | -0.038 ± 0.013 | [-0.045, -0.030] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 10 | -0.313 ± 0.013 | [-0.321, -0.306] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / framer / mean minus framer / soft_credibility / false_ejection_rate | 10 | 0.005 ± 0.007 | [0.002, 0.009] | 0.03711 | 0.1855 |
| f2_tenseed_crew7 / framer / median minus framer / soft_credibility / false_ejection_rate | 10 | 0.267 ± 0.015 | [0.258, 0.276] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / framer / trimmed minus framer / soft_credibility / false_ejection_rate | 10 | 0.280 ± 0.018 | [0.270, 0.292] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / framer / sharp_credibility minus framer / soft_credibility / false_ejection_rate | 10 | 0.059 ± 0.006 | [0.056, 0.063] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / framer / dependence_aware minus framer / soft_credibility / false_ejection_rate | 10 | 0.099 ± 0.009 | [0.094, 0.105] | 0.001953 | 0.07031 |
| f2_tenseed_crew7 / framer / hypothesis minus framer / soft_credibility / false_ejection_rate | 10 | -0.168 ± 0.011 | [-0.174, -0.161] | 0.001953 | 0.07031 |

### f3_cycle

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f3_tenseed_crew3 / defender effect / false_ejection_rate | 10 | -0.275 ± 0.065 | [-0.315, -0.238] | 0.001953 | 0.01758 |
| f3_tenseed_crew3 / adaptation / false_ejection_rate | 10 | -0.048 ± 0.090 | [-0.099, 0.007] | 0.1309 | 0.1309 |
| f3_tenseed_crew3 / net change / false_ejection_rate | 10 | 0.227 ± 0.096 | [0.171, 0.283] | 0.001953 | 0.01758 |
| f3_tenseed_crew5 / defender effect / false_ejection_rate | 10 | -0.209 ± 0.071 | [-0.250, -0.168] | 0.001953 | 0.01758 |
| f3_tenseed_crew5 / adaptation / false_ejection_rate | 10 | 0.135 ± 0.044 | [0.110, 0.161] | 0.001953 | 0.01758 |
| f3_tenseed_crew5 / net change / false_ejection_rate | 10 | 0.345 ± 0.077 | [0.303, 0.392] | 0.001953 | 0.01758 |
| f3_tenseed_crew7 / defender effect / false_ejection_rate | 10 | -0.264 ± 0.055 | [-0.296, -0.231] | 0.001953 | 0.01758 |
| f3_tenseed_crew7 / adaptation / false_ejection_rate | 10 | 0.112 ± 0.061 | [0.077, 0.149] | 0.001953 | 0.01758 |
| f3_tenseed_crew7 / net change / false_ejection_rate | 10 | 0.376 ± 0.047 | [0.349, 0.404] | 0.001953 | 0.01758 |

### f4_cycle

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| f4_tenseed_crew3 / defender effect / any_false_ejection_rate | 10 | 0.275 ± 0.267 | [0.131, 0.443] | 0.003906 | 0.05273 |
| f4_tenseed_crew3 / adaptation / any_false_ejection_rate | 10 | 0.234 ± 0.242 | [0.087, 0.373] | 0.02539 | 0.1523 |
| f4_tenseed_crew3 / net change / any_false_ejection_rate | 10 | -0.041 ± 0.307 | [-0.227, 0.135] | 0.6738 | 1 |
| f4_tenseed_crew3 / defender effect / coalition_game_win_rate | 10 | 0.715 ± 0.256 | [0.556, 0.860] | 0.001953 | 0.05273 |
| f4_tenseed_crew3 / adaptation / coalition_game_win_rate | 10 | 0.659 ± 0.235 | [0.518, 0.792] | 0.001953 | 0.05273 |
| f4_tenseed_crew3 / net change / coalition_game_win_rate | 10 | -0.056 ± 0.060 | [-0.095, -0.025] | 0.001953 | 0.05273 |
| f4_tenseed_crew3 / defender effect / mean_false_ejections | 10 | 0.302 ± 0.341 | [0.125, 0.523] | 0.007812 | 0.05469 |
| f4_tenseed_crew3 / adaptation / mean_false_ejections | 10 | 0.215 ± 0.303 | [0.034, 0.387] | 0.04102 | 0.2051 |
| f4_tenseed_crew3 / net change / mean_false_ejections | 10 | -0.086 ± 0.414 | [-0.337, 0.148] | 0.502 | 1 |
| f4_tenseed_crew5 / defender effect / any_false_ejection_rate | 10 | 0.135 ± 0.050 | [0.107, 0.166] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / adaptation / any_false_ejection_rate | 10 | 0.007 ± 0.046 | [-0.019, 0.035] | 0.6191 | 1 |
| f4_tenseed_crew5 / net change / any_false_ejection_rate | 10 | -0.127 ± 0.019 | [-0.138, -0.115] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / defender effect / coalition_game_win_rate | 10 | 0.607 ± 0.115 | [0.544, 0.678] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / adaptation / coalition_game_win_rate | 10 | 0.491 ± 0.109 | [0.432, 0.559] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / net change / coalition_game_win_rate | 10 | -0.116 ± 0.027 | [-0.132, -0.100] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / defender effect / mean_false_ejections | 10 | 1.273 ± 0.205 | [1.158, 1.395] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / adaptation / mean_false_ejections | 10 | -0.425 ± 0.171 | [-0.524, -0.324] | 0.001953 | 0.05273 |
| f4_tenseed_crew5 / net change / mean_false_ejections | 10 | -1.698 ± 0.091 | [-1.749, -1.642] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / defender effect / any_false_ejection_rate | 10 | 0.031 ± 0.010 | [0.024, 0.037] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / adaptation / any_false_ejection_rate | 10 | 0.000 ± 0.007 | [-0.004, 0.004] | 1 | 1 |
| f4_tenseed_crew7 / net change / any_false_ejection_rate | 10 | -0.031 ± 0.011 | [-0.038, -0.024] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / defender effect / coalition_game_win_rate | 10 | 0.659 ± 0.030 | [0.642, 0.676] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / adaptation / coalition_game_win_rate | 10 | 0.510 ± 0.019 | [0.500, 0.522] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / net change / coalition_game_win_rate | 10 | -0.149 ± 0.022 | [-0.162, -0.137] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / defender effect / mean_false_ejections | 10 | 1.866 ± 0.109 | [1.799, 1.926] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / adaptation / mean_false_ejections | 10 | -1.007 ± 0.082 | [-1.056, -0.959] | 0.001953 | 0.05273 |
| f4_tenseed_crew7 / net change / mean_false_ejections | 10 | -2.873 ± 0.087 | [-2.926, -2.822] | 0.001953 | 0.05273 |

### hypothesis_adaptation

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| counterattack_hyp_n7_r1 / mean adaptation / false_ejection_rate | 10 | -0.038 ± 0.064 | [-0.075, 0.001] | 0.09375 | 0.5625 |
| counterattack_hyp_n7_r1 / mean matched adaptive difference / false_ejection_rate | 10 | -0.090 ± 0.064 | [-0.126, -0.050] | 0.003906 | 0.04297 |
| counterattack_hyp_n7_r1 / mean transfer / false_ejection_rate | 10 | -0.046 ± 0.066 | [-0.083, -0.005] | 0.06641 | 0.5625 |
| counterattack_hyp_n7_r1 / hypothesis adaptation / false_ejection_rate | 10 | 0.296 ± 0.076 | [0.252, 0.342] | 0.001953 | 0.02734 |
| counterattack_hyp_n7_r1 / hypothesis matched adaptive difference / false_ejection_rate | 10 | 0.228 ± 0.083 | [0.181, 0.278] | 0.001953 | 0.02734 |
| counterattack_hyp_n7_r1 / hypothesis transfer / false_ejection_rate | 10 | -0.097 ± 0.108 | [-0.160, -0.033] | 0.02344 | 0.2344 |
| counterattack_hyp_n7_r1 / hypothesis / hypothesis minus mean / mean / false_ejection_rate | 10 | 0.317 ± 0.080 | [0.271, 0.365] | 0.001953 | 0.02734 |
| counterattack_hyp_n9_r1 / mean adaptation / false_ejection_rate | 5 | 0.003 ± 0.060 | [-0.036, 0.056] | 0.9375 | 1 |
| counterattack_hyp_n9_r1 / mean matched adaptive difference / false_ejection_rate | 5 | -0.063 ± 0.053 | [-0.097, -0.018] | 0.125 | 0.5625 |
| counterattack_hyp_n9_r1 / mean transfer / false_ejection_rate | 5 | -0.013 ± 0.056 | [-0.052, 0.033] | 0.6875 | 1 |
| counterattack_hyp_n9_r1 / hypothesis adaptation / false_ejection_rate | 5 | 0.493 ± 0.040 | [0.461, 0.523] | 0.0625 | 0.5625 |
| counterattack_hyp_n9_r1 / hypothesis matched adaptive difference / false_ejection_rate | 5 | 0.402 ± 0.024 | [0.383, 0.421] | 0.0625 | 0.5625 |
| counterattack_hyp_n9_r1 / hypothesis transfer / false_ejection_rate | 5 | -0.172 ± 0.017 | [-0.186, -0.161] | 0.0625 | 0.5625 |
| counterattack_hyp_n9_r1 / hypothesis / hypothesis minus mean / mean / false_ejection_rate | 5 | 0.465 ± 0.066 | [0.407, 0.512] | 0.0625 | 0.5625 |

### loop_dynamics

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain / any_false_ejection_rate | 10 | 0.129 ± 0.046 | [0.103, 0.157] | 0.001953 | 0.04883 |
| multigen_dependence_n7_r8_k4 / late_minus_early / any_false_ejection_rate | 10 | -0.011 ± 0.018 | [-0.021, -0.001] | 0.1074 | 0.8594 |
| multigen_dependence_n7_r8_k4 / parity_gap / any_false_ejection_rate | 10 | -0.008 ± 0.040 | [-0.031, 0.015] | 0.5273 | 1 |
| multigen_dependence_n7_r8_k4 / matrix_adaptation_gain / coalition_game_win_rate | 10 | 0.445 ± 0.070 | [0.405, 0.486] | 0.001953 | 0.04883 |
| multigen_dependence_n7_r8_k4 / late_minus_early / coalition_game_win_rate | 10 | -0.064 ± 0.047 | [-0.092, -0.036] | 0.003906 | 0.07031 |
| multigen_dependence_n7_r8_k4 / parity_gap / coalition_game_win_rate | 10 | 0.007 ± 0.112 | [-0.057, 0.075] | 0.8555 | 1 |
| multigen_none_n5_r8_k10 / matrix_adaptation_gain / any_false_ejection_rate | 10 | 0.255 ± 0.062 | [0.219, 0.292] | 0.001953 | 0.04883 |
| multigen_none_n5_r8_k10 / late_minus_early / any_false_ejection_rate | 10 | -0.061 ± 0.102 | [-0.120, -0.001] | 0.08789 | 0.791 |
| multigen_none_n5_r8_k10 / parity_gap / any_false_ejection_rate | 10 | 0.048 ± 0.066 | [0.011, 0.088] | 0.03906 | 0.5078 |
| multigen_none_n5_r8_k10 / matrix_adaptation_gain / coalition_game_win_rate | 10 | 0.576 ± 0.050 | [0.549, 0.607] | 0.001953 | 0.04883 |
| multigen_none_n5_r8_k10 / late_minus_early / coalition_game_win_rate | 10 | 0.002 ± 0.042 | [-0.022, 0.027] | 0.9043 | 1 |
| multigen_none_n5_r8_k10 / parity_gap / coalition_game_win_rate | 10 | 0.133 ± 0.163 | [0.042, 0.232] | 0.02148 | 0.3652 |
| multigen_none_n7_r1_k10 / matrix_adaptation_gain / false_ejection_rate | 10 | 0.300 ± 0.032 | [0.282, 0.319] | 0.001953 | 0.04883 |
| multigen_none_n7_r1_k10 / late_minus_early / false_ejection_rate | 10 | -0.013 ± 0.068 | [-0.051, 0.028] | 0.5625 | 1 |
| multigen_none_n7_r1_k10 / parity_gap / false_ejection_rate | 10 | 0.033 ± 0.035 | [0.012, 0.052] | 0.02148 | 0.3652 |
| multigen_none_n7_r8_k10 / matrix_adaptation_gain / any_false_ejection_rate | 10 | 0.213 ± 0.033 | [0.194, 0.233] | 0.001953 | 0.04883 |
| multigen_none_n7_r8_k10 / late_minus_early / any_false_ejection_rate | 10 | -0.000 ± 0.019 | [-0.012, 0.011] | 0.9453 | 1 |
| multigen_none_n7_r8_k10 / parity_gap / any_false_ejection_rate | 10 | 0.053 ± 0.061 | [0.016, 0.088] | 0.0293 | 0.4395 |
| multigen_none_n7_r8_k10 / matrix_adaptation_gain / coalition_game_win_rate | 10 | 0.501 ± 0.050 | [0.472, 0.530] | 0.001953 | 0.04883 |
| multigen_none_n7_r8_k10 / late_minus_early / coalition_game_win_rate | 10 | -0.027 ± 0.062 | [-0.066, 0.007] | 0.2051 | 1 |
| multigen_none_n7_r8_k10 / parity_gap / coalition_game_win_rate | 10 | 0.116 ± 0.137 | [0.036, 0.196] | 0.03125 | 0.4395 |
| multigen_none_n9_r8_k3 / matrix_adaptation_gain / any_false_ejection_rate | 5 | 0.032 ± 0.018 | [0.018, 0.047] | 0.0625 | 0.75 |
| multigen_none_n9_r8_k3 / late_minus_early / any_false_ejection_rate | 5 | 0.001 ± 0.015 | [-0.011, 0.013] | 0.875 | 1 |
| multigen_none_n9_r8_k3 / matrix_adaptation_gain / coalition_game_win_rate | 5 | 0.418 ± 0.088 | [0.349, 0.488] | 0.0625 | 0.75 |
| multigen_none_n9_r8_k3 / late_minus_early / coalition_game_win_rate | 5 | -0.070 ± 0.030 | [-0.098, -0.054] | 0.0625 | 0.75 |

### reward_controls

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| multigen_rwbal_n7_r1_k4 / c_stage_mean minus reference h4 / false_ejection_rate | 5 | -0.003 ± 0.054 | [-0.045, 0.040] | 0.9375 | 1 |
| multigen_rwbal_n7_r1_k4 / d_stage_mean minus reference h4 / false_ejection_rate | 5 | 0.113 ± 0.054 | [0.077, 0.160] | 0.0625 | 1 |
| multigen_rwbal_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / false_ejection_rate | 5 | -0.122 ± 0.099 | [-0.204, -0.049] | 0.0625 | 1 |
| multigen_rwbal_n7_r1_k4 / c_stage_mean minus reference h4 / coalition_favorable_rate | 5 | 0.001 ± 0.037 | [-0.028, 0.031] | 0.75 | 1 |
| multigen_rwbal_n7_r1_k4 / d_stage_mean minus reference h4 / coalition_favorable_rate | 5 | 0.181 ± 0.054 | [0.141, 0.227] | 0.0625 | 1 |
| multigen_rwbal_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / coalition_favorable_rate | 5 | -0.186 ± 0.080 | [-0.254, -0.126] | 0.0625 | 1 |
| multigen_rwfe_n7_r8_k2 / c_stage_mean minus reference h2 / any_false_ejection_rate | 5 | 0.035 ± 0.034 | [0.008, 0.061] | 0.1875 | 1 |
| multigen_rwfe_n7_r8_k2 / d_stage_mean minus reference h2 / any_false_ejection_rate | 5 | 0.044 ± 0.085 | [-0.019, 0.107] | 0.5 | 1 |
| multigen_rwfe_n7_r8_k2 / matrix_adaptation_gain minus reference h2 / any_false_ejection_rate | 5 | -0.011 ± 0.076 | [-0.071, 0.049] | 0.6875 | 1 |
| multigen_rwfe_n7_r8_k2 / c_stage_mean minus reference h2 / coalition_game_win_rate | 5 | 0.073 ± 0.047 | [0.038, 0.112] | 0.0625 | 1 |
| multigen_rwfe_n7_r8_k2 / d_stage_mean minus reference h2 / coalition_game_win_rate | 5 | 0.180 ± 0.089 | [0.116, 0.258] | 0.0625 | 1 |
| multigen_rwfe_n7_r8_k2 / matrix_adaptation_gain minus reference h2 / coalition_game_win_rate | 5 | -0.108 ± 0.118 | [-0.210, -0.019] | 0.125 | 1 |
| multigen_rwmeet_n7_r1_k4 / c_stage_mean minus reference h4 / false_ejection_rate | 5 | 0.020 ± 0.056 | [-0.022, 0.067] | 0.5 | 1 |
| multigen_rwmeet_n7_r1_k4 / d_stage_mean minus reference h4 / false_ejection_rate | 5 | 0.081 ± 0.050 | [0.044, 0.121] | 0.0625 | 1 |
| multigen_rwmeet_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / false_ejection_rate | 5 | -0.067 ± 0.095 | [-0.135, 0.015] | 0.1875 | 1 |
| multigen_rwmeet_n7_r1_k4 / c_stage_mean minus reference h4 / coalition_favorable_rate | 5 | 0.029 ± 0.050 | [-0.010, 0.068] | 0.3125 | 1 |
| multigen_rwmeet_n7_r1_k4 / d_stage_mean minus reference h4 / coalition_favorable_rate | 5 | 0.108 ± 0.069 | [0.048, 0.158] | 0.0625 | 1 |
| multigen_rwmeet_n7_r1_k4 / matrix_adaptation_gain minus reference h4 / coalition_favorable_rate | 5 | -0.091 ± 0.102 | [-0.170, -0.012] | 0.25 | 1 |

### static_learned

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| depstatic_n7_r1 / C0 / mean / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.048 ± 0.020 | [-0.066, -0.033] | 0.0625 | 1 |
| depstatic_n7_r1 / C0 / median / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.167 ± 0.044 | [0.133, 0.202] | 0.0625 | 1 |
| depstatic_n7_r1 / C0 / trimmed / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.132 ± 0.036 | [0.104, 0.160] | 0.0625 | 1 |
| depstatic_n7_r1 / C0 / sharp_credibility / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.057 ± 0.020 | [0.043, 0.075] | 0.0625 | 1 |
| depstatic_n7_r1 / C0 / dependence_aware / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.008 ± 0.009 | [-0.015, -0.002] | 0.125 | 1 |
| depstatic_n7_r1 / C0 / hypothesis / crew_rule minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.082 ± 0.019 | [-0.098, -0.069] | 0.0625 | 1 |
| depstatic_n7_r1 / C0 / dependence_aware / both minus C0 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.004 ± 0.013 | [-0.007, 0.014] | 0.5625 | 1 |
| depstatic_n7_r1 / C1 / mean / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.036 ± 0.015 | [-0.046, -0.023] | 0.0625 | 1 |
| depstatic_n7_r1 / C1 / median / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.171 ± 0.077 | [0.113, 0.232] | 0.0625 | 1 |
| depstatic_n7_r1 / C1 / trimmed / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.131 ± 0.050 | [0.091, 0.166] | 0.0625 | 1 |
| depstatic_n7_r1 / C1 / sharp_credibility / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.052 ± 0.029 | [0.030, 0.074] | 0.0625 | 1 |
| depstatic_n7_r1 / C1 / dependence_aware / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.012 ± 0.007 | [-0.018, -0.007] | 0.0625 | 1 |
| depstatic_n7_r1 / C1 / hypothesis / crew_rule minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.017 ± 0.033 | [-0.042, 0.011] | 0.375 | 1 |
| depstatic_n7_r1 / C1 / dependence_aware / both minus C1 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.026 ± 0.030 | [0.002, 0.050] | 0.25 | 1 |
| depstatic_n7_r1 / C2 / mean / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.034 ± 0.015 | [-0.046, -0.022] | 0.0625 | 1 |
| depstatic_n7_r1 / C2 / median / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.184 ± 0.056 | [0.139, 0.224] | 0.0625 | 1 |
| depstatic_n7_r1 / C2 / trimmed / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.153 ± 0.032 | [0.128, 0.177] | 0.0625 | 1 |
| depstatic_n7_r1 / C2 / sharp_credibility / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.062 ± 0.020 | [0.045, 0.076] | 0.0625 | 1 |
| depstatic_n7_r1 / C2 / dependence_aware / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.006 ± 0.006 | [-0.011, -0.002] | 0.125 | 1 |
| depstatic_n7_r1 / C2 / hypothesis / crew_rule minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.031 ± 0.033 | [-0.055, -0.005] | 0.125 | 1 |
| depstatic_n7_r1 / C2 / dependence_aware / both minus C2 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.020 ± 0.009 | [0.013, 0.027] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / mean / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.035 ± 0.010 | [-0.042, -0.028] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / median / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.185 ± 0.056 | [0.146, 0.231] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / trimmed / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.148 ± 0.027 | [0.127, 0.168] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / sharp_credibility / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.068 ± 0.013 | [0.059, 0.078] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / dependence_aware / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.004 ± 0.006 | [-0.009, 0.001] | 0.25 | 1 |
| depstatic_n7_r1 / C3 / hypothesis / crew_rule minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | -0.030 ± 0.017 | [-0.043, -0.016] | 0.0625 | 1 |
| depstatic_n7_r1 / C3 / dependence_aware / both minus C3 / soft_credibility / crew_rule / false_ejection_rate | 5 | 0.034 ± 0.013 | [0.024, 0.044] | 0.0625 | 1 |

### static_rules

| Signed contrast | Seeds | Mean difference ± SD | 95% CI | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| depstatic_n7_r1 / truthful / mean minus truthful / soft_credibility / false_ejection_rate | 5 | 0.000 ± 0.000 | [0.000, 0.000] | 1 | 1 |
| depstatic_n7_r1 / truthful / median minus truthful / soft_credibility / false_ejection_rate | 5 | 0.391 ± 0.017 | [0.378, 0.403] | 0.0625 | 1 |
| depstatic_n7_r1 / truthful / trimmed minus truthful / soft_credibility / false_ejection_rate | 5 | 0.266 ± 0.009 | [0.260, 0.274] | 0.0625 | 1 |
| depstatic_n7_r1 / truthful / sharp_credibility minus truthful / soft_credibility / false_ejection_rate | 5 | 0.000 ± 0.000 | [0.000, 0.000] | 1 | 1 |
| depstatic_n7_r1 / truthful / dependence_aware minus truthful / soft_credibility / false_ejection_rate | 5 | 0.000 ± 0.000 | [0.000, 0.000] | 1 | 1 |
| depstatic_n7_r1 / truthful / hypothesis minus truthful / soft_credibility / false_ejection_rate | 5 | 0.264 ± 0.009 | [0.257, 0.270] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / mean minus lone_liar / soft_credibility / false_ejection_rate | 5 | -0.019 ± 0.005 | [-0.023, -0.015] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / median minus lone_liar / soft_credibility / false_ejection_rate | 5 | 0.288 ± 0.022 | [0.269, 0.303] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / trimmed minus lone_liar / soft_credibility / false_ejection_rate | 5 | 0.200 ± 0.018 | [0.185, 0.213] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / sharp_credibility minus lone_liar / soft_credibility / false_ejection_rate | 5 | 0.059 ± 0.007 | [0.054, 0.065] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / dependence_aware minus lone_liar / soft_credibility / false_ejection_rate | 5 | 0.053 ± 0.011 | [0.045, 0.062] | 0.0625 | 1 |
| depstatic_n7_r1 / lone_liar / hypothesis minus lone_liar / soft_credibility / false_ejection_rate | 5 | 0.073 ± 0.018 | [0.057, 0.085] | 0.0625 | 1 |
| depstatic_n7_r1 / alibi / mean minus alibi / soft_credibility / false_ejection_rate | 5 | -0.069 ± 0.012 | [-0.080, -0.060] | 0.0625 | 1 |
| depstatic_n7_r1 / alibi / median minus alibi / soft_credibility / false_ejection_rate | 5 | 0.022 ± 0.023 | [0.004, 0.040] | 0.1875 | 1 |
| depstatic_n7_r1 / alibi / trimmed minus alibi / soft_credibility / false_ejection_rate | 5 | 0.053 ± 0.029 | [0.029, 0.074] | 0.0625 | 1 |
| depstatic_n7_r1 / alibi / sharp_credibility minus alibi / soft_credibility / false_ejection_rate | 5 | -0.009 ± 0.009 | [-0.016, -0.003] | 0.125 | 1 |
| depstatic_n7_r1 / alibi / dependence_aware minus alibi / soft_credibility / false_ejection_rate | 5 | -0.093 ± 0.013 | [-0.104, -0.084] | 0.0625 | 1 |
| depstatic_n7_r1 / alibi / hypothesis minus alibi / soft_credibility / false_ejection_rate | 5 | -0.253 ± 0.015 | [-0.266, -0.243] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / mean minus framer / soft_credibility / false_ejection_rate | 5 | -0.012 ± 0.006 | [-0.016, -0.007] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / median minus framer / soft_credibility / false_ejection_rate | 5 | 0.039 ± 0.010 | [0.031, 0.047] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / trimmed minus framer / soft_credibility / false_ejection_rate | 5 | 0.169 ± 0.012 | [0.160, 0.178] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / sharp_credibility minus framer / soft_credibility / false_ejection_rate | 5 | -0.017 ± 0.005 | [-0.020, -0.012] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / dependence_aware minus framer / soft_credibility / false_ejection_rate | 5 | 0.122 ± 0.012 | [0.112, 0.131] | 0.0625 | 1 |
| depstatic_n7_r1 / framer / hypothesis minus framer / soft_credibility / false_ejection_rate | 5 | -0.177 ± 0.013 | [-0.188, -0.168] | 0.0625 | 1 |
