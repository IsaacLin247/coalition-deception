# Analysis plan for the post-audit replication

This plan preserves the September 2026 post-audit analysis specification. The
clean repository was packaged after some replication outcomes were available;
its new source hash and protocol timestamp identify this portable distribution,
not a new prospective preregistration or a completed study. The frozen `submission_protocol.json` specifies jobs, seeds,
budgets and executable source hashes. No results from the historical directory,
smoke tests or incomplete jobs enter the replication tables.

## Inclusion, units and reporting

All jobs and seeds in the frozen protocol are required. A job is usable only if
its successful completion marker matches the frozen source hash, all required
artifacts exist, its run provenance matches, and its required result cells are
present. Retained checkpoints must also deserialize with restricted loading, contain finite tensors matching the saved architecture, record the expected update counts, and match the final numbered checkpoint parameters. File and parameter fingerprints are written to `checkpoint_integrity.json`. Missing seeds, invalid schemas, or missing comparison cells block final
inference. `--interim` permits explicitly labeled descriptive tables of complete
jobs and missing-job status, with all inferential p-values and confidence
intervals suppressed. It does not reduce any planned denominator or change the
required comparisons. Seed tables preserve original seed IDs and input paths.

The independent unit is a complete training seed for learned experiments and
an independently seeded evaluation batch for the scripted F2 sweep. Multiple
generations, matrix cells, and episodes are repeated observations within this
unit. Report each cell's mean and sample SD across seeds. A single observed seed
has an undefined between-seed SD, not zero. No episode-level test is used as a
substitute for seed-level inference.

For every planned contrast, pair identical seed IDs before subtracting. Report
the mean difference, sample SD, and percentile 95% confidence interval from
20,000 bootstrap resamples of entire paired seed differences. Bootstrap RNG is
fixed by a hash of the contrast identifier, independent of output order. The
two-sided exact test enumerates all sign assignments to the observed differences
and compares the absolute mean, retaining difference magnitudes. This sign-flip
test requires symmetry/exchangeability under the null; pairing alone does not
establish that assumption. With five seeds the smallest attainable two-sided
unadjusted p-value is 0.0625. Confidence intervals are pointwise and do not
provide familywise coverage.

With ten seeds the minimum two-sided raw p-value is 0.001953125. A Holm family
of 26 or more tests whose smallest attainable p-value is this ten-seed minimum
cannot pass the first rejection threshold at 0.05, even if every observed sign
agrees. Five-seed tests cannot reject at raw 0.05. Families are not shrunk to
obtain significance; these resolution limits are not evidence of absence or
equivalence. Effect sizes and their uncertainty remain central.

Holm adjustment controls multiplicity separately within each named comparison
family below. The complete planned family size is retained; missing comparisons
cannot make another contrast more significant. Unplanned comparisons are
descriptive/exploratory and do not acquire a confirmatory label. The machine
readable `planned_comparisons.json` lists every signed expression, metric,
family, and expected seed ID, constructed before result files are loaded.

## Outcomes and fixed contrasts

False-ejection metrics must keep their denominators distinct. In one-round
conditions `false_ejection_rate` is the probability of an innocent ejection in
the single meeting. In multi-round conditions its label is terminal-meeting
false ejection. `any_false_ejection_rate` is the probability of at least one
innocent ejection anywhere in the game, including incident-free meetings.
`mean_false_ejections` averages the number per game;
`false_ejections_per_meeting` pools false ejections and meetings within each
seed before seed-level aggregation. The incident-bearing and incident-free
splits remain explicitly separate. Game wins mean coalition parity with at
least one coalition survivor. Creator survival requires a real incident and is
a per-episode, incident-gated joint probability, not one minus the per-episode
creator-ejection probability.

- **f1_learning:** For each of the three population sizes, final minus initial
  false-ejection probability and coalition-favorable probability. Both endpoints
  come only from `result.json`'s `initial_holdout` and `final_holdout`, evaluated
  with seed base 1987654321. The initial endpoint is the untrained actor.
  Monitored curves use the separate 987654321 stream, with no checkpoint
  selection. Curve x values are `updates_completed`, falling back to `update+1`.
- **f2_condition_effects:** At each population size and each of the seven rules,
  lone-liar, alibi, and framer minus truthful false-ejection probability.
- **f2_rule_effects:** At each size, under alibi and framer conditions, each
  non-soft rule minus soft credibility false-ejection probability.
- **f3_cycle:** At each size, C0-D0 minus C0-D1 (defender effect), C1-D1 minus
  C0-D1 (coalition adaptation), and C1-D1 minus C0-D0 (net change), for
  single-meeting false ejection.
- **f4_cycle:** The same three expressions at each size for whole-game
  any-false-ejection probability, mean false-ejection count, and coalition game
  wins. Terminal-meeting false ejection is reported descriptively alongside them.
- **budget:** At each planned extended budget, compare C0-D1 and C1-D1 against
  the 400-update reference on seeds 0–4. Use single-meeting false ejection for
  r=1; any-false-ejection probability and game wins for r=8.
- **ablation:** For each of the three removal conditions, removal minus default:
  single-meeting false ejection for r=1, and any-false-ejection probability and
  game wins for r=8. A separate **ablation_behavior** family covers same-target
  voting, coalition false-claim rate, and incident-gated creator survival for
  both round settings. Hidden-partner removes partner identity from actor observations, masks,
  and missing-message features; the original audit is summarized in the
  repository-level `cahnges.md`.
- **hypothesis_adaptation** and **dependence_counterattack:** For each non-soft
  defense and size, C[defense]-defense minus C[soft]-defense (adaptation),
  C[defense]-defense minus C[soft]-soft (matched adaptive difference), and
  C[defense]-soft minus C[soft]-soft (transfer), all for false ejection. In the
  hypothesis family, also compare C[hypothesis]-hypothesis minus C[mean]-mean.
- **loop_dynamics:** For each undefended/defended main loop and primary metric,
  test the seed's mean matrix adaptation gain
  `mean_g(M[g,g]-M[g-1,g])` against zero. Also test late minus early C-stage
  means using nonoverlapping first/last windows of `min(3,floor(k/2))`
  generations, and the parity contrast when both parity categories exist.
  Primary metrics are false ejection for r=1 and any-false-ejection probability
  plus game wins for r=8. Parity excludes neighboring generations and D0;
  k=3 has no opposite-parity nonneighbor cells, so no parity test is defined.
- **dependence_loop:** Compare the defended k=4 loop with the undefended n=7,
  r=8 reference truncated to generations 0–4 on the same ten seeds. Compare
  C-stage mean, D-stage mean, and matrix adaptation gain for whole-game
  any-false-ejection probability and game wins.
- **reward_controls:** Compare each reward variant to the n=7 reference on
  exactly seeds 0–4, truncating both to its prespecified k=4 (single meeting)
  or k=2 (multi-round) horizon. Compare C-stage mean, D-stage mean, and matrix
  adaptation gain. Single-round metrics are false-ejection and
  coalition-favorable probabilities; multi-round metrics are
  any-false-ejection probability and game wins. Supplementary matched reference
  summaries use these five seeds, not the reference's full ten-seed mean.
- **static_rules:** For every scripted condition at n=7, each non-soft rule
  minus soft false-ejection probability. **static_learned:** For each C0–C3,
  each non-soft rule and the combined dependence rule/tally minus the soft
  crew rule, using the matching seed. The strength sweep is descriptive with
  every preplanned strength shown; no best strength is selected.

All remaining recorded metric cells, trajectories, full crossplay matrices and
probe-set policy distances are descriptive. Titles state setting and outcome,
not a presumed direction, defense success, capture, stationarity, or convergence.

## Exploratory finite-pool response gap

For each new multigeneration matrix and generation k, report
`G_k = max_i M[i,k] - min_j M[k,j]`, with i,j spanning the complete observed
policy pool from that run. Thus the pool includes policies trained later than k;
this is a retrospective pool comparison, not an online stopping criterion.
Use the applicable probability outcomes (single-meeting false ejection or
whole-game any-false-ejection/game-win probability). Report the selected row and
column, matrix size, and episodes per cell. Do not reinterpret an empirical
maximum as a best response over all policies.

For Monte Carlo uncertainty, each bounded probability cell with N independent
episodes receives a simultaneous Hoeffding band with
`epsilon = sqrt(log(2*m/0.05)/(2*N))`, where m is the number of matrix cells.
Apply max/min to the clipped cell bands to bound each gap. The union bound
allows correlated cells from shared evaluation seeds; it assumes independent
episode draws within each cell. These conservative bands account for empirical
extremum selection within the observed matrix. Across-seed summaries describe
replication variability; they do not replace the within-matrix bands. Neither a
small gap, a nonsignificant trend, nor similar policies establishes true
exploitability, equilibrium, or convergence. Policy-pool coverage, changing
training opponents, and finite evaluation precision remain explicit limits.
