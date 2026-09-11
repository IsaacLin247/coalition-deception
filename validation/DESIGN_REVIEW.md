# Independent design review: prospective defense validation supplement

Drafted and reconciled 11 September 2026, after the completed corrected study and
before this supplement's development or production outcomes. The frozen,
machine-readable supplement protocol governs execution. This is not a
preregistration of the historical work.

## Scope and interpretation

Retain the previous 303 contrasts, their 15 families, and all original results
unchanged. They motivated this follow-up and supplied rough variance references.
Do not pool their seeds into the new confirmatory analysis. The new study asks
whether attack adaptation remains observable after explicit development of the
hypothesis defense, and independently tests the previously observed separation
between coalition victory and innocent-ejection outcomes. It does not isolate
the effects of the five changes from the archived game implementation.

Keep the revised environment and feasibility scorer fixed. Tune only the final
tie action and declared membership threshold. The score remains a heuristic;
development of a decision threshold does not turn it into a calibrated posterior.
Every simulation and training job runs on `ssh desktop`.

## Candidate set and development selection

Use the full Cartesian grid of tie handling `{index, random, skip}` and membership
threshold `{0.35, 0.50, 0.65, 0.80}`, with honesty gate weight 0.5. A tie means exact
equality of the complete existing ranking tuple `(membership, creator, own
evidence)`. Thus random and skip only replace the final index tie breaker; they
do not discard the creator or private-evidence ranking. Index at threshold 0.50
must reproduce the published rule. Random selection uses a declared separate
policy RNG stream; invalid/absent candidates never become legal through a tie.
Keep the current plurality tally fixed, since changing that tally would change
another component of the benchmark.

Development uses eight independently seeded evaluation batches, IDs 100–107.
Every candidate faces seven declared tasks: the four scripted conditions
(truthful claims, lone liar, alibi, framer) in the synthetic F2 meeting environment,
plus the original corrected study's mean-trained, soft-trained, and
index-hypothesis-trained attacks in the spatial F8 single-round environment.
The two environments have incompatible observation dimensions; do not evaluate
a frozen spatial attack in the synthetic environment. Development batch 100+i
uses the old training-seed-i checkpoint from each learned attack class, i=0–7.
Thus there are exactly 24 old development checkpoints, all pinned by SHA256.
These older learned policies are development opponents only. Each task contributes
one equal-weight cell by first averaging its eight batches. Use all declared checkpoints;
never select the most favorable attacks after inspecting candidate performance.
Freeze the exact checkpoint identities and number of episodes before evaluating
the candidate grid. Existing attack outcomes may inform this design, but no final
supplement seed, final opponent, or final evaluation stream enters selection.

For every candidate and every development task, retain false-ejection rate,
creator-ejection rate, no-ejection rate, and honest-voter skip/tie rates. Creator
capture is the unconditional joint probability of a real incident and ejection
of its creator. The synthetic F2 setting supplies an incident, but spatial F8 can
have incident-free episodes; these episodes stay in the denominator and cannot
count as creator captures. Use identical episode seeds across candidates within
each task. This is an explicitly declared mixed-task selection utility, not a
claim that the seven cells share an environment or incident frequency. Scores do
not select checkpoints or modify training. Use exactly 1,000 episodes per cell:
12 candidates × 7 tasks × 8 batches = 672,000 development episodes.

The proposed selection rule protects against a trivial abstention solution:

1. A candidate is eligible only if its truthful-condition creator capture is at
   least `max(0.20, legacy_truthful_capture - 0.05)` and its equal-weight mean
   creator capture across all seven opponent cells is at least the legacy
   rule's corresponding capture minus 0.05. Here legacy means the corrected
   feasibility rule with index ties and threshold 0.50, evaluated on precisely
   the same development batches.
2. Among eligible candidates, minimize the largest false-ejection rate across
   the seven opponent cells. Break selection ties by lower equal-weight mean
   false-ejection rate, then higher truthful creator capture, then the fixed
   candidate order `index, random, skip`, with thresholds ascending within each.
3. If no candidate satisfies the capture conditions, use the original index/0.50
   rule as the prespecified fallback and explicitly record development failure.
   Do not relax a condition or introduce another threshold after seeing results.

These are decision criteria, not statistical tests of defense quality. The
capture floors are task-specific, declared tradeoffs, not universal standards.
Mean and soft defenses should also be displayed in the development table as
context. Keep all twelve rows and all selection inputs, even if the winner is
the legacy defense or development does not improve truthful performance.

Because truthful refers to claim contents and the coalition still controls
its testimony selection and ballots, label that control precisely. A low
false-ejection rate together with low capture is a tradeoff, not proof of a
useful defense. Under the unchanged plurality tally, honest abstention can also
enable coalition ballots to eject an innocent; report that interaction directly.

## Fixed primary family

Use five crew plus two coalition agents. Let `D0h` be the published corrected
index/0.50 hypothesis defense, `D*` the deterministic development selection,
and `C_soft`, `C_0h`, `C_*` fresh attacks trained for the same fixed budget against
soft credibility, D0h, and D*, respectively. Each matched seed receives each
training condition; training is fresh from initialization, not warm-started from
the development library. For the eight-round experiment, preserve the corrected
matched cycle's definitions of C0, D1, C1, rewards, budgets, and legal observations.

The six two-sided primary tests form **one new Holm family at alpha 0.05**:

| ID | Per-seed signed quantity | What it can establish |
|---|---|---|
| P1 | Synthetic F2 truthful-script FE(D*) − FE(D0h) | Whether development's truthful-condition error improvement generalizes. |
| P2 | Spatial F8 FE(C_*, D*) − FE(C_soft, D*) | Adaptation to the selected fixed defense. |
| P3 | Spatial F8 FE(C_*, D*) − FE(C_0h, D0h) | Relative vulnerability after separately matched attack training. |
| P4 | Win(C1, D1) − Win(C0, D1), eight rounds | Coalition victory change after adaptation. |
| P5 | Mean innocent ejections(C1, D1) − Mean innocent ejections(C0, D1), eight rounds | Change in total innocent-ejection burden. |
| P6 | [Win(C1,D1) − Win(C0,D1)] − [Any-FE(C1,D1) − Any-FE(C0,D1)], eight rounds | Direct difference between changes in two probability outcomes. |

Report all six regardless of direction or significance. P3 compares the combined
defense-plus-matched-training procedures; it is not a fixed-attack causal effect
of changing a tie breaker. P6 is a difference in probability changes, not an
individual-game causal mediation claim. It does not establish that any-FE is
unchanged. In particular, a nonsignificant any-FE effect is never called equivalence.

Retain canonical random/0.50 and skip/0.50 defenses in final evaluation regardless
of which candidate is selected. Each final static seed evaluates all four scripts
against six defense slots: original index/0.50, random/0.50, skip/0.50, selected,
mean, and soft. Each final adaptive seed freshly trains five attacks against
original index/0.50, random/0.50, skip/0.50, selected, and soft, and evaluates the
entire five-by-five crossplay. All receive the same fixed 400-update training
budget. If selected equals a canonical defense, keep the duplicated named slot
and training stage; disclose that identical specifications and RNG seeds can
produce identical policies. Do not silently change the family or sample size.
The canonical random/skip crossplay is prespecified descriptive sensitivity
evidence, with no additional confirmatory tests or selection among p-values.

Report correct/creator ejection, no-ejection, voter skip/tie rates, full fixed
crossplay, and single-round transfer descriptively. Label all extra population
sizes or unplanned comparisons exploratory. Retaining a small follow-up family
is legitimate because its entire definition precedes these new outcomes; it is
not a retrospective replacement for the large historical families.

## Independent seeds, streams, and stopping

Fix the final sample size before any final run. **Use 72 fresh paired
training seeds**, IDs 1000–1071. At 64 seeds the conservative power
for a five-percentage-point effect falls to about 78% when SD is 1.5 times the old
hypothesis-adaptation SD; 72 gives about 84%. This sample size has limited
power if the new defender produces substantially greater between-seed variability.
No outcome-dependent sample expansion is permitted.

Use separate, explicitly recorded streams for development episodes, training,
optional monitoring, and untouched final evaluation. A proposed final evaluation
base is 2,987,654,321, but the runner must verify how offsets map to actual episode
seeds and guarantee disjoint streams and valid integer ranges. Matching integer
training seed IDs across treatments creates a blocking convention, not proof
that policies follow identical random paths. Freeze initial actor checkpoints
and all budget endpoints; never select the best checkpoint using final outcomes.

Every planned seed must complete all required cells. Infrastructure interruption
permits deterministic resumption/retry of the same seed and same source; record
all attempts. Algorithmic failures are outcomes requiring disclosure, not grounds
to discard or replace a seed. A numerical or semantic defect blocks the analysis
and requires a documented protocol amendment before affected runs restart.
No stopping or additional seeds depend on p-values, effect directions, or CI width.
Final inferential tables remain disabled while a planned cell is missing.

Use exactly 1,000 untouched final episodes per policy cell. With independent
unpaired Bernoulli draws, a difference between two 1,000-episode proportions has
worst-case Monte Carlo SD about 0.0224.
Common episode seeds change covariance and therefore actual precision. Episodes
reduce evaluation noise; they do not increase the number of training replicates.

## Statistical method and power sensitivity

For each contrast form one difference per complete seed, report its mean and
sample SD, and use the two-sided exact paired-mean sign-flip test. Each final cell
has exactly 1,000 episodes, so each primary difference is an integer numerator
divided by 1,000. Reconstruct numerators from raw episode counts, verify every
denominator, and enumerate the distribution of signed integer sums by dynamic
programming with arbitrary-size integer multiplicities. This computes all
2^72 assignments without materializing them. It preserves magnitudes and zeros;
it is neither a signs-only binomial test nor a floating-point tolerance test.
The raw minimum at 72 nonzero same-signed differences is 2/2^72. Apply Holm to
exactly six p-values with stable prespecified contrast IDs.

Exact enumeration is a computational property, not proof of the null assumption.
Sign flipping requires symmetry/exchangeability of complete seed differences
under the null; matched seed IDs alone do not establish it. Report that
assumption, display the seed-level differences, and include a paired t-test
sensitivity analysis without changing the primary method in response to results.
Report pointwise 95% percentile intervals from 20,000 resamples of complete paired
seeds, with fixed contrast-specific RNG seeds. They do not provide familywise
coverage merely because p-values receive Holm adjustment.

`power_analysis.py` computes approximate planning power using the noncentral-t
benchmark at the conservative first Holm threshold alpha = 0.05/6. This is not
an exact power calculation for the primary sign-flip test. It describes a single
contrast at that threshold, not the probability all six reject. Historical SDs
are planning references only:

| Historical seed difference | SD |
|---|---:|
| Hypothesis adaptation, single meeting | 0.076054 |
| Eight-round win adaptation | 0.109207 |
| Eight-round innocent-ejection-count adaptation | 0.171090 |
| Eight-round win adaptation minus any-FE adaptation | 0.071469 |

Five percentage points and 0.10 ejections per game are explicit planning effect
targets; they are not equivalence margins and do not dictate interpretation of
smaller estimates. New defense selection can alter variance substantially. For
a five-percentage-point target, n=72 power is approximately 93% at SD 0.10, 84%
at SD 0.11408, and 55% at SD 0.15. At n=64 those values are 89%, 78%, and 48%.
For 0.10 ejections/game and SD 0.20, n=72 power is approximately 93%.
Do not claim uniformly adequate power for every plausible outcome distribution.

The computation follows the one-sample t statistic on complete paired
differences and the noncentral-t alternative. See the [NIST discussion of sample
size and detectable changes](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm)
and [SciPy's documented one-sample t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
The full script records its historical input SHA256 and SciPy version and can
emit either Markdown or JSON without reading any prospective result.

An independently seeded statistical Monte Carlo check uses 1,000 synthetic
normal alternatives per SD, rounds them to the planned 1/1,000 lattice, and draws
8,191 independent random sign assignments per alternative. The plus-one p-value
approximates the primary exact sign-flip distribution. At n=72 and effect 0.05,
the estimated powers are:

| Assumed seed-difference SD | Estimated power | Monte Carlo standard error | Monte Carlo 95% Wilson interval |
|---|---:|---:|---|
| 0.100000 | 0.942 | 0.0074 | [0.926, 0.955] |
| 0.114081 | 0.834 | 0.0118 | [0.810, 0.856] |
| 0.150000 | 0.578 | 0.0156 | [0.547, 0.608] |

These intervals measure Monte Carlo precision conditional on the synthetic
normal scenario, not uncertainty about the true power in the game. Finite random
sign counts add approximation error near the threshold; final tests use the
exhaustive dynamic program. RNG seeds are 2026091101–2026091103. The complete
receipt is `POWER_ANALYSIS.json`; reproduce it with
`python validation/power_analysis.py --format json --mc-trials 1000`.

## Evidence required before manuscript claims change

Freeze candidate-grid protocol and source before development; freeze D* and its
complete selection receipt before final attack training/evaluation. Preserve the
original source fingerprint separately. Record new source hashes, exact actor
initializations/final checkpoints, seed mappings, commands, wall times, completion
markers, evaluation episode records, and all failed attempts. Check the default
candidate against the unchanged old implementation and test tie RNG/abstention
semantics before the production freeze. Compare independently recomputed
per-seed primary contrasts with the reporting analyzer.

The eventual paper must describe this as an independently seeded follow-up
informed by the completed audit. If the selected defense remains poorly
calibrated or eliminates the apparent adaptation effect, report that result and
narrow the claims. A successful software run cannot by itself establish that
the paper is ready for submission.
