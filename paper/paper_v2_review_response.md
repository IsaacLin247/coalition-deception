# Paper v2: review corrections

Updated 15 September 2026. These revisions address the five review points in the
shorter [manuscript](paper_v2.md), generated [TeX](paper_v2.tex), and [PDF](paper_v2.pdf).
The original long article remains unchanged. No training, game rules, checkpoints,
or released experimental outcomes were changed; the added decomposition uses
existing published seed summaries.

## 1. Voting rule and abstention

Methods now states that skip ballots are excluded from the candidate tally. A
unique candidate leader can be ejected even when most players skip; all-skip
ballots or a tie among leading candidates produce no ejection. Results connects
this rule to the abstention diagnostic: coalition votes can decide the outcome
when honest voters all skip. This matches
[`_step_vote`](../code/src/social_collusion/env/transition.py), rather than treating
skip as a competing candidate with a veto.

## 2. Reward and incident decomposition

Methods and Results now explain that F3's `balanced_ejection` reward gives the
defender **+1 in every incident-free game, even after innocent ejection**. F4
optimizes terminal victory, with no separate innocent-ejection penalty. Therefore
neither increased overall false ejection in F3 nor frequent innocent ejections
alongside improved victory in F4 directly measures susceptibility to deceptive
testimony. The reward implementation and frozen training protocol were checked,
including six synthetic terminal-state reward cases; no games were generated.

For the fixed coalition against the learned defender, the decomposition is:

| Population | Overall false ejection | Incident and false ejection | No incident and false ejection | Share of false ejections without an incident |
|:--|--:|--:|--:|--:|
| 3+2 | 0.4635 | 0.0202 | 0.4433 | 95.6% |
| 5+2 | 0.4031 | 0.0903 | 0.3128 | 77.6% |
| 7+2 | 0.4747 | 0.1552 | 0.3195 | 67.3% |

The first three numeric columns all use **all 10,000 games per cell** as the
denominator. They satisfy P(F) = P(F and I) + P(F and not I), where F is false
ejection and I is an incident. The last column is a share of false ejections,
not an incident-conditional error rate.

The additional descriptive finding is that defender training reduces the joint
incident-and-false-ejection rate at every population, while overall false ejection
increases. At 5+2, the overall increase is **+0.2091 = −0.0775 + 0.2866**: the
increase is confined to incident-free games. This does not isolate a testimony
effect or prove that the reward caused the errors, since incidence and the
population of incident-containing games also change. Four 3+2 learned-defender
seeds contain no incidents, making their incident-conditional rates undefined.

The [evidence receipt](data/reward_incident_decomposition.json) retains all 12
crossplay cells, reconstructed seed counts, defined conditional denominators,
reward cases, source hashes, and all three defender-change decompositions.
It verifies 72 published means and 120 seed-level decompositions. The manuscript
also uses the verified 5+2 initial incident rate, 0.9626.

## 3. Vote agreement and false claims

The heading, abstract, metrics, results, and conclusion now describe **raw
same-target vote agreement**. Any skip counts as nonalignment; the rate includes
meetings with both coalition ballots recorded. Shared parameters, observations,
and incentives can produce agreement without communication causing it, and no
matched-independent learned-policy baseline is reported. The false-claim metric
is FALSE-labelled actions divided by all coalition claim actions, including
non-informative actions. A falling fraction therefore does not show greater
truthfulness among informative claims. These definitions match
[`collusion_metrics.py`](../code/src/social_collusion/metrics/collusion_metrics.py).

## 4. Limits of the mechanism evidence

The abstract, mechanism section, and conclusion retain the narrow finding:
scripted alibi errors occurred without excessive total coalition credibility
weight in the tested records. The scoring interventions retain initial evidence
but permit new responses and votes. Their effects therefore do not identify a
unique causal channel or locate all failures in claim-to-candidate scoring.
Condition-dependent benefits and adverse effects remain reported, consistent
with the [mechanism review](../data/reviews/MECHANISM_FINDINGS.md).

## 5. Exploratory interpretation and contribution

The abstract and conclusion explicitly frame the study as exploratory. F4's
victory changes are described as observed rates; its 27-test Holm family still
has no rejection. The post-audit contrast specification, five-seed bootstrap
limits, and multiplicity restrictions remain explicit. Prior social-deduction
and Among Us-style communication research is acknowledged; the contribution is
adaptive-defense crossplay, separate victory and innocent-harm outcomes, and
direct diagnostics, rather than introducing learned social-deduction agents.

The prospective 72-seed follow-up is separate from these findings. Its desktop
jobs finished on 12 September and all 224 jobs passed local collection and
independent reanalysis on 15 September. Exact counts and sign-flip/Holm p-values
matched; remaining floating-point results matched the specified tolerance.
**Its outcomes are not incorporated in paper v2.** Repository status wording now
reflects this completion and the pending scientific integration; see the
[status record](../validation/RUN_STATUS.md).

## Reproduction

From the repository root, using the installed project environment:

```sh
python paper/verify_reward_incident_decomposition.py
bash paper/build_v2.sh
python paper/verify_paper_v2.py --output paper/data/paper_v2_validation.json
```

The v2 checks preserve the original article's hashes and mathematical proof,
reconstruct the figure values and incident decomposition, and check tables,
references, captions, and PDF text bounds. No new simulation is needed for these
review corrections.
