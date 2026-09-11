# Truthful F2 hypothesis errors: frozen-model review, 9 September 2026

The reviewed failure examples follow the declared feasibility score and voting rule; they do
not demonstrate erroneous exclusion of the true coalition. This is a bounded explanation of
observed mechanisms, not attribution of the entire F2 error rate or proof that every possible
implementation path is correct. Production source, parameters and results were not changed.

## Candidate count and score geometry

At 5+2, there are seven original agents. Excluding the publicly murdered victim and the
honest observer leaves **M=5 candidate members**, with coalition size m=2. An exchangeable
membership score is therefore **2/5=0.4**, not 0.5. This follows from frozen
`src/social_collusion/policies/hypothesis_crew.py:126–127,164–165,184–217`.

For a useful **restricted special case**, suppose all accepted records are feasible and the
feasibility of placing candidate j at the scene is an indicator c_j independent of which
other candidate is in H and of the optional honesty gates. Then

    q(H) = (1/2) sum_{j in H} c_j,
    S = sum_j c_j,
    Z = sum_{|H|=2} q(H) = ((M-1)/2) S.

For S>0, the normalized creator score is c_j/S and the membership score is

    member_j = 1/(M-1) + [(M-2)/(M-1)] (c_j/S).

At M=5 this is `member_j = 0.25 + 0.75 * creator_j`. Hence the declared
`member > 0.5` decision threshold corresponds, **in this special case only**, to
`creator > 1/3`. One, two or three equally feasible creators receive membership scores
1, 0.625 or 0.5 respectively. Two ambiguous candidates both clear the threshold, while
three equally feasible candidates lead to a skip. This formula is not asserted for arbitrary
transcripts: optional accepted claims can make creator feasibility depend on H and its gates.

`hypothesis_calibration_probe.py` directly checked those three small observer views plus the
exchangeable 0.4 baseline against the frozen module; `hypothesis_calibration_probe.json`
records outputs and source/probe hashes. It imports the immutable source and verifies the
module hash against the frozen protocol. These checks generated no F2 sweep.

## What the score retains and discards

Frozen `hypothesis_crew.py:173–206` tests whether each accepted record plus a proposed
`LOC(j,t)=marker` has **any** satisfying assignment. Both an agent known to have been at
the scene and an agent merely not ruled out there can contribute one. It does not count the
relative probability of their possible paths or model incident/victim/isolation selection.
Those omissions are documented in `audit/pre_submission_hypothesis.md`; the implementation
matches that declared heuristic rather than a normalized generative posterior.

The archived scorer explicitly gave confirmed scene placement a factor R (four rooms here)
and an unplaced candidate factor one; see
`audit/original/pre_submission/src/social_collusion/policies/hypothesis_crew.py:167–196`.
The repair removed this unjustified uniform-room likelihood ratio. Removing it can flatten
ranking between confirmed and merely possible scene occupancy. It also removed victim
hypotheses and smoothing and repaired constraints, so the overall old/new performance change
cannot be assigned to this single alteration from the available bounded check.

Target selection uses exact lexicographic order `(membership, creator, own evidence, -id)`
and a membership threshold 0.5 (`hypothesis_crew.py:236–246`). A mathematical tie can therefore
be resolved to a lower-index innocent; that behavior is distinct from a floating-point tie
perturbation.

## Two reproduced truthful-condition failures

The independent replay probe selected the first ten false ejections in the first fixed
64-episode seed-0 batch. Across their forty honest voter views, every true pair had positive
score, all truthful claims plus private facts and the actual creator placement were jointly
feasible, and no actual claim, private-fact or timeline violation was found. See
`f2_truthful_hypothesis_probe.json` for the complete records and source identity. This is a
case audit conditioned on failure, not an estimate of failure-mechanism frequencies.

- **Episode 7, evidence seed 1987654321:** true coalition {2,5}, creator 5, innocent ejected 4.
  Voters 0,1,6 each give candidates 4 and 5 membership 0.625, creator 0.5 and own evidence 1.
  Every candidate-pair weight in each of those three views exactly matches the restricted
  two-feasible-creator formula above. Creator 5 truthfully reports seeing the victim at the
  incident room and time; agent 4's truthful scene observation concerns an earlier time and
  does not place it at the incident instant. Both remain feasible under the score. The final
  exact tie breaker picks agent 4, and those three honest votes help eject the innocent.
- **Episode 9:** true coalition {5,6}, creator 5, innocent ejected 0. All four honest voters
  skip, with maximum membership scores 0.4 or 0.4375. The coalition's two ballots both name
  agent 0, and the default plurality tally ejects that agent. No honest hypothesis voter
  chose the innocent in this example.

The latter mechanism depends on the experiment's precise control. **“Truthful” refers to
claim contents, not a fully cooperative coalition policy.** Frozen
`scripted_coalition.py:181–199,218–225,239–250` selects supported claims while filtering some
self-incriminating testimony and prevents its ballots from targeting either coalition member.
The default config string `majority` is implemented as plurality over agent targets with skips
excluded (`env/transition.py:499–513`); four skips do not defeat two votes for one agent.
Thus increasing honest abstention need not reduce collective false-ejection probability.

The examples support reporting limitations of feasibility ranking and its interaction with
strategic coalition ballots. They do not justify tuning the frozen threshold after seeing the
new result, relabeling the scorer as calibrated Bayesian inference, or attributing the full
truthful-condition change to one repair. No production rule change or extra sweep is needed
to report the completed result honestly.
