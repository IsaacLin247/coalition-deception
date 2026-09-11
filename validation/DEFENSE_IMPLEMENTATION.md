# Supplementary hypothesis-defense decision variants

This extension addresses whether the published feasibility defense's behavior depends
on its final tie rule and membership decision threshold. It imports the existing
benchmark's `HypothesisCrew`, `coalition_posterior`, public-role constraints, and
private-evidence function. It does not modify their source, candidate-coalition
enumeration, room solver, score normalization, observation rules, claims, movement,
training collection, or vote aggregation. The study is a prospective supplement to
the completed benchmark, not a replacement for its archived results.

## Decision rule and controls

For each living candidate other than the observer, the policy forms the exact
lexicographic tuple `(membership score, creator score, own evidence)`. A candidate is
eligible only if the top membership score is **strictly greater** than the selected
threshold. Threshold equality causes abstention. These are normalized heuristic
scores, not calibrated probabilities; threshold selection alone does not establish
probabilistic calibration.

The policies differ only when multiple candidates share the exact maximum tuple:

| Final tie setting | Action after the threshold is passed |
| --- | --- |
| `index` | Select the lowest candidate ID, reproducing the published rule. |
| `random` | Select uniformly by a seeded pseudorandom function of the observer's permitted view. |
| `skip` | Abstain when two or more candidates share the maximum tuple. |

The comparison grid is `index`, `random`, and `skip`, crossed with thresholds
`0.35`, `0.50`, `0.65`, and `0.80`. The honesty-gate parameter is held at `0.50` in
that grid. The interface also permits a separately specified honesty gate in
`[0, 1]`; it passes that parameter directly to the unchanged scoring function.
An all-zero inconsistent view still abstains. Near-equal floating point scores are
not converted into ties: the implementation uses exact tuple equality and adds no
epsilon or tolerance. A unique highest tuple is unaffected by the final tie setting.

The environment's existing no-skip fallback remains intact. If skipping is disabled,
an abstaining proposal becomes a vote for the first legal candidate. Supplementary
comparisons must therefore keep `allow_skip_vote=True` when interpreting the `skip`
condition as an abstention treatment.

## Reproducible randomization and information access

`random` uses a separately supplied policy randomization seed, an explicitly
serialized observer view, and the exact tied candidate list. SHA-256 produces an
integer seed for an explicit NumPy `PCG64` generator; `integers(len(tied))` selects
one tied candidate. The version is `observer-view-sha256-pcg64-v1`.

The observer view contains public alive status, public role disclosures, public
inactive times, round number, report information, public claim contents, and the
observer's permitted positive and negative location evidence. The incident time
appears only when the configuration makes it public after a report. Private marker
information appears only when the observer saw the marker. A blind policy excludes
its private location and marker information.

The key excludes the simulator's episode seed, hidden roles and incident identities,
unseen trajectories, private coalition symbols, and privileged truth/support labels
on claims. It also excludes the response list, ballots, phase, and sequential
speaking position. The underlying hypothesis policy does not use those response
fields to rank living candidates. Omitting them preserves the same accusation and
vote when the observer's relevant evidence is unchanged.

The random selector never consumes the shared engine or learner RNG. Repeated calls,
diagnostic inspection, batch order, policy reconstruction, and replay therefore do
not redraw an otherwise identical decision or shift coalition/environment draws.
The run harness must record the policy seed and use separately specified training
and evaluation seed streams.

This is a **fixed pseudorandom function of the view within a replicate**, not an
independent new draw every time the same view occurs. The same permitted view and
policy seed reuse a target across episodes. That choice is explicit, avoids hidden
episode-state access, and keeps response/vote consistency without mutable caches.
It can make repeated views predictable to an adaptive opponent; results should not
be described as covering every stochastic tie implementation. Uniformity under
agent relabeling is distributional over independent policy seeds, not exact
same-seed permutation equivariance. A finite synthetic relabeling test checks for
gross concentration on a low ID; it is not a mathematical proof of invariance.

## Why abstention may still permit false ejection

The benchmark's `majority` setting implements plurality over non-skip ballots.
Abstentions are omitted from the tally. Consequently, if every honest observer
abstains while two coalition members vote for the same innocent, the innocent has
the unique positive plurality and is ejected. The `skip` defense changes individual
decisions; it does not impose a majority quorum or give skip ballots a veto. A
synthetic test applies the existing ballot transition to this exact configuration
and verifies the outcome. The supplementary study must report false ejections,
correct ejections, and non-ejections together rather than assume higher abstention
means stronger defense.

## API and diagnostics

```python
from validation.defense_policy import make_defense

spec = {
    "name": "hyp_random_t050",
    "tie_break": "random",
    "threshold": 0.50,
    "honesty": 0.50,
}
crew = make_defense(spec, seed=policy_seed, record_decisions=True)
```

`DefenseSpec` is the typed alternative to the dictionary. `tie_seed=` aliases
`seed=`; supplying both is rejected. The policy inherits the original truthful
claim selection, free-play movement, and action-mask handling. It overrides the
decision target and, optionally, records diagnostics at the actual `vote_index`
call. Standard inherited options such as `blind` and `accuse` are supported.

`crew.decision(state, voter)` returns an immutable `DefenseDecision` containing
`target`, `tied_candidates`, `top_membership`, and `above_threshold`. This call does
not advance any RNG or append an audit record.

With `record_decisions=True`, `crew.audit_records` receives one record per
`vote_index` call, captured before ejection. Fields include round, voter,
population, top membership score, final tie size and candidate IDs, threshold,
honesty gate, tie setting, threshold status, proposed target, actual target, and
skip status. The harness should enable recording for evaluations, consume the
records, and clear the list explicitly between retained batches. It should not
enable unbounded diagnostic accumulation during long training runs. Extra
diagnostic calls to `vote_index` would produce extra records; use `decision` for
inspection. Records do not include hidden game metadata or empirical outcome
labels; the harness may join outcomes after the game.

The in-memory record also carries `_state_token=id(state)` exclusively for the
runner's diagnostic join. `VecEnv` advances each state in place, so the runner
maps retained terminal states to their episode indices and drops records from
the final vector batch's unretained surplus games. It removes `_state_token`
before writing JSON and adds `episode_index` and `eval_seed`. Object identity
never enters the observer key, target, randomization, or learning observation.

## Validation performed

Run from the clean repository root with its normal benchmark dependencies:

```sh
PYTHONPATH=code/src:. python -m pytest validation/tests/test_defense_policy.py -q
```

The 23 synthetic unit tests pass. They cover exact ties and near-ties, the strict
threshold, creator and own-evidence priorities, living/self exclusions, agreement
with the frozen baseline on bounded real-score views, all-zero views, gate
forwarding, response/vote consistency, RNG isolation, reconstructed-policy replay,
hidden-state and blind-view invariance, ID relabeling distributions across 2,048
fixed policy seeds, skip/plurality interaction, pre-ejection recording, no-skip
fallback, and invalid specifications. Tests use constructed observer views and a
single isolated ballot tally; no training or Monte Carlo game simulation was run
locally for this implementation check. The supplementary game study is run on the
authorized SSH desktop.
