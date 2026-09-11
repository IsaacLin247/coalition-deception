# Additional validation after publication review

This prospective supplement follows the completed corrected study. It addresses
the hypothesis rule's tie handling and threshold choice, and the insufficient
resolution of many earlier statistical families. It does not replace, pool, or
relabel the earlier 303 comparisons. The previous data release remains intact.

All research simulation and training for this supplement runs on the desktop
reached through `ssh desktop`. The original engine and training code under
`../code/` remain unchanged. New decision rules are defined in
[defense_policy.py](defense_policy.py); they use the same feasibility scores,
information constraints, and plurality tally.

## Fixed design

- **Development:** twelve hypothesis variants combine index, random, or skip
  handling of exact final ties with thresholds 0.35, 0.50, 0.65, and 0.80. Mean
  and soft credibility provide reference results. Eight development batches
  evaluate four synthetic-meeting scripted conditions and three spatial learned
  attack types. The learned attacks are 24 pinned checkpoints from the completed
  study, used only for development. There are 784,000 development episodes.
- **Selection:** retain candidates meeting the declared creator-capture floors;
  minimize the worst of seven development task error rates, followed by the
  declared tie breakers. Preserve all candidate results and use the original
  defense if none is eligible. The selection is hashed before final jobs start.
- **Fresh final evidence:** 72 independent replicate IDs, 1000–1071. The static
  comparison evaluates six defenses against four scripted conditions. Adaptive
  training uses five defenses and a full 5-by-5 crossplay matrix. The eight-round
  response cycle trains C0, D1, and C1 and evaluates all four relevant pairings.
  All cells contain 1,000 prescribed episode indices; each training stage has
  400 PPO updates with 32 completed episodes per update.
- **Primary analysis:** exactly six predefined seed-paired contrasts, with one
  Holm family. Exact mean sign-flip tests use integer episode-count differences
  and arbitrary-size assignment counts, avoiding both the previous resolution
  problem and numerical enumeration of 2^72 assignments. Pointwise bootstrap
  intervals retain all 72 seeds. The random/skip sensitivity comparisons beyond
  the six primary contrasts remain descriptive.

The maximum budget is 576 learner stages and 230,400 PPO updates. Final crossplay
contains 3,816,000 episodes, in addition to development and monitored training
evaluation. If selection duplicates a canonical defense, its separately named
stage still uses the same fixed budget and paired seed; identical results are
reported as such, without treating duplication as independent evidence.

[DESIGN_REVIEW.md](DESIGN_REVIEW.md) explains the primary effects, development
criterion, assumptions, and limits of the power calculation.
[POWER_ANALYSIS.json](POWER_ANALYSIS.json) retains the reproducible planning
calculation. Statistical power depends on the unknown variance of new outcomes;
the fixed seed count is not a guarantee of significance or a promise to expand
the study until a result rejects.

## Reproduction

Install the dependencies described in [the code guide](../code/README.md). Restore
the 24 development input checkpoints using the source release paths and hashes
in [the input manifest](inputs/development_attacks.json). These checkpoint bytes
are preserved in the previous public data release and are not stored in Git.
The supplementary power calculation and paired-t sensitivity also use SciPy;
the desktop analysis environment adds SciPy 1.17.0 without changing its existing
NumPy or PyTorch installations.

Run commands from the repository root with the installed Python environment:

```sh
python -m pip install scipy==1.17.0
python -m pytest validation/tests
python validation/runner.py status --protocol validation/protocol.json \
  --results results/validation_20260911 --phase development
python validation/pipeline.py --protocol validation/protocol.json \
  --results results/validation_20260911 --slots 12
```

The pipeline completes development, records the deterministic selection, runs
every final job, and invokes the gated analysis. It stops on failure. Completed
jobs can be verified and skipped on restart; incomplete attempts are retained
and require an explicitly recorded recovery. No completion marker is accepted
without the prescribed source, selection, checkpoints, raw records, and budgets.

The transport helpers can package and verify an isolated SSH workspace. They
do not alter the original study's desktop directory or global machine settings.
Machine-specific paths, launch receipts, and local operational logs stay outside
the public source tree. Preserve the frozen source while computation is running;
an implementation change requires a separately recorded protocol amendment and
new source identity.

The manuscript will incorporate this supplement only after its final analysis
passes verification. A completed computation alone does not establish that the
selected defense is stronger or that every original scientific claim survives.
