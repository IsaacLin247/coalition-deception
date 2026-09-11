# Independent review of the prospective supplement analyzer

Reviewed 11 September 2026, before production freeze, by a separate review agent
from the analyzer author. Scope: `analyze.py`, `inference.py`, the reconciled
`design_metadata.json`, development selection boundaries, and the power receipt.
This is an internal code and arithmetic review, not external peer review or a
judgment that the eventual empirical findings support publication. No research
simulation was run for this review.

**Result: no unresolved blocking analysis issue was found in the reviewed
revision.** Two issues raised during review were corrected before these final
checks: the output family name now matches the declared `post_review_validation`
family, and every retained evaluation configuration is compared in full against
the resolved frozen job and opponent settings. The latter prevents a change in
private-history access, incident visibility, objective, or synthetic/spatial
regime from passing merely because population size and round count agree.

## Primary estimands and integer arithmetic

All six signed expressions agree with the design, including cell orientation:

| ID | Verified expression |
|---|---|
| P1 | Synthetic truthful false-ejection count: selected minus original index/0.50. |
| P2 | Spatial false-ejection count against selected defense: selected-trained minus soft-trained attack. |
| P3 | Spatial matched false-ejection count: selected-trained/selected minus original-trained/original. |
| P4 | Eight-round coalition wins: C1–D1 minus C0–D1. |
| P5 | Eight-round total innocent-ejection count: C1–D1 minus C0–D1. |
| P6 | P4 minus the C1–D1 versus C0–D1 change in any-innocent-ejection count. |

Every expression divides its per-seed integer numerator by the common 1,000-game
denominator. P5 therefore measures ejections per game; the other expressions
measure probability differences. P6 does not subtract counts of innocent
ejections from a win probability, and it does not equate a nonsignificant harm
effect with no harm effect.

An independent evaluator constructed all six expressions directly from the JSON
metadata, without using the analyzer's `PRIMARY` tuple, on 72 synthetic mixed-sign
count cohorts. The integer vectors and their signed means matched exactly.

The sign-flip implementation preserves difference magnitudes and zero-seed
multiplicities. Its subset-sum dynamic program uses Python arbitrary-size integer
counts, so the `2^72` assignment count does not overflow a 64-bit integer. The
greatest-common-divisor reduction rescales an integer lattice exactly. The tail
includes equality at the observed absolute sum. It is an exact mean-sign-flip
distribution, not a signs-only test.

Independent arithmetic checks covered **110 additional cases**: 104 random
integer vectors with 2–14 entries compared with brute-force sign enumeration,
and six 72-entry equal-magnitude cases compared with direct binomial-coefficient
tail counts. All matched exactly, including the unanimous-sign minimum
`2/2^72` and the zero-observed-sum probability one.

The output contains one six-member primary Holm family. Canonical random/skip
cells and other crossplay entries receive descriptive summaries, not additional
confirmatory families. Paired-t sensitivity uses the same six expressions and
is explicitly labeled sensitivity. Its p-values do not replace the primary
sign-flip p-values. The pointwise 20,000-resample bootstrap intervals retain
their stated coverage limitation.

## Raw outcomes and complete-cohort gates

The production path requires all 72 specified seeds, 1000–1071, for each of static,
adaptive, and matched-cycle jobs: 216 final jobs in total. It additionally requires
all eight development jobs, their completion manifests, frozen source identity,
and a reproducible selection receipt. Missing, substituted, duplicate, or
reordered seed/cell records cannot reduce a test's denominator or the Holm family.
Failure removes this analyzer's stale derived result tables and emits a status
file with inference disabled.

For every final cell, the analyzer checks all 1,000 episode indices and reconstructs
the terminal plurality result from role, alive-status, and ballot vectors. The
reconstructed electorate includes an agent ejected at that meeting and excludes
earlier inactive agents. Skips are excluded from the plurality tally, preserving
the declared game rule. Creator capture requires a real coalition incident
creator; it is not counted from two absent sentinel indices. Coalition victory
is reconstructed from terminal living-role counts.

Whole-game any-false-ejection and total innocent-ejection outcomes are reconstructed
from the full ordered meeting ejection records and the fixed role vector. They
are not recovered from the terminal-meeting flag. These checks verify recorded
meeting ejections; retained data do not contain every intermediate ballot, so
this is not an independent replay of every earlier meeting's tally. The terminal
tally is independently reconstructed.

Summary means must match reconstructed counts without rate rounding or zero
imputation. Decision-audit records must identify exactly the retained pre-ejection
honest electorate and match its actual votes, resolved parameters, and episode
identities. The full environment check now applies the same static-opponent
override precedence as execution and normalizes JSON list/tuple representation
without changing values.

The actual job builder with the current input manifest and metadata was inspected:
it produces eight development jobs and three 72-seed final cohorts, with six
static defense slots, five adaptive training slots and their complete crossplay,
and the three-stage eight-round cycle. Learned stages specify 400 updates. These
are the declared production settings; the review does not rely on reduced smoke
jobs as evidence that the full experiment has finished.

## Development isolation and engineering smoke

The analyzer recomputes the development selection from all source-pinned raw
development outcomes using `selector.select`. The stored receipt must match this
decision and all recorded inputs; only its creation timestamp is excluded from
that equality comparison. Every final job's recorded start must be at or after
the frozen selection timestamp. Old learned checkpoints enter the declared
development tasks and are not treated as new final training replicates.

A reduced engineering smoke requires both top-level `smoke_only: true` and the
matching analysis flag. Its explicit reduced cohort and denominator are checked
against the smoke protocol, and its status and Markdown retain the smoke label.
Removing one flag fails the analysis; removing both does not admit the reduced
jobs into the required production 72-seed/1,000-episode grid. Smoke and production
also have distinct protocol and source-pinned artifact identities. Consumers must
retain those identities and the smoke label when using exported tables.

## Power receipt and limits

`POWER_ANALYSIS.json` exactly matches the SHA256 recorded in the current design
metadata. The receipt does not claim to contain a hash of `power_analysis.py`;
the reviewed script's actual hash is recorded below and the production runtime
source manifest separately pins that script.

Four normal-theory benchmark values were independently recomputed by numerically
integrating the normal rejection probability over a chi-square variance estimate,
instead of calling the script's noncentral-t power function:

| Seeds | Effect | SD | Independent power |
|---|---:|---:|---:|
| 64 | 0.05 | 0.10 | 0.894423280160 |
| 72 | 0.05 | 0.10 | 0.933130965408 |
| 72 | 0.05 | 0.114080673210 | 0.838691024867 |
| 72 | 0.05 | 0.15 | 0.548094277684 |

All differences from the reported noncentral-t calculation were below `2.2e-14`.
The separately seeded Monte Carlo sign-flip planning receipt preserves its
scenario assumptions, random-sign approximation, and Monte Carlo uncertainty.
Neither calculation establishes the actual variance or power of a newly selected
defense. Exact p-value computation likewise does not establish null symmetry:
joint sign invariance remains a substantive assumption, and paired seed IDs
alone do not prove it.

After the two review corrections, the targeted analysis, inference, and selection
suite passed **54 tests**. These counts overlap the wider supplement test suite
and must not be added to it as distinct tests.

## Reviewed file identities

| File | SHA256 |
|---|---|
| `analyze.py` | `c5d923830c511247fd1e4163ca6fa9fb97a8ef2d9f91ff95c420082f64dfbfa7` |
| `inference.py` | `c05eb3152ccd63b9b73e543c145f78e2e0a938c27d6384ba134dce9ab77a3d36` |
| `selector.py` | `605844849d26d72e6255e8241aeb3a94718291e39b9ac04b6309c0ce3345ebf3` |
| `jobs.py` | `5eb77317d5ecdea2c046750242a96b0d8bf4f1bbdd9e2ce555a23f1f7e69dd33` |
| `runner.py` | `d62a0275a77b1f4ab3fe645e1887bd00d2bb879423da3a8c4b7f5e59428fe40c` |
| `design_metadata.json` | `6f7bb54863c8ea6adf34ab78f4ea7bc824f11f5aa8cc925edf58ac36fcd22814` |
| `power_analysis.py` | `c277d4c169e947e7ad1a19201beef0c94e20d8db7f1affb63405feb938d1a26b` |
| `POWER_ANALYSIS.json` | `b39cecc7b149d9db32c5d11547ce5ea7580f8b5753bac7c5b201dc2c1fb9eefe` |
