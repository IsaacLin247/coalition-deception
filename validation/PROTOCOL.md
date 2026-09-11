# Frozen protocol for additional validation

The production design was frozen on **2026-09-11T19:31:50.151134+00:00**, before any
production development or final evaluation was launched. This is a prospective
supplement motivated by the completed corrected study, not a preregistration of
the original research.

- Protocol SHA-256: `fa017c32f1e99da67644b5c1823fa406c7a9d0325b1331c8cdacf2135a01f02e`
- Executable/configuration source SHA-256: `95e13e33f671270685281eb7cc20ff15f7f0686a38e7edcefa3401807d655402`
- Pinned source files: 82
- Jobs: 224 (eight development, 216 final)
- Final replicate IDs: 1000–1071; all 72 required
- Primary comparisons: P1–P6, one six-member Holm family
- Development episodes: 784,000
- Final comparison episodes: 3,816,000
- Training stages: 576; 400 updates and 32 complete episodes per update
- Planned total PPO updates: 230,400

The exact candidate specifications, input checkpoint hashes, development
criterion, jobs, evidence streams, contrast expressions, and inference
assumptions are in [protocol.json](protocol.json). The deterministic selection
record is created only after every development job verifies, and is hashed
before final jobs start. Final outcome values are not inspected until all
216 final jobs pass their completion gates.

The [design review](DESIGN_REVIEW.md), [power calculation](POWER_ANALYSIS.json),
[independent analysis review](INDEPENDENT_ANALYSIS_REVIEW.md), and
[desktop smoke validation](SMOKE_VALIDATION.md) document the preparation. Smoke
evidence is separately identified and excluded from the scientific cohorts.

The original 303 comparisons and frozen benchmark engine are preserved. New
random/skip sensitivity comparisons outside P1–P6 are descriptive. This study
can report no improvement, no detected adaptation, or unfavorable effects;
none of those outcomes changes the cohort, budgets, or statistical family.
