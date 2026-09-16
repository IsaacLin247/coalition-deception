# Desktop execution status

## Completion update — 15 September 2026

All eight development jobs and 216 final jobs completed on the desktop on
12 September. Local collection of all 224 jobs and independent reanalysis
completed on 15 September at 20:35 UTC. Reanalysis checked 3,816,000 episode rows
and 4,490,695 meeting rows with no incomplete jobs or reported problems. Primary
counts and exact sign-flip/Holm p-values matched the desktop analysis; remaining
floating-point results matched at absolute tolerance 1e-12 and relative tolerance
1e-10. The [completion receipt](completion_receipt.json) retains protocol,
selection, and source identities.

Scientific interpretation and manuscript integration of these prospective
results remain pending. The current `paper_v2` reports the earlier corrected
240-job study, including the subsequent review corrections.

## Launch record — 11 September 2026

Production launched **11 September 2026 at 19:33:17 UTC** (14:33 CDT) as
`WCCValidation20260911` on the user's desktop. The scheduled task continues after
SSH disconnects. The initial check confirmed all eight development jobs running
with no failures; the queue permits up to twelve concurrent final jobs.

The code and protocol were published in commit
[`f906227e950b8240d17edabf7800c605717addc2`](https://github.com/IsaacLin247/coalition-deception/commit/f906227e950b8240d17edabf7800c605717addc2),
tagged
[`validation-protocol-2026-09-11`](https://github.com/IsaacLin247/coalition-deception/tree/validation-protocol-2026-09-11/validation),
before production launch. The unauthenticated public protocol download, local
protocol, and desktop protocol all matched SHA-256
`fa017c32f1e99da67644b5c1823fa406c7a9d0325b1331c8cdacf2135a01f02e`.
All 24 development input checkpoints and the source manifest were verified on
the desktop before starting.

The desktop runs Python 3.12.10, NumPy 2.5.2, PyTorch 2.14.0+cpu, and SciPy
1.17.0. Each research worker uses one computation thread. SciPy was added for
the prespecified statistical sensitivity analysis; NumPy and PyTorch were not
changed. Full machine and launch receipts are retained locally.

The automated sequence is development, frozen selection, all 216 final jobs,
then complete-cohort analysis. A separate local monitor checks operational
progress, collects completed jobs with checksums, and reanalyzes the complete
cohort. It does not expose partial final effect estimates for model selection.

The preceding launch record is retained for provenance; the dated completion
update above supersedes its operational status. The [protocol](PROTOCOL.md) and
[study guide](README.md) describe the fixed design and reproduction commands.
