# Desktop integration validation

The complete development → frozen selection → final evaluation → gated analysis pipeline passed on the desktop on 11 September 2026. These deliberately small runs test software integration. They contribute no scientific observations, defense selection, effect-size estimates, or training checkpoints to the production study.

The smoke protocol retained all twelve development candidates and both references, all seven development tasks, all six final static defenses, the complete five-defense adaptive training/crossplay grid, and the eight-round matched response cycle. It used development batches 100–101 with a separate evidence stream, final training seeds 3000–3001, 16 evaluation episodes per cell, and two updates of four completed games per learner. Production uses its separately frozen seeds and budgets.

| Verification | Completed |
| --- | ---: |
| Development jobs | 2 |
| Final jobs | 6 |
| Evaluation cells | 302 |
| Retained evaluation episodes | 4,832 |
| Completed meeting records | 5,094 |
| Learner stages | 16 |
| PPO updates | 32 |
| Completed training games | 128 |
| Retained checkpoints | 64 |
| Desktop synthetic unit tests | 91 |
| Exported files independently hash-verified | 430 |

All eight completion markers passed independent local verification after transport. The final analyzer independently validated 106 final cells, 1,696 episode records, and 1,958 meeting records before calculating the six planned test forms. It reconstructed outcomes from ballots and role labels, checked meeting totals, linked every recorded hypothesis decision to a retained honest ballot, reproduced development selection, and verified that final jobs began after selection was frozen. The frozen source passed a further check after the complete pipeline exited successfully.

The pipeline took 52.44 seconds with four available worker slots: development 20.33 seconds, selection 0.37 seconds, final jobs 30.48 seconds, and analysis 1.26 seconds. These reduced-budget timings are engineering measurements, not a reliable runtime forecast for trained production policies.

The desktop environment used Python 3.12.10, NumPy 2.5.2, PyTorch 2.14.0+cpu, SciPy 1.17.0, and pytest 9.1.1. The first pytest preflight resolved an older editable engine installation. Setting the test process's `PYTHONPATH` to the snapshot's `code/src` corrected the invocation; all 91 tests then passed. The runner itself explicitly inserts its frozen engine path before importing research modules. No research job failed, no source was edited after freezing, and no global desktop settings changed.

Executed source identity:

```text
Pinned executable/configuration inputs: 82
Source SHA-256:
dcfc2ca26bf399b26682011a8ecc77e13da0af85d8fb9315bb13352512066e5c
Smoke protocol SHA-256:
a0c56dec30a8d586186fc1c8201c6279095c9ad90c300b3cdb5a763535143c0b
Source bundle SHA-256:
af36ca8f3ec86f9c0f311da6da1974c72c4ab4ce7a07bc26c039c04b7bfdbacc
Result archive SHA-256:
76d86b9a91e763d42fc49e31142f83c2e8fa773b4ac03094fae5335347e49a8f
```

The immutable result archive contains 57,258,407 bytes. An interrupted download resumed through SFTP and matched both the complete archive checksum and every embedded file checksum. Machine-specific launch records, operational logs, exact per-job timing, source inventories, and raw smoke artifacts are retained separately in the local validation operations archive.

After this execution, two analyzer-only checks were strengthened before production: the reported Holm family label was aligned with the protocol, and the full resolved environment configuration was checked for every cell. The updated analyzer independently rechecked all 302 archived cells, 4,832 episodes, and 5,094 meetings. All six numerical analysis outputs were unchanged. This follow-up retains the executed smoke's original source identity; it does not relabel its provenance as the later production source.

```text
Updated analyze.py validator SHA-256:
c5d923830c511247fd1e4163ca6fa9fb97a8ef2d9f91ff95c420082f64dfbfa7
```

The final local synthetic suite, including the additional analysis guards, passed 96 tests. Its later validator identity is distinct from the 82-file source fingerprint used to generate the desktop smoke observations.
