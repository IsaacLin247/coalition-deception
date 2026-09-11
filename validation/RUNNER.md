The validation supplement imports the published engine in `code/` without changing its source. Its own protocol, results and hashes remain separate from the 240-job corrected study. Execute research simulation and training on the authorized `ssh desktop` host.

Create `design.json` with `python validation/jobs.py --inputs validation/inputs/development_attacks.json --metadata validation/statistical_design.json --out validation/design.json`. The optional metadata must describe the prospectively specified primary comparisons; it cannot override job budgets or the selection rule. Review the materialized design before freezing it.

After every runtime module is final and the 24 development checkpoints are present, freeze exactly once:

```sh
python validation/runner.py freeze --design validation/design.json --protocol validation/protocol.json
```

This records the exact design-file checksum and every engine/configuration/runtime Python source checksum. Checkpoint paths are relative to the package; their individual SHA-256 digests and complete environment configurations are explicit design inputs. Checkpoint environment equality is checked before inference, including the private-history observation schema. Development combines four synthetic meeting tasks and three spatial learned-opponent tasks; these are separate declared tasks, never an attempt to run a spatial checkpoint in the synthetic environment.

Run the development stage, then select once using all eight complete batches. Freeze the selector output before running final jobs:

```sh
python validation/runner.py run --protocol validation/protocol.json --results results/validation_20260911 --phase development --slots 12
python validation/selector.py --protocol validation/protocol.json --results-root results/validation_20260911 --out validation/selection.json
python validation/runner.py run --protocol validation/protocol.json --selection validation/selection.json --results results/validation_20260911 --phase final --slots 12
```

The design contains eight development jobs and 216 final jobs. Development evaluates all 12 hypothesis candidates plus soft/mean references against seven opponent/task types. Final data contain 72 independent seeds for each of the static, adaptive, and eight-round matched-cycle experiments. Final static evaluations retain index, random, and skip ties at threshold 0.50, the selected defense, and soft/mean references. Adaptive jobs train five fresh coalitions against soft, the three canonical tie variants, and the selected defense, and evaluate their full 5×5 crossplay matrix. These additional canonical tie comparisons are descriptive; the six primary comparisons remain unchanged. If selection returns a canonical variant, its duplicate stage and cells are retained with the same initialization and declared settings. Cycle jobs train fresh C0, D1, and C1 with the published seed offsets of 0, 10000, and 20000, and evaluate the full 2×2 matrix. Every learner receives exactly 400 updates of 32 completed games. Initial and every-50-update checkpoints are retained. Initial/monitor evaluation uses an isolated policy and evidence stream; all final cells use the declared 1,000 fixed-index evidence draws.

Each job writes a `run.json` provenance record, atomic `result.json`, one complete raw JSON file per evaluation cell, training histories, all required checkpoints, and finally a `complete.json` marker. Raw files preserve every episode in index order, all completed meeting outcomes, role labels, actual ballots, ejected IDs, and available defense decision-time diagnostics. Non-finite conditional descriptive statistics are represented as null; no episode or seed is filtered. Static and adaptive result entries identify opponents and defenses explicitly. Completion validation checks the exact planned cell set, complete evidence indices, primary outcome agreement, all training episode/update totals, required checkpoint presence, and hashes every artifact.

The concurrent queue holds a phase-specific exclusive lock, keeps process/log/status records, and fails closed on source changes, missing inputs, malformed completion records, and unsuccessful subprocesses. It does not silently retry, overwrite partial histories, select favorable seeds, or stop for statistical significance. To restart after an operational failure, preserve the failed job directory under a timestamped `_attempts/` directory, document the reason, and rerun the same frozen job. Do not remove an active lock; inspect its recorded host/process before handling an abandoned lock.

Use `status` to inspect completion, and `verify` to rehash every artifact:

```sh
python validation/runner.py verify --protocol validation/protocol.json --selection validation/selection.json --results results/validation_20260911 --phase final
```

Final numerical outcomes remain uninspected until every final job passes verification. Training progress may be monitored operationally through completed-update counts and process health. Do not inspect or report interim final contrasts. The original study's exploratory results are not pooled into the new confirmatory tests.
