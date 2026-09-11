# Independent F8 raw-record verification

The frozen F8 diagnostic correctly calculates its recorded flags, but the original archive gate only recomputed aggregate ratios from those saved flags. The editable archive builder now calls `verify_vote_diagnostic.py` for every raw episode before accepting per-seed summaries. The verifier uses only the Python standard library and does not import the diagnostic or game engine.

It independently checks:

- The exact production game sizes, training seeds, episode indices, trained/evaluated rule identities, checkpoint identity, and fixed evidence/actor seed formulas. Shared but incorrectly generated seeds therefore fail verification.
- Actual plurality, crew-only plurality, and coalition-only plurality; distinct ballot-necessity and coalition-only-sufficiency flags; false ejections, coalition alignment, skip counts, vote counts, and all ballot denominators.
- Every honest pre-ejection voter's inclusion, including an honest voter later ejected; consistency between the pre-vote alive mask, final ballots, and saved vote actions.
- Incident and creator-ejection counts from the saved incident record; coalition claim/false-claim/no-information counts from the saved pre-vote claims.
- Finite normalized membership/creator score vectors, maximum membership score, zero-score flags, top-two coalition membership with deterministic tie ordering, creator top-pick correctness, and undefined flags for zero-score views.

These checks recompute confidence-related flags from the retained feasibility-score vectors. They do **not** rerun the hypothesis solver or establish that its scores are exact generative posterior probabilities. Likewise, claim-count validation uses saved truth labels; it is not an independent replay of claim truth classification. The archive separately verifies the immutable engine, protocols, required training jobs, checkpoint/history hashes, raw file hashes, complete cell set, common reset hashes, per-seed aggregates, and across-seed mean/sample SD. Ballot removal holds recorded votes fixed and does not establish that agents would behave identically if other voters were removed.

The standalone production semantic check is:

```sh
python -B audit/submission/verify_vote_diagnostic.py --results audit/submission/vote_diagnostic_results
```

The command requires the complete 135-cell production identity set and 500 episodes per cell. Its semantic check complements the archive's source/checkpoint gate; it does not replace that provenance gate. Smoke data require an explicit `--smoke` flag, which the archive never supplies.

Validation on the existing smoke fixture (`vote_diagnostic_smoke_validated`, no new experiments): all nine cells, 27 episodes, and 114 honest score views passed. Targeted diagnostic/verifier/archive regressions passed 48 tests in 4.03 seconds. Tests reject deliberately changed necessity/sufficiency, abstention and confidence flags, omitted ejected voters, inconsistent vote actions, changed seeds, malformed scores, and smoke identities at the production gate.

The archive also now compares the full actual executable/configuration inventory against the frozen source manifest, rejecting added `.py`, `.yaml`, or `.yml` files under `src`, `scripts`, `configs`, or `experiments`, as well as changed/missing listed files. The actual 82-file frozen inventory passes. Analysis and checkpoint-integrity outputs must identify the exact current `checkpoint_integrity.py` and `checkpoint_plan.py` helpers. Full local/desktop package and Torch build manifests accompany the original environment records.

No frozen F8 script, frozen source, protocol, checkpoint, production output, or waiting production process was modified. The independent verifier and this note are included in the archive helper/documentation allowlists. Before-edit copies of the archive builder and its tests are retained locally in `review_before_f8_semantic_guards/`.
