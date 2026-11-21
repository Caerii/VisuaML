# VisuaML Replay changes

## 0.19.0rc1

- Added held-out threshold intervention diagnosis, independent seed replications, and exact follow-up replays.

## 0.18.0rc1

- Added a portable token intervention specification and a two-decision transformer replay with a verified failure.

## 0.17.0rc1

- Added calibrated threshold-side transfer and a matched-margin test across feedback and threshold edits.

## 0.16.0rc1

- Added additive, rule-preserving feedback interventions and a dose-response control.

## 0.15.0rc1

- Added a delayed-rule intervention with runnable follow-ups and an abstention branch.

## 0.14.0rc1

- Added a crossed-rule intervention study with a factorized control and runnable tool consequences.

## 0.13.0rc1

- Added calibrated subspace interventions with exact follow-up tool replay.

## 0.12.0rc1

- Added checked PyTorch activation hooks and an executable WASM-feedback intervention study.

## 0.11.0rc1

- Reduced direct capture overhead while preserving canonical hashes and row provenance.
- Added indexed child lookup with support for reading previous request trees.
- Added a parallel 40-million-observation benchmark with immutable worker reuse.

## 0.10.0rc1

- Added direct indexed journal capture and stable request-prefix comparison.
- Added worker manifests that link disjoint ranges without copying observations.
- Added multi-candidate comparison, row audit, and a batch picker in the viewer.

## 0.9.0rc1

- Added reusable journal indexes with block checksums for repeated comparisons.
- Added full index verification and removed PyTorch startup from journal-only CLI commands.

## 0.8.0rc1

- Added streaming comparison of differently partitioned worker journals.
- Added shard and line provenance for the first differing observations.

## 0.7.0rc1

- Added ordered, streaming pipeline comparisons and named PyTorch-to-ONNX Runtime output checks.
- Added an end-to-end pipeline fixture with independent model, guest, and handoff failures.
- Admitted Rust linker global exports and identified unsupported instructions by function and offset.

## 0.6.0rc1

- Added WASM, ONNX, and PyTorch artifact descriptions.
- Added aligned PyTorch export comparison with first differing tensor and activation.
- Admitted active data and linker metadata in the bounded direct-call profile.

## 0.5.0rc1

- Added the installable `visuaml-replay` command with a bundled report and a reference runtime check.
- Added expected-return reports and a failing-test capture example.
- Kept version 1 checkpoint, resume, and local preview reports readable. Expected-return reports require a release that recognizes that recipe.

## Compatibility

Use the documented `visuaml.replay` exports, CLI, and versioned JSON formats
for integration. The tensor layout and research implementation modules are
internal. A report identifies the guest, execution profile, and graph it
requires. A runtime with a different execution identity rejects the restore.
Keep the producing wheel or repository revision with an archived report.
