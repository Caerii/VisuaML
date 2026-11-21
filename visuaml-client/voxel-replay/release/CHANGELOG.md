# VisuaML Replay changes

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
