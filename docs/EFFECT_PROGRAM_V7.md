# VisuaML v7: bounded stateful effects and measurement evidence

**Status:** opt-in research dialect, not a general-purpose Python/MLIR compiler, not a photonic backend, and not a formal proof system.

## Why this exists

VisuaML's v6 `stateful_ir.py` validates one specific sequence of neural-assembly operations. v7 adds an extensible but deliberately closed interpreter for multiple mutable tensor state cells, ordered effects and finite repeated execution. It does not replace v6 or alter the default VisuaML canvas/API.

### New modules

- `compiler_lab/effect_ir.py`: typed state/input declarations, bounded `Repeat` blocks, local SSA, type inference, write revisions, linear effect tokens, resource admission, fingerprints, conservative effect preservation.
- `compiler_lab/effect_runtime.py`: deterministic NumPy reference execution, caller-state isolation, copy-on-write updates, state digests and iteration receipts.
- `compiler_lab/effect_verification.py`: executable differential checks of **all state cells** after a rewrite; witness testing is not universally quantified semantic equivalence.
- `compiler_lab/calibration_evidence.py`: disjoint holdout evaluation for the v6 unitful affine latency model, admission gates and provenance-bound estimates. This is **not yet wired into** the graph-cut optimizer.
- `backend/examples/effect_program_v7_demo.py`: two connected neural areas, each with weights, bias and winner state; A feeds B over three deterministic rounds. Hardware is not simulated.

## Typed instruction dialect

A `Repeat` block has local SSA values, with every state read/write consuming and producing a unique **linear effect token**. Reads state the current relative revision of their cell (initially 0 each iteration), and each write increments it. Repetition is finite and statically bounded; mutable state persists between iterations, but temporary SSA values do not.

Admitted operations: `read`, `write`, `identity`, `add`, `subtract`, `relu`, `tanh`, `matmul`, `gather_sum`, deterministic `topk_indices`, `hebbian_update`, and `refraction_update`. The latter two implement a restricted weight update and bias update, not full assembly-calculus semantics.

The runtime validates exact state tensors (shape/dtype/finiteness), rejects repeated or out-of-range indices, and copies state before executing. If a later iteration fails, the original inputs remain untouched and no output state is delivered. JSON reports bind program and state digests. The digest is reproducibility evidence, not a digital signature or an attestation of external experimental provenance.

## Research acceptance

A five-step recurrent program is compared with VisuaML's existing v6 `run_assembly_schedule` oracle with Hebbian plasticity and refraction active; weights, bias and winners agree on the selected fixture. This is **not** full parity with `Caerii/assemblies`' stateful engines: the existing v6 cross-repo check covers a narrower non-refraction case. To claim broad parity, extend pinned cross-repo tests to multi-area scheduling, recurrent plasticity, refraction, inhibition and varying ties/seeds and compare each committed state.

The differential checker requires effects to be structurally identical and compares all resulting state cells on one or more concrete witnesses. It rejects a deliberately wrong arithmetic rewrite, and accepts an inserted identity with exact equality.

## Latency evidence gate

v6 `calibrate_latency` fits `T_us = base + a * work + b * bytes` using declared observations. The v7 gate requires **nonoverlapping measurement IDs** and **held-out, in-domain inputs**, reports max/mean absolute and relative errors, and rejects a failed report when requested for planning. Synthetic observations are blocked by default unless `allow_synthetic=True`. Source kinds are provider-declared; lab measurements still require independent source authenticity, timestamp and device calibration procedures. The gate currently yields per-operator prediction objects; **a complete device-vs-host partition still requires measured boundary transfers and per-node cost integration**.

## Reproduction

From the repository root after dependencies are installed:

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_effect_program_v7.py backend/tests/test_calibration_evidence_v7.py
PYTHONPATH=backend python backend/examples/effect_program_v7_demo.py
```

## Next steps

1. Pinned upstream parity on actual `Caerii/assemblies` for refraction and multi-area schedules, using each engine's declared semantics and write ledger.
2. Dynamic conditional control flow with explicit effect-join semantics; do not introduce speculative execution of learning writes.
3. Verifiable migration between this restricted dialect and future canonical IR, including formalized loop invariants.
4. Calibration-to-placement integration with per-node cost functions, measured host/device operator profiles and boundary-transfer evidence; fail closed on extrapolation.
5. Expose effect traces and holdout gate reports in the React compiler inspector, clearly distinguishing execution from simulation and speculation.

### Resource admission

The validator caps state and input tensor memory, shapes, operation counts, repeated instruction expansion and aggregate estimated work (currently at most 80 million abstract work units per admitted program). These static limits reduce accidental abuse but do **not** constitute a secure execution sandbox or a complete runtime memory/time bound. Run externally supplied programs inside a separately isolated, resource-limited service.
