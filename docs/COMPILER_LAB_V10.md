# VisuaML Compiler Lab v10 — executable integration, not universal compilation

v10 links existing compiler components across **five admitted, tested boundaries**:

1. **Canonical semantic envelope** (`semantic_ir.py`): normalizes FX `ProgramIR` and stateful `EffectProgram` into one inspectable and digestible schema with typed inputs, regions, explicit state effect tokens, provenance, and output names. The original executable dialect remains authoritative. This is a shared semantic *inspection layer*, not yet a general lowering implementation for arbitrary Python or MLIR.
2. **Declarative neural-assembly schedule** (`assembly_schedule.py`): accepts up to 16 areas, 64 directed fibers, and 24 ordered projection steps. It produces bounded typed `EffectProgram` instructions, preserving ordered source-fiber accumulation, per-fiber learning rates, weight ceilings, winner selection, state revisions and effect-token sequences. The supported subset has explicit float32 dense matrices, integer neuron IDs, and deterministic lowest-index ties. It does **not** implement stochastic recruitment, normalization, custom winner policies, refraction, or arbitrary dynamic flow. A separate optional test checks the pinned `NumpyExplicitEngine`.
3. **Executable hybrid dispatch** (`hybrid_execution.py`): graph nodes run on PyTorch CPU (host) or a separately dispatched NumPy CPU adapter (illustrative device), with real software tensor copies on boundaries. It measures wall time, operator dispatch, logical transfer sizes, and a conservative retained-memory value, then validates actual outputs against the PyTorch reference. This is *not* real hardware offload or proof of performance gain.
4. **Hardware-aware scheduling** (`schedule_sim.py`): deterministic FIFO resource schedule with independent host/device lanes, one serialized copy link, optional repeated-transfer caching, explicit memory budget admission, and modeled cost assumptions. It returns a feasible schedule, **not** globally optimal scheduling. All microsecond values are hypothetical inputs, not observed hardware latency.
5. **Experimental coherent optical lowering** (`optical_dialect.py`): lowers one real `nn.Linear` into a passive complex field matrix (normalized spectral norm), explicit required gain, and electronic bias. The opt-in surrogate explores attenuation, phase quantization, amplitude clipping, and illustrative Gaussian noise. It records exact limitations; it is not Ayo hardware, silicon photonics fabrication geometry, or measured laser/ADC/DAC power.

`v10_lab.py` produces a deterministic, JSON-safe integration report for the fixed trusted model and three-area learning schedule. The existing `/compiler` API attaches this report as `v10` under an **allowlisted** scenario request. The existing React page displays the resulting trace, modeled schedule, and limitations without interpreting modeled economics as measured speedups.

## Local execution

```bash
cd visuaml-client
PYTHONPATH=backend python -m pytest -q backend/tests/test_v10_integration.py
PYTHONPATH=backend python -c 'from visuaml.compiler_lab.v10_lab import run_v10_lab; import json; print(json.dumps(run_v10_lab(seed=19), indent=2))'
VISUAML_COMPILER_LAB_ENABLED=1 pnpm run api
pnpm run dev
```

The upstream parity test requires `neural-assemblies` installed. The project CI installs a pinned package revision in its separate reference workflow; the default local test suite skips it if absent. Do not confuse the v10 NumPy device adapter with Ayo's hardware.

## Security and scientific boundaries

- The fixed `/api/compiler-lab/analyze` request never accepts arbitrary Python/model source or untrusted bytecode.
- The older model-import/upload endpoints execute local Python code; **untrusted imports must be disabled by default** and unsafe development opt-in kept local. For true hosted arbitrary-model support, use a hardened out-of-process execution boundary with resource isolation, upload restrictions, and network/filesystem separation, not mere string validation.
- Graph provenance hashes are integrity fingerprints, not signatures or formal equivalence proofs.
- FX model capture runs Python code; only trusted models should be captured.
- The current shared semantic envelope is not a lossless unrestricted ONNX/MLIR round-trip and does not represent symbolic-shape guards beyond recorded input metadata.
- Observed CPU witness results are not universal numerical equivalence. Float tolerances, data-dependent control flow, stochastic algorithms, training gradients, and novel state effects require additional validation.

## Next integration decisions

1. Make the shared semantic model the first-class typed input for every lowering pass, preserving the full input/output program signatures.
2. Replace the illustrative NumPy 'device' with a genuine independently instantiated hardware or external simulator adapter.
3. Add measured per-device **directional**, batching-aware transfer traces and explicit device residency before making system-level cost claims.
4. Build MLIR translation with round-trip tests using an actual parser/runtime rather than emitting decorative MLIR-looking strings.
5. Add formal verification for narrow supported algebraic rewrites, training gradients and multi-area refraction/inhibition parity.
