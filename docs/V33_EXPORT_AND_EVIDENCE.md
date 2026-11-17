# v33 — Exported computation and evidence-carrying transformations

*A useful compiler accumulates knowledge that survives individual kernels, models, and experiments.*

This is an **experimental, bounded integration milestone**. The existing v32 generated native C++17 compiler accepts a closed subset of the shared Semantic IR. v33 adds (1) a strict static `torch.export.ExportedProgram` importer and (2) a numerical transformation-evidence registry that can bind before/after programs to two independent oracles and preserve negative counterexamples.

## 1. Exported program admission, not casual tracing

`torch.export` produces an ATen graph, lifted parameters, graph-signature input/output contracts, and potentially dynamic shape constraints. These components cannot be flattened into an unguarded node list without losing assumptions.

The `native_export_frontend_v33.py` importer receives an **already-produced** `ExportedProgram` and one trusted finite CPU `float32` example. It does not call the module forward, and it refuses dynamic range constraints, input/buffer mutation outputs, buffers/constants, multiple user inputs/outputs, and unrecognized ATen targets.

The initial admitted vocabulary is intentionally tiny:

| Functional ATen | Generated Semantic IR | Limit |
| --- | --- | --- |
| `aten.linear.default` | `matmul` | rank-2, bias-free, explicit transposed weight |
| `aten.mm.default`, `aten.matmul.default` | `matmul` | rank-2 only; operands must be known, or a source-bound lifted weight parameter |
| `aten.conv2d.default` | `conv2d` | NCHW/OIHW, ungrouped, undilated, zero-padded, no bias |
| `aten.relu.default` | `relu` | pure, shape-preserving |
| `aten.add.Tensor` | `add` | exact equal shapes, unit alpha, no broadcasting |

Every value must have static float32 tensor metadata. The existing v32 native admission layer is invoked before code compilation, enforcing single-region SSA definitions, shape contracts and resource limits. Parameter tensors are **explicit named runtime inputs**, with a SHA-256 of their exact contiguous bytes included in a canonical export-contract fingerprint. The binding's `validated_inputs()` method rejects changed parameters or runtime input shapes before entering the native ABI.

A captured static contract is valid only for its declared geometry and dtype. The current importer **does not translate dynamic `torch.export` shape guards**; those exports fail closed. Silently treating a symbolic tensor as fixed would be scientifically unsound.

### Trust boundary

The upstream `torch.export.export(model, ...)` call can execute the model's Python during capture. It must therefore be used only with **trusted local modules**. Passing `trusted=True` to the VisuaML importer is explicit acknowledgement, not sandboxing. No arbitrary uploaded modules, unknown remote code, unsafe pickle, or custom operator execution are admitted by this workflow. Python callers and installed native compilers remain in the trusted computing base.

## 2. Reusable transformation evidence

The `transformation_evidence_v33.py` ledger introduces a deliberately narrow first rule, `relu(relu(x)) → relu(x)`, with SSA single-use and effect-free guards. The second result name remains stable and the first intermediate is eliminated only when no other consumer needs it.

The witness evaluator compiles both the source and candidate programs into **separate native C++17 binaries**, executes them on bounded concrete tensors, checks both against independently supplied NumPy and PyTorch reference implementations, and checks them against each other. It records:

- Original and rewritten Semantic IR hashes, generated C++ and native binary fingerprints, compiler flags and target.
- Exact declared rewrite identifier and numeric tolerances.
- Input SHA-256 for each witness and maximum absolute errors for the oracle-oracle, baseline-oracle, candidate-oracle and candidate-baseline comparisons.
- Explicit `passed_finite_witnesses` or `rejected` status; rejected records retain a bounded concrete counterexample for reproduction.
- A canonical JSON record digest and a local integrity-check function.

This is **finite empirical evidence**, not a formal equivalence proof or a cryptographically signed publication. The SHA-256 allows local accidental-modification detection; an adversary could replace a record and recompute its digest. Authenticated publication, signer policy and independent approval must be separate future gates.

The current rule is deliberately conservative. Future rewrites should be independent, registered operators with explicit preconditions and tests, not arbitrary code submitted to the evidence evaluator.

## 3. Evidence-gated search, not automatic adoption

The `optimization_search_v33.py` harness builds on those receipts. It first verifies a candidate with two independent numerical references; a rejected candidate **never enters performance measurement**. After successful finite witnesses, it recompiles the source and candidate programs, replays the previously tested input, verifies numerical parity again and records alternating-order warm CPU measurements for both generated native binaries.

It preserves individual samples, medians, build/compiler provenance, the embedded numerical receipt and the candidate rule identity. Every result is labeled `exploratory_timing_after_finite_numerical_witnesses`, not `production_faster`. An inapplicable rule returns a `not_applicable` record rather than pretending a transformation occurred, and an oracle disagreement returns `rejected_no_performance_measurement` with its counterexample.

This closes one small but real loop: **capture → admit → propose → independently verify → measure → retain the evidence**. It is deliberately not a general e-graph optimizer, autoscheduler, signed certificate authority, native device profiler or formal translation validator.

## 4. Reproduce the end-to-end witness

With a trusted installed C++17 compiler and the pinned Python dependency set:

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_native_export_frontend_v33.py backend/tests/test_transformation_evidence_v33.py backend/tests/test_optimization_search_v33.py
python -m pytest -q backend/tests
PYTHONPATH=backend python backend/examples/export_evidence_demo_v33.py --seed 3303 --trials 7 --output /tmp/visuaml-v33-evidence.json
```

The demonstration uses three locally defined trusted synthetic modules: a residual MLP, a residual Conv2D network and a double-ReLU program. It exports the first two, compiles both through v32's graph-generated native engine, checks against eager PyTorch on two different input tensors per model, and records the generated C++ source and full compiler fingerprints. The double-ReLU program is rewritten and separately cross-checked against NumPy and PyTorch; its accepted finite witness then gates an experimental before/after native timing comparison. The [v33 GitHub workflow](../.github/workflows/native-evidence-v33.yml) publishes a versioned JSON artifact upon successful CI.

**These are synthetic models, not two pinned real pretrained architecture families.** They do not establish arbitrary `torch.export` support, dynamic-shape compilation, low-level GPU code generation, generalized stateful effects, speedup over external production compilers or physical hardware economics.

## 5. Research evidence in the browser

The new read-only [`/research` route](../visuaml-client/src/ui/ResearchEvidence/ResearchEvidencePage.tsx) loads a versioned static [v33 observation snapshot](../visuaml-client/public/evidence/v33-observation.json). It does **not** need the Python API or execute user-submitted models. The page displays the source revision, source-generated native binary fingerprints, finite numerical errors, accepted rewrite witness status, and the observed non-winning experimental timing. It links to the exact GitHub CI workflow that produced the original data.

This small public JSON is an explicitly **curated summary** of a verified CI artifact, not the entire numerical receipt or a cryptographically signed record. The full verified CI JSON is preserved at [benchmarks/export_evidence_observation_v33.json](../benchmarks/export_evidence_observation_v33.json) with a separate SHA-256 [origin manifest](../benchmarks/export_evidence_observation_v33.origin.json). The original GitHub Actions artifact has limited retention; this permanent source-bound record is the durable reproducibility source. The public snapshot is checked against its complete archived bytes by Python regression tests. The page validates the read-only snapshot schema and refuses unknown provenance links rather than silently rendering unsupported content.

GitHub source changes alone do **not** establish deployment. The user-visible research route becomes accessible on `visuaml.com` only after the production host deploys an appropriate client build and supports direct SPA navigation. Hosting verification remains in [issue #8](https://github.com/Caerii/VisuaML/issues/8).

## 6. Architecture of the compounding loop

```text
trusted PyTorch exported program
    ↓ signature, parameters, shape guard admission
SemanticProgram with source and weight hashes
    ↓ closed-world operator validation
generated C++17 native execution
    ↓ independent NumPy / eager PyTorch witnesses
evidence record: pass / reject / counterexample
    ↓ reviewed acceptance gate (future)
reusable search and transformation records
```

The next high-value development is to make this a *general experiment protocol* across more op families and real pinned checkpoints, with provenance-backed storage, transformation search, replay, a reviewer authority layer and a Compiler Lab visualization connecting individual operations to source, generated code and counterexamples.

The **public roadmap weights remain unchanged**. The approved public acceptance percentage must never rise merely because a research PR exists.

Deployment to `visuaml.com/roadmap` remains a separate unverified production task in [issue #8](https://github.com/Caerii/VisuaML/issues/8). 
