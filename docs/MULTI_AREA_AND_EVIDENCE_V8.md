# VisuaML v8 — reciprocal assembly parity and evidence-gated placement

Two independent additions live on the **feature branch**, not on `main`.

## 1. Reciprocal neural assemblies (restricted)

`multi_area_bridge.py` constructs a typed, bounded EffectProgram for ordered
A→B then B→A projections. Every projection:

- takes a snapshot of source winners and explicit connection weights;
- computes target raw input by summing active source rows;
- uses deterministic lowest-index tie-breaking for k-WTA;
- applies local Hebbian multiplication with a ceiling;
- writes weights and winners back in explicit effect-token order.

The second projection reads B winners **after** the first has written them.
State arrays are copied; callers' arrays are not mutated. A deliberately
restricted admission requires concrete float32 nonnegative weights no greater
than the ceiling, valid stable neuron IDs, and fixed k.

The local test suite compares five consecutive reciprocal iterations against an
independently coded scalar NumPy oracle. A *separate optional* test imports
`Caerii/assemblies` and compares with its real `NumpyExplicitEngine` under a
fixed input-connectome and schedule. A GitHub Actions pinned-reference job
runs this optional test with an explicit upstream commit.

**Not supported:** stochastic connection generation, changing network topology,
input noise, refraction, Long-Range Inhibition, column scaling, normalization,
GPU kernels, distributed execution or general dynamic control flow. Passing
this test does **not** establish full neural-assemblies equivalence.

## 2. Hardware evidence into a per-node optimizer

`evidence_placement.py` consumes **passing held-out latency evaluations** from
v7's calibration gate. The new adapter estimates host/device operator latency
individually for each graph node using the declared work and output bytes.
This changes v3's previous per-op constant override: two matmul nodes with
different work now receive different model-predicted costs.

To avoid pretending to know transfer costs, a *separate* fitted and validated
`transfer` link model is required. Its prediction is evaluated as one transfer
at a producer's tensor size. Missing device/host operator evidence, failed
holdouts, mixed synthetic/measured labels, or out-of-domain work/bytes are
rejected. Synthetic evidence must be explicitly enabled for experiments.

The planner's time unit is **microseconds**. It uses the existing exact binary
min-cut optimizer for its **additive objective**, with node-specific unary
costs and per-edge transfer costs. It is *not* a latency measurement, not a
real hardware backend, not a proof of optimal schedule under concurrency, and
not an Ayo semiconductor benchmark.

The output includes SHA-256 fingerprints for the input IR, capability contract,
evidence datasets and output cost assumptions. Source labels remain
provider-declared; hashes prove consistency with a report, not authenticity.

## Run checks

From `visuaml-client`, in a suitable Python environment:

```bash
python -m pytest -q backend/tests/test_multi_area_bridge_v8.py backend/tests/test_evidence_placement_v8.py
```

To exercise actual upstream parity, install the *pinned* `Caerii/assemblies`
reference version and re-run `test_multi_area_bridge_v8.py`. This test is
skipped without that dependency; CI must report the result separately.

## Next engineering gates

1. Broaden reference parity to independent source areas, multiple simultaneous
   fibers, and specific homeostatic regimes with proven admission boundaries.
2. Measure and validate actual backend kernels and transfers, using raw traces
   and holdout protocols rather than fictitious cost coefficients.
3. Introduce explicit state-effect capabilities into backend lowering and
   require executable proof-of-correctness witnesses before promotion.
4. Surface graph-level evidence receipts and numerical uncertainty in the
   React Compiler Lab so scientists can inspect *why* a lowering was chosen.
