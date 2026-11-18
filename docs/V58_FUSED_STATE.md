# v58 fused operand and local transitions

The optional `use_fused_state=True` path factors the v58 straight-line
stack and register gate equations. It computes the two pop results from
the original cells and two successive mixed one-hot heights. The final
prefix mask clears consumed cells. Push uses the gated inactive slot.
The register contraction combines the live write-mode gate with the
selected local gain. The existing general v55 operand transitions remain
available to the default v58 path and structured executors.

The fused path uses the same admitted binary tensors, order of bound
checks and selected operand gains. It checks live local read and write
gains even when a candidate register action is discarded by the final
mode. It validates the final stack and register bank. The ALU,
program-counter routing and tracing logic use their existing paths.
The implementation is a factored NumPy state contraction and can run
with either ALU backend.

## Matched measurements

One matched timed run covered all 197 admitted straight-line witnesses
using the compiled ALU for both baseline and candidate. Each witness
used one warmup and three timed runs for each mode. The harness
alternated timing order across cases. Profile and allocation collection
ran after timing.

| Mode | Unfused case median | Fused case median | Paired geometric-mean speedup |
| --- | ---: | ---: | ---: |
| Traced | 14.96 ms | 9.29 ms | 1.59× |
| Trace omitted | 13.70 ms | 8.17 ms | 1.67× |

The fused path improved 191 traced cases and 187 untraced cases.
Worst observed regressions were 1.73× traced and 1.64× untraced. Those
samples are retained in the raw records. Profile attribution now
shows exact-binary checks and Python-side routing among the prominent
remaining costs. The measurements concern the admitted CPU research
corpus.

## Verification

Full trace hashes covering all 3,387 states, results, instruction counts,
token identities and coefficient hashes matched the archived compiled-ALU run. The
harness also checked complete traced states against the unfused path,
v51, v56 and v57, and compared final outputs with original Wasmtime.
Unit tests cover every stack depth and both pop/retain gates, active
push and retain, local reads and writes for every slot, operand gain
ablations, local gain ablations, and stack underflow and overflow.
The original ten interventions were detected with the fused state and
compiled ALU enabled for the v58 cases. The four-worker backend suite
finished with 1,202 passed and 20 skipped.

## Reproduce

Use the pinned research environment and a C++17 compiler. Run from the
producing commit identified by the artifact manifest.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py --compiled-alu --fused-state --profile --output "$out/fused.json"
python visuaml-client/backend/scripts/benchmark_faults_v58.py --compiled-alu --fused-state --require-all-detected --output "$out/faults.json"
```

[Artifact manifest](../benchmarks/results/v58-fused-state-863cc60/manifest.json)
