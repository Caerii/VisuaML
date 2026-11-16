# Compiler Lab v15 — CPU performance engineering

## Scope and honest interpretation

This patch improves the *existing* pure NumPy-AOT CPU interpreter and the bounded
1–4-layer synthetic decoder. It does not claim GPU, real photonic, pretrained
checkpoint, inference-serving, or industry-best throughput. It does not change
the supported operator set or eliminate the shape-specialization requirement.

## What was measured

Before optimizing, cProfile of 15 repeated two-layer single-token invocations
showed substantial interpreter overhead from a full dictionary scan after
**every** instruction, repeated SHA-256 verification, and repeated SSA admission
for every decoded token. The baseline performed 724,426 function calls in
0.306 seconds under the profiler (profiling adds substantial overhead).

The v15 implementation:

1. Replaces the `execute()` per-instruction retained-array dictionary scan with
   an incrementally maintained byte count. This preserves its original
   *logical retained-value* accounting (including tensor aliases), avoiding the
   previous O(number_of_values × instructions) scan.
2. Adds `PreparedNumpyExecutor`. Initialization deep-copies the manifest, takes
   byte-backed immutable snapshots of constants, validates their digests, and
   performs full structural and resource validation once. Repeated calls still
   check every runtime input for shape, dtype, byte budgets and token range.
   The existing `execute()` and `run_bundle()` entry points still reverify
   untrusted bundles on each call. Prepared snapshots must not be treated as a
   sandbox or an immutable compiled artifact signature.
3. Reuses a prepared executor for each stacked-decoder shape-specialization.
   Model checkpoint digest guards remain in place; changing weights requires
   creating a new driver. Source-bound v13/v14 evidence is regenerated.
4. Lowers bounded float32 Conv2d into a read-only convolution-window view and
   groupwise BLAS matrix multiplication, with a 64 MiB im2col scratch admission
   and 256 MiB output admission. Non-float32 / oversized cases preserve the
   earlier explicit kernel-loop implementation.

## Representative matched local results

Single process, OpenBLAS one thread, CPU-only, Python 3.13, NumPy 2.3.5,
PyTorch 2.10.0+cpu. These measurements were made with **25 warm timed trials**
per case, after warmup; compilation and shape-specialization costs excluded.
Values are median milliseconds. No confidence intervals or cross-hardware
comparisons are claimed.

| Synthetic workload | v14 baseline | v15 optimized | Ratio |
|---|---:|---:|---:|
| 1-layer, 1-token decoder | 1.71126 | 0.66927 | 2.56x |
| 2-layer, 1-token decoder | 3.36929 | 0.96532 | 3.49x |
| 4-layer, 1-token decoder | 7.98369 | 1.84482 | 4.33x |
| Conv2d 16→32, 48×48, 3×3 | 3.53460 | 1.05181 | 3.36x |

A separate 70-trial decoder run measured 2.63x / 2.82x / 4.02x for
1 / 2 / 4 layers. This variation is a reminder that CPU microbenchmarks are
sensitive to system load and sample size, particularly for submillisecond
workloads. The benchmark script reports only the current version; comparisons
must use identical hardware, BLAS settings and inputs.

## Reproduction

From `visuaml-client` with Python dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=backend python backend/examples/compiler_perf_bench_v15.py --trials 50 --output perf.json
PYTHONPATH=backend python -m pytest -q backend/tests/test_aot_performance_v15.py
PYTHONPATH=backend python -m pytest -q backend/tests
```

The source's one-shot SHA-256 certificates for v13 and v14 are regenerated on
changes to the compiled interpreter: they capture source identity and finite
numerical witnesses, not an all-input formal proof. The benchmark does not
establish real checkpoint performance or any claimed physical speedup.

## Remaining major performance gaps

- Compile-once dynamic KV cache / paged attention; current per-past-length
  specialization still incurs expensive compile time and cache duplication.
- Fused RMSNorm/RoPE/attention/MLP kernels; current NumPy executes many small
  temporary arrays per layer and materializes repeated KV heads.
- Memory planning/liveness/arena allocation with alias-safe lifetime semantics.
- Auto-tuned CPU backends, vectorized quantization, GPU kernels, and verified
  mixed-precision propagation.
- Prefill throughput, batch-size sweeps, sustained tokens/s, p50/p95 streaming
  latency, peak resident memory, cold compilation latency, and energy baselines.
- Real pretrained weights and tokenizer parity, then hardware calibration and
  physically representative optical models.

Do not merge performance changes on throughput alone: correctness, input guards,
full regression CI, and source-bound evidence regeneration are hard gates.
