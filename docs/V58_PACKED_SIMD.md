# v58 packed AVX2 batches

`PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True,
use_packed_simd=True)` enables the experimental packed AVX2 batch path.
It shares a program across eight independent `uint32_t` lanes and reuses
the scalar data-step template, bounded stack depth, routing gains, and
arithmetic candidate equations. AVX2 is detected at runtime.

SIMD covers complete, untraced eight-input groups with sufficient fuel.
The admitted straight-line routing rows have an input-independent
successor. Every SIMD step checks the live routing coefficients and uses
the same data-gain checks as scalar execution. The scalar path handles
short tails, traced requests, unavailable AVX2, and groups needing
per-input fuel or trap handling. Each input retains its own stack words,
locals, and result. Both arms return the same public result records.

## Warm throughput

Two serial runs timed both batch modes on three programs with 4, 14,
and 82 steps. Every row summarizes paired medians from five samples per
program and arm. Size one calls the same packed scalar `run` in both
arms. Preparation-inclusive measurements prepare a fresh program in
each arm at the start of a call. Both arms use the same loaded library.

| Inputs | Warm speedup | With preparation |
| ---: | ---: | ---: |
| 1 | 0.98–1.00× | 0.93–0.96× |
| 8 | 1.05–1.28× | 0.92–0.95× |
| 32 | 1.20–1.25× | 0.78–1.29× |
| 128 | 1.11–1.30× | 1.02–1.16× |
| 1,024 | 1.22–1.27× | 0.99–1.23× |

At 1,024 untraced inputs the median program reached **314,583–408,150 inputs per second**.
Warm speedup was **1.22–1.27×** across the three paired workloads. Smaller
batches and preparation-inclusive timings showed more variation. AVX2 stays opt-in.

A separate 21-pair benchmark at size 1,024 measured **1.99×** with
preallocated buffers at the native call boundary and **1.25×** through the public API.
The native-call measurement excludes Python input validation, array allocation,
and construction of the per-input result records.

Traced calls use the scalar path in both arms. Their archived timings
are retained as a control and fluctuate with trace allocation and host
load. Timed programs reuse reviewed input values to fill each batch.

## Verification

Both runs matched **197 inputs and 3,387 trace states** against the
packed scalar path, archived traces, and original WASM in Wasmtime.
The benchmark repeated each admitted fixture's witnesses in untraced
groups to exercise the vector path even on programs with few inputs.
Packed scalar and SIMD agreed on **312 coefficient
interventions** and **1,576 fuel cases**. The focused tests cover all
six ALU operations, signed boundary values, incomplete groups, short
fuel limits, and live routing and data effects. The backend suite
finished with **1,260 passed and 20 skipped**.

Disassembly of the benchmark's kernel binary confirms 256-bit packed
integer add, subtract, multiply, and equality instructions in the
AVX2 function. The scalar lowering's arithmetic check also passed
at this revision.

## Reproduce

Use the source revision in the manifest, the pinned research
environment, and a C++17 compiler. Performance replication requires
an AVX2-capable host.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
for run in 1 2; do
  python -m scripts.benchmark_native_batch_loop_v58 --compare-simd --output "$out/runtime-$run.json"
done
```

The archive includes the preallocated timing, disassembly check,
fuel and coefficient replay, and summary scripts.

[Artifact manifest](../benchmarks/results/v58-packed-simd-83515bc/manifest.json)
