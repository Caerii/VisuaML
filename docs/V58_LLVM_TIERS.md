# v58 LLVM optimization tiers and held-out replay

The optional LLVM specializer accepts `optimization_level=0`, `1`, or
`2`. O2 remains the default. All levels share source admission and emit
the same pre-optimization IR. The level sets both the LLVM pipeline
and target machine optimization settings.

## Setup and throughput

Two timing runs each covered the 35 admitted programs from the existing
scaling corpus. The harness rotated engine order, executed copied and
resident buffers, and checked outputs against the original scalar WASM
in Wasmtime. It used 15 timed samples and one warmup per program, engine,
batch size and buffer mode. LLVM process initialization was measured
separately and took 30–56 ms.

| LLVM tier | Median setup after initialization | Median copied ns/value at batch 16,384 | Paired warm speedup vs. Wasmtime |
| --- | ---: | ---: | ---: |
| O0 | 2.32–2.34 ms | 2.22–2.24 | 1.54–1.55× |
| O1 | 11.14–11.74 ms | 0.531–0.532 | 8.06–8.16× |
| O2 | 11.71–11.99 ms | 0.516–0.523 | 8.24–8.29× |

For these programs O0 cut initialized compile time roughly fivefold,
at the cost of about fourfold lower large-batch throughput relative to
O2. O1 and O2 were close on both measures. These are CPU API timings.

## Held-out decisions and direct replay

A simple selection model used median cost on 27 public and scaling
programs, then evaluated eight held-out programs from the other timing
run. It used a known batch schedule and an explicit cache state.
Swapping the timing runs as calibration and evaluation changed the
recommended cold bulk tier between O1 and O2, and the recommended
cached interactive backend between Wasmtime and C++. That instability
motivated direct replay before considering an automatic policy.

The direct replay held out the same eight programs, used three timed
runs per engine and scenario, and included per-program setup and copied
requests. Every engine ran each schedule on each program. Cache seeding,
engine initialization and result checks were outside the timing. The
final output for each requested batch size was checked against the
original scalar WASM. The selection rule was fixed before replay and
used the first timing run's public and scaling programs.

| Cache state and workload | Model selected | Fastest on held-out programs | Worst time relative to fastest measured |
| --- | --- | ---: | ---: |
| Cold interactive | LLVM O0 | 5/8 | 1.40× |
| Cold moderate | LLVM O0 | 4/8 | 2.27× |
| Cold bulk | LLVM O1 | 4/8 | 1.16× |
| Cached interactive | Wasmtime | 4/8 | 2.44× |
| Cached moderate | C++ | 8/8 | 1.00× |
| Cached bulk | C++ | 8/8 | 1.00× |

Cold means no per-program cached artifact in an initialized process.
Cached means matching C++ and Wasmtime artifacts already exist.
The interactive schedule has 45 batches, mostly of one value.
Moderate has 240 batches including 100 of 16,384 values, and bulk has
1,600 batches including 1,200 of 16,384 values. Batched calls use the
same input prefix for each size and deterministic mixed request order.

O0 is useful when LLVM is already initialized and a program will see a
short sequence of requests. The high worst-case costs on moderate and
cached-interactive programs support retaining explicit backend selection.
For longer uncached work the replay favored O1 or O2. In the sampled
cached moderate and bulk cases, C++ won all sixteen program/scenario
comparisons. Production dispatch requires a wider workload mix and a
policy that accounts for initialization, cache availability and reuse.

## Validation and reproduction

The artifact manifest matches each tier's pre-optimization IR hashes
with the independent scalar Z3 receipts. Z3 proved 33 expressions over
all i32 inputs. The two longer programs retained finite execution
coverage. The proof tool caught the deliberate mutation, and the
reference fault campaign detected all ten injections.

Use the pinned research environment with `llvmlite==0.46.0` and a C++17
compiler. The manifest records the exact producer for each artifact.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_native_batch_v58.py --llvm-tiers --output "$out/run-1.json"
python visuaml-client/backend/scripts/benchmark_native_batch_v58.py --llvm-tiers --output "$out/run-2.json"
python visuaml-client/backend/scripts/evaluate_v58_tier_selection.py --calibration "$out/run-1.json" --evaluation "$out/run-2.json" --output "$out/model.json"
python visuaml-client/backend/scripts/replay_v58_tier_selection.py --calibration "$out/run-1.json" --evaluation "$out/run-2.json" --output "$out/replay.json"
```

[Artifact manifest](../benchmarks/results/v58-tiers-c99f696/manifest.json)
