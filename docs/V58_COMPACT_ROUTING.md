# v58 compact PC routing and binary admission

The exact-binary check still compares every ordinary float64 value with
zero and one. For arrays of up to 128 elements, it counts matching
elements. Larger arrays keep the previous boolean reduction. The
threshold was chosen after an exploratory run showed a slowdown on the
largest table when counting was used at every size. Array subclasses
continue through the existing NumPy dispatch in the shared validator.

The opt-in `use_compact_routing=True` backend snapshots plain float64
arrays and validates the canonical dense source-token tables before
encoding each active row as two target indices and three guard bits.
Each native call verifies its live PC and condition, evaluates the v49
ReLU zero-test with the supplied AND coefficients, then mixes the
selected row's live guard and path gains. A valid result becomes the
one-hot PC consumed by the executor. Invalid outputs fail closed.
The dense router remains the default. Both paths use the same WASM
admission and v58's straight-line execution contract.

## Measured components

The isolated binary benchmark used nine timed samples per size and case,
with 2,048 checks in each sample. For valid arrays of 4, 32 and 97
elements, the candidate measured 1.42×, 1.48× and 1.29× faster than the
former reduction. Some large and invalid cases measured slower. The
exploratory and final results both retain those timings.

The isolated routing microbenchmark exercised active rows plus the
terminal sink on countdown and GCD, with zero and signed nonzero
conditions. The occupied route coordinates and diagnostic values
matched the dense router before timing. Compiled setup was warmed.
The compact admission timings include checking the source-bound token
tables and building the program-specific compact representation.

| Fixture | Dense median per route | Compact median per route | Compact admission and preparation |
| --- | ---: | ---: | ---: |
| Countdown | 97.63 μs | 6.39 μs | 0.52 ms |
| GCD | 80.71 μs | 7.92 μs | 0.50 ms |

These fixtures exercise the shared routing equations for structured
source. The end-to-end v58 executor benchmark uses its existing admitted
straight-line corpus.

## End-to-end replay

Both runs compared dense and compact routing with the same binary
validator, compiled ALU and fused state. Each run timed three paired
calls per mode for all 197 admitted inputs. The no-trace compact
path also omits construction of the per-step routing diagnostics.

| Run and mode | Dense case median | Compact case median | Paired speedup | Faster cases | Worst regression |
| --- | ---: | ---: | ---: | ---: | ---: |
| First, traced | 8.51 ms | 7.43 ms | 1.16× | 149/197 | 1.62× |
| First, no trace | 7.32 ms | 6.35 ms | 1.16× | 150/197 | 1.48× |
| Repeat, traced | 8.73 ms | 7.59 ms | 1.16× | 157/197 | 1.48× |
| Repeat, no trace | 7.61 ms | 6.39 ms | 1.19× | 161/197 | 1.69× |

A timing attempt that overlapped with the regression suite was excluded.
The recorded repeat ran after the test workers had finished.
The two runs matched all 3,387 archived states, results, token hashes,
and coefficient hashes. The original ten fault injections were detected.
Focused routing tests also verified that changing the guard or zero-test
gains changes the live route and that a missing path gain, invalid PC,
nonbinary input, or altered targets fail closed. The full backend
suite finished with 1,207 passed and 20 skipped.

## Reproduce

Use the pinned research environment and a C++17 compiler at the
producer commit in the manifest.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_binary_routing_v58.py --output "$out/micro.json"
python visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py --compiled-alu --fused-state --compact-routing --profile --output "$out/runtime.json"
python visuaml-client/backend/scripts/benchmark_faults_v58.py --compiled-alu --fused-state --compact-routing --require-all-detected --output "$out/faults.json"
```

[Artifact manifest](../benchmarks/results/v58-compact-route-0e843ad/manifest.json)
