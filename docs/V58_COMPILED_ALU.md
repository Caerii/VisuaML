# v58 compiled constructed ALU

The opt-in `use_compiled_alu=True` path evaluates all six existing v49
float64 gate candidates in a shared C++ CPU kernel. It reads the admitted
AND, XOR, adder, and carry-staircase coefficients and their live binary
inputs. Candidate selection, output gain checks and data movement remain
in the v58 gate path. This preserves the same shared-weight transition and
covers the same admitted straight-line WASM. The NumPy ALU remains the
default, including support for experimenting with alternate arithmetic
coefficients. The compiled path explicitly admits the canonical v49
arithmetic coefficients and rejects modified ones at that boundary.

## Wall time and correctness

A single matched run timed both ALU implementations, with and without
tracing, for all 197 admitted scaling witnesses. Each recorded mode
used three timed calls after warmup. It checked 3,387 states against the
NumPy v58 execution, v51, v56 and v57, plus results from original
Wasmtime execution. Source, route tokens, microop tokens and weight
hashes matched the archived baseline. Focused tests compare all six
candidate vectors on signed edges and deterministic random inputs.

| Mode | Median case time | Paired speedup vs. same-mode NumPy | Paired speedup vs. original traced baseline |
| --- | ---: | ---: | ---: |
| Traced | 14.62 ms | 2.18× | 2.68× |
| Trace omitted | 13.29 ms | 2.30× | 2.95× |

The compiled path was faster than same-mode NumPy on 196 of 197 witnesses
in each mode. The largest same-mode no-trace regression was 2.30× on
`generated_07` at input `-1`. That case also showed large timing changes
in the traced NumPy and compiled measurements. The raw records retain
the outlier. A focused 15-pair replay on the same program and input
measured 2.07× faster without tracing and 2.10× faster with tracing.
The replay leaves the original outlier unexplained.

The compiler and library setup took 219 ms once per process. Across the
faster pairs, the median estimated break-even was 13 calls in either
trace mode relative to optimized NumPy. Compiler
flags, target and binary checksum are recorded. The traced inclusive
profile placed ALU evaluation at 0.58 seconds, down from 5.66 seconds
in the optimized NumPy profile. Operand transitions were the largest
remaining inclusive component at 2.23 seconds. These profiler totals
overlap other categories. Separate raw timings cover token setup,
coefficient initialization and trace serialization. Python allocation
peaks are captured separately per first input and exclude native RSS.

The ten original fault interventions were detected. The two v58
interventions ran with the compiled ALU enabled, and eight v57 cases ran
on their original engine. Additional tests check modified arithmetic
coefficients and output gains with the compiled candidate path.

## Reproduce

Use the pinned research environment and a local C++17 compiler. From
the producing code commit listed in the manifest, run

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py --compiled-alu --profile --output "$out/compiled.json"
python visuaml-client/backend/scripts/benchmark_faults_v58.py --compiled-alu --require-all-detected --output "$out/faults.json"
```

[Artifact manifest](../benchmarks/results/v58-compiled-alu-b41a9ff/manifest.json)

## Test suite throughput

On the same source, the backend suite took 238.92 s serially and
85.53 s with four pytest workers, a 2.79× wall-time improvement. Both
runs had 1,198 passes and 20 skips. The slowest individual serial
cases involved native CPU prefill and Compiler Lab service execution.
The duration lists and complete logs are archived. Four workers fit the
measured environment's four-core quota.

For the same coverage with parallel execution, install the optional
`pytest-xdist==3.8.0` in the research environment and run from
`visuaml-client`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q -p no:cacheprovider --tb=short --durations=20 backend/tests
python -m pytest -q -p no:cacheprovider --tb=short --durations=20 -n 4 backend/tests
```
