# v58 prepared native execution loop

`PreparedMicrostepProgram(program, use_native_loop=True)` runs the admitted
straight-line program in one C++17 call. The optional per-instruction
native mode remains available as the comparison path. The loop mode uses
the same compact routing tables, live route gains, decoded data rows, and
data-gate implementation. It evaluates all six ALU candidates on every
data step. Routing runs before the data transition, and fuel includes the
final structural step.

The loop owns its stack and locals for one invocation. Input encoding
starts with an empty, zero-filled stack and binary i32 locals. The shared
data transition admits the selected gains and output words before writing,
checks both stack bounds and clears the inactive suffix. Its successful
outputs preserve the active-prefix invariant. The loop can use that
invariant to read the top word directly. The standalone microstep still
validates the externally supplied stack. Conditional route validation
still runs for each step. Prepared data and route snapshots are shared
across invocations.

Traced execution writes bounded integer state snapshots and floating-point
route details, then reconstructs the existing trace dictionaries in Python.
Untraced execution allocates no trace buffers. The v58 admission boundary
remains straight-line WASM.

## Warm latency

Each of two serial runs measured five calls per input and mode after
warmup. The paired arms reused prepared programs and used the same
compiled library, gate coefficients, and routing. The table gives medians
of per-input wall-time medians and geometric means of per-input speedups.

| Run and mode | Per-instruction native | Native loop | Paired speedup | Faster inputs |
| --- | ---: | ---: | ---: | ---: |
| 1, traced | 1,345.1 μs | 70.9 μs | 18.46× | 197/197 |
| 1, untraced | 509.2 μs | 46.6 μs | 11.34× | 197/197 |
| 2, traced | 1,350.7 μs | 71.4 μs | 18.50× | 197/197 |
| 2, untraced | 504.7 μs | 46.5 μs | 11.30× | 197/197 |

Median one-time preparation was **1.38–1.42 ms** for the loop and
**1.43–1.44 ms** for per-instruction native execution. The shared
library build and load, **0.72–0.77 s**, was outside the measured
warm calls and preparation. See the manifest for exact host and library
versions.

All inputs were faster in both modes and runs. The smallest paired
speedup was 2.66×. A separate same-session comparison against prepared
Python measured **87.4× untraced** and **71.0× traced**
across the same 197 inputs. That comparison timed both arms directly.

## Verification

Both timing runs matched all **3,387** archived full trace states across
197 inputs and 35 programs. Results, step counts, provenance, and trace
hashes matched the per-instruction native path. The harness also compared
core trace fields against v51, v56, and v57 and checked results against
Wasmtime running the original WASM.

The established campaign detected **10/10** injections, with the native
loop active in the two v58 cases. The separate coefficient replay matched
all **312** outcomes across operand, local, microstep, and
ALU output gains. Of those, 258 trapped, 6 changed a trace or
result, and 48 retained baseline behavior. The focused suite also
ablated an active routing gain and checked the live condition and exact
fuel boundary.

The fuel sweep compared **1,576** executions across the corpus,
including 788 expected fuel traps and 788 successful
executions, with and without tracing. The full backend suite passed
**1,231 tests**, with 20 skipped.

## Remaining costs

Route dictionary assembly accounted for 13% of inclusive traced
time in the separate warm Python profile. The profile groups the C++
execution with the Python call that invokes it.

The program now crosses the Python/C++ boundary once per invocation,
giving independent-input batching a single entry point at which to
measure reuse and amortization. The traced profile includes the Python
cost of building the public dictionaries.

## Reproduce

Use the producing source revision in the manifest, the pinned research
environment, and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
opts=(--compiled-alu --fused-state --compact-routing --indexed-pc --prepared --native-microstep --native-loop)
for run in 1 2; do
  python -m scripts.benchmark_constructed_runtime_v58 "${opts[@]}" --repeats 5 --output "$out/runtime-$run.json"
done
python -m scripts.benchmark_faults_v58 "${opts[@]}" --require-all-detected --output "$out/faults.json"
```

The archive includes scripts for verifying paired trace identities,
comparing against prepared Python, and checking coefficient and fuel behavior.

[Artifact manifest](../benchmarks/results/v58-native-loop-16c2814/manifest.json)
