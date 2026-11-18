# v58 native prepared data microstep

`PreparedMicrostepProgram(program, use_native_microstep=True)` evaluates
the prepared data microstep in one C++17 call per data instruction.
The executor still routes the PC through the existing live compiled router
and checks the final structural `end` in Python. The native data kernel
uses the selected typed row and the actual fixed coefficients for stack
pops, all six ALU candidates, local read and write gates, and the final
stack push. It contracts the one-hot state and address gates to their
active coordinates and writes the owned state buffers only after the
microstep passes its checks.

Preparing a program binds immutable source-authenticated rows and arrays
to the native ABI. Coefficient type and layout checks and the compiled
arithmetic coefficient admission run at preparation. The binary operand,
microstep, ALU output, and local gains are still checked and applied live.
Each call creates its own stack, locals, PC, fuel, and trace. The existing
prepared Python path remains the default and serves as the reference.
Native microsteps require the compiled ALU and fused state.

## Warm latency

The comparison covers 197 inputs across 35 straight-line WASM programs.
Both modes reuse the same prepared program per workload, and both use
compiled ALU candidates, fused state, and compact indexed routing.
The benchmark recorded five calls per input and mode after warmup and
alternated timing order across inputs. The timings below are medians
of per-input wall-time medians. Speedup is the geometric mean of the
per-input ratios.

| Run and mode | Prepared Python | Native | Paired speedup | Faster inputs |
| --- | ---: | ---: | ---: | ---: |
| 1, traced | 5.607 ms | 1.380 ms | 4.07× | 197/197 |
| 1, untraced | 4.235 ms | 0.513 ms | 7.95× | 197/197 |
| 2, traced | 5.773 ms | 1.437 ms | 3.82× | 196/197 |
| 2, untraced | 4.434 ms | 0.547 ms | 7.95× | 197/197 |

The lone traced slowdown (2.14×) was replayed with 15 interleaved
calls per arm. The native path ran 3.38× faster in the replay.

Median one-time preparation was **1.49–1.50 ms** for the native mode
and **1.32–1.33 ms** for the prepared Python path.
Both arms used the same shared library. Its measured build and load took
0.63–1.00 s before the timed preparations and calls. The measurements
used x86-64 Linux, Python 3.12.14, NumPy 2.1.3, and one BLAS thread.

## Verification

The two timing runs matched all **3,387** archived full trace states,
including route values, as well as result, step count, and token and
fixed-weight identities. The harness also rechecked full records against
the current prepared Python path, compared core state fields against
v51, v56, and v57, and checked output against original WASM in Wasmtime.
The established campaign detected **10/10** injected faults, with the
native data gate active for the two v58 cases.

A separate coefficient experiment compared 312 single-coordinate
interventions spanning operand, local, microstep, and ALU output gains.
Both engines returned the same trap type and message in 258
cases, the same changed trace or result in 6, and the same
baseline behavior in 48. The raw records include controls
whose changed coefficient was inactive.

The backend suite finished with **1,223 passed and 20 skipped**.
Focused checks cover stack bounds, malformed tokens, run isolation under
concurrency and after fuel exhaustion, caller-owned snapshots, and live
local, operand, and ALU gains.

## Remaining cost

A separate warm profile attributes 56% of inclusive untraced time
to validated top-of-stack reads used by routing and final result decoding.
Trace and result i32 decoding account for 57% of inclusive traced
time. These are profiler shares.

The next experiment is to run indexed routing and data microsteps in one
native loop, keeping the routed PC, fuel and per-step evidence. That
would remove one Python boundary per microstep, share admission of the
live operands between routing and data, and give independent-input
batching one native entry point per prepared program and batch.

## Reproduce

Run the source-bound benchmarks from the producing commit recorded in the
manifest with the pinned research environment and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
opts=(--compiled-alu --fused-state --compact-routing --indexed-pc --prepared --native-microstep)
python -m scripts.benchmark_constructed_runtime_v58 "${opts[@]}" --profile --repeats 5 --output "$out/runtime-1.json"
python -m scripts.benchmark_constructed_runtime_v58 "${opts[@]}" --repeats 5 --output "$out/runtime-2.json"
python -m scripts.benchmark_faults_v58 "${opts[@]}" --require-all-detected --output "$out/faults.json"
```

The archive includes the warm profiler, coefficient parity replay, and
trace comparison scripts.

[Artifact manifest](../benchmarks/results/v58-native-microstep-4e1cb86/manifest.json)
