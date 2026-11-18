# v58 serial native batch execution

`PreparedMicrostepProgram(program, use_native_loop=True).run_many(...)`
shares one admitted program and makes one native call for a batch of
admitted inputs. The native batch calls the same loop for each input,
in order, with a fresh PC, operand stack, and locals. It returns a
`MicrostepExecution` for success and a `MicrostepBatchFailure` containing
the corresponding exception for an input-level failure.

```python
prepared = PreparedMicrostepProgram(program, use_native_loop=True)
outcomes = prepared.run_many([7, -17, 42], fuel=[400, 1, 400], capture_trace=False)
```

A single integer fuel limit applies to every input. An iterable supplies
one limit per input. The trace defaults to the format used by `run`.
Untraced batches allocate no trace buffers. Traced batches stage
post-step records in buffers whose size grows with the number of inputs
and the admitted program length.

## Throughput

Two serial runs measured five samples of each mode after warmup, with
one prepared program per timing arm. The three workloads had 4, 14, and
82 steps. Batch inputs cycle through each workload's reviewed arguments.
Both arms return full execution records. For the preparation-inclusive
time, each measured call prepares one program and then processes the
whole input list. The shared-library build and load precede all timings.

The table reports the range across runs of the geometric mean of the
three paired speedups. Values above 1 favor batching.

| Inputs | Warm untraced | Warm traced | With preparation, untraced | With preparation, traced |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.76–0.88× | 0.88–0.93× | 0.92–0.97× | 0.93–1.17× |
| 8 | 1.00–1.05× | 0.84–1.06× | 0.93–0.94× | 0.92–1.19× |
| 32 | 1.09–1.09× | 1.17–1.30× | 0.87–1.04× | 0.94–1.12× |
| 128 | 0.99–1.19× | 1.23–1.25× | 0.99–1.14× | 1.26–1.35× |
| 1,024 | 1.06–1.09× | 1.12–1.18× | 1.05–1.12× | 1.08–1.17× |

At 1,024 inputs, untraced batching delivered 16,363–18,640 inputs per second across the
three-workload median. Median total batch time, including preparation, was 57.2–59.7 ms.
The 4-step workload gained 1.24–1.30× warm and untraced at this size, while the
82-step workload measured 0.99–1.02×. The smaller batches showed variation across
runs. At size one, the batch API was slower in both warm modes.

## Verification

Both runs verified **197 inputs and 3,387 trace states** against the
native single-input loop, archived trace hashes, and original WASM in
Wasmtime. Focused tests covered traced and untraced mixed batches with
invalid inputs, differing fuel limits, fuel exhaustion, live routing
and final-stack traps, exact result order, and concurrent calls using
one prepared program. The backend suite finished with **1,236 passed**
and **20 skipped**.

## Native profile

An instrumented C++ executable replayed 160 untraced batches of 1,024 inputs
for `heldout_133`. The ALU candidate function accounted for 50%
of the sampled native self time. Its signed comparison helper accounted for
another 15%. The profile used the same admitted rows and
coefficients. The archived throughput measurements used the normal shared
library. This profile points to evaluating the same six ALU candidates across
inputs as the next performance experiment.

## Reproduce

Use the manifest's producing revision, the pinned research environment,
and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
for run in 1 2; do
  python -m scripts.benchmark_native_batch_loop_v58 --output "$out/runtime-$run.json"
done
```

The archive contains the raw two-run timing data, the summary script, and
the separate instrumented native profile.

[Artifact manifest](../benchmarks/results/v58-native-batch-loop-99f0aa9/manifest.json)
