# v58 prepared constructed execution

`PreparedMicrostepProgram` admits one straight-line WASM program and reuses
its compact router, immutable decoded data rows, live gate coefficients,
and six provenance digests. Every invocation initializes its own operand
stack, locals, PC, and trace. Routing still evaluates the conditioned gate
at every PC. The data microstep still evaluates the fixed-weight candidates
using the snapshotted coefficients. The existing one-shot API still
supports the dense, compact, and indexed modes.

```python
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

prepared = PreparedMicrostepProgram(program)
first = prepared.run(-17)
second = prepared.run(42, capture_trace=False)
```

The prepared API selects compact indexed routing, compiled ALU candidates,
and fused state by default. Callers can turn off the compiled ALU or fused
state independently. Preparation validates every data microop row, so
a bad decoder row can fail during construction. Supported coefficient
overrides supplied before preparation still reach the live gates.

## Repeated latency and setup

Both arms used compiled ALU, fused state, compact routing, indexed PC, and
the same precompiled microop tokens. One arm admitted each call. The other
reused a prepared object per program. Each arm ran five timed calls per
input and mode after a warm call, with alternating arm order across inputs.
Timings are casewise wall-clock medians.

| Run and mode | Indexed median | Prepared warm median | Paired speedup | Faster cases | Largest slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, traced | 6.89 ms | 5.22 ms | 1.33× | 180/197 | 1.44× |
| 1, untraced | 5.70 ms | 4.16 ms | 1.35× | 185/197 | 1.96× |
| 2, traced | 6.63 ms | 5.09 ms | 1.29× | 187/197 | 1.56× |
| 2, untraced | 5.59 ms | 4.08 ms | 1.36× | 189/197 | 1.37× |

The comparison covers **197 inputs** across 35 programs. Median
one-time preparation across programs was **1.26–1.34 ms**. Shared-library
initialization was timed separately before program preparation and runs.
All times were measured on x86-64 Linux with NumPy 2.1.3, Python 3.12.14,
and one BLAS thread.

Charging each case its program's median preparation cost, prepared execution
is estimated to match or beat the indexed path by the second untraced call
for **163–172 of 197 inputs** and by the fourth for **176–185 inputs**.
The corresponding traced counts are 164 and 173–182. These estimates use
per-input medians and assume the same charge and savings on each call.

The largest slowdowns were replayed with 15 alternating paired samples
per case. The prepared replay was 1.28× as fast with tracing
and 1.45× as fast without tracing on those inputs.
The artifact preserves the original and replay timings.

## Verification

Both runs matched all **3,387** full trace states, output values, step
counts, and corpus token and coefficient identities from the archived
indexed run. The current benchmark also checks each full trace against
the one-shot indexed path, compares core trace fields against v51, v56,
and v57, and checks results against the original WASM in Wasmtime.
The fault campaign detected **10/10** established injections. It used the
prepared executor for the v58 cases. Focused tests covered a shared
session under concurrent calls, a successful call after a fuel trap,
post-preparation caller mutations, and malformed token rejection.

The backend suite finished with 1,215 passed and 20 skipped in 93.44 s.
The first suite run surfaced
a structural guard that inspected only the old entry point. The updated
guard verifies that both entry points delegate to the shared loop and
still checks the gate calls and absence of opcode-specific data dispatch.
The measured executor source is identical across those two revisions.

## Next performance target

A separate warm profile attributed 87% of inclusive untraced
execution time to the shared data microstep. This includes operand and local
transitions, binary checks, and the compiled ALU call.

The next experiment is to move the complete live data microstep into
one native boundary, keeping the same candidate gates, runtime fault
checks, and trace witnesses. The prepared object gives that kernel a
single place to bind admitted rows and fixed coefficients and gives the
benchmark a warm baseline. Measure latency first and separately measure
throughput across independent inputs with one program preparation.

## Reproduce

Use the pinned research environment and a C++17 compiler at the
producing commit recorded in the manifest. Use the verification commit
for the backend suite and focused regression checks.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
for run in 1 2; do
  python -m scripts.benchmark_constructed_runtime_v58 \
    --compiled-alu --fused-state --compact-routing --indexed-pc --prepared \
    --repeats 5 --output "$out/runtime-$run.json"
done
python -m scripts.benchmark_faults_v58 \
  --compiled-alu --fused-state --compact-routing --indexed-pc --prepared \
  --require-all-detected --output "$out/faults.json"
```

Run the archived `summarize.py` with the archive directory or fresh
output directory as its argument from the repository root to check trace
hashes and rebuild the paired timings and crossover summary.

[Artifact manifest](../benchmarks/results/v58-prepared-gates-b563bb4/manifest.json)
