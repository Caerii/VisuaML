# v58 constructed runtime validation and tracing

This branch starts from the offline verification evidence at
`322cd520fb440790659c9e3f62962b9e427803fc`. It preserves the source-bound
archives and keeps the separate source-specialization work on its own
branches.

The runtime uses exact equality to zero or one to admit ordinary float64
arrays. Shape and dtype checks stay at each original gate. NaN,
infinities and nonbinary values fail the same admission boundaries.
Array subclasses keep the existing NumPy finiteness and membership
dispatch. The v58 caller may pass `capture_trace=False`. That option skips
materializing trace rows after each transition. State updates, fuel,
gates, result and provenance still use the same execution loop.

The profile covers the full set of 197 straight-line witnesses from the
scaling corpus. Each measured program was checked against v51, v56, v57,
and original Wasmtime. The comparison also checks complete v58 trace
hashes, source, route token and microop token identities, and fixed
coefficient array hashes against the archived baseline implementation.

The baseline profile attributed about 6.01 seconds to ALU evaluation,
3.71 seconds to NumPy membership checks, and 3.50 seconds to operand
transitions. These are inclusive times which can overlap. The candidate
reduced the traced profiler's executor total from 15.22 to 10.95 seconds.
The ALU remained the leading inclusive cost at about 5.66 profiled
seconds. The raw records preserve the full call breakdown. Timing
occurs before profiling. Separate samples cover token construction,
fixed initialization, trace serialization and Python allocation peaks
for the first input per program. These peaks do not measure native
allocation or total resident memory.

Run from the producing code commit and pass the archived reference
backend on `PYTHONPATH` for the baseline runs. The harness verifies
loaded runtime sources against the corresponding producing Git commit.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
out=$(mktemp -d)
mkdir "$out/reference"
git archive 322cd520fb440790659c9e3f62962b9e427803fc visuaml-client/backend | tar -x -C "$out/reference"
script=visuaml-client/backend/scripts/benchmark_constructed_runtime_v58.py
base="$out/reference/visuaml-client/backend"
candidate="$PWD/visuaml-client/backend"
PYTHONPATH="$base" python "$script" --baseline --profile --output "$out/base-1.json"
PYTHONPATH="$candidate" python "$script" --profile --output "$out/candidate-1.json"
PYTHONPATH="$candidate" python "$script" --output "$out/candidate-2.json"
PYTHONPATH="$base" python "$script" --baseline --output "$out/base-2.json"
for trial in 1 2; do
  PYTHONPATH="$candidate" python "$script" --compare "$out/base-$trial.json" "$out/candidate-$trial.json" --output "$out/comparison-$trial.json"
done
PYTHONPATH="$candidate" python visuaml-client/backend/scripts/benchmark_faults_v58.py --output "$out/faults.json"
```

[Artifact manifest](../benchmarks/results/v58-validation-adaa307/manifest.json)

Across two independently launched baseline/candidate pairs, traced
execution improved by 1.26–1.30× by paired geometric mean. Omitting
trace materialization on the candidate was 1.31–1.36× faster than the
original traced baseline. Some individual witnesses slowed down. The
largest observed regression with tracing was 1.49× across the two
runs. Each pair matched 3,387 full instruction states and all 197
results and traces. The original fault campaign detected all ten
injections. Both tracing modes retained the two applicable v58 checks.
