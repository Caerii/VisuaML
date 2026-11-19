# v58 packed structured WASM control

```python
prepared = PreparedMicrostepProgram(
    program, use_native_loop=True, use_packed_scalar=True, use_structured_control=True,
)
result = prepared.run(argument)
```

This mode covers the full existing bounded v51 control family: `block`,
`loop`, `if`, `else`, `end`, `br`, `br_if`, and `return`. The admitted
function shape is `run(i32) -> i32`, with i32 locals, empty block
signatures, explicit fuel, and reviewed branch depths 0–11. The reference
dialect requires exactly one live operand at return. Structured batches
run packed scalar lanes.

## Ownership

- Source-bound v52 route rows and live guard and path gains determine the
  next PC from the actual packed top operand. The condition is sampled
  before `if` or `br_if` consumes it.
- The v54 frame-event matrix and conditional gains select retain, push,
  pop, branch, or return. The v53 action gain gates the selected
  transition. Packed frame coordinates retain the source entry, kind,
  and operand base. Branching restores that base and preserves the
  target frame when its label is a loop.
- The same packed v58 data step implements ordinary instructions.
  Source metadata supplies lexical frame endpoints and structural
  checks. Traces expose post-step stack, locals, complete frame
  coordinates, the live route, and the selected frame event.

Preparation admits source-linked routing and microop tokens, snapshots
the supplied frame coefficients, and records their digest. A prepared
program can be reused for independent single inputs or batches. Trace
storage is bounded by the supplied fuel, including loop traces that
execute more steps than the source contains.

## Evidence

Two serial runs matched **42 inputs across 8 programs**
and **1,665 full states**, including PC, operands, locals, frame
coordinates, route values, and gate events. The replay checks host v51
state, v57 gate state and events, and Wasmtime results. It includes signed
i32 boundary inputs, both `if/else` arms, nested early return, a branch
targeting an `if` label, a branch retaining its target loop, and a
12-frame nest.

The extra branch matrix ran **36 witnesses** across all 12 legal
nesting depths in nested blocks for both `br` and `br_if`, with taken
and untaken conditional cases. Traces matched v57, and result values
matched Wasmtime.
The main replay compared **88 fuel-boundary cases** per run and
verified a nonterminating program's fuel exhaustion independently in
Wasmtime.

Zeroing the live `if` zero-guard gain changed the first divergent next
PC **from 7 to 4 at step 3**. The same input changed its result
**from 22 to 11** in both gate executors under this intervention. Frame
push and conditional event ablations also trapped. The archive preserves
raw WASM bytes, inputs, transitions, and the first-divergence record.

## Latency

These are medians across per-input warm measurements in two serial runs.
The v57 gate reference prepares its tensors on each invocation. The v58
arms reuse prepared programs. The timings describe that combined change
in representation and preparation.

| Program | v57 traced | Packed traced | Packed untraced |
| --- | ---: | ---: | ---: |
| `ifelse` | 16.4–16.7 ms | 68.4–77.8 µs | 14.9–15.2 µs |
| `countdown` | 104.5–107.3 ms | 296.9–352.9 µs | 20.9–23.5 µs |
| `gcd12` | 106.4–134.5 ms | 401.9–417.7 µs | 20.8–20.9 µs |
| `nested_branches` | 14.8–15.1 ms | 45.5–51.8 µs | 15.3–30.9 µs |
| `nested_return` | 5.9–8.9 ms | 39.5–40.3 µs | 14.4–14.6 µs |
| `nested_loop_target` | 23.9–25.4 ms | 122.5–166.8 µs | 15.1–16.2 µs |
| `if_label_target` | 16.7–23.1 ms | 67.0–93.8 µs | 14.8–15.2 µs |
| `maximum_nesting` | 29.5–33.5 ms | 155.1–231.8 µs | 14.0–15.2 µs |

The geometric mean paired traced speedup over these witnesses was
**220–253×**. A separate run that prepared a fresh program in
each arm measured **12.9×** traced. The backend suite finished with **1,272 passed and
20 skipped**.

## Reproduce

Use the source revision in the manifest, the pinned research environment,
and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
for run in 1 2; do
  python -m scripts.verify_packed_control_v58 --repeats 2 --output "$out/runtime-$run.json"
done
```

The verifier exposes the fixture, frame replay, and first-divergence
functions for future branch, loop, and recovery experiments. The
archive also supplies the branch-depth generator and summary script.

[Artifact manifest](../benchmarks/results/v58-packed-structured-control-4b0c13e/manifest.json)
