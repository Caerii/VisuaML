# Research execution ledger

## v57/v58 offline audit

The audit started at PR #46, `f7326d4f709ff997f822849e4bd650c88a30f2d1`.
It contains PR #45, `1be8614b8b8adc1de6e5aab3af1ff989c902dc40`, which
descends from `main` at `f0b883f5ed99cddd5999843c93b82bdb7d8fa2fd`.
There were 13 changed files from main to PR #45 and 30 more changed paths
from PR #45 to PR #46. The checkout contains no active Actions workflows.

The recorded evidence was produced at
`c3dd17fde2e996d29c1270136b543b6fee1011fc`, tree
`1cfc0fe02893796ed8ae227de136f121e9d25b81`. The runtime was Linux x86-64
with Python 3.12.14, NumPy 2.1.3, PyTorch 2.9.1+cpu, Wasmtime 36.0.0,
Z3 4.15.3, pytest 8.3.4, and Ruff 0.11.13.

| Gate | Status | Observation |
| --- | --- | --- |
| v57 state parity | Verified | 78 focused tests. Across 19 source programs and 138 witnesses, v57 agreed with v56 and v51 on 3,026 instruction states and with original Wasmtime on outputs. A separate replay validated archived WASM and recomputed the v51 states from its bytes. |
| v58 straight-line parity | Verified | 67 focused tests covered state parity, original Wasmtime, scaling workloads and fault handling. The smoke runner recorded four witnesses across two programs and 56 transitions. v58 covered two witnesses because one program uses structured control. |
| Fault campaign | Verified | All ten activated injections were detected or rejected. |
| Strict recovery | Verified | 13 source programs, including eight held-out programs, received typed-source Z3 equivalence results. Independent replay checked 156 finite executions and the recorded v58 traces. |
| Failure-complete recovery | Verified | 14 attempts, 13 verified, one cubic grammar rejection, and zero invalid attempts. A separate archive audit verified the denominator and rechecked all successful receipts. |
| Frontend | Verified | The production build passed under Node 20.20.2 and pnpm 10.34.9. All 20 frontend tests passed. |

### Diagnosed failures

- On the PR #46 source, the offline v58 preflight stopped before tests
  because Ruff rejected `RUFF_NO_CACHE=1`. After using the supported
  boolean it reported three lint errors. The corrected entrypoint now
  passes syntax and the v58 lint target.
- The first v57 focused run reported 76 passes and two failures. A
  missing ALU selector reached the v56 ALU error path before being rejected.
  The fixed decoder checks selector presence against the admitted opcode.
  A separate negative test assigned the value already in the token. It
  now changes the bit before exercising canonical admission.
- Six documentation link checks referenced workflow files removed from
  main. The docs now reference the exact historical Git revision and all
  11 link checks passed.
- The first round-trip producer emitted 13 solver receipts, but the
  separate verifier rejected the archive. All hypotheses rechecked as
  UNSAT, but SMT-LIB serialization hashes depended on earlier solver
  queries. Each source proof now gets its own Z3 context and the strict
  and failure-complete archives replayed in separate processes.
- Extra Ruff coverage on the changed v50 module flagged an existing
  `E731` assignment in `emit_canonical_wasm`. The offline v58 lint target
  passed.

### v58 execution ownership

| Mechanism | Constructed control | Host responsibilities |
| --- | --- | --- |
| Instruction selection | The v52 one-hot PC gate emits the live next PC. The v57 token table supplies the selected data opcode fields. | The host admits canonical tokens, tracks fuel, fetches source opcode metadata for traces and final-end validation, and proposes the top word or a zero vector for routing. |
| Stack and locals | Fixed effect and value masks select pops, pushes, and register updates. The returned operand and local tensors become the next live state. | NumPy allocates intermediate arrays and evaluates the candidate paths. The host initializes input local 0, validates outputs, and materializes trace fields. |
| ALU | The constructed selector chooses among six exact bit-circuit candidates. | The implementation evaluates all candidates and checks tensor shape and binary membership. |
| Control and termination | Live PC selection applies to admitted straight-line execution. | The host admits and validates the final `end`, converts the result to a signed integer, and rejects non-admitted structured control. Control frames remain in the older engines. |

The finite trace evidence applies to the declared corpus. The Z3
checks apply to typed source and recovered candidate semantics. The
next architecture milestone is a unified data-and-control transition,
with structured control, linear memory, and calls still outside v58's
admitted datapath.

### Reproduce and replay

Run from the repository root on Linux with the pinned runtime above.
Copy the evidence directory to an external location before checking out the
producing commit so the archived inputs remain available. Keep the producing
checkout clean. For example, from the commit that contains this ledger:

```bash
records=$(mktemp -d)
cp -r benchmarks/results/v58-offline-c3dd17f/. "$records"
git switch --detach c3dd17fde2e996d29c1270136b543b6fee1011fc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider'
export PYTHONPATH="$PWD/visuaml-client/backend"
python "$records/replay_v57_reference.py" "$records/v57-evidence.json.gz"
python visuaml-client/backend/scripts/verify_neural_roundtrip_evidence_v58.py \
  --input "$records/v58-roundtrips.json"
python visuaml-client/backend/scripts/verify_recovery_campaign_v58.py \
  --input "$records/v58-recovery-campaign.json"
```

To generate fresh records outside the checkout:

```bash
out=$(mktemp -d)
python visuaml-client/backend/scripts/verify_offline_v57.py \
  --output "$out/v57.json"
python visuaml-client/backend/scripts/verify_offline_v58.py \
  --output-dir "$out/v58" --repeats 2 --faults \
  --roundtrip --roundtrip-holdout --recovery-campaign --recovery-campaign-holdout
(cd visuaml-client && python -m pytest -q --tb=short backend/tests)
```

The v57 producer compares v57 with v56, v51, and Wasmtime. The
separate archived-byte verifier checks the recorded state against v51
and the output against Wasmtime. It rejected a copy with a corrupted
WASM header. The archive includes v57 fixed-weight and token hashes.
The supplementary v58 identity record compares the fixed decoder,
ALU, PC, and microstep digests returned for 16 generated programs.

The [manifest](../benchmarks/results/v58-offline-c3dd17f/manifest.json)
records both compressed and raw checksums where applicable. Independent
replay checks target the original producing source. Rerun timings to
characterize a new machine or source revision.

## Backend regression

The corrected-source run passed **1,182 tests**, skipped **20**, and
reported zero failures. An earlier full run at `d7588de` started before
the documentation and selector corrections and reported 1,174 passes
and eight failures. Its failures were the six missing workflow links
and the two v57 tests described above.

## Scaling baseline and optimization target

The completed scaling run used one warmup and five measured calls per
engine and input, with explicit one-thread settings. It covered 53
programs (19 public, eight held-out, and 26 scaling workloads), 238
witnesses, and 4,983 reference transitions. v58 ran on 197 witnesses.
For the 41 structured-control witnesses the v58 entry is absent. A
separate `tracemalloc` pass recorded Python allocation peaks for one
input per program. These values do not measure total process memory.

The table uses only the same 197 admitted witnesses for all engines.
Each column shows the median across witnesses of the individual
five-call statistic.

| Engine | Wall median (ms) | Wall p95 (ms) | Process CPU median (ms) |
| --- | ---: | ---: | ---: |
| v51 interpreter | 2.198 | 2.671 | 2.196 |
| v56 ALU executor | 29.230 | 34.602 | 29.226 |
| v57 microops | 32.097 | 38.014 | 32.011 |
| v58 microsteps | 40.994 | 47.536 | 40.993 |
| Wasmtime via Python | 0.021 | 0.031 | 0.020 |

On paired v56/v58 inputs, v58 took **1.40×** as long by geometric mean,
with a worst observed slowdown of **1.99×**. v58 was slower on 195 of
197 paired witnesses. These are measurements of the local traced
Python/NumPy implementation and the warm Wasmtime Python binding.
The Python executor timings include trace work. Wasmtime timings exclude
module setup, and CPU frequency was not controlled.

Reproduce the timed baseline separately from other heavy workloads:

```bash
python visuaml-client/backend/scripts/benchmark_neural_vm_v58.py \
  --output "$out/scaling.json" --repeats 5 --warmups 1 --scaling --memory
python visuaml-client/backend/scripts/summarize_neural_benchmarks_v58.py \
  --input "$out/scaling.json" --output-json "$out/scaling-scorecard.json" \
  --output-csv "$out/scaling-scorecard.csv"
```

A five-call profile of `arithmetic_chain(32)` at input `-17` showed
0.670 s cumulative in ALU candidate execution and 0.399 s cumulative in
NumPy membership checks. These are inclusive profiler costs from one
workload. The next benchmarkable change is to reduce repeated binary
membership validation while preserving tensor admission, full state
parity, and the same corruption controls. Runtime optimizations and a
unified structured-control datapath are untested.

The representation-cost study independently rechecked its strict proof
archive, then measured 78 original-versus-recovered pairs across 13
programs. It alternated call order within each engine. Coverage,
per-pair timings, binary sizes, and conditional recovery amortization
are recorded in the representation-cost artifact. The tested change
was the symbolic WASM representation. Execution weights were held
fixed.

## v58 constructed runtime profiling and validation

The original traced baseline was rerun from the offline evidence
revision. Exact binary comparisons improved traced v58 by 1.26–1.30×
across 197 paired witnesses. Trace omission uses the same live gates.
[Profile and verification report](V58_CONSTRUCTED_VALIDATION.md)

## v58 compiled constructed ALU and suite throughput

The optional compiled gate equations improved traced execution 2.18×
over the same-mode optimized NumPy path and moved the leading profiled
cost to operand transitions. All 3,387 states and 197 witness results
matched the reference. Four pytest workers reduced complete-suite wall
time from 239 s to 86 s with the same 1,198 passes and 20 skips.
[Compiled ALU and test throughput report](V58_COMPILED_ALU.md)

## v58 fused state contractions

Factoring the stack and local equations reached 1.59× traced and 1.67×
untraced paired speedups against the compiled-ALU path. Full trace
hashes matched all 3,387 archived states, and the original ten fault
interventions remained effective.
[Fused state benchmark](V58_FUSED_STATE.md)

## v58 deliberately ambiguous recovery

Seven composed threshold sources forced an incorrect initial hypothesis
and a successful solver refutation. Six recovered after two to five
refinements. The out-of-grammar control was rejected and retained in
the same archive. A second run matched the semantic evidence.
[Counterexample stress report](V58_CEGIS_COUNTEREXAMPLES.md)

## v58 compact PC routing and binary admission

The native router evaluated the same live zero-test, guard and path gates
with compact source-bound target coordinates. Isolated countdown and GCD
routes were 10–15× faster. Two matched executor runs preserved full
traces for 3,387 states. Exact match counting is scoped to small tensors.
[Compact routing and binary admission report](V58_COMPACT_ROUTING.md)

## v58 indexed PC and microoperation reads

The optional indexed path carries the validated live gate coordinate
into the next source-bound row and reuses admission of immutable
decoder weights within execution. Warm row reads fell from about
39–40 μs to 1.5 μs in component tests. Two executor runs measured
1.11–1.12× paired untraced speedups and preserved all 3,387 archived
states. Setup cost and per-input slowdowns remain in the evidence.
[Indexed execution report](V58_INDEXED_GATES.md)

## v58 reusable constructed execution

The prepared executor admits routing, decoded rows, and fixed coefficients
once per source program. Warm untraced execution was 1.35–1.36× faster
across 197 paired inputs over two runs. Preparation cost, per-input
slowdowns, crossover estimates, all 3,387 matched states, and the ten
fault injections are recorded in the evidence. The warm profile directs
the next experiment to the shared data microstep.
[Prepared execution report](V58_PREPARED_GATES.md)

## v58 native prepared data microstep

One C++ call contracts the stack, local, ALU and result gates for a
prepared data row. The paired warm speedups were 7.95–7.95× untraced
and 3.82–4.07× traced across 197 inputs. The archive retains all
3,387 matched states, the established fault campaign, and differential
interventions on live gate coefficients.
[Native data microstep report](V58_NATIVE_MICROSTEP.md)

## v58 native prepared execution loop

A single native call now runs the compact router and live data gates
for a prepared program. Warm execution improved 11.30–11.34× untraced
and 18.46–18.50× traced over per-instruction native execution. The
evidence includes a direct comparison with prepared Python, matched
traces and fault outcomes, and a corpus-wide fuel boundary sweep.
[Native execution loop report](V58_NATIVE_LOOP.md)

## v58 serial native batches

Prepared programs now run independent inputs in one native call and
return ordered results and traps. The evidence compares warm and
preparation-inclusive throughput at five batch sizes, preserves all
3,387 trace states, and samples the remaining native ALU cost.
[Serial native batch report](V58_NATIVE_BATCH_LOOP.md)

## v58 packed scalar execution

The prepared native loop stores binary i32 state in machine words and
uses an exact canonical ALU lowering. The two-run archive compares single
calls and batches, checks traces, coefficient effects and fuel behavior,
and records the arithmetic proof and direct prepared-Python timings.
[Packed scalar execution report](V58_PACKED_SCALAR.md)

## v58 packed AVX2 batches

Eight packed lanes share each straight-line program's admitted step
logic. Warm 1,024-input throughput improved 1.22–1.27× at the API and
1.99× in the separate preallocated kernel measurement. The archive
includes emitted vector instructions, trace and result parity, and
fuel and coefficient replays.
[Packed AVX2 batch report](V58_PACKED_SIMD.md)

## v58 packed structured control

The prepared packed loop now covers all eight structured-control opcodes
in the bounded v51 dialect. Replays include full frame and data states,
every branch depth, fuel boundaries, and a live route-gain intervention
localized to its first divergent state. The verifier preserves original
WASM fixtures and can be reused for later lowering and recovery work.
[Packed structured control report](V58_PACKED_STRUCTURED_CONTROL.md)

## v58 packed memory and bounded recovery

The packed structured executor supports bounded i32 load and store.
The prefix-sum witness preserves the original WASM, access replay,
live write-gain intervention, and counterexample-guided search over
three fold operations. Two identical runs checked full page equivalence
across 781 finite-domain inputs.
[Packed memory and recovery report](V58_PACKED_MEMORY.md)

## v58 shared memory-region verification

One terminal comparator rejected both a forward-only recovery and a
forward-specialized compiler proposal on the same overlapping copy.
Guarded bulk lowering, access-order replay, partial effects on traps,
and bounded SMT model claims are recorded independently.
[Memory-region contract report](V58_MEMORY_REGION.md)

## v58 copy recognition and trap replay

The guarded move lowering derives roles, descriptor addresses, and
fuel from a matched WASM grammar. Independently written WAT and 24
padded local assignments replay against Wasmtime. Three near-misses
use source execution. Memory trap traces preserve the completed
prefix and identify the widened failing address.
[Copy recognition report](V58_COPY_RECOGNITION.md)

## v58 stateful WASM scale

An opt-in packed profile raises the source and untraced fuel limits
and uses source-bound linear-size route coordinates. Full WASM Sudoku,
directed graph BFS, and matrix multiplication were compared to Wasmtime
and independent references. The archive preserves full-page hashes,
curated Sudoku proofs, seeded scale sweeps, and a near-cap fuel run.
[Stateful scale report](V58_STATEFUL_SCALE.md)

## v58 structured NumPy memory

The opt-in array path reuses the fixed-weight v57 state transitions
with validated indexed v52 routes and NumPy linear memory. A rollback
fixture, three runs of the original Sudoku binary, and directed BFS
match packed C++ and Wasmtime. The archive includes a placement and
reversal, exact page hashes, fault and fuel replay, a standalone Rust
Sudoku checker, and paired timings for the frame contraction.
[Structured NumPy memory replay](../benchmarks/results/v58-array-memory/README.md)

## v58 Rust voxel guest and fixed-bit multiply

Original Rust-generated WASM runs a voxel collision tick with the
structured NumPy executor, packed C++, Wasmtime, and Node WebAssembly.
Four deterministic fixtures match full memory. Grouping the NumPy
multiply partial products preserved the ReLU carry gates and improved
paired Sudoku and BFS timings. A bounded Escargot replay matched the
packed trace and memory cutoffs.
[Rust guest and fixed-bit ALU replay](../benchmarks/results/v58-voxel-tick/README.md)

## v58 interactive voxel replay

A React Three Fiber scene runs the original one-agent Rust WASM and a
shared two-agent guest. Chromium inputs can be exported and replayed
offline. Both fixtures match VisuaML NumPy and Wasmtime after each
tick, including the shared claim that leaves both positions unchanged.
[Interactive replay and measurements](../visuaml-client/voxel-replay/README.md)

## v58 terrain histories and sustained replay

The terrain guest resolves simultaneous movement against pre-tick
occupancy, claims, and materials, then commits A's terrain edit. A
counterfactual omits the edit and keeps B blocked. Both histories and
a 72-tick sparse and crowded sweep match Chromium, Wasmtime, and
VisuaML NumPy. The first-mismatch report preserves the input and
full pages. A profile-guided unit-retain frame contraction improved
paired terrain and BFS cutoff runs.
[Terrain scene, replay commands, and evidence](../visuaml-client/voxel-replay/README.md#terrain-edit-and-counterfactual)

## v58 unit operand push/pop contraction

A diagnostic profile of terrain and BFS motivated sparse contraction
of unit-gain push and pop. It removes both dense candidate stacks
from those transitions while preserving buffer ownership. Paired
runs improved sparse and crowded terrain, full BFS, and one-blank
Sudoku. The causal histories retained exact traces and full pages.
[Operand profile, replay checks, and paired samples](../visuaml-client/voxel-replay/evidence/operands/README.md)

## v58 prepared operand pop

The canonical array profile prepares unit pop for its invocation-owned
operand state. Checked push and truncation maintain the invariant across
control-flow joins, and dynamic underflow remains checked. Paired
terrain, full BFS, and Sudoku runs improved median execution latency.
Exact causal traces, fuel and trap prefixes, and Wasmtime pages matched.
[Prepared pop profiles, counters, and paired samples](../visuaml-client/voxel-replay/evidence/prepared-pop/README.md)

## v58 prepared fixed-gate ALU selection

Canonical source-bound add and signed less-than bind their existing
v49 circuits during invocation setup. Eligible sites skip five
unused arithmetic candidates. Paired terrain, complete BFS, and
one-blank Sudoku replays improved under the same prepared-pop
baseline. Exact pages, traces, fuel and traps were retained.
[ALU profiles, work counts, and raw replay timings](../visuaml-client/voxel-replay/evidence/prepared-alu/README.md)

## v58 prepared instruction regions

Prepared multiply, a source-bound const/add pair, and an adjacent
local-read extension were measured separately. The regions retain
the original live fixed gates and materialize just the final stack
state. Terrain, full BFS, and Sudoku retained exact pages and source
steps, with reference traces at every tested fuel boundary.
[Region profiles, dispatch counters, and paired samples](../visuaml-client/voxel-replay/evidence/prepared-regions/README.md)

## v58 compiled fixed-gate frame events

The canonical untraced array profile can bind the v54 event selector
to a dedicated compiled gate symbol. Its graph matches the full
144-case canonical truth table. Terrain, full BFS, and Sudoku
retained exact pages and source steps with packed fallback blocked.
The warmed speedups and cold compilation cost are recorded separately.
[Compiled gate graph, replay checks, and measurements](../visuaml-client/voxel-replay/evidence/compiled-frame-event/README.md)
