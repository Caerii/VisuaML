# VisuaML Replay 0.19 release candidate

Send one JSON report with the exact WASM bytes, continuation, reproduction recipe, and expected replay digest. A recipient can run it with the pinned graph in this repository. The CLI executes and checks the report. The browser displays saved records.

## Install the CLI

From the repository root, with Python 3.12:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
visuaml-replay doctor
visuaml-replay example passage-report.json
visuaml-replay inspect passage-report.json --after-step 16 --steps 3
```

The package bundles the reference graph and Passage report. The CLI works
outside the checkout and the reference path requires no C++ compiler.
To hand the installed tool to someone, build a wheel with
`python -m pip wheel . --no-deps -w dist` and share the wheel and the report.
The release candidate is installable from this source tree or its wheel.
See [changes and compatibility](CHANGELOG.md) and the [contributor guide](CONTRIBUTING.md).

## Inspect imported artifacts

Use `visuaml-replay describe` for a WASM binary, an ONNX graph, or a trusted
PyTorch export. See the [artifact import walkthrough](chapters/import-artifacts.md)
for the activation mismatch and unchanged Rust module examples.

## Compare a model and tool pipeline

Use the [pipeline walkthrough](chapters/pipeline-replay.md) to compare
PyTorch with ONNX Runtime and locate model, WASM, and host handoff differences
in a shared report. The JSONL interface is available for other harnesses.

## Replay an activation intervention

The [intervention walkthrough](chapters/activation-interventions.md)
uses a checked PyTorch hook and a Rust WASM feedback value to replay a
routing switch and its controls.

The [routing subspace study](chapters/routing-subspaces.md) tests a smaller
activation edit and executes both resulting WASM tool choices.

The [crossed-rule study](chapters/crossed-routing.md) checks whether the
patch transfers a comparison outcome across different recipient rules.

The [delayed-rule study](chapters/delayed-rules.md) reuses one donor
activation under three rules and verifies the two resulting WASM calls.

The [additive routing study](chapters/additive-routing.md) tests a
calibrated numeric feedback edit while preserving the recipient's rule.

The [margin-transfer study](chapters/margin-transfer.md) uses only the
donor's threshold side and infers the edit magnitude from the recipient.

The [transformer intervention study](chapters/compositional-interventions.md)
freezes a token edit on calibration cases and replays a second tool decision.

## Capture a failing test in CI

[The runnable test example](examples/ci/check_guest.py) exits with status 1
because the test expects 43 and the WASM guest returns 42. It writes a single
report from inside the helper. Run the sender from the repository root, then
run the recipient command in any directory with the CLI installed.

```sh
python visuaml-client/voxel-replay/release/examples/ci/check_guest.py --report failure.json
visuaml-replay open failure.json --output reproduced.json
```

Configure your CI runner to upload `failure.json` on test failure. A
successful `open` means it reproduced the report. Read
`expectation_met`, `expected_return`, and `returned` in its result to see
the failing assertion. Try the passing control with `--expected 42`.
For your own guest, `visuaml-replay pack CHECKPOINT REPORT --expect-return N`
records a return expectation in the recipe.

A report embeds the guest binary and its entire 64 KiB memory page.
Use synthetic inputs for publicly shared examples. Check your team's
artifact access controls before uploading a report containing real data.

## Run the source-tree examples

Use Python 3.12 and install PyTorch CPU 2.9.1, NumPy 2.1.3, and Wasmtime 36.0.0. The live counterexample demo additionally uses z3-solver 4.15.3.0.

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.1.3 wasmtime==36.0.0
cd visuaml-client/voxel-replay
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/visuaml_replay.py open release/examples/suspended-passage-report.json \
  --output passage-result.json
python tools/visuaml_replay.py open release/examples/compiler-divergence-report.json \
  --output comparison.json
npm ci
npm run dev -- --port 4180
```

In the first report, the passage is already open at import. The helper returns the original material `7` to its caller, and the saved continuation takes 17 source steps without repeating the terrain store. In the second report, the preview first differs after PC `160` with stacks `[256]` and `[0]`. The suffixes then agree.

Open [WASM Replay](http://localhost:4180/replay-viewer.html) and load the comparison or either shipped example. The [Pocket console](http://localhost:4180/pocket.html) and [Suspended Passage](http://localhost:4180/voxel-call-checkpoint.html) remain playable separately.

To bring your own module:

```sh
python tools/visuaml_replay.py capture guest.wasm checkpoint.json --argument 0 --after-step 3
python tools/visuaml_replay.py pack checkpoint.json report.json
# Send report.json and the producing repository commit.
python tools/visuaml_replay.py open report.json --output result.json
```

Use `--memory` with exactly 65,536 initial bytes and `--fuel` to set the initial allowance. The capture boundary must be a running source boundary. For a supported compiler preview at a `local.get` pair, use `pack --preview-local N`. A recipient needs only one file and the repository revision. Earlier checkpoint bundles still open through the same CLI.

## Try the compiled runner

The reference runner is the default. On Linux with a C++17 compiler and the
CPU PyTorch install above, try the handoff from the same directory:

```sh
python tools/visuaml_replay.py demo --output /tmp/passage-report.json
python tools/visuaml_replay.py open /tmp/passage-report.json
```

The demo writes one report after the compiled runner stops inside the Rust
helper. It checks the resulting checkpoint against the shipped reference
and also runs the continuation in the reverse direction. The second
command starts a separate process, verifies the report digest, and
prints the original source trace. A recipient can use the reference
runner without a compiler. The compiled path builds its CPU operator on
first use and caches it in the current user's private temporary directory.
Use `CXX` to select an alternative compatible compiler.

- Use `capture --runner compiled` to reach a checkpoint on your own guest.
- Use `resume REPORT.json --runner compiled` for final status, fuel, result,
  and memory digest. This checks the imported checkpoint and produces a
  summary. Use `open REPORT.json` to verify the report's ordered trace and
  expected result digest.
- Use `Replay.run(state, steps=N, runner="compiled")` from Python to stop at
  a source-instruction budget. The runner bounds execution batches to at
  most 64 steps and shortens the last batch to reach the requested cut.
  For the trace from any imported boundary, use `Replay.finish(checkpoint)`.

[Read the executable handoff chapter](chapters/runner-handoff.md).

## Jump to a source instruction

To view the Rust helper return and the caller's next two instructions:

```sh
python tools/visuaml_replay.py inspect release/examples/suspended-passage-report.json \
  --after-step 16 --steps 3 --output passage-window.json
```

The report was captured after source step 14. The command advances two
steps, then records the next three transitions. The first row is the
helper return, with `[65512, 7]` on the caller's stack. Omit
`--after-step` to start at the saved checkpoint. Add `--runner compiled`
to seek faster through a long prefix on Linux. The selected runner
advances without collecting observations. The reference runner then
reconstructs up to 256 trace rows. Traps and exhaustion show up as
transitions even when they retire no source instruction.

Use the viewer's **Inspect the Rust helper return** button, or load your
generated JSON with the file picker. The window records which runner
reached its starting boundary and which reconstructed its rows. Its
scope is the checkpoint's ordinary continuation. To run a report's
recipe, including a compiler preview, and verify its expected result
digest, use `open`.

Python callers can use `inspect_window(report, after=16, limit=3)`.
[Example window](examples/passage-window.json)

## Read and run the library

Start with [The helper returns 7 after it writes 0](chapters/suspended-helper.md).
The chapter reads the shipped report, tests restore and fork independence,
and finds the first resumed divergence when a live local is lost.

```sh
PYTHONPATH=../backend python -m doctest release/chapters/suspended-helper.md
PYTHONPATH=../backend python tools/benchmark_replay_capture.py
```

| Code | Responsibility |
| --- | --- |
| [Profile and frozen transition](../../backend/visuaml/compiler_lab/frozen_calls_v58.py) | Admit bytes as a validated program image and execute the source-step semantics. |
| [Runtime](../../backend/visuaml/replay/runtime.py) | Load a trusted local graph, start runs, advance source transitions, and collect observations. |
| [Checkpoint codec](../../backend/visuaml/replay/checkpoint.py) | Validate portable state and restore an invocation-private page and caller frames. |
| [Reports](../../backend/visuaml/replay/reports.py) | Package and verify portable artifacts and compare ordered observations. |
| [CLI](../tools/visuaml_replay.py) | Read and write files, expose commands, and render results. |

Use `from visuaml.replay import Replay` to validate a module and reuse its
program image across invocations. `start` and `restore` create independent
live states. `advance` is for a source transition without a journal entry.
`step` includes a source observation and `finish` collects the suffix from
a checkpoint. The live state is owned by the invocation. Retain a serialized
`checkpoint` when a boundary must survive further writes. `open_bundle`
and `check_report` handle the existing portable version 1 artifacts. The
default graph comes from the source tree; callers may supply a trusted local
graph path when constructing a runtime. The report never supplies that path.

The capture benchmark reuses one loaded runtime. It pairs 32 warm captures
of the same 14-source-step prefix, including final serialization and
excluding setup, Wasmtime verification, and filesystem I/O. Full prefix
observations took **10.32 ms** median versus **5.47 ms** for unobserved
capture. Both paths emitted the same checkpoint in every pair. This is a
single-fixture CPU result. [Raw samples](chapters/capture-timings.json)

## Supported semantics

The maintained entry point uses the frozen source-step direct-call profile. Its loader validates one fixed 64 KiB memory, the exports `run(i32) -> i32` and `memory`, up to eight unary i32 functions, and acyclic direct calls to lower-numbered functions. The opcode allowlist and stack/control limits are enforced by the canonical decoder. Imports, floating point, dynamic memory growth and general Rust standard-library binaries fall beyond this profile. The example cartridge legalizer can help with specific LLVM forms.

The graph owns live PC, operand and caller state, checked memory access and fuel. The host validates and prepares bytes, supplies initial memory, drives source transitions, and serializes observations. The optimized runtime remains a research path, while this release uses the single-step graph as its execution contract. Packed fallback is blocked. Capture verifies a normally completed run against Wasmtime on return and full page. Interior source boundaries use the VisuaML reference path.

The version 1 checkpoint binds the embedded program, retired source count and nested continuation with a canonical JSON hash. The nested continuation separately binds the graph, WASM and prepared image, and stores remaining fuel, stack, locals, frames and memory. The outer runnable report has its own version 1 schema. Its expected digest covers the ordered replay, final status, and page identity, or both runs and the first divergence for a comparison. Matching it confirms reproduction with the supplied artifact. It does not authenticate the sender. The comparison mode models one proposed observation for a `local.get` pair. It reports the first ordered divergence and keeps both suffix traces. It is a test harness for a bounded execution-plan family. General program-fix comparison and external-effect replay are outside this profile.

## Outside trial

The handoff task is to bring an independent small unary i32 module, capture inside a direct call, send the resulting bundle to another reader, then ask that reader to open it in a new process. Record the producing commit, guest binary, exact command, expected boundary and outcome. If admission fails, include the first loader error and the smallest build that reproduces it. If execution disagrees, include the original bytes and evidence from Wasmtime.

The automated smoke test opens both release examples and constructs a new WAT module after loading the pinned graph from the repository. It captures inside a call, packages the report, removes the sender's separate guest and checkpoint files, and opens the report in a fresh process. It also verifies rejection of a wrong expected result digest.

```sh
python tools/test_runnable_report.py
```

An independent contributor and recipient have yet to complete the external trial.

The [intervention diagnosis](chapters/intervention-diagnosis.md) replays the
original failed switches and tests the calibrated patch on two independent
checkpoints.

The [live-tool intervention replay](chapters/live-tool-interventions.md) traces
an actual first WASM return through the controller's second request.

The [rank-one comparison study](chapters/comparison-direction.md) tests local
calibration and raw-coordinate transfer across matched-training controllers.
