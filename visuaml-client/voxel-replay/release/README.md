# VisuaML Replay 0.2 release candidate

Send one JSON report with the exact WASM bytes, continuation, reproduction recipe, and expected replay digest. A recipient can run it with the pinned graph in this repository. The CLI executes and checks the report. The browser displays saved records.

## Install and try

Use Python 3.12 and install PyTorch CPU 2.9.1 NumPy 2.1.3, and Wasmtime 36.0.0. The live counterexample demo additionally uses z3-solver 4.15.3.0.

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
