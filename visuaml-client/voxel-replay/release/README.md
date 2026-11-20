# VisuaML Replay 0.1 release candidate

Send a JSON checkpoint with the exact WASM bytes embedded. A recipient can reopen it with the pinned graph in this repository. The CLI executes imported bundles and the browser shows exported traces and checkpoint registers. The browser is a saved-record viewer.

## Install and try

Use Python 3.12 and install PyTorch CPU 2.9.1 NumPy 2.1.3, and Wasmtime 36.0.0. The live counterexample demo additionally uses z3-solver 4.15.3.0.

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.1.3 wasmtime==36.0.0
cd visuaml-client/voxel-replay
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/visuaml_replay.py open release/examples/pocket-checkpoint.json
python tools/visuaml_replay.py compare release/examples/terrain-checkpoint.json \
  --preview-local 2 --output comparison.json
npm ci
npm run dev -- --port 4180
```

Open [WASM Replay](http://localhost:4180/replay-viewer.html) and load the comparison or either shipped example. The [Pocket console](http://localhost:4180/pocket.html) and [Suspended Passage](http://localhost:4180/voxel-call-checkpoint.html) remain playable separately.

To bring your own module:

```sh
python tools/visuaml_replay.py capture guest.wasm report.json --argument 0 --after-step 3
python tools/visuaml_replay.py open report.json
```

Use `--memory` with exactly 65,536 initial bytes and `--fuel` to set the initial allowance. The capture boundary must be a running source boundary. Carry the bundle, the release commit, and the exact commands when filing a report.

## Supported semantics

The maintained entry point uses the frozen source-step direct-call profile. Its loader validates one fixed 64 KiB memory, the exports `run(i32) -> i32` and `memory`, up to eight unary i32 functions, and acyclic direct calls to lower-numbered functions. The opcode allowlist and stack/control limits are enforced by the canonical decoder. Imports, floating point, dynamic memory growth and general Rust standard-library binaries fall beyond this profile. The example cartridge legalizer can help with specific LLVM forms.

The graph owns live PC, operand and caller state, checked memory access and fuel. The host validates and prepares bytes, supplies initial memory, drives source transitions, and serializes observations. The optimized runtime remains a research path, while this release uses the single-step graph as its execution contract. Packed fallback is blocked. Capture verifies a normally completed run against Wasmtime on return and full page. Interior source boundaries use the VisuaML reference path.

Version 1 binds the embedded program, retired source count and nested continuation with a canonical JSON hash. The nested continuation separately binds the graph, WASM and prepared image, and stores remaining fuel, stack, locals, frames and memory. The comparison mode models one proposed observation for a `local.get` pair. It reports the first ordered divergence and keeps both suffix traces. It is a test harness for a bounded execution-plan family. General program-fix comparison and external-effect replay are outside this profile.

## Outside trial

The handoff task is to bring an independent small unary i32 module, capture inside a direct call, send the resulting bundle to another reader, then ask that reader to open it in a new process. Record the producing commit, guest binary, exact command, expected boundary and outcome. If admission fails, include the first loader error and the smallest build that reproduces it. If execution disagrees, include the original bytes and evidence from Wasmtime.

The fresh-process checks here use repository examples. An independent contributor and recipient have yet to complete this trial.
