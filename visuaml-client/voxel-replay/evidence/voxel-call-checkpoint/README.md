# Suspend and resume inside a Rust helper

This experiment uses the existing frozen direct-call graph and a 64 KiB build of the Rust passage guest. The witness adds an explicit next-tick move input, a word recording the material returned by the helper, and a caller commit counter. The same WASM and graph run the uninterrupted reference and both resumed histories.

The first agent clears material 7 at byte 65512 on source step 14. The checkpoint is taken immediately after the store. Memory is already 0, the callee's local 1 is still 7, the next PC is 11, call depth is 1, and 1,986 of the original 2,000 fuel units remain. The callee operand stack is empty here. The suspended caller frame holds the pending store address and call argument. A second round trip checks a later boundary where the callee itself has 7 on its operand stack.

## Checkpoint and checks

Version 1 stores every VM register, as well as the full operand stack, locals, caller frames, depth, and a base64 copy of the complete WASM page. The encoded input is JSON data. Import verifies the graph, WASM, and prepared program image identities, the canonical payload hash, memory size, register shapes, and live state bounds. It restores a new private page and the original fuel. Version 1 is scoped to the frozen direct-call machine and its state schema.

The export runs in its own process. Each resume starts another process and restores the same file. Both match the uninterrupted trace at all 17 remaining source boundaries, including full state and memory hashes, source PCs and access records. At step 17 the helper returns 7 to the restored caller. The caller stores it and increments its commit counter once. No resumed instruction calls the helper again or repeats the terrain store. The first tick also matches Wasmtime's uninterrupted result. Wasmtime is used as a completed-tick reference.

The two copies differ only in their input at the beginning of the following tick. The passage is open in both. The Wait input leaves the second agent on the near side. The Walk input moves it through the opening. Both finish with two caller commits. Wasmtime matches both final pages, and packed fallback is blocked in every graph run.

## Inspect and replay

From `visuaml-client/voxel-replay`, run `npm ci` and `npm run dev -- --port 4180`. Open [Suspended Passage](http://localhost:4180/voxel-call-checkpoint.html) to step the saved verified trace and select a next-tick input. To execute a downloaded checkpoint in a fresh process:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_voxel_call_checkpoint.py
python tools/replay_voxel_call_checkpoint.py --resume evidence/voxel-call-checkpoint/checkpoint.json --move 1
```

Use `--move 0` for Wait. To rebuild the witness with Rust's WASM target installed, run `bash tools/build_voxel_call_checkpoint.sh`.

[GIF](suspended-passage.gif) · [Checkpoint](checkpoint.json) · [Exact replay](replay.json) · [Rust source](../../guest/passage_call.rs)

[Find a way through](../voxel-fork-search/README.md) searches future inputs from this suspended call.
