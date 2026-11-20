# Export, relocate, and fork a Pocket checkpoint

A seven-state table drives the existing Rust tape helper inside the same linked WASM and frozen VisuaML graph. This table can cross both sides of its starting window. The Universal Tape program keeps its own left-edge convention and retains its original table.

The guest first commits one setup transition. The resulting state is 6 with head at logical coordinate 0 and symbol 1. Pocket exports the page at that point. The pending transition would overwrite symbol 1 with 2, enter state 1, and move left. The left-edge attempt changes only status and scratch next-head. The verifier checks every byte and explicitly checks that logical tape, head, state and count stayed put. It does the same at the later right edge.

## Restore contract

The version 1 JSON contains the full base64-encoded 64 KiB page, its SHA-256, the WASM, graph, program and dense table hashes, and the logical origin, window length, head, state and committed count. The graph hash is the expected offline backend identity. In the browser, the run uses linked WASM and compares against recorded VisuaML page hashes.

The importer checks identities, the page hash, its actual table bytes and logical metadata before instantiating fresh memory. For left growth, it moves the tape bytes to higher physical offsets, inserts blanks, advances the guest's head index, and subtracts the amount grown from the host's logical origin. The status, control state and count come from the pre-attempt page. The scratch next-head is rebased along with the head and overwritten by the next guest step. Right growth changes the length and initializes the new cells. The guest retries the whole pending transition. These relocations remain within one fixed WASM page.

## Two futures

The verifier exports in one Node process and imports the resulting file twice in another. Both copies grow left, commit the pending transition, then grow right later in the run. The clean copy agrees at every remaining logical step with a run that had enough room from the start. After the second expansion, its full pages match the wide reference.

The edited import changes one symbol at logical coordinate 0, from 1 to 2. The first transition after import, **committed transition 2**, enters state 4 instead of state 1 and writes 1 instead of 2. Both copies reach state 5 and halt at logical coordinate 1 with five committed transitions. Their final tapes over coordinates −1, 0 and 1 are `112` and `212`. Each import owns its own memory, and the tests verify that running the edited copy leaves the clean copy and original checkpoint unchanged.

The replay checks full pages for all 20 attempted transitions across Node WASM, Wasmtime and the unchanged frozen graph, with packed fallback blocked. A direct transition reference checks the exact updates, including both blocked attempts. Tests also cover overlapping relocation and rejection of wrong identities, altered bytes and inconsistent metadata.

## Try it

From `visuaml-client/voxel-replay`, run `npm ci` and `npm run dev -- --port 4180`. Open [Checkpoint Fork](http://localhost:4180/pocket-tape-fork.html). Export a checkpoint, close and reopen the page, import the same file into both panels, then run each copy.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_pocket_tape_fork.py
npm test
```

[Watch the GIF](checkpoint-fork.gif) · [Download the checkpoint](checkpoint.json) · [Exact replay](replay.json)
