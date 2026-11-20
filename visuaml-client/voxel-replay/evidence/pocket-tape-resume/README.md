# Checkpoint and extend Pocket's tape

The same program, `R R M H`, runs twice on the fixed Universal Tape table. Both start with input `1`. One gets two simulated tape cells. The other gets five.

The short run retires 54 universal transitions and reaches the pending physical move that would carry out the second `R`. The guest returns status 2 because the next physical head is outside the logical tape. That attempt takes 192 WASM source instructions. It updates only status and scratch next-head. It commits no tape write, head move, state change, or transition count.

The host saved the complete 65,536-byte page immediately before the attempt. At **Extend and resume**, it restores that clean page into a fresh instance of the same linked WASM, changes the little-endian tape length at page byte 264, and appends three blank symbols. Blank is symbol 0, so those words were already zero in this checkpoint. The runtime retains a fixed 64 KiB WASM page while the logical window grows inside it.

The restored page equals the wide run's page before the pending transition. The guest retries that transition, converts the newly accessible blank into a marked zero, finishes the simulated move, executes `M`, and halts. No interpreter state is reconstructed from the decoded view.

| Simulated instruction | Head | Padded decoded tape | Committed universal transitions |
| --- | ---: | --- | ---: |
| First R | 1 | 10000 | 38 |
| Second R | 2 | 10000 | 78 |
| M | 2 | 10100 | 118 |
| H | 2 | 10100 | 131 |

The histories agree on every full page after the expansion, as well as all decoded simulated boundaries when the short window is conceptually padded with blanks. Both retire 27,248 WASM source instructions on the committed path. The failed attempt adds 192 only to the short run's actual work. Wasmtime and the frozen graph agree on every attempted transition, including the stopped page, with packed fallback blocked. The graph, linked WASM, and fixed table hashes are recorded in the replay.

## Run the comparison

From `visuaml-client/voxel-replay` run `npm ci` and `npm run dev -- --port 4180`. Open [Universal Tape](http://localhost:4180/pocket-universal.html#resume-title), then select **Start tape comparison**. **Advance both runs** advances to the next simulated boundary or the window stop. **Extend and resume** restores the checkpoint. The next advance retries the move in the widened window. The browser runs linked WASM and compares hashes with the saved offline proof.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_pocket_tape_resume.py
npm test
```

[Watch the GIF](tape-resume.gif) · [Pre-attempt checkpoint](checkpoint.bin) · [Exact replay](replay.json) · [Program and window sizes](../../fixtures/pocket-tape-resume.json)
