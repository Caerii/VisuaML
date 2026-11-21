# Rolling context with a frozen word decoder

The unchanged `764def4` word decoder runs a post-freeze countdown and a late branch with at most 512 active tokens. The host exports at completed instruction boundaries, rebuilds the initial prefix from live machine state, and starts a new cache. Both four-address and two-address live sets match the scalar reference at every source transition.

## Checkpoint invariant

A version-1 checkpoint binds a program image and decoder artifact by SHA-256. It stores the next PC, R0, four latest committed memory values, live status, and global retired-instruction count. R1 is always zero in this ABI. The replay includes the immutable program image. The four initial memory entries supply defaults, and every address must have an explicit word in a resumed image. This profile has no external inputs during execution. The runner can change the host segment budget because it does not alter the next guest instruction or total retired accounting.

Export uses the host audit decoder to read completed output frames. It applies each committed write, extracts the final PC and R0, and serializes the result. It discards a trailing incomplete frame. Restore validates identities and state, packs the unchanged CODE frames and four memory COMMIT frames, appends the saved PC and register, and prefills a fresh cache. It does not evaluate the next instruction. The checkpoint and repacking are host operations.

The invariant follows the word decoder's refinement relation. A fresh cache populated from the checkpoint has the same code, register, PC, and value for every future observable address. For each admitted subsequent instruction, attention recovers the current operands and the graph emits the same successor. A newer matching store dominates the replacement initialization. Context positions can differ between histories while preserving exact offset selection and code or address match under the existing score and bit-recovery bounds. This is a conditional argument for this ISA and a committed boundary. It relies on no external inputs and numerical recovery within the fixed context envelope.

## Controls and counterexamples

- The program writes 7 to address 2 before its countdown. The loop repeatedly overwrites address 0. The saved word at address 2 survives all resets and feeds the late branch. In the 128-iteration runs the late LOAD retires at instruction 515.
- A short run fits in one window. Its full-history continuation and compacted continuation match token for token. Removing its superseded initial address-0 COMMIT also produces the same continuation.
- Eleven partial cuts before the early store's COMMIT keep address 2 at 0 and publish no matching cache key. Restoring at the last committed instruction reproduces the rest of the original stream, including the discarded partial bytes.
- Editing retained address 2 from 7 to 0 changes the saved state immediately. The first dependent effect appears at retired instruction 259 in the 64-iteration run. LOAD at PC 6 sees 0, so the next branch chooses PC 8 rather than PC 10. All earlier states agree with the unedited run when the deliberately changed word is excluded. The continuation matches a scalar run started from the edited checkpoint.
- Import rejects a checkpoint missing an address. A deliberately broken exporter instead fills the omitted address 0 with zero. The first resumed LOAD at PC 2 retires as instruction 3 and returns 0 instead of 64. The saved replay locates this failure and cross-checks the resulting short trace with the scalar oracle.

## What scales

| Live addresses during loop | Countdown | Retired | Generated tokens | Resumptions | Peak active tokens | Total ms per step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 263 | 3,156 | 10 | 504 | 7.29 |
| 2 | 128 | 519 | 6,228 | 21 | 504 | 6.62 |
| 4 | 64 | 263 | 3,156 | 10 | 504 | 6.71 |
| 4 | 128 | 519 | 6,228 | 21 | 504 | 6.71 |

The two variants have the same image and execution lengths. The four-live-word variant reads addresses 1 and 3 after the late branch, while the other rereads address 2 there. This changes future-read liveness independently of countdown length. Both variants use a fixed four-word snapshot and the same prefix length, so this measures current runtime behavior with fixed storage. A sparse snapshot representation would be a different experiment.

The 216-token reconstructed prefix holds the program and initial state. Each full segment retires 24 instructions and reaches 504 active positions, with 2,161,152 logical KV bytes. Checkpoint JSON is 241–244 bytes, excluding the separately addressed program image. Across these runs, per-segment medians are 231–261 microseconds for restore packing and validation and 117–124 microseconds for checkpoint construction, parsing and serialization, including committed frame audit. The total timing includes prefix construction, prefill, decode, and export. It excludes model loading and reference verification. Samples come from one complete CPU run per configuration on one thread and are descriptive.

Cumulative emitted tokens grow with instruction count. Active attention and KV stay within the original context envelope. This is a bounded-window continuation result.

## Reproduce

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
PYTHONPATH=.:../backend python tools/replay_rolling_context.py --open evidence/rolling-context/replay.json
npm run dev -- --port 4180
```

The CLI replays the frozen weights and compares the archive. Open `/replay-viewer.html` and select **Open rolling context replay**. [Replay and checkpoints](replay.json) · [Holdout program](holdout.json) · [Raw timings](timings.json) · [Frozen decoder](../word-decoder)
