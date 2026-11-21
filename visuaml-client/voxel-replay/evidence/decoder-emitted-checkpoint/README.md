# Decoder-emitted checkpoints

One frozen TorchScript artifact executes the original word instruction stream and exposes a checkpoint readout over its token-level KV cache. The readout emits a complete 60-token block in one forward call. It uses the identical four memory frames and one live-state frame consumed by the host-built checkpoint path. No guest execution occurs during the readout. The revised artifact was frozen at `e48c7d0` before the new holdout was authored.

## Checkpoint phase

The host requests a checkpoint at a live COMMIT and reserves 60 token slots before starting a segment. The checkpoint phase reads the existing cache without adding the output block to it. Four fixed queries attend to the memory layer by address and recency. Five relative-position queries attend to the token layer for the latest next PC and R0 bytes. Byte packing, frame construction, and marker emission are tensor operations in the exported graph. The context bound and score margins carry over from the word decoder. A matching address leads nonmatches by at least 16 over 512 positions, and exact-position gathers have margin at least 24. Binary64 bit recovery uses the original 1/4 and 3/4 thresholds.

The host checks the boundary and framing, concatenates the same immutable CODE prefix with the emitted block, and starts a fresh KV cache. It preserves decoder and program identities in the replay and counts retired frames at the host boundary. HALT ends the run. The host neither calculates nor rewrites checkpoint payloads on the emitted path. The independent auditor decodes complete execution frames and builds its expected prefix separately. The emitted replacement is already constructed before comparison. The byte comparison is part of verification and is excluded from measured emitted-path construction time.

Each checkpoint begins with a fixed 60-token frame format and carries the four latest words, next PC, and R0. The existence of a live COMMIT supplies execution status. This is a readout at a host-requested boundary, with all four addresses retained. The experiment leaves learned retention, autonomous checkpoint scheduling, and token-by-token autoregressive output of the checkpoint block open.

## Exact controls

- The original 519-instruction run matches the scalar reference and archived rolling trace. With identical segment limits, both checkpoint paths generate identical execution tokens in every segment. All 27 emitted replacement prefixes match the independent host-built prefix exactly.
- The post-freeze 154-instruction program copies `0x12345678` into an early live address and repeatedly overwrites its separate countdown counter. The held word is read after seven resumptions. The scalar and both host paths agree at every instruction.
- Every one of the 60 strict prefixes of the checkpoint block is rejected. All 11 partial-store cuts reject a checkpoint request. The previously valid block remains available.
- At the snapshot after the early store, removing the superseded initialization frame gives identical checkpoint bytes.
- Replacing the retained word with zero changes the LOAD at source instruction 151 and sends the branch to PC 8. Omitting the counter frame is rejected. Simulating a faulty exporter that replaces it with zero causes the first resumed LOAD, instruction 3, to read 0 instead of 37. Both validly framed counterfactual continuations match independently initialized scalar continuations.

## Cost

The original program generates 6,228 instruction tokens either way. The emitted path adds 1,620 checkpoint tokens across 27 readouts. With a 216-token prefix, each full segment retires 19 instructions, reaching 444 active execution tokens and 504 positions when the output reserve is counted. The active cache is at most 1,903,872 logical bytes at 444 positions. The old schedule without a reserve retires 24 instructions per full segment, reaches 504 active positions, and needs 21 resumptions.

| Workload | Emitted state | Host, matched segment limit | Host, original schedule |
| --- | ---: | ---: | ---: |
| 519 instructions | 3,510 ms | 3,406 ms | 3,463 ms |
| 154 instructions | 1,018 ms | 1,025 ms | 1,089 ms |

On the 519-instruction run, medians per resumed segment were **0.143 ms checkpoint emission**, **0.113 ms validation and prefix splice**, and **8.14 ms fresh-cache prefill**. Matched host construction took **0.288 ms median** with **7.77 ms** prefill. Total includes generation, prefix build, readout where applicable, prefill, and source-frame audit. It excludes loading, reference verification, and the independent prefix comparison on the emitted path. These are descriptive single-thread measurements of one warmed whole replay per path.

## Reproduce

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
PYTHONPATH=.:../backend python tools/replay_decoder_emitted_checkpoint.py --open evidence/decoder-emitted-checkpoint/replay.json
npm run dev -- --port 4180
```

Open `/replay-viewer.html` and choose **Open emitted checkpoint replay**. [Frozen IR](decoder.ir) · [Tape and checkpoint replays](replay.json) · [Raw timings](timings.json)
