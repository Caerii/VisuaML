# Sparse checkpoints for the word decoder

The frozen sparse decoder admits 1–16 addresses, each holding a 32-bit word with default value zero. The address range is declared when admitting a program and carried with its immutable image in the replay. This decoder was frozen at `de37a96` before the high-address program was authored. The original four-word decoder and checkpoint remain the reference for the shared instruction subset.

## Representation and readout

The checkpoint is a sorted list of nonzero memory frames followed by the live PC and R0 frame. Every frame is 12 existing vocabulary tokens. An absent address reads as zero. A committed write of zero supersedes its prior value; the next checkpoint leaves out that address. Every nonzero word is retained regardless of whether the remaining program uses it.

The decoder scans one query per declared address against the old memory cache. Matching COMMIT keys gain 16,384, addressless default COMMITs gain 8,192, and position adds at most 8,176 over 512 positions. A matching record wins over a default, and a default wins over unrelated positions, with minimum margin 16. Among matching keys the latest position wins. At least the live-state COMMIT supplies a default candidate. Its payload is zero, as are other valid addressless COMMIT payloads. The existing ReLU recovery circuit recovers bits under the binary64 score and kernel margin assumptions described for the word decoder.

The tensor graph filters zero words, emits surviving addresses in ascending order, and reads PC and register from the last committed frame. The host passes the bound and capacity, validates framing, concatenates CODE with the emitted frames, and starts a fresh cache. It never inserts default memory frames or rewrites retained values. Program admission rejects addresses outside the declared range. The existing decoder graph covers the maximum 16 addresses; the range adjusts the checkpoint scan and the admitted operands, not the compiled width.

The host reserves `12 * (capacity + 1)` positions before execution. The graph rejects unfinished stores, requests without enough reserve, and retained-word overflow. The long holdout reserves 60 positions for up to four words plus live state. Partial output cannot replace a valid snapshot because the host requires complete framing and the final live-state row.

## Exact replay

The post-freeze program writes 7 to address 15, erases a prior word at address 14, then repeatedly overwrites a separate counter. Across **155 instructions and seven resumptions**, every retired state matches the scalar reference. Removing retained address 15 from the converged checkpoint changes the next load to zero and sends the branch to PC 10.

Four valid histories vary stale words and countdown length. They join after 8, 12, 16, and 20 instructions. All write back to zero and emit the identical 36-token checkpoint holding addresses 1 and 15 and the live register/PC. Each continuation matches the version that runs directly from the full history. Two four-address controls independently match the dense checkpoint's meaning and emitted continuation. Their sparse checkpoints are 24 and 48 tokens, versus 60 in the fixed format.

## Three independent axes

Each microbenchmark cell contains 16 CPU samples on one thread. Scan measures address enumeration and value lookup. Emission measures canonical filtering, framing and PC/register readout. Both use the frozen tensor entry points. Timing prefill uses a fresh cache and the actual compacted tokens. Layer KV figures count logical elements, excluding allocator overhead.

| Axis and control | Sweep | Checkpoint tokens | Median scan | Median restart prefill |
| --- | --- | --- | --- | --- |
| Address range, same 120-position history and two words | 8, 12, 16 | 36 in all cases | 58, 61, 66 µs | 0.81–0.82 ms |
| Retained words, same 120-position history and range 16 | 0, 2, 4, 8 | 12, 36, 60, 108 | 63–72 µs | 0.58–2.12 ms |
| Old history, same range and same final two words | 300, 348, 396, 444 positions | 36 in all cases | 102–116 µs | 4.44–6.55 ms |

Emission medians were 69–79 µs in the short-image axes. Fresh logical KV in the density sweep grew from 107,712 to 538,560 bytes as the prefix gained retained entries. The scan computes one attention row per address over the entire existing cache. Its work is proportional to declared addresses times source positions even when few words survive. This small sweep characterizes the bounded range on one machine.

The end-to-end holdout generated 1,860 execution tokens. It reached 444 active positions, or 504 including reserve, and 1,992,672 logical KV bytes. In one CPU replay total generation, checkpointing and splice time was about 1.09 s. Median combined checkpoint readout was 0.225 ms and median prefill was 5.90 ms across resumed segments. Loading and independent verification are outside the timer. [Raw timing samples](timings.json)

## Reproduce

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
PYTHONPATH=.:../backend python tools/replay_sparse_checkpoint.py --open evidence/sparse-checkpoint/replay.json
```

Use `--write` to regenerate the replay and benchmark the three axes. [Exact replay](replay.json) · [Post-freeze program](holdout.json) · [Frozen graph IR](decoder.ir)
