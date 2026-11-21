# Paired histories and immutable CODE cache reuse

The checkpoint decoder from `7d2a1b1` is unchanged. This experiment checks whether it emits the same restart state for histories that converge, then reuses the immutable CODE cache at restart. Both paths use the same four-word checkpoint and 512-token context.

## Three histories, one checkpoint

The same program runs three times with different initial values at address 2 and a countdown at address 3 that takes one, two, or three iterations. It overwrites address 2 twice and clears address 3 through different sequences of committed writes. The surviving store to address 2 occurs at instruction four. It is five, nine, or thirteen instructions old at the join.

The runs arrive at PC 9 after 9, 13, and 17 retired instructions. R0 is 7 and memory is `[11, 7, 7, 0]`. Each run emits the same 60-token checkpoint and the same continuation. The complete histories and continuations agree with the scalar reference. The retired totals are host bookkeeping; the guest has no instruction that reads them or the host segment budget.

An intervention changes only retained address 2 from 7 to 0. The prediction was fixed in the holdout manifest. The first resumed LOAD returns 0, changing emitted token three and sending the following branch to PC 11 instead of PC 13. The intervened trace matches a scalar continuation initialized with the edited word. All 60 strict prefixes of a checkpoint are rejected. All 11 unfinished-store cuts reject checkpoint emission.

## What is reusable

Each restart prefix contains 156 immutable CODE tokens followed by 60 state tokens. At every decoder layer a position depends only on input positions up to itself. When the artifact, tokens, order, and starting position are equal, later state tokens cannot change the CODE keys and values. That lets the host cache only CODE once and apply prefill to the 60 state tokens at the original offset. Old execution and memory tokens are outside the reused cache. There is no change to the checkpoint readout or instruction semantics.

The implementation binds the in-memory cache to the decoder instance, artifact hash, exact CODE tokens, and position zero. It rejects wrong code, a wrong artifact, a shifted position, and an attempt to bind execution/state tokens. Layer-by-layer keys, values, and suffix logits compare exactly with fresh prefill at every archived restart boundary. The reused CODE cache is unchanged after each replay.

## Paired measurements

Four alternating warm CPU pairs per workload, on one thread. Both paths reserve 60 slots for checkpoint emission, so they retire the same source work with the same segment limits. Raw samples include one-time CODE prefill, restart prefill, checkpoint emission, and total latency. Total timing includes prefix validation, execution, and frame audit. It excludes weight loading, test-only equality checks, and scalar reference verification.

| Workload | Retired / resumptions | Full restart prefill | Reused restart prefill | Full total | Reused total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing long holdout | 519 / 27 | 8.01 ms | 2.02 ms | 3.417 s | 3.247 s |
| Independent post-freeze holdout | 135 / 7 | 9.06 ms | 2.08 ms | 0.925 s | 0.897 s |

The table reports pooled restart medians and whole-run medians. On the 519-instruction run the CODE seed cost 3.30 ms median once, and checkpoint emission remained near 0.15 ms per resume. Restart prefill fell about 75%, and median whole-run latency fell about 5%. The independent program counts down by two and uses different addresses and arithmetic. Its traces match at every retirement too.

Both paths reach 444 active execution positions and 504 including the output reserve. Peak active logical KV is 1,903,872 bytes in both. Reuse also retains a separate 668,928-byte logical CODE cache. The figures count tensor elements and do not measure allocator residency. The current append implementation materializes new active cache tensors.

## Reproduce

From `visuaml-client/voxel-replay` in the research environment:

```sh
PYTHONPATH=.:../backend python tools/replay_checkpoint_prefix_reuse.py --open evidence/checkpoint-prefix-reuse/replay.json
npm run dev -- --port 4180
```

The verifier reruns one full pair per workload and checks the archive. Use `--write` to collect four alternating timing pairs. Open `/replay-viewer.html` and select **Open checkpoint prefix reuse replay** to inspect or download the saved evidence. [Paired histories and exact replay](replay.json) · [Raw timings](timings.json) · [Frozen decoder](../decoder-emitted-checkpoint)
