# Demand recovery for a pending Rust world read

Three forks import the same small checkpoint. It contains the suspended Rust helper, the certified target's regional grid, and a binding to the tick-zero world index. The host supplies the fork's observation input before the helper reads it. The local fork uses the original target at (14, 8). The remote fork requests (2, 8). A third fork requests (13, 8), a retained coordinate whose dependency cone is outside the original guarantee. All three execute the same newly compiled Rust artifact and the same frozen direct-call graph.

## Pending read and recovery

The helper opens the gate once at PC 10. The host commits that edit before the first synchronous automaton tick and records it in the event ledger. On each tick the observer invocation runs until **PC 38**, immediately before the mailbox LOAD. No observation value has been delivered. The pending snapshot preserves the private page, operand stack, PC, fuel, and call state; the broker records the requested coordinate and tick. The world clock remains at the requested tick throughout recovery.

The recovery source is a seekable 720-byte tick-zero world image with a bound index of row digests. Only required rows are read and checked. The broker selects the exact radius-`tick` Manhattan cone for the requested cell and applies the tick ledger in order. At step `k` it computes only the radius-`(tick-k)` cone. A gate edit in that cone is applied at its original tick. This suffices for the snapshot's radius-one, sealed, synchronous rule and the declared single-gate event schedule. Recovery supplies a single signal bit, restores the pending VM state, and executes the LOAD without replaying the helper or taking another world tick.

If the source is unavailable, a read cannot complete. Both a missing checkpoint source and a missing event ledger stop at the same pending VM checkpoint before any observation is supplied. The negative fixtures model absence explicitly. The shared artifact and source are separate downloads so the local fork can run with zero source bytes.

## Results

- At **(2, 8), tick 1**, the source cone contains five cells. One cell update recovers 1, and the guest stores action 1. The observer's trace and complete memory page match the full-world reference.
- At **(13, 8), tick 5**, replay includes the gate write from PC 10 before tick one. The correct signal is 1. Using the tick-four result would deliver 0 and leave action off. The stale-tick control supplies the tick-four result to the same pending tick-five VM state and records the different branch and unchanged action.
- All six observation invocations in each of the three forks match the full-world run at every guest boundary, including the paused pre-load state, branch, action store, and final page digest. Completed invocations also agree with Wasmtime. Packed fallback is blocked.
- The local fork never reads the world source. Both expansion forks start from the identical shared artifact and keep their own row cache and VM pages.

## Work and latency

One recorded CPU run per fork, one thread. Suspension is measured from the pending broker request through recovery, VM checkpoint restore, and mailbox delivery, ending just before execution of the LOAD. It excludes initialization, the first pre-load VM prefix, and later verification. Read bytes are file payload bytes read, including the index on first use. Row granularity means some retrieved bytes lie outside the exact cone. Repeated reads retain authenticated rows while recomputing cell updates for the requested tick.

| Fork | Additional source bytes | Recomputed cell-ticks across six reads | Observed suspension range |
| --- | ---: | ---: | ---: |
| Local | 0 | 0 | 0.42–0.51 ms |
| Remote | 1,855 | 281 | 0.57–0.83 ms |
| Delayed | 1,855 | 301 | 0.55–0.68 ms |

The first remote recovery reads 1,455 bytes, including the 1,335-byte index, and one 120-byte span of three rows. The shared checkpoint JSON is 89,566 bytes and includes the original 64 KiB guest page in its VM envelope. [Raw per-read samples](timings.json)

## Reproduce

From `visuaml-client/voxel-replay` with the existing research Python environment:

```sh
PYTHONPATH=.:../backend python tools/replay_demand_region.py
PYTHONPATH=.:../backend python tools/replay_demand_region.py --fork remote --checkpoint evidence/pocket-demand-region/shared-checkpoint.json
npm run dev -- --port 4182
```

Open `/pocket-coupled.html` and use **Recover a pending observation** to inspect the saved read, recovered cone, edit, and result. Use `bash tools/build_demand_region.sh` with Rust's WASM target installed to rebuild the guest. [Replay, including pending snapshots](replay.json) · [Shared checkpoint](shared-checkpoint.json) · [Source index](world-index.json) · [Source rows](world.bin)
