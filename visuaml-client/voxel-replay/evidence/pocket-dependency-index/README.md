# Queryable Pocket dependencies

The benchmark reuses the frozen Rust WASM guest, source checkpoint, and exact automaton from demand recovery. The existing dependency-key path and uncached replay remain references. A reverse index maps a potential edit `[tick, x, y]` to observations it could affect. Each observation records its tick-zero source cells and possible effect locations. Its replay proof also gives the actual relevant effects and the digest of the sealed event prefix.

The index includes potential edits in the backward cone even if the original history had no edit there. For a materialized edit at tick `e`, a read at tick `t` can depend on it only within radius `t-e`. Every indexed cache lookup still validates the authoritative ledger and binds the answer to query, source checkpoint, rule, and materialized effects. The baseline projects those effects directly. The indexed path uses precomputed effect-slot membership and preserves event order. A missing ledger leaves the LOAD at PC 38 pending. The exactness contract relies on the scheduler supplying all committed edits through the read tick.

## Edited histories

Closing (17, 10) before tick 1 invalidates none of the six target reads. The edited run reuses every answer, and all guest boundaries match the full-world run.

In the second edit, removing the remote wall at (2, 8) invalidates the first remote observation. Re-executing the guest changes its eighth source instruction, LOAD at PC 38, from 0 to 1. The Rust action produces a **new** gate-open event before tick 2. The prior trace has no such event. The replay records the discovered observation-to-action-to-edit edge and looks up the gate event. It affects the tick-6 consumer read at (13, 8). The first five consumer answers are reused. Tick 6 is recomputed and its LOAD at PC 38 reads 1 instead of 0. All resumed boundaries and final pages again match the independent full-world run.

The index identifies potential invalidations. Guest execution still runs to reproduce or discover the branch effects. This witness has one scheduled guest producer and one consumer, with deterministic wall edits. It gives no certificate for effects the scheduler cannot enumerate.

## Work and latency

Four alternating CPU pairs on one thread. Each trial has 18 forks and 108 reads at three overlapping coordinates. Both policies use the same cache keys, achieve **84 hits**, read **11,130 source bytes**, and recompute **1,553 cell-ticks** per trial. The indexing change reduces some lookup work while retaining exactly the same reuse policy.

| Median total per trial | Direct dependency key | Indexed |
| --- | ---: | ---: |
| Ledger access, validation and digest | 2.19 ms | 1.97 ms |
| Dependency lookup, including first index construction | 2.16 ms | 1.77 ms |
| Answer binding and cache validation | 1.24 ms | 1.16 ms |
| Source bytes loading and verification | 3.08 ms | 2.63 ms |
| Cell recomputation | 1.80 ms | 1.71 ms |
| VM restore and delivery | 51.19 ms | 48.61 ms |
| Total pending time | 63.06 ms | 59.68 ms |
| Resumed, traced guest execution | 539.93 ms | 525.84 ms |

The ledger already lives in memory. The source-loading row measures indexed file reads and row verification. Each row is a median of totals, so rows need not sum to the median pending time. The timed guest suffix includes per-instruction evidence hashing. World advancement, full-world oracle execution, and setup are outside these phase timers. VM restore and delivery dominate the observed suspension. These four samples characterize this fixture and give no general performance guarantee. [Raw phase samples](timings.json)

## Inspect and rerun

From `visuaml-client/voxel-replay` in the research environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_dependency_index.py --effect 2 9 8
python tools/replay_dependency_index.py --read 2 8 1
python tools/replay_dependency_index.py --open evidence/pocket-dependency-index/replay.json
```

The first command lists the consumer read at (13, 8) on tick 6. The second lists the remote read's source cells and the new effect it produced. The final command reruns both edited histories and one comparison trial. Use `--write` for four timing pairs. [Indexed replay](replay.json) · [Earlier cache policies and stale-answer controls](../pocket-shared-recovery)
