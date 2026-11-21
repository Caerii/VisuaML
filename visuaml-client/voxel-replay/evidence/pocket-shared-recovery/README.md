# Shared recovery across divergent world histories

Six forks start from the same combined checkpoint and use the frozen demand-region Rust guest. Each consumer reads (13, 8) on six ticks. Two forks also use a remote Rust observer before scheduling the consumer's gate event. The benchmark compares uncached recovery, a full-history key, and a projection of the sealed event ledger.

## Binding and completeness

The full-history key binds the combined source checkpoint identity, indexed source image, source tick, world rule hash and dimensions, the requested position and tick, and the ordered ledger prefix. The dependency key binds the same source and query data plus the ordered materialized wall edits in the backward cone. An edit applied before update `e` can influence a read at tick `t` only if its edited cell is within Manhattan radius `t-e`. Within a tick the original edit order is preserved. The baseline computes the same proof metadata for inspection and bypasses answer storage.

The runtime seals one ordered ledger row after assembling the scheduled edits for each tick. For the admitted rule, materialized walls and the tick-zero source capture everything the grid update can observe. The proof records the ledger digest, sealed tick, relevant edit sequence, and producing ledger digest. Certifying a projected hit depends on this authoritative scheduler supplying all materialized edits. The digest binds the record it receives; ledger availability and completeness are runtime assumptions, not a service that independently authenticates arbitrary external histories. A missing ledger or unsupported event cannot justify a hit, and the read remains pending. Cache entries are private in-memory objects. Bindings reject stale answers; this is not a defense against an attacker forging an answer and its binding together.

## Where answers are shared

- An identical fork shares all six reads under both keyed policies.
- Closing the wall at (17, 10) creates a different ledger. Full-history caching misses all six reads while the dependency policy shares all six. The edited cell lies outside every queried cone.
- Leaving the signal gate closed shares the first four answers with the open history. At ticks 5 and 6 the gate is inside the cone and the closed fork recomputes.
- Two shortcut histories differ at the remote cell (2, 8). Rust reads the cell on tick 1. Its recorded action determines whether the scheduler opens the gate at (9, 8) before tick 2. At tick 6, the consumer sees 1 in the action-on history and 0 in the action-off history. A diagnostic projection of only the originally spatial edits would give the same key. Including the materialized gate event distinguishes the histories. The replay records the producing Rust LOAD checkpoint, action PC, page digest, and later gate edit.

A wrong-tick answer and a wrong-branch answer are both injected into the actual broker at the fifth pending read. Both leave Rust at PC 38, with only ticks 1–4 supplied. A cached prior answer also cannot satisfy a request with an unavailable ledger. Every successful producer and consumer trace, including instruction records and final pages, is compared with a full-world run. Completed observer invocations also match Wasmtime. Packed fallback is blocked.

## Work and latency

Four alternating CPU trials per policy, 38 reads each, one thread. These are medians over aggregate per-trial counts and suspension time. Source bytes include repeated reads of each fork's authenticated row index. Row caches are private to each branch; answer reuse avoids those reads entirely when there is a hit. Suspension includes proof construction, source replay if needed, and VM checkpoint restore through mailbox delivery. It excludes setup, world advancement, and full-world verification.

| Policy | Cache hits | Source bytes read | Cell-ticks recomputed | Total suspension |
| --- | ---: | ---: | ---: | ---: |
| Uncached | 0 | 14,040 | 1,808 | 26.42 ms |
| Full history | 7 | 12,185 | 1,506 | 25.32 ms |
| Dependencies | 27 | 8,475 | 680 | 21.40 ms |

Projection saves replay work in this workload. Every observation still incurs the paused VM restore, so cut cells and total suspension improve by different amounts. [Per-read timing samples](timings.json)

## Reproduce

From `visuaml-client/voxel-replay` in the research environment:

```sh
PYTHONPATH=.:../backend python tools/replay_shared_recovery.py --open evidence/pocket-shared-recovery/replay.json
npm run dev -- --port 4182
```

Use `--write` to regenerate all four timing trials. Open `/pocket-coupled.html` and select **Explain cache decision** to inspect the divergent cases. [Exact replay and provenance](replay.json) · [Source checkpoint and world index](../pocket-demand-region)
