# Live selection of memory work

The [frozen budget-64 graph](machine.pt) extends the selected-ALU baseline
from `04894a7`. Its SHA-256 is
`52234c828b67e30b7d42b34e5b1cb24634f02ddbd78e92a7d45e1c63b4773e9d`.
Guest WASM, terrain rules, and the execution budget stay fixed.

The baseline used `where` for some data selection and gathered four guest
bytes on every ordinary step. The new [IR](machine.ir) puts the effective
address, byte gather, and word assembly in the true arm of a `prim::If`
driven by the fetched opcode. The false arm returns zero placeholders. For
an ordinary non-memory instruction the access tensor is the empty event.
Kind zero means no memory access, so the old incidental payload words have
no audit meaning. The audit still compares all words for loads, stores, and
fault events. For memory instructions the prior checked bounds, store
commit, and access trace remain intact. No Python callback or native packed
path participates.

## Profile and timing

A complete audited tick contains 462 memory rows in 1,650 terrain steps.
Full BFS has 2,334 memory rows in 41,591 steps. The graph skips 1,188 and
39,257 unnecessary four-byte gathers, respectively. It also skips address
assembly and the access-event stack on those ordinary non-memory rows.

The baseline 64-step terrain profile attributed 3.2% of CPU self-time to all
gathers combined, an upper bound of about 6 ms on a 187 ms tick for gathers
alone under that profile. This includes the live operand and local gathers
that the change retains. On the same instrumented prefix total gather
calls fall from 256 to 208, `where` from 874 to 778, and access-event
`stack` from 64 to 16. The BFS prefix cuts gathers from 256 to 198 and
event stacks from 64 to 6. Actual stores are unchanged. Profiling is
separate from wall timing. [Raw profiles and paired timings](timings.json)

| Workload | Baseline → selected-memory median | Terrain p95 | Agent-ticks/s |
| --- | ---: | ---: | ---: |
| 8 agents, 64 KiB | 184.91 → 173.43 ms | 196.32 → 182.50 ms | 42.8 → 46.1 |
| 8 agents, 16 MiB | 188.08 → 174.69 ms | 193.85 → 178.82 ms | 42.4 → 45.8 |
| Full BFS | 4.816 → 4.456 s | — | — |

Terrain has 21 warm samples per variant after excluding the first tick,
with nearest-rank p95. BFS has two full completions per variant. Terrain
uses 26 host invocations at 1,650 steps per tick, and BFS uses 650 at
41,591 steps. Both variants use the same input binary and page at each
paired comparison. All timed outputs matched Wasmtime byte for byte.
At 173–175 ms, terrain remains about 3.5× over the 50 ms target.

The graph froze in 36.5 ms and loaded in 11.2 ms in the paired run.
Image preparation took 1.4–4.0 ms. Full snapshots took about 3 µs at
64 KiB and 0.85 ms at 16 MiB. Preparation, page initialization, snapshots,
verification, and trace collection are outside the execution timings.

## Replay

[Source traces and full-coverage opcode counts](replay.json) verify both
Terrain Edit histories, nested calls and helper traps, fuel cuts around
the rollback write, and the Rust helper edit. Every small-witness prefix
matched the single-step VM state, including full memory and frame tensors.
The Terrain Edit histories matched Wasmtime and the saved checked NumPy
digests and step counts. Packed fallback was blocked.
93 focused Python tests and five browser replay tests passed.

Run from `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch 2.9.1 and Wasmtime:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_frozen_memory_selection.py
python tools/bench_frozen_state_reuse.py --memory
```
