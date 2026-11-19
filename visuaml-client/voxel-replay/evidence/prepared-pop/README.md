# v58 prepared operand pop

The default array-memory executor constructs an invocation-local stack
after source, route and microop token admission. Only the canonical
coefficient and token profile is eligible. Supplied gate weights or
tokens, legacy execution, and explicit `prepare_operand_pop=False`
keep the checked stack. No prepared metadata is shared between
programs, supplied by callers, or retained across invocations.

| Check | Dependency | Prepared behavior |
| --- | --- | --- |
| Pop action shape and one-hot token, default value and truncation tokens, active gain and type eligibility | Invocation's canonical execution profile | Fixed by the internal pop method and the default profile selected at entry |
| Binary cells, canonical height and zero inactive rows | Owned live-state invariant | Check at stack construction, retain checked push and truncation, and preserve the invariant by the unit pop construction |
| Nonempty stack | Live execution | Retained in the executor and in the prepared stack method |
| Public argument, page and fuel; dynamic i32 memory address and fuel cutoff | Public or live execution | Existing admission, bounds and fuel behavior |

The already admitted stack height selects a live row. Pop uses the
same fixed down matrix, copies the cells and popped word independently,
and clears the retired row. The binary invariant survives every
incoming edge because push and truncate still validate their output,
including truncation during branch restoration. Reads and trace
enumeration cannot mutate a live state. The executor does not retire
multiple source instructions at once. There is no native fallback.

## Diagnostic ceiling

The baseline profiler warmed each workload before timing. A separate
line-traced pass split checked-pop input and output scans, static
token and coefficient setup, live depth and resource checks, copy
and row operations, and the fixed height product. Memory-opcode
bodies were measured separately. The line categories include their
callees and are disjoint within the sampled pop. Driver exclusive
time and remaining gate costs come from a separate cProfile run of
the prepared variant.

| Sample | Checked pops | Owned-state scans | Static setup and unit eligibility | Height matrix | Copy and row | Memory opcode bodies | Diagnostic ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse terrain, one tick | 1,160 | 56.85 ms | 36.33 ms | 5.08 ms | 2.98 ms | 7.33 ms | 1.079× |
| Crowded terrain, one tick | 888 | 42.91 ms | 28.81 ms | 2.81 ms | 2.28 ms | 6.63 ms | 1.077× |
| BFS, 4,000 steps | 3,145 | 148.52 ms | 100.62 ms | 9.41 ms | 8.10 ms | 4.68 ms | 1.069× |
| One-blank Sudoku | 783 | 36.54 ms | 25.32 ms | 2.37 ms | 1.98 ms | 1.96 ms | 1.073× |

The ceiling assumes that all static setup and repeated state scans
in pop vanish and all other line-traced costs stay fixed. It is an
Amdahl estimate from instrumented data, with tracing overhead in
the denominator. Use the paired execution runs for speed claims.
Prepared cProfile puts driver exclusive time at roughly 2–4%.
The fixed ALU remains the largest measured named subsystem
at about 22–23% on terrain and 40% on the BFS prefix. This evidence
does not support starting straight-line host dispatch grouping in
this experiment.

## Reproduction and exactness

From `visuaml-client/voxel-replay` with the research Python environment:

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/profile_pop.py
python tools/prepared_pop_pairs.py sparse --pairs 6
python tools/prepared_pop_pairs.py crowded --pairs 6
python tools/prepared_pop_pairs.py bfs --pairs 2
python tools/prepared_pop_pairs.py sudoku --pairs 6
python -m pytest -q tools/test_prepared_pop.py tools/test_operand_replay.py tools/test_terrain_verify.py
```

The paired tool discards a warmup pair and alternates order. Setup
through the first route is timed separately on every invocation.
Each terrain pair measures four consecutive ticks from the same
initial world. Counter passes run after timings. One state scan is
added per prepared invocation and two are removed per prepared pop.
The counter also records checked and prepared pop calls directly.
BFS timing uses the full 41,591-step graph and the separate counter
pass uses a 4,000-step prefix.

The checked and prepared paths reproduce both saved terrain histories
and identical detailed traces, including loop backedges. Per-tick
pages match Wasmtime, as do the final full BFS and Sudoku pages.
The focused checks also cover exact fuel before and after the store,
a page-straddling trap and completed trace, malformed public
arguments, caller-supplied coefficients, state ownership, and the
existing first-divergence injections. Verification blocks packed
fallback.

## Paired results

| Workload | Samples per path | Source steps | Warm median checked | Warm median prepared |
| --- | ---: | ---: | ---: | ---: |
| Eight sparse agents | 24 ticks | 1,650 per tick | 744 ms | 685 ms |
| Eight crowded agents | 24 ticks | 1,250 or 1,362 per tick | 607 ms | 576 ms |
| Full BFS seed 0 | 2 runs | 41,591 | 24.70 s | 23.38 s |
| One-blank Sudoku | 6 runs | 893 | 572 ms | 516 ms |

Sparse terrain rose from 10.72 to 11.60 agent-ticks/s and crowded
terrain from 13.08 to 13.86. Sparse p95 fell from 784 to 730 ms.
Crowded p95 went from 661 to 686 ms even as the median improved.
These nearest-rank p95 estimates use 24 samples per path. The
optimized sparse and crowded tails are still 14.6× and 13.7×
the 50 ms budget.

The short workloads were measured one at a time on CPU 0 in the
reported repeat, with one numerical-library thread. The initial
same-host concurrent screen had more variation and showed a crowded
median regression. Its raw samples are retained in
[concurrent-screen](concurrent-screen). Full BFS was pinned to CPU 2
during that screen, and both optimized samples were faster than both
checked samples. Two BFS pairs measure the observed median transfer
and give little information about its tail.

| Counter sample | Prepared pops | Checked pops on prepared path | Operand-state scans avoided |
| --- | ---: | ---: | ---: |
| Sparse terrain, four ticks | 4,640 | 0 | 9,276 |
| Crowded terrain, four ticks | 3,712 | 0 | 7,420 |
| BFS, 4,000-step prefix | 3,145 | 0 | 6,289 |
| One-blank Sudoku | 783 | 0 | 1,565 |

Median per-call setup through the first route was approximately
2.4–2.5 ms on terrain. The larger observed setup increase among
the median estimates was 0.37 ms on BFS, while each measured warm
latency saving was tens of milliseconds or more. Preparation is
invocation-local, so the observed savings recover its cost within
the same invocation. There is no cross-run amortization claim.

[Exclusive profile and phase timings](profile.json) · [Sparse samples](sparse.json) · [Crowded samples](crowded.json) · [BFS samples](bfs.json) · [Sudoku samples](sudoku.json)
