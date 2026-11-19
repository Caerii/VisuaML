# v58 unit push/pop contraction

The dense operand transition builds both height and cell candidate
stacks for all four actions on every call. In the admitted unit
push/pop case, the one-hot action and fixed gain select one candidate
exactly. The new path locates its row from the validated one-hot
height, copies the cell array once, and writes or clears only that
row. The height still comes from the original fixed up/down matrix.
A pop returns its own copy of the selected word. Retain, truncate,
non-unit active gains, and ndarray subclasses follow the dense path.

The change applies to traced and untraced execution. Source dispatch,
route and frame gates, local and arithmetic gates, memory management,
and fuel accounting are unchanged. The operand data still lives in
the same bit-sliced NumPy state. The selected cell update is now a
host-indexed copy equivalent to the unit-gain contraction. No source
instructions are grouped. The sparse path does not call the packed
executor.

## Diagnostic profile

The baseline diagnostics warmed each input, then instrumented one
eight-agent crowded tick and the first 2,000 instructions of archived
BFS. Line tracing and cProfile ran in separate passes. These are
instrumented costs. Candidate arithmetic and the temporary arrays it
produces share the same NumPy expressions. Candidate-stack calls
are measured separately, as is the final gain contraction.

| Baseline category | Crowded tick | BFS prefix |
| --- | ---: | ---: |
| Candidate arithmetic and its temporaries | 32.3 ms | 58.4 ms |
| Candidate-stack allocation and copying | 37.6 ms | 67.6 ms |
| Final gain and candidate contraction | 23.0 ms | 43.1 ms |
| Operand admission and postchecks | 163.0 ms | 293.0 ms |
| WASM memory opcode body | 5.8 ms | 2.6 ms |

The executor and operand adapter exclusive times in [profile.json](profile.json)
show the host overhead separately. Driver self time includes its other
inline bookkeeping. The line-traced memory measurement covers address
calculation, bounds and NumPy byte access. Those two figures use
different instrumentation and should not be subtracted from each other.

| Diagnostic sample | Push/pop transitions | Candidate-stack allocations before | After |
| --- | ---: | ---: | ---: |
| Crowded terrain, 1,250 source steps | 1,777 | 3,716 | 162 |
| BFS prefix, 2,000 source steps | 3,142 | 6,476 | 192 |

The counters count the two `np.stack` calls made by the dense path.
They exclude other temporaries and the remaining independent output
copies. The remaining stacks come from truncations.

## Verification and reproduction

The before and after paths use the same WASM and input pages. Each
terrain tick checks full memory against Wasmtime and the same saved
input history. Full BFS and the one-blank Sudoku run check each result,
source step count, and every byte of the final page against Wasmtime.
The original and optimized paths also reproduce the saved edit and
no-edit histories with identical detailed traces. Fuel cutoffs around
the witness store and the page-straddling store trap compare the full
completed trace, trap site, and page. Independent buffer ownership,
an aliased push input, and the existing first-divergence tests are
covered by the focused regressions.

From `visuaml-client/voxel-replay`, use the existing research Python
environment.

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/profile_operands.py
python tools/operand_pairs.py sparse --pairs 6
python tools/operand_pairs.py crowded --pairs 6
python tools/operand_pairs.py bfs --pairs 2
python tools/operand_pairs.py sudoku --pairs 6
python -m pytest -q tools/test_operand_replay.py tools/test_terrain_verify.py \
  ../backend/tests/test_neural_operand_storage_v55.py
```

Each run starts with a discarded warmup pair and alternates order.
Terrain pairs each replay four consecutive ticks from the same world
snapshot, giving 24 measured ticks per variant and scene. The warm
execution boundary is the first route. Setup through the first
route is recorded separately for each call. The experiment does not
batch independent worlds. The raw samples include timings and
source-step counts.

## Paired results

| Workload | Samples per path | Source steps | Median before | Median after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight sparse agents | 24 | 1,650 per tick | 818 ms | 717 ms | 1.14× |
| Eight crowded agents | 24 | 1,250 or 1,362 per tick | 667 ms | 600 ms | 1.11× |
| BFS seed 0 | 2 | 41,591 | 27.64 s | 25.86 s | 1.07× |
| One-blank Sudoku | 6 | 893 | 628 ms | 584 ms | 1.08× |

Sparse throughput rose from 9.6 to 11.0 agent-ticks/s. Crowded
throughput rose from 12.0 to 13.1. Sparse p95 fell from 902 to
761 ms and crowded p95 from 714 to 692 ms, about 15.2× and 13.8×
the 50 ms tick budget at 20 Hz. These are nearest-rank p95
estimates from 24 samples per path. BFS and Sudoku sample sizes
are too small to characterize the tail.

Median per-call time through the first route was around 2.5 ms on
terrain in both variants. Raw setup and execution measurements
are recorded independently. Each workload was pinned to one CPU
with one numerical-library thread. The four timing processes ran
on separate CPUs in the same environment.

[Raw sparse samples](sparse.json) · [Raw crowded samples](crowded.json) · [Raw BFS samples](bfs.json) · [Raw Sudoku samples](sudoku.json)
