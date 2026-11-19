# v58 prepared const/add regions

The starting comparison is the d1ff1df prepared-ALU and prepared-pop
path. The original Rust and archived WASM binaries, terrain rules,
and frontend are unchanged. Configuration flags isolate prepared
multiply, the two-instruction region, and the adjacent extension.

## Selection from the warmed profile

| Sample | Source steps | const/add | local.get/const/add | Multiply still selected through generic ALU |
| --- | ---: | ---: | ---: | ---: |
| Sparse terrain tick | 1,650 | 65 | 48 | 0 |
| Crowded terrain tick | 1,250 | 53 | 36 | 0 |
| Full BFS | 41,591 | 2,142 | 2,111 | 4,381 |
| One-blank Sudoku | 893 | 70 | 25 | 124 |

Pair coverage was about 8% of each sampled terrain tick and 10.3%
of full BFS source steps. Full BFS cProfile attributed 6.66 s to the
generic ALU and 1.05 s to already-selected circuits, 5.94 s to frame
events, 3.61 s to frame-state transitions, 3.03 s to checked operand
transitions, 2.89 s to routing, and 0.75 s to exclusive driver time.
A separate line trace measured about 46 ms in memory opcode bodies.
These instrumented slices have different overhead and should not be
added to derive the execution time. The terrain profiles are in the
same directory.

The generic ALU still evaluated all six candidate circuits for
multiply. Its mean per-call profiled cost, minus the retained
multiply circuit, suggested about 3.5 s of avoidable circuit work
at the 4,381 selected multiply sites in full BFS. Charging the
pair all baseline routing, frame and operand overhead across both
its steps gave a rough region opportunity of about 1.7 s. Both
figures are diagnostic estimates. The larger multiply opportunity
motivated measuring prepared multiply as its own stage. That path
uses the existing ReGLU products, grouping, and carry staircase.

## Region semantics

Canonical source admission supplies the immediate bits, existing
prepared add, and validated route coordinates. Each region needs
unconditional sequential edges and no alternate entry to any
interior source PC. It has no memory access or internal branch.
Custom tokens and coefficients use the original per-instruction
path. Traced calls and insufficient fuel or stack capacity use
individual source steps.

The pair keeps the constant and add result in local temporaries.
It reads the admitted live top word, invokes the existing fixed
ReLU adder, and copies the cell tensor once while replacing that
word. The entry and exit height are equal. This removes the
transient const push, two pop states, and a separate result push.
Routing and frame transitions for the two unconditional retain
instructions are proved from the canonical profile rather than
materialized. Each successful region call retires two source steps.

The adjacent extension adds a leading local.get when its own route
qualifies and neither interior PC has another predecessor. It
executes the original fixed local read and fixed add, retains the
unchanged local bank, and materializes only the final checked
push. It retires three source steps. Python dispatches the bound
regions and advances PCs and fuel. NumPy handles binary vectors
and cell copies, and the selected gate circuits compute from live
values. Packed C++ fallback is blocked in the verification runs.

| Counter sample | Source steps | Host dispatches before → pair → adjacent | Full cell copies avoided, pair / adjacent |
| --- | ---: | ---: | ---: |
| Sparse terrain, four ticks | 6,600 | 6,600 → 6,340 → 6,148 | 780 / 972 |
| Crowded terrain, four ticks | 5,224 | 5,224 → 5,012 → 4,868 | 636 / 780 |
| Full BFS | 41,591 | 41,591 → 39,449 → 37,338 | 6,426 / 8,537 |
| One-blank Sudoku | 893 | 893 → 823 → 798 | 210 / 235 |

A host dispatch is one individual source dispatch or one region
call. Full operand-cell copies are derived from executed transitions.
The baseline uses four such copies for a pair and five for a
triple. Both regions produce one final cell array. Counts exclude
small word copies, heights, and intermediate frame arrays. The
full BFS coverage pass verifies the same final page as Wasmtime
and is separate from timing. The staged counter pass also records
BFS prefix calls and arithmetic candidates.

## Measurements

Each set alternates order after a discarded warmup set and uses
identical inputs across variants. Terrain uses the original and
mirrored four-tick logs. Preparation through the first route and
warm execution are recorded independently. Timing workers were
pinned to separate CPUs with one numerical-library thread.
Instrumentation runs separately.

| Workload | Samples per path | Warm median d1 configuration → multiply → pair |
| --- | ---: | ---: |
| Eight sparse agents | 32 ticks | 526 → 525 → 489 ms |
| Eight crowded agents | 32 ticks | 423 → 426 → 394 ms |
| Full BFS | 2 runs | 19.22 → 15.80 → 14.66 s |
| One-blank Sudoku | 8 runs | 413 → 332 → 286 ms |

Multiply had no eligible sites on the terrain runs. Its terrain
differences reflect timing variation. Sparse p95 in the staged
campaign moved from 584 to 541 ms and crowded from 456 to 434 ms.
Median terrain setup was about 2.7 ms at baseline and 2.9 ms with
the pair. The raw samples preserve setup for every workload.

| Workload | Samples per path | Separate warm median pair → adjacent |
| --- | ---: | ---: |
| Eight sparse agents | 32 ticks | 491 → 482 ms |
| Eight crowded agents | 32 ticks | 389 → 384 ms |
| Full BFS | 2 runs | 15.12 → 14.21 s |
| One-blank Sudoku | 8 runs | 290 → 286 ms |

The adjacent experiment is a separate campaign. Its sparse and
crowded throughput rose from 16.05 to 16.51 and from 20.67 to
21.11 agent-ticks/s. Terrain p95 moved from 553 to 519 ms sparse
and from 421 to 419 ms crowded. These are 10.4× and 8.4× the
50 ms budget. The small terrain and Sudoku median gains should be
read at the observed sample sizes. Two BFS pairs give no useful
tail estimate.

## Verification and replay

The original edit and no-edit histories match Wasmtime byte-for-byte
after each tick, and match the baseline's completed detailed traces.
Full BFS and Sudoku require exact final pages, return values, and
source steps. Focused checks cover signed i32 wrapping, multiply
against the full-candidate and integer references, fuel at every
boundary of both regions, traced fallback, a bypass branch,
custom configuration, live stack overflow, and the original
page-straddling store trap. First-divergence tests remain active.

From `visuaml-client/voxel-replay`:

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/profile_regions.py bfs
python tools/prepared_region_pairs.py sparse --pairs 8
python tools/prepared_region_pairs.py crowded --pairs 8
python tools/prepared_region_pairs.py bfs --pairs 2
python tools/prepared_region_pairs.py sudoku --pairs 8
python tools/prepared_region_pairs.py bfs --pairs 2 --extend
python tools/full_region_coverage.py
python -m pytest -q tools/test_prepared_regions.py tools/test_terrain_verify.py
```

Use the same terrain and Sudoku arguments with `--extend` to
reproduce the adjacent campaign. The independent full AI Escargot
check was not run during this experiment.

[Staged raw timings](staged) · [Adjacent raw timings](adjacent) · [Profiles](profile) · [Full BFS dispatch counts](full-bfs-coverage.json)
