# v58 compiled frame-event gate

The warmed 574e6c5 profile showed that the existing three-instruction
local/const/add region used about 1–2% of elapsed profile time.
The v54 frame-event gate occupied about 28% on terrain and 24%
on full BFS. It runs at each individual source dispatch outside
the pre-existing NumPy const/add regions, making it the higher
coverage circuit for the first compiled experiment.

| Baseline sample | Frame event | Frame state | Routing | Operand transitions | Exclusive driver | Memory body, separate line trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse, four ticks | 864 ms | 567 ms | 426 ms | 409 ms | 130 ms | 24.6 ms |
| Crowded, four ticks | 710 ms | 464 ms | 343 ms | 337 ms | 106 ms | 20.3 ms |
| Full BFS | 5,113 ms | 3,171 ms | 2,501 ms | 2,705 ms | 680 ms | 43.6 ms |

The raw cProfile also records selected ALU, adder, signed comparator,
local-read, and prepared-region times. Inclusive samples overlap.
Memory body timing uses separate line tracing and is not additive.
From the profiled frame-event share, removing the entire selector
would predict roughly 1.38× on terrain and 1.31× on full BFS.
These are diagnostic Amdahl estimates.

## Inspectable graph and boundary

Preparation validates source and route data through the original
executor. It contracts the canonical one-hot opcode against the
default event matrix and its if/end/br_if coefficients, giving an
immutable eight-double row per source PC. The native symbol
`visuaml_frame_event_gate` in `constructed_alu_v58.cpp` accepts only
that row, four live or lexical binary predicates, and a pointer to
one five-double output vector. It computes the original v54 graph:

1. `enter_if = sequential + has_else - ReLU(sequential + has_else - 1)`.
2. Compute the conditional push, pop and branch shifts by multiplying
   the prepared coefficients by `enter_if`, active-frame, and
   branch-taken predicates, respectively.
3. Subtract the sum of the shifts from retain, add one shift to each
   corresponding output component, and preserve the return component.
4. Admit exactly one active output and return its coordinate.

The invocation owns and reuses the output vector. NumPy's frame-state
circuit consumes it synchronously. The kernel does not receive a PC,
operand stack, guest memory, or fuel. Python still routes, dispatches,
retires source steps, and handles traps. The library also contains
existing packed symbols, but this path binds only the frame-event
symbol, and verification blocks the packed executor. The result
records a shared-library SHA-256 when the path is bound.

Opt in with `use_compiled_frame_event=True`. The default NumPy path
remains available. Custom profiles and detailed traces use the
reference selector. The compiled gate is bounded by one source
instruction. The existing two- and three-instruction NumPy regions
still fall back instruction by instruction when fuel is insufficient,
and tracing exposes the original source PCs.

## Isolation and physical work

Eight paired samples of 5,760 calls each yielded an isolated median
of 75.81 µs through the NumPy interface and 0.901 µs through the
compiled interface, including the Python call and ctypes crossing.
The sample cycles all 144 canonical opcode/guard combinations.
A separate allocation probe counted 34,560 explicit NumPy array
constructors over 5,760 calls. These constructors took about 11.9 ms
in that probe. This excludes implicit temporaries and validation
cost, which the isolated reference interface includes. The compiled
interface creates no per-call NumPy arrays.

| Verified counter pass | Source steps | Compiled gate calls | Explicit per-event NumPy arrays removed |
| --- | ---: | ---: | ---: |
| Sparse terrain, four ticks | 6,600 | 5,888 | 35,328 |
| Crowded terrain, four ticks | 5,224 | 4,656 | 27,936 |
| Full BFS | 41,591 | 35,196 | 211,176 |
| One-blank Sudoku | 893 | 728 | 4,368 |

These runs count only the six explicit NumPy array constructions at
each selector boundary. The row table and output buffer are allocated
once per invocation. Existing source-region dispatch counts and
operand materialization are the same in the two paths.

## Verification

An exhaustive truth table compares all 9 canonical opcode tokens
and all 16 binary predicate combinations, including combinations
that normal WASM does not emit. The kernel has no arithmetic word
inputs. Surrounding tests still check i32 wraparound using the
existing add circuit, identical completed traces and fuel cutoffs
at every boundary of both prepared instruction regions, bypass
branches, custom-weight fallback, and the page-straddling store trap.
Both saved terrain histories and exact per-tick pages match
Wasmtime. Full BFS and Sudoku match full pages, returns and steps.
Injected divergence regressions remain in the focused suite.

From `visuaml-client/voxel-replay` with the research environment:

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/bench_frame_event_gate.py
python tools/cold_frame_event_replay.py
python tools/profile_compiled_region.py bfs
python tools/profile_compiled_region.py bfs --compiled
python tools/compiled_frame_event_pairs.py sparse --pairs 10
python tools/compiled_frame_event_pairs.py crowded --pairs 10
python tools/compiled_frame_event_pairs.py bfs --pairs 2
python tools/compiled_frame_event_pairs.py sudoku --pairs 8
python -m pytest -q tools/test_compiled_frame_event.py tools/test_terrain_verify.py
```

## Whole-program results

Every pair uses the same source binary, input log, prepared ALU and
NumPy instruction regions. The measured sets alternate run order
after one warmup pair. Terrain uses two deterministic logs and four
consecutive ticks per paired replay. Reported samples use the final
row binding with dimension checks. The earlier screen is retained.
Each worker was pinned to its own CPU with one numerical-library
thread. Shared-host timing varied across the two campaigns.

| Workload | Samples per path | Source steps per sample | Warm median NumPy → compiled | Observed p95 NumPy → compiled |
| --- | ---: | ---: | ---: | ---: |
| Eight sparse agents | 40 ticks | 1,650 | 524 → 431 ms | 638 → 515 ms |
| Eight crowded agents | 40 ticks | 1,250 or 1,362 | 396 → 296 ms | 483 → 370 ms |
| Full BFS seed 0 | 2 runs | 41,591 | 14.29 → 11.79 s | 14.30 → 12.65 s |
| One-blank Sudoku | 8 runs | 893 | 368 → 264 ms | 434 → 304 ms |

The earlier screen medians were 493 → 345 ms sparse, 388 → 280 ms
crowded, 13.91 → 11.26 s BFS, and 304 → 219 ms Sudoku. Two BFS pairs
and eight Sudoku pairs do not characterize their tails.

In the final campaign, sparse throughput increased from 14.66 to
18.75 agent-ticks/s and crowded from 19.85 to 26.55. The reported
p95 improvement closes about 21% of the sparse and 26% of the
crowded gap to 50 ms. The remaining ratios are 10.3× and 7.4×
the target. Median warm preparation through the first route
increased about 0.17–0.44 ms. These setup figures exclude the
separately measured library build.

The isolated cold build and load of the existing shared C++ module
took 1.07 s. The timing workers saw about 0.92–1.32 s. Using the
final warm medians and including per-tick setup, the isolated cold
cost amortizes after about 12 sparse or 11 crowded ticks. A full
BFS run saves more than one cold build in both observed pairs.
A separate 16-tick sparse replay measured the build inside its
first executor call. The compiled path first pulled ahead on tick
8 and finished in 6.71 s cumulative executor time versus 7.83 s
for NumPy, with all 16 full-page hashes verified. This is one
cold replay. Its observed crossover and the median-based estimate
use different workloads and timing samples.

After compilation, the crowded profile spends about 436 ms in
frame-state updates and 327 ms in routing across four ticks, with
the compiled event binding around 11 ms. Full BFS spends about
3.53 s in frame-state updates and 2.78 s in routing, with the
compiled event binding around 83 ms. Frame-state materialization
is the next common profiled target. Compilation alone did not
change the number of source dispatches or the coverage of the
existing prepared regions.

Full AI Escargot was not run during this experiment.

[Final paired samples](pairs) · [Earlier screen](screen) · [Profiles](profile) · [Isolated gate and allocation samples](isolated.json) · [Cold 16-tick replay](cold-replay.json)
