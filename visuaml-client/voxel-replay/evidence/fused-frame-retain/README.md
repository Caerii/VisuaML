# v58 fused frame-event retain

The d076cae reference uses the compiled v54 selector and a checked
NumPy v53 transition after each individual source dispatch. The
existing prepared instruction regions, guest binaries, world
rules, and frontend remain unchanged.

## Baseline profile

Hottest common frame action was retain. It used the original
unit-retain shortcut, which still validates the current tensors,
encodes and validates frame tokens and weights, and copies depth,
kinds, bases, and restored operand height. The driver then decodes
depth and slices lexical metadata. Full-warmed cProfile samples
attribute the v53 transition about 26% of sparse terrain time,
25% of crowded terrain time, 20% of full BFS time, and 19% of
one-blank Sudoku time. The compiled selector occupies below
1% in each of these profiles.

| Counter sample | Source steps | Retain | Push | Pop | Branch | Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse terrain, four ticks | 6,600 | 5,216 | 252 | 32 | 388 | 0 |
| Crowded terrain, four ticks | 5,224 | 4,128 | 204 | 16 | 308 | 0 |
| Full BFS | 41,591 | 32,637 | 256 | 190 | 2,113 | 0 |
| One-blank Sudoku | 893 | 688 | 11 | 3 | 25 | 1 |

The counts exclude source steps inside existing prepared regions,
whose frame actions were already contracted. The line-level probe
separates frame token encoding, admission, depth decoding,
candidate transition work, and the retain copies. It instruments
a full terrain sample and Sudoku. Its BFS sample is a 4,000-step
prefix, while the cProfile comparison and timings use the full
BFS invocation. Line-probe costs are inclusive of called
NumPy operations and should not be added to cProfile measurements.

In the sparse line probe, retain consumed 473 ms and all changing
events 175 ms within v53. State and token admissions together
accounted for 458 ms across all events. The explicit retain copies
and result construction accounted for 17 ms. In the BFS prefix,
retain and changing events accounted for 278 and 67 ms, and the
explicit retain copies about 9 ms. This explains why removing the
checked retain call matters more than removing its copy alone.

## Compiled boundary and ownership

The C++ symbol `visuaml_frame_event_retain` first evaluates the
existing exact v54 event equations. A canonical one-hot retain
with a unit action gain makes the v53 contraction the identity:
every other candidate has coefficient zero. The fused copy arm
materializes that identity in the second private slab during
the same native call. The resident arm leaves the first slab
in place across consecutive source instructions. Both use the
same live and lexical predicates. None of the original WASM
frame, stack, memory, or fuel operations is interpreted by the
kernel.

On a state-changing event the kernel returns the event token.
Python still constructs checked v53 inputs and calls the original
NumPy transition, including loop-target selection, branch
restoration, and frame-bound checks. The checked output is copied
back into the owned slab. The identity can retain the slab
because construction validates the initial frame and the only
other producer is that checked transition. On retain, the
reference return height is the live operand height, which the
driver only observes for branch. Neither arm allocates it on
retain. Python owns lexical metadata, bounds, trap handling,
guest memory and source fuel. The original compiled selector
remains the default when explicitly enabled. Select the
experiments with `compiled_frame_retain="copy"` or
`compiled_frame_retain="resident"` and
`use_compiled_frame_event=True`. Trace and custom-weight calls use
the existing checked path.

Each invocation has disjoint internal storage. The driver reads
reshaped views synchronously and never publishes them as trace or
result objects. The copy arm uses two slabs and swaps views, so
private prior views may be overwritten on later instructions.
The resident arm only replaces bytes when importing a checked
state. Tests cover independent invocations, nonaliased initial
and checked arrays, and immutable previously collected traces.

## Results

| Workload | Samples per arm | Warm median selector → fused copy → resident |
| --- | ---: | ---: |
| Eight sparse agents | 32 ticks | 339 → 271 → 273 ms |
| Eight crowded agents | 32 ticks | 276 → 220 → 219 ms |
| Full BFS, 41,591 steps | 2 runs | 10.84 → 9.57 → 9.34 s |
| One-blank Sudoku | 8 runs | 214 → 187 → 173 ms |

Sets alternate order after warmup, use the same original and mirrored
input logs, and time setup separately through the first route.
Workers were pinned to separate CPUs. In the reported warm campaign,
median terrain setup was roughly 3.2–3.3 ms. Sparse throughput was
23.50 → 28.46 → 28.83 agent-ticks/s and crowded throughput was
29.09 → 36.66 → 36.72. Observed terrain p95 was 363 → 356 → 314 ms
sparse and 306 → 245 → 237 ms crowded, over 32 ticks per arm.
The simpler fused-copy path remains 7.1× and 4.9× the 50 ms
tail budget. The resident arm did not provide a repeatable
terrain median gain over copy. Its observed tail improvement is
worth recording, and the small sample cannot establish it as a
reliable effect. Two full BFS samples per arm do not characterize
the tail.

The number of native calls is unchanged across all three arms.
For the full BFS counter pass, all paths make 35,196 native event
calls. The fused paths remove 32,637 checked NumPy retain calls,
leaving 2,559 checked state changes. They skip four token arrays,
three frame-copy arrays, and the unused height copy per retain,
or 261,096 explicit arrays across BFS. The fused copy arm still
copies 2,792 bytes per retain, about 91.1 MB total on BFS. Resident
removes those copies. Both fused arms import about 7.15 MB from
checked boundaries, including initial state. Counts exclude
short-lived validation temporaries.

Cold compilation and load of the shared module was about
1.07–1.17 s across workers. In a separate 16-tick sparse cold
comparison, the selector, copy, and resident arms took 6.92,
5.81 and 5.65 s cumulative executor time. Both fused arms were
ahead at tick 1 in this single run. All arms pay for the same
module build. First-call setup varied from 0.97 to 1.07 s, so
the first-tick crossover is an observation rather than a
guarantee. Full pages matched Wasmtime after every tick.

The evidence supports stopping at the fused transition for this
milestone. The resident implementation remains an opt-in research
variant. The terrain medians and materialization profile do not
support widening residency just to remove these copies. After
fusion the crowded profile puts routing (313 ms) and operand
transitions (285 ms) ahead of the remaining frame updates (109 ms).
The fused full-BFS profile shows the same ordering by group, with
operand transitions at 2.88 s, routing at 2.65 s, and remaining
frame updates at 0.65 s.

## Replay and reproduce

Both terrain histories, full BFS, and Sudoku compare exact pages,
returns, and source steps. Tests cover all 13 live depths on retain,
all binary event guards for the retain source, checked push/pop,
return, a loop backedge, fuel before and after each source boundary
in the witness, traced fallback, a bypass branch, and the
page-straddling memory trap. The existing injected-divergence tests
are included. Packed fallback is blocked in these runs. Full
AI Escargot was outside the paired and correctness runs.

From `visuaml-client/voxel-replay` with the research environment:

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/profile_compiled_region.py bfs --compiled
python tools/profile_frame_materialization.py bfs
python tools/compiled_frame_event_pairs.py sparse --pairs 8 --fused
python tools/compiled_frame_event_pairs.py crowded --pairs 8 --fused
python tools/compiled_frame_event_pairs.py bfs --pairs 2 --fused
python tools/compiled_frame_event_pairs.py sudoku --pairs 8 --fused
python tools/cold_frame_event_replay.py --fused
python -m pytest -q tools/test_fused_frame_retain.py tools/test_terrain_verify.py
```

[Paired samples](pairs) · [Compiled selector profiles](baseline-profile) · [Materialization line profiles](materialization) · [Fused copy profiles](copy-profile) · [Cold replay](cold-replay.json)
