# v58 prepared fixed-gate ALU selection

The default array-memory invocation binds source-admitted `i32.add`
and `i32.lt_s` sites to a circuit before the first route. Preparation
uses the existing IndexedMicroops decoder, then checks that the
source instruction, one-hot selector, and unit gain agree. The
binding is private to the validated program and canonical
coefficient profile. Custom tokens and coefficients use the existing
six-candidate path. Pass `prepare_alu=False` for the paired baseline.
Both paths retain prepared operand pop.

At runtime Python looks up the source PC. The original v49 ripple
carry and signed comparison functions consume live bit vectors.
Their ReLU gates, carry thresholds, and signed comparison recurrence
are unchanged. The comparison still returns a 32-bit binary word,
and both inputs and the result are checked. At prepared sites the
invocation omits the five other candidates, the NumPy candidate
stack, and the one-hot output mixing, whose unit coefficient was
established at setup. No native packed fallback or region dispatch
is used.

## Diagnostic counts and timing

Profiles ran separately after warming the same inputs. Terrain
diagnostics use four consecutive ticks. BFS profiling and counters
use 4,000 source steps. BFS timing uses the full 41,591-step graph.
Both unprepared signed comparison directions evaluate at every
generic ALU tick, even for add. The multiply ReLU staircase also
runs on every generic tick. Together those were the two largest
candidate costs in all four baseline profiles. Baseline cProfile
attributed roughly 22% of terrain time and 40–41% of the BFS/Sudoku
samples to the full-candidate ALU.

| Counter sample | Dynamic ALU ticks | Prepared add / lt | Arithmetic candidates before → after |
| --- | ---: | ---: | ---: |
| Eight sparse agents | 884 | 676 / 204 | 5,304 → 904 |
| Eight crowded agents | 724 | 564 / 156 | 4,344 → 744 |
| BFS prefix | 1,231 | 402 / 206 | 7,386 → 4,346 |
| One-blank Sudoku | 297 | 123 / 25 | 1,782 → 1,042 |

At each prepared site the removed candidate stack is one
`np.stack` call. The AND and XOR call counts in the profiles cover
the full invocation, including routing. For example, sparse terrain
went from 471,096 to 239,100 AND calls and from 58,344 to 6,996 XOR
calls. The profiles count these calls and candidate invocations.
They do not count every transient NumPy allocation.

An instrumented Amdahl estimate for removing only the discarded
circuit evaluations was approximately 1.22× sparse, 1.23× crowded,
1.19× for the BFS prefix, and 1.21× Sudoku. It uses the baseline's
per-circuit mean times and prepared-site frequencies, deducts each
selected circuit, and leaves selector and array setup costs in the
denominator. It separates add from the add nested inside subtract
and approximates ALU zero-test cost with its per-call mean. This is
an estimate from profiled data. The paired measurements also capture
the omitted one-hot contraction and candidate arrays.

## Alternating paired measurements

Each workload discarded a warmup pair and alternated the run order.
World pairs alternated between the original four-tick log and its
mirror, with five pairs per log. Timing runs were pinned to separate
CPUs with one numerical-library thread. The recorded warm interval
starts at the first route. Setup to that boundary is recorded
separately. Median setup increased by about 0.19–0.38 ms across
workloads.

| Workload | Samples per path | Steps per sample | Warm median before → after | Observed p95 before → after |
| --- | ---: | ---: | ---: | ---: |
| Sparse terrain, eight agents | 40 | 1,650 | 700 → 531 ms | 771 → 584 ms |
| Crowded terrain, eight agents | 40 | 1,250 or 1,362 | 565 → 419 ms | 613 → 459 ms |
| Full BFS seed 0 | 2 | 41,591 | 23.34 → 18.69 s | 23.47 → 18.82 s |
| One-blank Sudoku | 8 | 893 | 540 → 407 ms | 592 → 504 ms |

The crowded p95 improved in this experiment with its own paired
baseline. The world p95 estimates use 40 ticks per path. For BFS and
Sudoku the observed p95 is a sample maximum and gives little
information about the tail. Sparse throughput improved from
11.26 to 14.85 agent-ticks/s and crowded from 14.23 to 19.04.
The optimized terrain tails are still 11.7× and 9.2× the 50 ms budget.
The next high-frequency generic ALU site on BFS and Sudoku is
multiply, accounting for 431 and 124 selected instructions in the
diagnostic samples. The terrain runs already prepare nearly all ALU
ticks, so broadening coverage there would remove little work.

## Exact replay and reproduction

Both terrain histories agree with the generic ALU on complete source
traces, and the verifier checks each full page against Wasmtime.
Full BFS and Sudoku timings require identical reference returns,
steps, and full final pages. Tests compare both arithmetic circuits
on 216 word pairs each against the generic path and integer
references, including signed boundaries, carry and wraparound.
Unsigned comparison is checked independently as a sanity check on
the unchanged signed comparator's shared subcircuit. The complete
eight-entry truth table of one full-adder bit including carry-in was
checked against the actual ReLU adder. This is exhaustive for that
primitive. The 32-bit word tests are sampled.

Other checks cover custom selectors or coefficients, malformed
inputs, exact fuel near the witness store, and the full completed
trace and effects of the page-straddling store trap. Packed fallback
is blocked. Chromium replayed both causal histories and matched the
72 archived world-scaling hashes. The Rust binaries, world rules and
frontend are identical to the starting revision.

From `visuaml-client/voxel-replay` with the research Python environment:

```bash
export PYTHONPATH=.:../backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/profile_alu_candidates.py
python tools/profile_alu_candidates.py --prepared
python tools/prepared_alu_pairs.py sparse --pairs 10
python tools/prepared_alu_pairs.py crowded --pairs 10
python tools/prepared_alu_pairs.py bfs --pairs 2
python tools/prepared_alu_pairs.py sudoku --pairs 8
python -m pytest -q tools/test_prepared_alu.py tools/test_terrain_verify.py
```

[Generic profile](profile-generic.json) · [Prepared profile](profile-prepared.json) · [Sparse](sparse.json) · [Crowded](crowded.json) · [BFS](bfs.json) · [Sudoku](sudoku.json)
