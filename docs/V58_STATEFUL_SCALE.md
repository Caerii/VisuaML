# v58 stateful WASM scale replay

The packed executor has an opt-in scale profile. Load original WASM with
`decode_control(raw, scaled=True)` and prepare it with the packed
structured executor. It admits up to **4,096 instructions** and
**50 million fuel steps per untraced invocation**. The 2,000-step trace
budget still applies. The profile retains the existing one-function,
i32-only opcode subset and fixed 64 KiB memory page.

Source-bound routes are built directly as indexed successor, alternate,
and guard coordinates. Route and microop storage grow linearly with
source length. At 4,096 instructions, route targets and guards take
131,104 bytes, including the terminal row. The legacy dense layout
and token hashes match the archives for 35 source programs.
A focused test executes the full static instruction limit.

## Full algorithms

- **Sudoku** runs iterative backtracking in WASM. WASM selects the next
  cell by minimum remaining candidates, tests digits, updates peer
  constraint counts, and restores the board on backtracking. The host
  supplies the givens, peer topology, and initial constraint counts.
  AI Escargot and the Norvig grid are solved in the same WASM program.
  A consistent unsatisfiable instance returns failure and preserves
  its original grid.
- **BFS** runs on 64-vertex directed graphs. WASM builds the traversal
  queue and shortest-path distance table. Seeded graphs include
  disconnected components and unreachable vertices.
- **Matrix multiplication** performs full i32 dot products for sizes
  up to 32×32. A second 823-instruction binary unrolls the 32-term
  inner loop and returns the same page for a matched 32×32 input.

## Measured scale

Each of two serial runs checked **4,116 cases** and
**414,224,557 packed source steps**. Return values and all 65,536 memory
bytes matched independently executed original Wasmtime binaries.
BFS and matrix outputs also matched independent Python references.
Sudoku solutions matched a separate search and an independent direct
row, column, box, and givens check. Z3 verified uniqueness for the four
curated solvable puzzles and unsatisfiability for the fifth. A traced
2,000-step fuel trap on AI Escargot replays the completed prefix.

| Workload | Cases | Source instructions | Packed steps | Packed median | Wasmtime median |
| --- | ---: | ---: | ---: | ---: | ---: |
| sudoku | 517 | 294 | 44,689,354 | 5.849–6.265 ms | 0.116–0.121 ms |
| bfs | 3,078 | 84 | 217,238,410 | 5.557–5.558 ms | 0.120–0.121 ms |
| matrix | 520 | 79 | 151,472,880 | 6.023–6.600 ms | 0.121–0.126 ms |
| matrix_unrolled | 1 | 823 | 823,913 | 62.660–62.694 ms | 0.371–0.864 ms |

The two runs matched input hashes, output page hashes, instruction
counts, results, and solved grids. Each row above gives a range of
the two per-family median API latencies. The timing excludes input
generation, proof checks, module instantiation, and preparation.
The packed call includes the existing private-page copy and writeback.
Wasmtime includes a fuel reset and full page transfer for each case.

A separate **8,212-case stress run** completed **820,889,636 packed
steps**. It reproduced every witness from the smaller sweeps and extended
their seeds. A single countdown invocation completed **49,999,999 steps**,
matched Wasmtime and the modular arithmetic reference, and trapped with
one step less fuel.

## Hard Sudoku examples

| Puzzle | Outcome | Exact packed steps |
| --- | --- | ---: |
| classic | solved | 79,754 |
| escargot | solved | 2,080,993 |
| norvig_grid | solved | 1,819,879 |
| euler_grid | solved | 82,201 |
| classic_unsatisfiable | unsatisfiable | 12,797 |

This suite measures execution scale and correctness. Wasmtime remains
faster for the unfused workloads measured here. The archived data
gives a matched baseline for later loop lowering. A performance
comparison with Perceptra would require a matched workload and
measurement setup.

## Reproduce

Use the source revision in the manifest with the pinned research
environment and a C++17 compiler.

```bash
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
for run in 1 2; do
  python -m scripts.verify_stateful_scale_v58 --bulk 4096 --output "$out/runtime-$run.json"
done
```

The archive contains the four original WASM modules and their WAT
sources, both complete witness sets, the Sudoku proof script, and the
backend log. The backend suite finished with **1,300 passed and
20 skipped**.

[Artifact manifest](../benchmarks/results/v58-stateful-scale-7125937/manifest.json)
