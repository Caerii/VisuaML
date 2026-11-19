# v58 structured NumPy memory replay

The opt-in `array_memory` argument on `execute_microop_routed_control`
runs a bounded structured WASM program against one caller-supplied 64 KiB
page. The same path runs the original rollback fixture, the archived
Sudoku binary, and archived BFS. Packed C++ and Wasmtime execute separately
as references. The replay and regression reject entry into VisuaML's
native packed executor.

| State or operation | Execution |
| --- | --- |
| Program counter | Python holds the integer coordinate. Source validation builds successor/alternate coordinates. Fixed v52 weights contract the guard and select the path in NumPy. |
| Branch predicate | The v49 fixed ReLU bit circuit tests the actual operand for zero. Python uses the resulting route for the next tick and supplies the lexical frame metadata. |
| Operand stack and locals | v55 operand storage and v57 local gates update NumPy bit arrays. The v53 frame circuit emits the restored stack height. Python applies the truncation through the operand adapter. |
| Arithmetic | The v56 fixed bit ALU computes candidates and selects the output with fixed gates. |
| Effective address and memory | NumPy packs address bits in little-endian order, interprets them as u32, and adds the source offset in u64. Python enforces the bounds guard and dispatches load/store. NumPy reads or assigns a four-byte slice of a zero-copy view of the caller's bytearray. |
| Host orchestration | Python drives fuel and ticks, dispatches by source opcode, maintains lexical metadata, and gathers evidence. The Sudoku harness supplies givens, peer topology, and initial counts. |

The memory operations use ordinary NumPy byte operations. The weights are
constructed, and this is a CPU array executor. The trace cap is 2,000
source steps. The long Sudoku rollback excerpt comes from the store
observer, and the packed executor corroborates its exact bytes at
untraced fuel cutoffs immediately before and after each recorded store.

The rollback fixture covers zero and nonzero load predicates, unaligned
little-endian writes, a loop and restoration, an out-of-bounds store, a
widened out-of-bounds load, and a fuel cutoff after a committed store.
Packed traces and page state match through each cutoff. Wasmtime checks
success and memory traps. Its fuel accounting is engine-specific.

## Replay

Run from the repository root with the v58 research environment active.
The Rust checker is standalone and has no third-party crate dependencies.

```bash
export PATH="/workspace/.visuaml-setup/research-venv/bin:$PATH"
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
witness=benchmarks/results/v58-array-memory
out=$(mktemp -d)
rustc --edition=2021 -D warnings -O "$witness/check_sudoku.rs" -o "$out/check_sudoku"
python "$witness/replay.py" --output "$out/replay" --profile \
  --case nearly_solved --case classic_unsatisfiable --case classic --case bfs \
  --rust-checker "$out/check_sudoku"
```

The Rust checker reads the first 81 little-endian i32 cells, checks all
givens and Sudoku constraints on successful runs, and independently
searches for a solution when the WASM reports unsatisfiability. Run the
Python replay without case flags for the five small witness cases alone.

## Verified results

| Case | Return | Source steps | Packed trace rows matched |
| --- | ---: | ---: | ---: |
| Sudoku, one blank | 1 | 893 | 893 |
| Sudoku, classic | 1 | 79,754 | 2,000 |
| Sudoku, archived unsatisfiable | 0 | 12,797 | 2,000 |
| BFS, seed 0 | 32 | 41,591 | 2,000 |

All four cases matched Wasmtime and packed C++ on the return value and
every byte of the final memory page. The packed source step counts matched.
The Rust checker passed all three Sudoku pages and rejected a deliberate
corruption of the solved one-blank page. On the unsatisfiable board,
zero-based source tick 6,001 at PC 218 placed 4 at byte address 88. Tick
7,891 at PC 194 cleared that cell. The returned board preserved the givens.

[evidence.json](evidence.json) records the producing commit, source hashes,
fixtures, matched small traces, and the observed placement and reversal.
The Sudoku binary SHA-256 is
`8df78d3cf72907abd4004e23ae9075682396ece4be3547631451f54effd311cf`.
AI Escargot remains a longer array-execution scale experiment.

The one-blank profile measured 4.5 ms through entry to the first route
and 740 ms from there through completion. The small witness measured
1.0 ms and 18.8 ms on the same boundaries. These snapshots include
replay harness overhead at the first boundary. The frame circuit
initially led the one-blank profile. Replacing generic membership
checks with direct binary comparisons and removing the path search
for its small exact tensor contraction reduced frame overhead.
Five alternating CPU-affined pairs after a warmup gave medians of
904 ms with the original frame function and 802 ms with the updated
function, a 1.13× end-to-end speedup within the NumPy executor.
The experiment swapped only `step_frames` and compared return value,
source steps, and the full page for every sample. The updated profile
puts the fixed bit ALU first at 0.374 s cumulative versus 0.186 s for
the frame circuit.

[Raw timing pairs](frame-timing.json) · [Profile](profile.txt)
