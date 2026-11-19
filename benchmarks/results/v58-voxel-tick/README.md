# Rust voxel guest and fixed-bit ALU replay

`voxel_tick.wasm` is byte-for-byte rustc 1.100.0 output. The guest reads
an agent position and a signed movement delta from one WASM memory page.
It looks up the destination voxel, records its material, and commits the
position only when the material is zero. Positions and deltas are byte
offsets aligned to four, and the host supplies a destination within the
four-cell row. Each field is a little-endian i32.

| Offset relative to the guest argument | Meaning |
| --- | --- |
| 0 | Agent position |
| 4 | Signed movement delta |
| 8 | Last destination material, written by the guest |
| 12 | Reserved |
| 16–31 | Four voxel materials |

The first Rust experiment emitted `i32.shl` to scale an i32 index by four.
Using byte coordinates removed it without rewriting the emitted WASM.
The remaining admission change allows wasm-ld's memory-first ordering
of the exact existing exports in the scaled profile. Encoding preserves
that order and the original binary identity. The guest uses the existing
i32 subset, one function, and one 64 KiB page.

The replay uses four cases with clear movement and collision in both
directions. It compares full memory and return values against Wasmtime,
the packed runtime, and independently constructed expected pages.
Source PCs, stack, locals, control labels, and accesses match packed
traces. The NumPy invocation disallows packed entry. Node 24.22.1
also reproduced all four full-page hashes through its WebAssembly API.

## Reproduce

Run from the checkout root using the research Python environment and
Rust 1.100.0 with the `wasm32-unknown-unknown` target installed.

```bash
export PATH="/workspace/.visuaml-setup/research-venv/bin:$PATH"
export PYTHONPATH=visuaml-client/backend PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
lab=benchmarks/results/v58-voxel-tick
out=$(mktemp -d)
"$lab/build_guest.sh" "$lab/voxel_tick.rs" "$out/voxel_tick.wasm"
cmp "$out/voxel_tick.wasm" "$lab/voxel_tick.wasm"
python "$lab/replay_guest.py"
node "$lab/replay_guest.mjs"
python "$lab/profile_alu.py" --case sudoku --case bfs --pairs 2
python "$lab/profile_alu.py" --case escargot
```

## What this establishes for a viewer

A React Three Fiber viewer can step this guest and render the i32
position, occupied cell, collision marker, and memory writes. A first
viewer could offer single-step, reset, and replay, with the same input
log exported for the NumPy executor and Wasmtime. Frame rendering,
simulation throughput, and multiplayer replication are separate
measurements. For multiplayer, use fixed ticks, ordered inputs, and
periodic state hashes, and measure rollback and synchronization before
scaling the voxel count.

## ALU results

The multiply optimization groups the triangular ReGLU products with one
NumPy reduction. It removes 32 Python lists and per-column reductions
per multiply evaluation, and computes the lower threshold vector once.
The carry recurrence still evaluates the same live ReLU thresholds.
All six fixed-gate candidates still run before one-hot selection.

| Archived workload | Original median | Grouped median | Speedup |
| --- | ---: | ---: | ---: |
| One-blank Sudoku, 893 steps | 781 ms | 733 ms | 1.06× |
| BFS seed 0, 41,591 steps | 34.36 s | 30.35 s | 1.13× |

Each workload used two alternating pairs pinned to CPU 0 with one
numerical-library thread. The timed region starts at executor entry and
includes preparation. Every run checked full memory, return value, and
source steps against packed C++. Profiling ran separately. Sudoku
selected multiply on 124 of 297 ALU ticks. BFS selected multiply on
4,381 of 12,956. Signed comparison now has the largest cumulative time
among ALU candidates, at 0.166 s for Sudoku and 7.876 s for BFS.
These are sample timings on one environment.

AI Escargot matched 2,000 traced source steps and the entire memory page
at a 20,000-step untraced cutoff. That prefix took 16.44 s in NumPy.
A linear extrapolation to the archived 2,080,993 steps gives about
29 minutes. The full NumPy solve remains unverified.

[Timing pairs and dynamic opcode counts](alu-timings.json) · [CPU profile](alu-profile.txt) · [Escargot prefix](escargot-prefix.json)
