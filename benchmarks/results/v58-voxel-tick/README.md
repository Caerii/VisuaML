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
traces. The NumPy invocation disallows packed entry.

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
