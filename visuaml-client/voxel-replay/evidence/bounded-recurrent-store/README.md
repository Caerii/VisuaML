# Bounded stores in the frozen recurrent machine

This experiment starts from `814030b`. The new [frozen graph](machine.pt)
was committed at `b46f1fb` before the new [Rust guest](../../guest/sparse_passage.rs)
was authored. The graph's SHA-256 is
`dd86889441548bd3364abf3cf4de2309dbbe00526152d30d9a838d1374c2e245`.
The [full-copy graph](../frozen-recurrent/machine.pt) remains the reference.

## Memory ownership

Each invocation receives a fresh byte buffer allocated from the caller's
initial bytes. The invocation owns that live buffer until it stops. Each
successor state borrows the same live bytes, so retaining a state object
preserves its scalar, operand and local tensors but leaves its page live.
Publish a detached, immutable byte snapshot with `snapshot_page` when
historical bytes are needed. Separate or concurrent invocations require
separate initial states. The graph reads its validated program image as
shared, read-only data.

The graph computes the unsigned effective address, checks the full
four-byte range, and commits `aten::scatter_.src` only for a running,
fueled, valid store. It captures the previous word before mutation.
The [frozen IR](machine.ir) and warmed TorchScript execution graph contain
the indexed mutation and no functional scatter, clone, or copy of live
memory. Each committed store writes four bytes in place and preserves the
storage pointer. The original path copies the entire old buffer for a
store. A published full snapshot still copies the entire buffer.

The loader accepts caller-specified fixed memories of up to 256 WASM pages
and checks the exact min and max against the binary. The host initializes a
buffer of the admitted size and supplies it to the graph. Existing callers
default to one 64 KiB page. The prototype uses the existing single-function
i32 instruction subset and fixed-size memory.

## Replay

[Ownership and boundary checks](ownership.json) compare the two graphs'
state tensors and memory access records at every step and verify retained
snapshots after later stores. An independent invocation keeps its initial
bytes. The cases cover overlapping and unaligned writes, the last four
bytes, a straddling store, an unsigned negative address, and exact fuel
cuts before and after the rollback write. Wasmtime checks the applicable
terminal values, bytes and traps. The checked NumPy comparison blocks packed
fallback and reproduces its archived trace for the rollback, trap, voxel and
branch witnesses. The [full replay record](replay.json) contains the graph
operators, binary and image hashes, exact traces, both Terrain Edit histories,
full BFS and one-blank Sudoku. Its traces and transfer records equal those
from the original graph.

## Scaling

Single-thread CPU PyTorch 2.9.1, 31 warmed samples per operation and
capacity. Primitive samples each contain 64 four-byte calls. Rust samples
each contain eight whole guest calls, with graph and image preparation
outside the timed loop. The Rust edit uses one store and 12 source steps.
The three linked binaries differ in their fixed memory limits and share the
same decoded instruction image. The edited material starts at the end of
the address space. On the next tick it reads as zero, while omitting the
edit leaves material seven. Wasmtime and both TorchScript graphs agree on
the entire page. [Raw timings, results, and hashes](scaling.json)

| Bytes addressable | Four-byte gather | Full-copy scatter | In-place scatter |
| --- | ---: | ---: | ---: |
| 4 KiB | 1.14 µs | 1.77 µs | 0.89 µs |
| 64 KiB | 1.14 µs | 2.94 µs | 0.89 µs |
| 1 MiB | 1.17 µs | 28.07 µs | 0.91 µs |
| 16 MiB | 1.39 µs | 840.91 µs | 1.14 µs |

| Guest capacity | Full-copy edit | Bounded edit | Bounded observe | Bounded full snapshot | Peak RSS during ticks, full-copy / bounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 KiB | 3.75 ms | 3.63 ms | 2.42 ms | 2.19 µs | 196.2 / 196.2 MiB |
| 1 MiB | 3.71 ms | 3.86 ms | 2.54 ms | 23.81 µs | 199.0 / 198.2 MiB |
| 16 MiB | 4.56 ms | 3.75 ms | 2.43 ms | 651.11 µs | 244.0 / 228.1 MiB |

RSS is the Linux process high-water mark from separate workers and includes
PyTorch and two initialized guest buffers per worker. The snapshot phase
raised the 16 MiB bounded worker peak to 244.1 MiB. Latencies are medians.
At 16 MiB the edit tick used about 18% less time, while the isolated
mutation avoided copying 16 MiB for each four-byte store. Several
milliseconds remain even for the 12-step tick. This is a sparse edit near
the end of a large address space, with a small active world.

The [two-function Rust guest](../../guest/passage_with_agent.rs) opens a
passage, then advances an agent on the next tick. Wasmtime returns `0` and
`1` in order. The frozen loader rejects its two-function module. The
compiler emitted a direct call from `run` to its material helper, giving
a concrete program for a future call-frame conformance test. A dispatch
profile would be needed before selecting a multi-step throughput experiment.

## Reproduce

From `visuaml-client/voxel-replay`, with the backend dependencies, CPU PyTorch
2.9.1 and Wasmtime installed. Rebuilding guests requires rustc 1.100.0 and
its wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_sparse_passage.sh
python tools/check_bounded_recurrent_store.py
python tools/replay_frozen_recurrent.py evidence/bounded-recurrent-store/machine.pt
python tools/bench_bounded_recurrent_store.py
```
