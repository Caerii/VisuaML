# One WASM guest rewrites another

The [Rust optimizer](../../guest/fold_constants.rs) examines three instruction
records in linear memory. It accepts two constants followed by an addition,
computes their wrapping i32 sum, and publishes one constant. The input is
three little-endian pairs of signed i32 tag and immediate words at byte
512. Tag zero denotes a constant and a negative tag denotes add, whose
immediate is ignored. At byte 536 the optimizer writes its output as
`(length, tag, immediate)`, with length zero for a nonmatch.

The host reads the emitted words, independently verifies the exact
constant-folding rule modulo 2³², and uses Wasmtime's WAT assembler to
encode the sequence into standard WASM. The assembler inserts the
sequence into a fixed harness that stores and reloads its result at byte
256. The guest performs the match and computes the replacement, and the
host encodes the already selected instructions. The rule check is complete
for this bounded three-instruction constant-only language.

The [optimizer](optimizer.wasm), [original](original.wasm),
[rewrite](rewritten.wasm), and [negative control](rejected-saturating.wasm)
all ran through the [same frozen budget-64 graph](../frozen-memory-selection/machine.pt).
Wasmtime independently confirmed returns and full pages.

| Program for `INT_MAX + 1` | Return | Bytes at 256 | First fuel budget that commits the store | Fuel to halt |
| --- | ---: | --- | ---: | ---: |
| Original | −2,147,483,648 | `00 00 00 80` | 5 | 8 |
| Accepted rewrite | −2,147,483,648 | `00 00 00 80` | 3 | 6 |
| Rejected saturating candidate | 2,147,483,647 | `ff ff ff 7f` | 3 | 6 |

The accepted pair returns the same signed value and produces identical
complete pages. Fuel is measured on each program's own instruction stream.
The negative control shows a concrete counterexample on the same input
page and argument. The optimizer itself used 36 source steps on the primary
case. That one-time work is separate from the two-step saving per execution
of the rewritten program.

The [replay](replay.json) records graph, image and binary hashes, the input
sequence, emitted sequence, every fuel boundary, additional operand cases
including negative overflow, and a rejected instruction pattern.

Run from `visuaml-client/voxel-replay` with CPU PyTorch 2.9.1, Wasmtime and
the backend dependencies. Rebuilding the optimizer requires rustc 1.100.0
and its wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_fold_constants.sh
python tools/replay_guest_rewrite.py --write-binaries
python tools/replay_guest_rewrite.py
```
