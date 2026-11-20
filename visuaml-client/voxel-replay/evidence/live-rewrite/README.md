# A WASM guest proposes live-input rewrites

The [Rust optimizer](../../guest/rewrite_live.rs) and every original and
candidate binary ran on the existing [frozen graph](../frozen-memory-selection/machine.pt).
Its SHA-256 is
`52234c828b67e30b7d42b34e5b1cb24634f02ddbd78e92a7d45e1c63b4773e9d`.
Packed fallback is blocked in the replay harness.

The input ABI is a bounded postfix expression with up to five tokens.
Each token is six active-low opcode lanes and one immediate word. ABIs
use little-endian i32 words. The host encodes the expression at byte 512.
The Rust guest matches the operation sequence, checks the constants, and
writes a candidate template ID and precondition ID at words 36 and 37
relative to that address. Precondition ID zero claims every i32 input.
A candidate ID of zero declines to rewrite. The host validates the
program and claim, checks the rewrite independently, and assembles the
emitted template into standard WASM. The host runs the rejected binary
only as a negative control.

## Proof and replay

The checker translates each expression independently into 32-bit
bit-vector operations and asks Z3 whether any input makes the results
differ. The proof covers one live i32 input and the given constant.
Addition and subtraction wrap, and the comparison is signed. The
expressions are straight-line and trap-free. A fixed outer harness
stores the expression result at byte 256 and reloads it. The proof is
of the expression result. Fuel and the step at which the harness
commits its store are measured separately for each binary.

| Emitted claim | Bit-vector result | Original → candidate source steps | First store fuel |
| --- | --- | ---: | ---: |
| `(x + (-17)) - (-17) → x` for every i32 | UNSAT for unequal outputs | 10 → 6 | 7 → 3 |
| `(x + 1) >s x → true` for every i32 | SAT at `x = 2147483647` | 10 → 6 | 7 → 3 |

The accepted pair returned `x` and produced identical full memory for
five replay inputs, including both signed extremes. At the rejected
claim's counterexample the original returned zero and wrote
`00 00 00 00`, while the candidate returned one and wrote
`01 00 00 00`. Wasmtime agreed with the frozen graph on all final
returns and memory. Every fuel budget through halt was checked. The
optimizer also declined the cancellation pattern when its constants
differed. The [replay record](replay.json) contains the guest emissions,
SMT-LIB queries, counterexample, binaries and program-image hashes, and
fuel-by-fuel evidence.

## Shipped terrain guest

The actual [terrain binary](../../guest/terrain_tick.wasm) has zero
matching cancellation sites. A separate [Rust probe](rust-cancellation-probe.rs)
compiled with the same optimization level reduces a wrapping
add/sub cancellation to `local.get` followed by `end`. The probe
shows what the compiler does with the expression in isolation.

A live decrement survives in terrain at source PCs 80 and 81, with
`i32.const -1; i32.add` after the load of a claim counter. The guest
emitted `x - 1`, and the checker proved it over every i32 counter.
The host kept the existing load and store and assembled the accepted
two-op suffix as `i32.const 1; i32.sub`. The alternative
[terrain binary](terrain-candidate.wasm) saved **zero** static and executed
instructions. Both histories matched the original and Wasmtime byte for
byte, as well as the saved terrain proof. The edit history took 377 and
370 source steps. The omitted edit took 370 and 365.

## Reproduce

Run from `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch, Wasmtime and `z3-solver==4.15.3.0`. Rebuilding Rust
binaries requires rustc 1.100.0 and the wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_live_rewrite.sh
python tools/replay_live_rewrite.py --write-binaries
python tools/replay_live_rewrite.py
python tools/replay_guest_rewrite.py
```
