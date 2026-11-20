# Frozen graph calls and returns

The [TorchScript graph](machine.pt) was frozen at `4164397`, before the
[two-function Rust passage scene](../../guest/passage_call.rs) was written.
Its SHA-256 is `f48971db1e02f610685803d650a0a5017e732ac4f222a84f7323ef53c75f93ec`.
The [replay](replay.json) records the guest and image hashes, graph operators,
function-local instruction indices, return coordinates and every memory event.
The raw Rust compiler output is passed unchanged to both engines.

The loader validates the original module in Wasmtime. It prepares lexical
routes and direct callee entries as immutable image data. The graph fetches
the instruction, saves the caller's return PC, locals and operand stack,
starts the helper with its own locals and stack, and writes the return value
into the caller's argument slot before resuming. A direct call and a return
each consume one unit of source fuel. Memory stays invocation-private and
shared across frames. The [graph IR](machine.ir) includes call handling,
return handling, the existing recurrent step and the indexed store.

The admitted ABI supports up to eight unary i32 functions with one i32
result, direct calls to lower-index functions, and a fixed-size memory.
This bounds frame depth at seven. The current structured control and
arithmetic subset still applies. The loader accepts the padded call index
emitted by rust-lld after Wasmtime validates the original binary. Recursive
and indirect calls require additional admission and runtime semantics.

## Two histories

The first agent's intent and the second agent's position share a three-word
record at byte 16,777,204. The helper reads the pre-edit material, optionally
clears it, and returns the pre-edit value. The caller uses that value to
resolve movement. The next tick reads the committed terrain.

| History | Tick | Notable source steps | Material | Second agent position |
| --- | ---: | --- | ---: | ---: |
| Edit | 1 | Call at 3, helper store at 14, return at 17 | 0 | 0 |
| Edit | 2 | Call at 3, return at 13, caller movement store at 17 | 0 | 1 |
| Omitted | 1 and 2 | Call at 3, return at 13 | 7 | 0 |

All four terminal pages matched Wasmtime byte for byte and matched independent
expected outcomes. A separate three-function fixture exercises nested calls,
an explicit return, caller operand and local restoration, 16 fuel boundaries,
and a straddling load trap inside the deepest helper. It restores stack values
to return 68 and preserves the page on the trap.

## Capacity and active agents

The same frozen graph ran the existing single-function terrain rules with
two and eight sparse agents at each capacity. The 16 MiB Rust build preserves
the decoded instruction image. Each configuration ran 16 ticks, with the
first excluded from timing summaries. Timings include initializing each
invocation's private bytes. Wasmtime, full-page comparison, and hashing ran
outside the timer. The independent movement model and Wasmtime agreed at
every tick. Packed fallback was blocked. [Raw samples](scaling.json)

| Capacity | Agents | Source steps per tick | Median tick | Agent-ticks/s |
| --- | ---: | ---: | ---: | ---: |
| 64 KiB | 2 | 450 | 167.78 ms | 11.72 |
| 16 MiB | 2 | 450 | 170.06 ms | 11.73 |
| 64 KiB | 8 | 1,650 | 605.57 ms | 12.94 |
| 16 MiB | 8 | 1,650 | 632.76 ms | 12.67 |

This sweep holds the active layout and inputs constant across capacities.
The timing is for the multi-agent terrain workload on the call-capable
graph, and the workload uses no helper calls. Increasing activity changed
instruction count and tick latency much more than changing capacity.
The run used 93 focused Python regressions and five browser replay tests.

## Reproduce

From `visuaml-client/voxel-replay`, with the backend dependencies, CPU PyTorch
and Wasmtime installed. Rebuilding guests requires rustc 1.100.0 and its
wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_passage_call.sh
python tools/replay_frozen_calls.py
python tools/scale_frozen_calls.py
```
