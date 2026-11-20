# Compiled ordinary non-memory CPU runs

This experiment starts from the prepared-effects-plus-lazy-fields
runner at `ed96d44`. The guest WASM and verified region plans are
identical between each pair of runs. Terrain uses the existing
64 KiB and 16 MiB binaries. The baseline, audit graph and new
[frozen graph](machine.pt) remain available independently.

## Lowering and replay contract

The graph passes the current PC, live state scalars, two bounded
operand vectors, the prepared program image and the remaining
invocation budget to one registered CPU operator. The
[C++ implementation](../../../backend/visuaml/compiler_lab/native/ordinary_cpu_v58.cpp)
fetches instruction rows by the live PC and runs consecutive
eligible non-memory source steps. It uses the same prepared pop and
push counts, conditional routes, i32 result expressions, lexical
restore heights and canonical zeroed stack suffix as the reference.
It decrements fuel once per committed source step. It returns the
new PC, stack height, bounded vectors, fuel, status and exact steps.

An additional prepared flag marks eligible rows. The run exits
before memory access, a call or nested return, a row marked for a
verified region, or an invocation/fuel boundary. The frozen
graph retains these transitions including byte addressing,
bounds and traps, and never passes the guest page to the CPU
operator. Ordinary control and ALU operate on local C++ scalars.
The CPU kernel has no WASM parser and uses the same source and
compiled library for every guest. The [lowered IR](machine.ir)
shows `visuaml_ordinary::run` within the budget loop, with no Python
callback. Packed fallback is separately blocked.

Copying the 24 stack and five local words into the scratch arrays
moves 232 bytes per compiled invocation. Only a vector that changed
gets a new output tensor, with at most 232 bytes copied back. These
are bounded vectors and do not grow with guest page capacity. The
existing reference stack and locals tensors remain unchanged and
published page snapshots keep their existing semantics.

All 1,650 source boundaries of an eight-agent tick match the
single-step graph. This includes boundaries around verified regions.
Both Terrain Edit histories match Wasmtime's returns and complete
memory and the archived NumPy proofs. Each timed BFS run also
matches Wasmtime and preserves its fuel and host-call count.
[Focused witnesses](witnesses.json) replay nested returns, a helper
trap, rollback fuel cuts, and both branches plus signed-minimum
arithmetic in a [guest authored after freeze](after-freeze.wat).
The latter was authored after commit `d99dbed`. 93 focused Python
regressions and five browser replay tests passed.

## Work removed

In separate full segmented replay, terrain had 860 ordinary steps,
including 640 non-memory steps. BFS had 24,001 ordinary steps,
including 21,667 non-memory steps. Smaller profiling budgets alter
region fit, and the compiled path can also decline a non-memory
row with a pending region or call. Saving 10 ms at 640 possible
steps requires about 16 µs per step, plausible in view of the
repeated scalar and field operators in the lazy path.

| Actual first 64-source-step profile | Terrain lazy → CPU | BFS lazy → CPU |
| --- | ---: | ---: |
| Scalar extraction `item` | 476 → 254 | 532 → 253 |
| Tensor `select` | 570 → 257 | 621 → 260 |
| Gather | 67 → 43 | 54 → 25 |
| Where | 16 → 7 | 14 → 2 |
| Functional scatter | 29 → 13 | 37 → 16 |
| In-place memory scatter | 7 → 7 | 2 → 2 |
| Registered CPU runs | 0 → 12 | 0 → 13 |

The budget-64 profile observes Torch runtime operators after
lowering, including scalar extractions still visible at the
boundary. The profiler does not count ordinary integer loads or
stores inside the native kernel as tensor operators. No
profiling or hashing runs inside the warm benchmark timer.

## Paired measurements

Each terrain capacity runs two distinct deterministic sparse input
logs, alternating starting direction. There are 64 warm ticks per
log per path. The table pools the 128 samples per path and uses
nearest-rank p95. The initial pages and inputs are identical within
each pair and timing order alternates. Timing includes the host
budget loop and simulation. Initialization, publication, oracle,
hashing, and tracing are measured separately or excluded. Source
steps and host invocations remain 1,650 and 26 in every terrain
tick, and 41,591 and 650 in every full BFS run.

| Workload | Lazy median / p95 | CPU median / p95 | Deadline misses, lazy → CPU | CPU agent-ticks/s |
| --- | ---: | ---: | ---: | ---: |
| Eight agents, 64 KiB | 56.52 / 61.90 ms | **37.44 / 40.45 ms** | 128 → 0 | 211.8 |
| Eight agents, 16 MiB | 56.14 / 61.98 ms | **37.49 / 41.91 ms** | 128 → 0 | 210.2 |
| Full BFS, six runs each | 1.415 s median | **0.710 s median** | — | — |

The [cold C++ build](build.json) took 8.10 s, registration 0.46 ms,
and graph freeze 67.8 ms. The shared object is rebuilt locally
using the checked-in source and CPU PyTorch headers. Its source,
binary hash, compiler and flags are inspectable. In the paired
process, preparing both ordinary image layouts took 0.28 ms
for 64 KiB terrain, 0.23 ms for 16 MiB terrain and 0.14 ms for BFS,
separate from the existing portfolio preparation. The eligible
lane adds eight bytes per image row. [Raw paired samples](paired.json)

## Continuous 20 Hz schedule with replay capture

Each capacity runs 240 ticks at scheduled 50 ms intervals on one
persistent private page. The harness copies the full page before
and after each tick, matching the existing browser demo's page-copy
cadence. It retains every twelfth checkpoint for full-page oracle
comparison and every tick's display ABI region for Wasmtime
verification. Oracle work runs after the timed schedule. The
latency is measured from scheduled start through the after-copy.
This measures the page publication cadence. Browser rendering and
the browser's byte-diff loop are outside this backend measurement.

| Capacity | Scheduled median / p95 | Misses | Simulation + input p95 | Two page copies p95 |
| --- | ---: | ---: | ---: | ---: |
| 64 KiB | **37.89 / 42.66 ms** | 6 / 240 | 40.73 ms | 0.085 ms |
| 16 MiB | 40.50 / 80.02 ms | 65 / 240 | 40.28 ms | 42.99 ms |

The 64 KiB replay meets the p95 target with six outlier deadlines.
The 16 MiB simulation still has headroom, and the full-page
publication cadence is the observed constraint there. The
individual component p95s use independent sample ranks.
[Raw scheduled replay and checkpoint hashes](continuous.json)

## Reproduce

From `visuaml-client/voxel-replay` with CPU PyTorch 2.9.1, Wasmtime
and the backend dependencies. A C++17 compiler is used to build
the native operator. The replay registers it automatically.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_compiled_ordinary.py
python tools/replay_compiled_ordinary.py --witnesses
python tools/replay_compiled_ordinary.py --continuous
```

Use `python tools/build_compiled_ordinary.py` to rebuild and freeze the
graph. The original lazy graph can still be frozen without registering
the extension.
