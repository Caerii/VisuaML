# Instruction and state reuse in the frozen graph

The reference is the published budget-64 artifact from `62711d1`.
The [row](row.pt) and [state](state.pt) variants were frozen at `9194748`
before the new [branch-and-helper program](after_freeze.wat) was authored.
The [selected](selected.pt) follow-up was frozen at `37daa07` before the
arithmetic witnesses were added. The program and machine hashes and
operator lists are in the [replay record](replay.json). The guest remains
data. The live graph PC fetches its row on every transition.

- **Row** passes the fetched row from call handling into the ordinary
  transition. The frame-row fetch on helper return is separate.
- **State** also carries the operand tensor when the instruction leaves it
  unchanged, carries locals except on `local.set` or `local.tee`, and
  constructs committed scalar state directly. A local write uses
  `index_copy` on the small locals vector. Stack updates retain the
  canonical zeros above the active height. Reused state tensors are
  read-only. Guest memory retains its invocation-private indexed-write
  contract and published byte snapshots remain independent.
- **Selected** also uses the live opcode to evaluate only its result
  expression. The exact integer tensor expressions for signed comparisons,
  wrapping arithmetic, constants, locals and loads remain in the graph.
  The route, trap and memory gates are unchanged.

The [row IR](row.ir), [state IR](state.ir), and [selected IR](selected.ir)
show the graph-owned branches and indexed store. Checks exclude Python
callbacks and full-page copy primitives from these IRs. All variants use
the same budget, guest binaries and world rules.

## Exact replay

The three variants matched the original single-step graph at every prefix
of the branch, helper return, helper trap, nested call, and rollback
witnesses. Fuel cuts at eight and nine steps retained the exact expected
memory. The new program can skip a helper with a branch, return through
the caller, or trap on a straddling store. The interleaving test mixes
different images, PCs and outcomes through the same artifact, compares
each step with the single-step reference, and checks memory ownership and
snapshot isolation. Selected arithmetic also matched 35 live operand
cases across signed comparisons, add, subtract and multiply, including
signed extremes and wraparound.

Both original Terrain Edit histories matched Wasmtime's full pages and
the archived NumPy source counts and digests. The benchmarks checked
all three variants against Wasmtime for return and full page on every
terrain tick and full BFS completion. Every full BFS completed at 41,591
source steps. Packed fallback was blocked. 93 focused Python tests and
five browser replay tests passed.

## Paired timing

Each run uses a single shared world. Timing excludes initialization,
preparation, profiling and evidence. The first terrain tick is dropped
from each mode and order rotates on identical inputs. BFS samples are full
completions after the artifacts ran terrain. The main run has 21 warm
terrain samples and three full BFS samples per mode. The separate ALU
run has 17 warm terrain samples and two full BFS samples per mode. Its
nearest-rank terrain p95 is the sample maximum. BFS tails are too sparsely
sampled to summarize.

| Workload | Baseline median | Row median | State median | Baseline → state p95 |
| --- | ---: | ---: | ---: | ---: |
| 8 agents, 64 KiB | 229.36 ms | 224.99 ms | 217.50 ms | 245.76 → 240.35 ms |
| 8 agents, 16 MiB | 224.06 ms | 220.33 ms | 213.55 ms | 270.30 → 224.89 ms |
| Full BFS | 5.850 s | 5.773 s | 5.496 s | — |

Row and state raised throughput from 34.6 to 36.2 agent-ticks/s at 64 KiB
and from 34.9 to 37.1 at 16 MiB. Terrain always consumed 1,650 source steps
in 26 host calls. BFS always used 650 host calls.

| Separate ALU run | State median | Selected median | State → selected p95 | State → selected agent-ticks/s |
| --- | ---: | ---: | ---: | ---: |
| 8 agents, 64 KiB | 223.10 ms | 186.87 ms | 243.02 → 200.69 ms | 35.6 → 42.9 |
| 8 agents, 16 MiB | 220.26 ms | 187.87 ms | 239.68 → 214.80 ms | 36.0 → 42.0 |
| Full BFS | 5.470 s | 4.764 s | — | — |

The selected graph reduced median by 15–16% on terrain and 13% on BFS
in its paired comparison. Terrain is still about 3.75× the 50 ms target.
Relative to the earlier reported 226 ms reference, the new 187–188 ms
observation closes roughly 22% of the original 176 ms gap. The tables
keep the paired experiments separate.

Artifact loading and image preparation are reported independently in the
[raw main data](timings.json) and [raw ALU data](selected-timings.json).
[Freeze timings](freeze.json) are recorded separately. Main benchmark
initialization medians were near 0.1 ms at 64 KiB and 1.2–1.4 ms at 16 MiB.
Full snapshot medians were 3 µs and 0.76 ms, respectively. Full tracing
still uses the unchanged single-step path with its separately measured
[audit cost](../frozen-multistep/evidence-costs.json). Cumulative main
benchmark peak RSS was 559 MiB, including all artifacts, the profiler,
verification and page buffers.

## Accounting for graph work

The profiler observes one warmed budget-64 prefix per case and variant.
At 64 KiB, instruction fetches fall from 128 to 64. All `select` events
(including instruction fields and tensor indexing) fall from 1,664 to
1,536. The state variant cuts `where` events from 1,216 to 874 and adds
four local index copies. In the separate ALU comparison, `stack` events
fall from 128 to 64 and positive self-memory attribution falls from about
101 KB to 83 KB. These profiler records attribute tensor allocation work
to operators. They are neither counts of upstream allocator calls nor
measurements of bytes physically copied.

In the baseline 64 KiB prefix, `index_select` was about 2% of CPU
self-time, suggesting only about 4 ms of headroom on a 226 ms tick.
`where`, `select` and `stack` together represented about 22%,
suggesting a larger target of about 49 ms if their attributed cost
were eliminated. These are rough estimates from an instrumented prefix,
with interpreter and profiler overhead also present. The selected
variant removes the candidate stack and its reduction while retaining
one access-event stack per instruction. No native packed executor or
Python callback runs inside any of these graphs.

## Reproduce

From `visuaml-client/voxel-replay` using the backend dependencies,
PyTorch 2.9.1 CPU and Wasmtime:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_frozen_state_reuse.py
python tools/bench_frozen_state_reuse.py
python tools/bench_frozen_state_reuse.py --selected
```
