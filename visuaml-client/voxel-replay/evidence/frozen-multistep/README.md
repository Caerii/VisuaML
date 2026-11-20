# Bounded multi-step frozen execution

This experiment starts from `a7e3f15`. The [single-step graph](../frozen-calls/machine.pt)
is the audit reference. The [multi-step graph](machine.pt) froze at
`84f685d` before the new [call-and-trap witness](after_freeze.wat) was authored.

The graph takes the validated image, live state and a transition budget from
0 to 64. The frozen [IR](machine.ir) contains a `prim::Loop`, the ordinary
call and return transitions, and the same indexed store. It carries the live
state through the loop. Python sees one result per invocation. The graph
exits when the budget has been used or the reference transition reports
halt, trap or fuel exhaustion. A fuel rejection consumes no source step.
If a budget ends exactly after the last available fuel is consumed, that
prefix still has the reference's running status. The next attempted
transition reports exhaustion without advancing the PC or mutating memory.
The result includes the last attempted access event. Full source traces
use the single-step audit path.

## Exact replay

The [replay bundle](replay.json) includes graph and guest hashes, operators,
full page digests and source traces. Every prefix of eight small witnesses
was compared against the single-step graph, including fuel, PC, frame
storage, stack, locals and the complete memory buffer. Checkpoints from
budgets 4, 16 and 64 were also compared. Both original Terrain Edit histories
matched the saved checked NumPy digests and source counts, and their
completed pages matched Wasmtime byte for byte.

In the Rust passage witness, a **single budget-64 invocation** enters the
helper at source step 3, clears the block at step 14, resumes the caller at
step 17 and halts at source step 21. The post-freeze guest stores the word
77, then either returns through its caller or traps on a straddling load.
The earlier committed store survives the trap. Wasmtime agrees on the
complete page. The rollback witness stops at fuel 8 with the original word
and at fuel 9 with the intermediate write still present. Nested return and
a trap inside the deepest helper are also checked. Packed fallback is blocked
in the replay and benchmark harnesses.

## Paired results

Budgets ran in rotating order on the same tick inputs. The table reports
milliseconds. Terrain has 16 warm samples per mode after excluding its
first tick. Its nearest-rank p95 is the maximum of those 16 samples. BFS
has two full completions per mode after a separate diagnostic, so its tail
estimate is omitted here. Every timed output page was checked against
Wasmtime. Source counts were 450 and 1,650 per terrain tick and 41,591 for
each full BFS run.

| Workload | Single step | Budget 4 | Budget 16 | Budget 64 | p95 single → 64 | Host calls single → 64 | Agent-ticks/s at 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 agents, 64 KiB | 163.35 | 89.94 | 67.92 | 61.87 | 217.73 → 70.58 | 450 → 8 | 31.94 |
| 8 agents, 64 KiB | 616.23 | 336.29 | 254.73 | 225.61 | 840.12 → 246.17 | 1,650 → 26 | 34.81 |
| 2 agents, 16 MiB | 168.84 | 92.38 | 68.78 | 62.94 | 356.83 → 72.62 | 450 → 8 | 31.32 |
| 8 agents, 16 MiB | 618.03 | 331.74 | 249.87 | 225.58 | 794.95 → 252.16 | 1,650 → 26 | 34.90 |
| Full BFS | 15897.96 | 8408.73 | 6412.57 | 5875.40 | — | 41,591 → 650 | — |

The eight-agent median is now about 4.5× the 50 ms target, down from about
12.3×. This is one shared world at a time. The timer excludes input and
state preparation, verification and evidence collection. Graph freeze took
28.6 ms, artifact load took 10.1 ms, validated image preparation took
1.2–4.4 ms, and median live-state initialization took about 0.09 ms at 64 KiB
and 1.1–1.3 ms at 16 MiB. Peak process RSS was 462 MiB across all workloads.
It includes both graphs, profiler, verification and page buffers, and is
a cumulative high-water mark. [Raw paired results](timings.json)

## Where the time goes

The warmed budget-zero call proxy took 115–126 µs, around one-third of
baseline time per source step. Removing that proxy cost alone suggests
roughly a 1.5× speedup. The measured gain is around 2.7×. The proxy includes
argument and result handling plus a zero access tensor, and omits the
host status check and execution-plan effects. It cannot attribute the
entire measured gain to the Python crossing.

In 64-transition diagnostic prefixes, TorchScript `forward` self-time
was 14–24% of profiler CPU time. This residual includes interpreter work
and profiler instrumentation. `where`, `stack` and `select` collectively
accounted for about 16–19%. Fetch ran twice per transition and
`index_select` accounted for 2–3%. Gather accounted for about 2%. The
actual indexed store accounted for 0.04–0.13% in these prefixes.
The next focused experiment is to reuse the already fetched instruction
row and reduce repeated tensor assembly around state updates. These
operator profiles are diagnostic samples, separate from wall timing.

An additional paired 64-step audit measured roughly 1 ms extra to collect
the Python source trace in most cases. A separate full 16 MiB snapshot
cost 1.08–4.91 ms in this sweep. These are host evidence costs, and neither
is included in the warm execution table. [Raw evidence costs](evidence-costs.json)

## Reproduce

From `visuaml-client/voxel-replay` with the backend dependencies, PyTorch
2.9.1 CPU and Wasmtime installed:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_frozen_multistep.py
python tools/bench_frozen_multistep.py
python tools/bench_frozen_multistep.py --evidence-costs
```
