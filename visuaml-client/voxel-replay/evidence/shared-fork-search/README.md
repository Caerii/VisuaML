# Shared fork search

The passage search completes the suspended tick once and traverses the binary input tree in depth-first order. Each internal node retains a tick-boundary page, the committed output history, and the retired count. The next invocation starts with that page and one edge action. A halted tick has no live registers to carry into the next invocation under this guest ABI. Sharing is keyed by traversal of identical input prefixes. Equal world states do not merge branches.

All 32 leaf histories match the flat search, including every tick's full-page hash, retirement count, return and world fields. The original minimum edit and tie-break select Walk at tick 1. The debugger still opens the store at PC 28.

| Measurement | Flat | Tree |
| --- | ---: | ---: |
| Executed future guest ticks | 160 | 62 |
| Executions of the suspended suffix | 32 | 1 |
| Peak retained checkpoints, including the root file payload | 1 | 6 |
| Live page-buffer upper bound | 256 KiB | 768 KiB |
| Measured peak Python allocations | 2,161,985 bytes | 1,967,837 bytes |
| Median paired search traversal, three samples each | 2,220 ms | 767 ms |

The page-buffer bound covers the active ancestry and transient tick copies. Python allocation peaks are from separate tracemalloc runs and exclude Torch native allocation accounting. The timing covers traversal, including checkpoint overhead and the suspended suffix, and excludes the follow-up detailed verification. Both modes use the same frozen graph and scheduling rules.

## Compiler regression

A real eight-agent terrain run supplies a checkpoint immediately before a verified pair of local reads. The same VM checkpoint importer restores independent pages. The accepted planner output runs on the existing region graph and agrees at exit. In the test harness, the deliberately rejected plan previews the second local at the first source boundary, then produces its declared final pair. Z3 rejects the preview and accepts the exit. The ordered comparison used by the world search catches the first bad boundary even though the full states reconverge.

At original PC 160 the reference stack is `[256]` and the rejected preview is `[0]`. After PC 161 both stacks are `[256, 0]`. The rejected preview is evaluated only in the regression harness.

## Reproduce

From `visuaml-client/voxel-replay` with the research environment active:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/search_voxel_forks.py --benchmark
python tools/search_voxel_forks.py
python tools/replay_compiler_fork.py
```

Use `--checkpoint` and `--session` with downloaded compiler artifacts to replay them in a fresh process. With the existing local Vite and fork-search servers running, open [Suspended Passage](http://localhost:4180/voxel-call-checkpoint.html). **Find a way through** runs the shared search. **Open first compiler difference** and **Compare region exit** show the saved counterexample.

[GIF](shared-forks.gif) · [Timing samples](benchmark.json) · [Compiler checkpoint](compiler-checkpoint.json) · [Compiler session](compiler-session.json)
