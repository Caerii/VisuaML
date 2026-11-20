# Guest-composed guarded-load block

Terrain and BFS WASM are unchanged. This experiment starts from the
guest-selected portfolio at `4e685a2`. The
[training site counts](../rule-portfolio/training.json) identify three
consecutive guarded-load pairs in terrain. The hottest single-entry
slice begins at image PC 9 and executed 33 times in the training tick.
BFS has no adjacent portfolio regions of this shape. Only the hottest
terrain site is enabled in the end-to-end performance comparison.

The [Rust planner](../../guest/compose_guarded_loads.rs) receives four
decoded instruction descriptors as active-low opcode lanes and source
PCs, then the two certified-region flags, a challenge word, and two
local-slot/offset descriptors. The little-endian i32 input starts at
byte 512. At byte 572 it emits a template ID, all four original PCs,
the read boundaries (2 and 4), an interior-observation policy, and the
two operand descriptors. Its fixed policy proposes composition for
two consecutive certified guarded loads in source order. The host
selects the hottest eligible site from the training counts, verifies
the guest's schedule rather than trusting its flags, verifies sequential
normal edges and excludes alternate entries into the three interior
rows. The original instructions and parent metadata remain in the
image for every fallback. No guest bytes change.

The [frozen lowering](machine.pt) checks fuel, invocation budget,
stack room, and both addresses before taking the four-step path. It
batches the eight byte lanes into one gather and materializes the
two final words in one stack scatter. Locals, frames and memory are
carried through. If the second load cannot execute, the first
certified region can still commit and control resumes at the second
region's original entry. The single-step runner remains the audit
path. An interior budget observes the original source boundary by
using ordinary transitions or a complete parent region in that
prefix.

## Proof and exact boundaries

The [rule record](rules.json) records Z3 results and includes runnable
SMT-LIB queries. Its digest binds the parent rules, composer,
checker, admission, graph and lowering. It proves the first
region's exit height and remaining fuel meet the second region's
entry, and that the batched result exactly reproduces the two
little-endian words, canonical stack, successor PC, fuel and last
access event. The fast-path premises include two legal addresses,
a canonical input stack with three free slots, and at least four
fuel and invocation slots. Locals, memory and frame tensors are
untouched. Every actual prefix at the selected real site and
throughout one audited tick is checked against the single-step
graph. Interior-prefix proof records describe delegation, while
replay verifies the implementation actually takes that path.

The [new transfer guest](after-freeze.wat) was written after freezing
the planner, graph, and certificate at `6dbdd48`. It admits the
same rule and loads 7 then 11. The rejected plan exposes 11 instead
of 7 after the first load. Z3 finds the mismatch at prefix two even
though both plans have final stack `[7, 11]`. The host rejects the
emitted negative-control policy. The fixture checks normal
completion, a trap at the first or second load, and fuel budgets
zero, one, two, three, four and 100. Three fuel units stop immediately
before the invalid second load. With enough fuel the second load
traps at source PC 3 and reports effective address 65,539. Wasmtime
confirms normal completion and both traps.

Both Terrain Edit histories match Wasmtime and their archived full
page digests and source counts. A separate full tick compares every
source boundary and relevant access event against the original
single-step graph. All warm terrain and BFS runs are checked against
Wasmtime for returns and entire pages. Packed fallback is blocked.

## Profiling and timing

The segmentation diagnostic runs the warmed original portfolio at
single-source and two-source budgets so it can attribute observed
calls to ordinary work, covered work, and covered work immediately
after another region. This introduces many extra host crossings.
The separate budget-zero proxy is about 133–135 µs. Subtracting it
once per call puts ordinary work at approximately 79% of adjusted
diagnostic time in terrain and 83% in BFS. These are estimates from
a segmented run and do not reconstruct the whole-tick latency.

| Diagnostic category | Terrain segments / source steps / elapsed | Full BFS segments / source steps / elapsed |
| --- | ---: | ---: |
| Ordinary or fallback | 860 / 860 / 299 ms | 24,001 / 24,001 / 8,321 ms |
| Covered after ordinary | 247 / 494 / 63 ms | 8,795 / 17,590 / 2,244 ms |
| Covered after covered | 148 / 296 / 38 ms | 0 / 0 / 0 ms |

Terrain PC 9's four-step local median went from **290.18 to 272.62 µs**
across 31 alternating samples of 16 invocations. The profile confirms
two row fetches versus one, four gathers versus three, two stack
scatters versus one, and four `copy_` events versus two. Applied 33
times, the local difference estimates **0.58 ms per tick** before
runner overhead. Even an idealized zero-cost version of that local
block would remove under 10 ms using these micro timings, which also
include the host crossing. More regions compose in terrain, but only
this site is active here.

| Warm workload | Published portfolio | Composed | Same graph with site disabled | Arithmetic only |
| --- | ---: | ---: | ---: | ---: |
| Terrain median | 108.16 ms | 108.98 ms | 109.32 ms | — |
| Terrain p95 | 110.92 ms | 111.77 ms | 113.19 ms | — |
| Full BFS median | 2.945 s | 3.050 s | — | 2.957 s |

Terrain has 24 warm samples per path. BFS has four full rotated
samples per path after the graph warmed on terrain. Terrain runs
the composed site 33 times among 1,650 source steps and 26 host
calls. BFS admits no site and keeps 41,591 source steps and
650 host calls. All warm paths use identical initial pages and
inputs, with image preparation, profiling, page publication and
verification outside the timer. There is no measured terrain gain
against the published portfolio. The median remains about 2.18×
the 50 ms target. The marginal same-graph result is consistent with
a small site-local saving. The composed runner is slower on BFS in
this sample, where it has no eligible work.

The offline proof took **177 ms** and checking the cached certificate
chain took **2.58 ms**. Initial plan/image preparation took 190 ms
for terrain and 7.75 ms for BFS. Extending the portfolio cost 10.47 ms
for terrain, including the guest proposal and independent Wasmtime
replay, and 0.13 ms for BFS, where no proposal ran.

In a separate sampled [ordinary-operator profile](ordinary-profile.json),
`select`, `where`, and `eq` collectively account for about 23% of
CPU self-time in both workloads. The next target is ordinary
opcode and state selection, especially the repeated tensor selects
and small masks, rather than a wider version of this byte gather.
The 93 focused Python tests and five browser replay tests passed.

## Reproduce

From `visuaml-client/voxel-replay`, with the backend dependencies,
CPU PyTorch, Wasmtime, Z3 and rustc 1.100.0 with its
wasm32-unknown-unknown target if rebuilding the planner:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_composed_loads.py
python tools/replay_composed_loads.py --transfer-only
python tools/profile_composed_ordinary.py
```

Use `bash tools/build_composed_loads.sh` to rebuild the guest planner
and `python tools/prove_composed_loads.py` to rerun the offline proof.
The [raw evidence](replay.json) includes warm timings, segmented profiles,
source prefixes, a rejected composition, trap cases and emitted plans.
