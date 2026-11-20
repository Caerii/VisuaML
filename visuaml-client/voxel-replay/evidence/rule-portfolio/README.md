# Profile-guided frozen rule portfolio

The terrain and BFS binaries are unchanged. The warm
[training profile](training.json) uses the existing pair and guarded-load
runner for elapsed time and a separate single-step audit for dynamic
opcode and PC counts. Terrain executes 193 adjacent local pairs and 242
local-plus-load pairs before accounting for overlap or budget boundaries.
Its new arithmetic pattern occurs 65 times. BFS instead executes 4,381
constant-plus-multiply, 2,239 constant-plus-signed-compare and 2,142
constant-plus-add pairs, a combined 8,762 candidate executions. The
existing rules fuse 33 BFS pairs and no loads.

## Selection and proof

The host generates bounded two-instruction candidates from the validated
source image. The old Rust planners still emit pair and load templates.
The new template recognizes `i32.const` followed by `i32.add`,
`i32.mul` or `i32.lt_s`. The host checks decoded instruction identity,
constant word, sequential PC edges, absence of branch guards and a
nonterminal second instruction in the same function. An independent entry
into a region interior can still execute the ordinary one-step graph.

For an overlap the host supplies the two scores to the
[Rust chooser](../../guest/choose_region.rs) at bytes 512 and 516 in
little-endian signed i32. The guest writes the chosen index at byte 520.
The score is training execution count multiplied by an integer cost
weight. Local pairs use weight one, while guarded loads and arithmetic
use weight two. These heuristic weights are fixed across guests. The
policy visits sites in PC order and resolves each overlap by retaining
the higher-scoring candidate. Ties retain the earlier site. The host
validates the output and rejects unknown, duplicate and overlapping
entries. The policy is a bounded greedy tradeoff. Uncontested sites are
admitted from the same proven templates. The [replay](replay.json)
includes the score inputs, guest outputs and a rejected overlapping
selection. It also compares an earliest-site nonoverlapping policy
applied to the existing rules.

The [versioned proof record](rules.json) records checked Z3 results and
contains independently rerunnable SMT-LIB queries. It contains no proof
object or proof trace. The checked-in digest binds the record to the
parent proof record, arithmetic semantics, symbolic model, admission
code, guest chooser and frozen lowering. The runtime admission path
checks those bindings and the proof status without rerunning Z3.
Changing a binding requires a new reviewed record.

The new model quantifies the constant word, every input stack word,
height, PC, and fuel. It requires a running, validated entry with a
nonempty stack, canonical zero tail, room for the intermediate push
and at least two budget and fuel units on the fast path. It models
the ordinary prefix and final source boundaries. The fast path
updates the original top slot directly. Signed 32-bit multiplication
is represented by its low 32-bit residue, which is identical to the
truncation of the runtime's signed 64-bit product. Neither arithmetic
instruction changes locals, frames or memory. A one-step invocation
or one-fuel cutoff still takes the source transition.

## Composition and replay

The planner, admission checker, graph and proof were bound at
`339ee99` before authoring the [transfer guest](after-freeze.wat).
It executes a guarded load, a constant arithmetic region and a local
pair consecutively. Each region was checked against the single-step
graph at the entry, at the one-step cutoff, and at the exit. The
load region still traps when its four-byte window straddles the page.
Both Terrain Edit histories retain their original source counts and
match Wasmtime's return and every page byte. One full terrain tick is
also audited at every original source boundary. Each timed full BFS
run retains 41,591 source steps and matches Wasmtime's final return and
page. Packed fallback is blocked for the replay and benchmark.
The parent-rule regressions include the eager-load rejection.

## Measurements

All comparisons rotate or alternate order on matching initial pages.
The timed interval includes the budget-64 execution and host loop,
with validation, traces, page publication and comparison outside it.
Terrain has 24 warm samples after dropping the first tick per mode.
BFS has five full runs per mode after the graphs are warmed on terrain.

| Mode | Eight-agent terrain median / p95 | Full BFS median | Terrain pairs / loads / arithmetic | BFS pairs / loads / arithmetic |
| --- | ---: | ---: | ---: | ---: |
| Prior runtime overlap handling | 121.61 / 132.57 ms | Same single eligible site | 149 / 177 / 0 | 33 / 0 / 0 |
| Existing rules, earliest nonoverlapping | 121.29 / 138.92 ms | 4.332 s | 147 / 175 / 0 | 33 / 0 / 0 |
| New rule only | 166.35 / 184.49 ms | **2.859 s** | 0 / 0 / 65 | 0 / 0 / 8,623 |
| Guest-selected portfolio | **110.02 / 114.30 ms** | 2.933 s | 86 / 239 / 65 | 33 / 0 / 8,623 |

All paths use 1,650 source steps and 26 host calls in terrain, and
41,591 steps and 650 host calls in BFS. The portfolio fuses 800 source
steps in terrain (48.5%) and 17,312 in BFS (41.6%). BFS ranges were
4.246–4.391 s for existing rules, 2.820–3.033 s for the new rule alone,
and 2.867–2.974 s for the portfolio. The arithmetic gain appears in
every full paired sample. The latter two ranges overlap.

The guest selects a different winner at six terrain overlaps. At
PCs 149 and 150 it receives scores 32 and 64 and chooses the later
load over the earlier local pair. In a separate matched comparison
with all three rules on both paths, earliest-site selection took
110.49 ms and the guest policy took 109.70 ms. The guest shifts actual
coverage from 147 pairs and 175 loads to 86 pairs and 239 loads, with
65 arithmetic fusions on both paths. The measured median difference
is under 1 ms, so this sample establishes the coverage change more
clearly than a latency advantage for the policy itself.
[Raw policy-only comparison](policy-timings.json)

A local two-step timing comparison on true entry states measured
425 to 217 µs on a terrain constant-add site and 445 to 235 µs
on a BFS constant-multiply site, over 25 alternating samples of
12 invocations per path. The runtime profile counts two row fetches
versus one, six total gathers versus one, and 24 `where` operations
versus zero at each site. The new path uses one bounded stack scatter.
Whole terrain latency fell by 11.6 ms against the prior runner. The
65 newly covered regions account for 130 steps, or 7.9% of the tick.
With uniform step costs, removing that work entirely at 122 ms
would correspond to about 9.6 ms. Real costs differ by instruction
and the portfolio also changes load coverage. The profiles retain
frequently executed store, load and branch sequences outside these
three templates. The current measured result is about 2.2× the
50 ms tick target.

The offline proof took 416 ms. Verifying both cached records and
bindings took 0.77 ms. Terrain site and image preparation took
173 ms including prior guest proposals and oracles, of which 1.44 ms
was existing site checks. The separate guest overlap-selection phase
took 21 ms. BFS site and image preparation took 6.50 ms and selection
took 0.01 ms, since it has no overlap. No per-site solver invocation
runs during these phases. The parent replay, 93 focused Python tests
and five browser replay tests passed.

## Reproduce

From `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch, Wasmtime and Z3. Rust guest compilation requires rustc
1.100.0 and the wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/replay_rule_portfolio.py
python tools/replay_rule_portfolio.py --transfer-only
python tools/bench_portfolio_policy.py
python tools/replay_reusable_region_rules.py
```

The cached rule records are sufficient for the replay. To rerun the
offline proof use `python tools/prove_rule_portfolio.py`. To collect
a fresh warm training trace use `python tools/replay_rule_portfolio.py --train-only`.
