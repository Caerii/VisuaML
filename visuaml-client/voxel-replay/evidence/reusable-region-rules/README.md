# Reusable local-read region rules

The [versioned rule artifact](rules.json) was archived at `8bbebbd`
before the [transfer guest](after-freeze.wat) was authored. The artifact
contains five Z3 UNSAT obligations, the SMT-LIB queries, and SHA-256
bindings to the symbolic model, VM semantic sources, single-step audit,
baseline, and [new frozen lowering](machine.pt). Admission checks the
checked-in [certificate digest](rules.sha256), version, and bindings
once per run. This is a trusted, reviewed artifact and a fixed lowering
template. Site admission uses ordinary structural checks and does not
invoke Z3. The frozen graph continues to read the unchanged WASM's
decoded image, with separate metadata for admitted region entries.

The [original pair planner](../guest-region-plan/planner.wasm) and the
[new load planner](../../guest/plan_local_load.rs) each receive two
instruction descriptors as active-low opcode lanes and operands, plus
a challenge word at byte 512. The guest emits a template identifier,
two operands, the first-prefix policy, and fuel charges at byte 532.
The host checks the plan against the decoded rows and the certified
template. It checks that both rows fall through in the same function
and that the second is nonterminal. Basic WASM validation covers
declared locals and memory arguments. No intermediate entry is
required to hit the region fast path. An independent entry to the
second row still executes its ordinary instruction or its own admitted
region.

## Proof scope and the rejected eager read

For the pair, the reusable proof quantifies both local indices, every
local value, every physical stack slot, PC, height, and fuel. It compares
the ordered operand results, PC and fuel after each original step.
The stack tail is canonical zero, the execution is running, and the
fast template requires two free slots and at least two fuel and
invocation slots. Locals, frames, and memory are carried unchanged.

The load combines a local read with an unsigned offset addition and
one little-endian four-byte load. The proof independently checks the
address guard, the one-step cutoff, the loaded result, and the access
event for arbitrary memory bytes. A failing address guard delegates to
the ordinary source transition, preserving the original trap and fuel
accounting. The first instruction also takes the ordinary path if the
invocation or the fuel budget would stop inside the region. Exhaustion
returns before any row fetch or gather. The original one-step graph
continues to provide the audit reference.

Z3 rejects an eager proposal for a 64 KiB page at input 65,532 and
offset 1. With one fuel unit, the correct execution completes the
local read, leaves 65,532 on the stack, and exhausts fuel before any
memory event. Giving the load permission to execute early would trap
at effective address 65,533. The negative-control replay shows both
observable outcomes and a profiler confirms zero fetches and gathers
after the corrected cutoff. The post-freeze guest also checks a
successful unaligned read, the straddling trap, an unsigned negative
address, and budgets of zero, one, two, and 100. Successful and trapped
runs agree with Wasmtime.

The same-program audit compares all 1,650 original terrain boundaries
with the single-step graph, including state and actual access events.
Both edit histories match Wasmtime and the saved NumPy page digests
and source counts. The original end-only pair proposal still fails
its first boundary and passes its second. Full BFS and every timed
terrain tick match Wasmtime's return and complete memory. Packed
fallback is blocked throughout.

## Preparation and performance

The two generalized rules took **209 ms once** to prove offline.
Checking the cached artifact and its bindings took **0.43 ms**. The
site-admission column includes row checks and metadata construction.
Planner and Wasmtime comparison time is measured separately, with
repeated proposals deduplicated within each image.

| Workload | Admitted pair sites | Admitted load sites | Other source rows | Site admission | Guest proposal and oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terrain | 20 | 25 | 126 | 1.47 ms | 162.57 ms |
| BFS | 1 | 0 | 83 | 0.16 ms | 6.26 ms |
| Post-freeze guest | 1 | 1 | 7 | 0.19 ms | 12.40 ms |

BFS has no eligible consecutive local-read and load pair in this binary.
No proposed rule is applied to other opcode sequences. The certificate
check and the site checks do not include guest proposal latency.

| Warm run | Baseline median | Pair only median | Both rules median | Source steps | Host calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-agent terrain | 170.20 ms | 150.68 ms | 122.35 ms | 1,650 | 26 |
| Full BFS | 4.288 s | 4.546 s | 4.305 s | 41,591 | 650 |

Terrain used 24 warm samples per path, rotating order on the same input.
Nearest-rank p95 was 180.55, 163.42 and 126.15 ms. Both rules reduce the
terrain median by 28.1% from baseline and 18.8% from pair only. The run
executes 149 pair and 177 load fusions per terrain tick, replacing 326
ordinary instruction-loop iterations. By contrast, full BFS executes 33
pair fusions, covers 0.16% of source steps and shows no measured gain in
two full runs per path. Timings exclude preparation, VM initialization,
verification, profiling, and page publication.

At one real terrain load pair the runtime profile counts two instruction
fetches versus one, seven gathers versus two, and 26 `where` operations
versus zero. The actual page read is retained. The fused path performs
one bounded operand-stack scatter. These are per-region operator counts,
not upstream allocations. The two previously published rewrite
demonstrations passed, alongside 93 focused Python tests and five browser
replay tests.

## Reproduce

From `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch, Wasmtime and Z3 4.15.3:

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_reusable_region_rules.sh
python tools/prove_reusable_region_rules.py
python tools/replay_reusable_region_rules.py
```

The published replay can be run directly against the archived proof
and binaries without running the offline proof command. Rebuilding
the Rust planner uses rustc 1.100.0 and its wasm32-unknown-unknown target.
The planner's input, emitted plans, per-PC bindings, negative witness,
and raw warm samples are in [replay.json](replay.json).
