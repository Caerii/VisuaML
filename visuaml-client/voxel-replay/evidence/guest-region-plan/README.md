# Guest-proposed specialization of unchanged terrain WASM

The [Rust planner](../../guest/plan_local_pair.rs) runs on the existing
frozen memory-selection graph. It receives two decoded instruction
descriptors starting at byte 512. Each contains an active-low
`local.get` lane and a local index. A fifth i32 word enables a
negative-control mode. The planner emits six i32 words at byte 532.
They hold the template ID, both final local indices in order, the
index observed at the first boundary, and the two fuel charges. It
declines a descriptor containing a load.

The host validates the original WASM, identifies adjacent pairs of
ordinary `local.get` instructions with sequential successors, and
passes those descriptors to the planner. It validates the template,
checks every distinct proposal with Z3, and binds accepted local
indices to their source PCs in a separate image table. The guest
chooses the template and schedule. The [frozen region runner](machine.pt)
lowers the accepted template to one vector gather and one scatter into
the bounded operand stack. No guest bytes are edited. The terrain
binary is the existing `f43859bb…` artifact, with its full hash in
[replay.json](replay.json).

## Prefix contract and negative control

The running state has two valid consecutive source rows, at least two
fuel units and two free stack slots. Inactive stack slots are zero, and
the locals contain i32 bit patterns. Under those preconditions the
independent Z3 model compares all 24 operand slots, height, image PC,
fuel, locals, status and empty memory access at prefix one and two.
Memory, frames and depth are shared unchanged by the two local reads.
The SMT queries compare a masked reference push with the plan's
indexed stack updates. They check the bounded template, with
runtime replays checking its binding to the real executor.

If either the invocation's source budget or its fuel would stop inside
the pair, execution takes the ordinary transition at the original PC.
The one-step audit artifact retains all original source boundaries.
The evidence checks a one-step observation and a fuel rejection inside
a selected pair. It compares all 1,650 boundaries of one eight-agent
tick against the single-step reference. Both Terrain Edit histories
also matched Wasmtime and the archived NumPy returns, full pages, and
exact source counts with packed fallback blocked.

The planner's challenge mode emits an end-only candidate. At terrain
PC 67, the counterexample replay pushes 0 at the first source boundary.
The rejected schedule would expose 128 there. Both plans finish with
the pair `[0, 128]` and the same total fuel charge. Z3 reports SAT
at boundary one and UNSAT at boundary two. The host only lowers plans
that pass both checks. The full proposals, SMT-LIB queries, hashes
and counterexample are archived in [replay.json](replay.json).

## Warm measurements on unchanged terrain

| Eight-agent sparse world | Frozen budget-64 baseline | Guest-planned region |
| --- | ---: | ---: |
| Median tick | 172.36 ms | 150.38 ms |
| Nearest-rank p95 | 186.58 ms | 161.00 ms |
| Source steps | 1,650 | 1,650 |
| Host invocations | 26 | 26 |
| Fused pairs per tick | 0 | 149 |

There were 24 warm paired samples after excluding the first tick.
The order alternated on identical inputs. Every timed page matched
Wasmtime byte for byte, and the reported latency excludes preparation,
verification and page publication. The measured median gain was 12.8%.
The 149 regions cover 298 steps, or 18.1% of the tick. With uniform
step costs, eliminating all covered work would give an illustrative
31 ms whole-tick ceiling. Step costs vary, and fusion still executes
the ordered operand results. Auditing with two-step calls allows
153 nonoverlapping pairs, with four opportunities lost to the
budget-64 invocation boundaries in the measured run.

A local benchmark used the actual entry state at terrain PC 149
and one two-step host invocation on either path. Over 31 alternating
samples of 16 invocations, median latency went from 415 to 204 µs.
A separate runtime profile counted two instruction-row fetches versus
one, six total gathers versus one, and 24 `where` operations versus
zero. The fused path adds one bounded operand-stack scatter. The
profiler records one `copy_` event within that local path. The full
page is carried through unchanged. These operator counts are separate
from wall timing.

In this run original image validation took 3.7 ms, and plan preparation
took 586 ms. Preparation included 96 ms for guest planning plus
Wasmtime checks and 487 ms in the independent solver. Preparation
deduplicates identical local-index pairs and runs once per image.
Loading the three artifacts took 46 ms. The accepted live-input
cancellation and rejected signed-growth case, the earlier
constant-fold regressions, 93 focused Python tests and five browser
replay tests passed.

## Reproduce

Run from `visuaml-client/voxel-replay` with the backend dependencies,
CPU PyTorch, Wasmtime and Z3. Rebuilding the planner requires rustc
1.100.0 and the wasm32-unknown-unknown target.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/build_region_plan.sh
python tools/replay_guest_region_plan.py
python tools/replay_live_rewrite.py
python tools/replay_guest_rewrite.py
```

To regenerate the region graph from source:

```python
from visuaml.compiler_lab.frozen_region_v58 import freeze_region_runner
freeze_region_runner().save("evidence/guest-region-plan/machine.pt")
```
