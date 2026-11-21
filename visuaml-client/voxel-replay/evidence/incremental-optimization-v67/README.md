# v67 Incremental proof-guided optimization and snapshot migration

The v64–v66 archives replayed unchanged, including 60 v64 and 48 v65 certificate imports, 240 exact tree edits, cyclic-width guards, the fuel-45 counterexample, and the v66 record-boundary migration. The two v67 quote rankers were frozen at c7ff011 before campaign fixtures were written. All experiments run locally.

## Incremental dependencies

The revision index holds typed logical slots. Each slot records its dependencies and reverse users. An edit invalidates the slot and its downstream closure; reads rebuild invalidated values on demand. A compiler adapter distinguishes source coefficients, memory effects, SMT obligations and checked results. Canonical content identities keep old checked obligations addressable even after their logical slots invalidate. The verifier uses its own scoped memo and independently regenerates the proof graph and module bytes. The inference adapter tracks factor and directed message dependencies and checks its two invalidated arrows against the existing exact tree cache and a fresh marginal. A message identity in this adapter carries no compiler proof authority.

## Revision campaigns

Four campaigns replay 16 revisions each. Local edits change a single lane coefficient; global edits replace every coefficient. Each sequence uses the same predetermined rolling or specialized schema under each comparison policy. Full regeneration rebuilds all dependencies and rechecks every obligation. Per-revision memo uses exact-content reuse only within each revision. Incremental retains logical and proof state across revisions. Each path publishes and independently verifies the same certificate root in a separate store. Measurement order alternates. Wall totals include invalidation, analysis, solver work, persistence, regenerated binary identity checks, and independent validation. Costs do not include workload execution.

| Edit sequence | Full fresh obligations / ms | Per-revision memo obligations / ms | Across revisions obligations / ms |
| --- | ---: | ---: | ---: |
| Local, 8 lanes | 128 / 250.2 | 122 / 251.3 | 45 / 155.4 |
| Local, 24 lanes | 384 / 689.1 | 346 / 658.5 | 77 / 188.1 |
| Global, 8 lanes | 128 / 280.7 | 126 / 284.7 | 106 / 243.9 |
| Global, 24 lanes | 384 / 732.6 | 355 / 705.7 | 219 / 487.2 |

After a local edit, the two dependence slots that use the changed lane must rebuild. Crossing a schema boundary refreshes proof slots even where effects are unchanged. A global edit rebuilds every effect. Identical obligation contents can still reuse a checked result on global edits. The local 24-lane sequence issues 24, 50 and 77 fresh obligations over its first 4, 8 and 16 revisions. These finite trajectories show reuse building up. They establish no asymptotic sublinear wall-clock claim, since each revision also requires analysis and publishing a root.

All 64 accepted roots revalidated in a fresh process with 256 distinct checked obligations in its import cache. Stale lane order, a forged check whose chain had been rehashed, and a changed emitted-binary identity were rejected.

## Reviewed rewrites and actual runtime

The reload implementation writes the original value to scratch and reloads it into the previous-value carry. Rolling eliminates that load by using the preserved local. Specialization additionally removes multiplication by one. The exact checker proves the scratch read-after-write identity and i32 modular term equality. Composition relies on disjoint data/scratch regions, bounds checked before writes, and the original carry relation. The regenerated WASM binds the implementation. These are completion-only rewrites. The v66 fission and snapshot-reorder schedules retain their own logical contracts; v67 does not add general loop fission, alias discovery, or arbitrary instruction reordering.

Both variants matched the independent scalar reference, Wasmtime and the frozen CPU tensor executor, including full-page checks and invalid counts. Packed fallback was blocked. Four full-capacity cases benchmark warmed, preinstantiated Wasmtime/Cranelift and native cc -O3, alternating shuffled order over 22 batches of 64 calls. The native implementation is an independent host baseline. On the first eight-lane layout, reload was 21.23 µs, rolling 18.59 µs, and native C 4.35 µs. Its paired mean saving interval was 2.26–3.03 µs. All four sampled intervals favored the reviewed replacement. Median-based break-even estimates range from 6,087 to 36,757 calls after charging the measured revision pipeline and fused deployment. The estimate excludes subsequent workload-specific integration costs. The two supplied binaries are that first case, regenerated from the certified artifacts.

## Migration between snapshot and output commit

A new host-level boundary lets a process stop after copying the pending record to scratch and before writing outputs. The checkpoint contains the original carry, cursor, full page, proof scope and remaining logical fuel. A fresh process independently reconstructs the prefix and switches from rolling commits to reverse store order for the pending record and suffix. Both schedules reach the same final page hash, cursor and carry. A stale layout, changed alias assumption, corrupted carry and completion-only proof are rejected. With one logical charge the snapshot can be written, but an attempted commit makes no further change. This does not authorize binary hot-swapping or source-instruction migration. A warmed validation-plus-resume median was 1.12 ms over 24 samples for a 131,378-byte packet. Fresh process launch is outside those timings.

## Learned proposals

An isolated quote experiment compares two learned rankers, cost-aware greedy and two-revision bounded enumeration with stationary quotes. The rankers learn the greedy modeled objective from synthetic inputs. They do not choose the fixed benchmark deployment sequence. Both seeds achieved about 23.33 aggregate one-step modeled units versus 23.34 for greedy; the two-step policy trades current modeled value for possible future reuse. The rankers took roughly 255 and 53 µs median versus 3 µs for greedy. On a cyclic ordering control, both previous learned order rankers, min-fill and min-degree choose width three. The broader v66 negative finding is intact. This experiment finds no learned economic advantage.

## Local replay

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_campaign_v67.py
    PYTHONPATH=.:../backend python tools/evaluate_campaign_v67.py --import-bundle
    PYTHONPATH=.:../backend python tools/migrate_snapshot_v67.py
    npm run dev

Open /dependency-fusion.html for the local/global revision timeline, invalidated and reused slots, proof costs, and downloadable snapshot checkpoint. It presents saved evidence. The Python commands re-execute and independently import the archives. Use --write to refresh outcomes and timings, and --benchmark on the migration tool to measure warmed restore.
