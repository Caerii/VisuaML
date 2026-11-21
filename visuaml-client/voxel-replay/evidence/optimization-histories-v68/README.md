# v68 Verified planning across optimization histories

The v64–v67 deterministic replays passed before this experiment, including all 64 v67 fresh/incremental comparisons and the snapshot-ready migration. Earlier archives remain untouched. The two learned reuse forecasts froze at 3f0bd31 before the history fixtures were written. During integration, the pointer lowering was adapted to the frozen executor's four-local ABI, then calibration and evaluation ran on that final lowering.

## Online planner and scope

A source revision node binds its coefficient layout and parent. A typed transition binds the revision, emitted binary and independently verified certificate. Per-policy proof caches retain admitted premises and complete program proofs inside a history. A resident execution instance is reused when its active layout and compiled variant match; the record-count profile is still an invocation input.

Policies see the current layout, proof budget, use count, a noisy workload-profile hint, and past edits and observations. They cannot see future revisions or the current hidden profile before sampling. The actions are keep, prove without deploying, or apply a reviewed rewrite; a separate benchmark request unhides the current profile and allows reuse of that observation on matching layouts. The benchmark runs a native sample and consumes the revision's optimization opportunity. When the profile changes, that observation can be stale. There is no synthetic reward for having a warm proof cache. It pays off only when a later proof or deployment actually uses it.

Greedy optimizes the current quote. Two-step DP and width-four beam work over bounded forecasts based on the current quote and expected layout recurrence. A value-of-information control buys a benchmark only if the forecast benefit on a likely revisit exceeds its estimated charge. The learned methods replace the smoothed edit-repeat forecast with a frozen MLP and use the same bounded planning objective. The clairvoyant exact DP knows every future edit and profile, and is a privileged bound under the accounting model. Its achieved policy is also independently checked. Replay limits each quote by the common modeled budget. It separately records measured assembly, planning, proof/lowering, independent validation, deployment, benchmark and invocation latency. Those wall timings are not substituted for the modeled tariff or treated as a second independent run of every requested use.

There are 15 held-out histories, five per family and six revisions per history. The profiles vary record count on one host. They do not represent testing on multiple hardware machines. Runtime benefits are empirical per-call differences from locked, paired native samples for each width and profile, multiplied by modeled use counts. Proof and deployment charges are modeled in microseconds and replayable. Each apply operation uses an independently admitted program. New deployments check the full page against the scalar reference before timed reuse. The complete raw cost breakdown is in timings.json.

## Campaign results

Realized modeled net value in milliseconds, summed across five paired histories per family:

| Policy | Repeated layouts | Bursty edits | Shifting layouts |
| --- | ---: | ---: | ---: |
| Greedy | 20.77 | 0.00 | −10.77 |
| Two-step DP | 16.70 | −2.80 | −12.49 |
| Beam | 22.90 | −14.75 | −19.76 |
| Value of information | 22.73 | −2.80 | −13.14 |
| Learned 681 | 22.90 | −13.14 | −14.26 |
| Learned 682 | 22.90 | −13.50 | −14.44 |
| Clairvoyant DP | 43.60 | 20.79 | 24.44 |

Lookahead and learned forecasting sometimes capitalize on a recurring layout. They also pre-prove or buy benchmarks that fail to pay for themselves when history changes. The learned methods each benchmark 17 times versus two under the value-of-information control. The learned Brier scores were 0.210 and 0.215 against 0.204 for the smoothed empirical control. Paired per-history mean intervals versus greedy span zero for both learned versions. Neither demonstrated a repeatable economic advantage. Negative totals under quote noise remain visible.

## Independent provenance and branching

The branch fixture diverges through two distinct layouts and returns to the base layout on the left. The reconverged revision has a different lineage identity, while its canonical code proof is identical. Sharing reduced fresh obligations from 32 to 18 across four events. The imported bundle independently validates each deployed binary, each branch event, and each recorded proof-to-revision transition. Rollback to a sibling is rejected and rollback to an actual ancestor preserves the other head. Full regeneration agrees for each distinct admitted scope.

The new pointer induction proof checks address progression and data/scratch bounds for the guarded native loop. The pointer variant retains the original carry and depends on the prior rolling proof. It is completion-only. The forged stride, proof-policy substitution and stale layout were rejected. Incorrect byte stride diverged at byte 28 in direct Wasmtime execution. The final pointer binary matched the scalar reference, Wasmtime and the frozen CPU tensor executor for the sampled bounded calls and full pages. Packed fallback was blocked. The two archived WASM binaries are the pointer and reload control for this case.

Calibration uses warmed preinstantiated Wasmtime/Cranelift and optimized host C, with 24 shuffled batches of 64 samples per width/profile. At eight lanes and full capacity, reload took 19.34 µs median, pointer 18.37 µs, and native C 3.17 µs. The paired mean saving interval for pointer was 0.77–1.13 µs. The sixteen-lane full-capacity median was slower than reload. The raw intervals remain available.

The v67 logical snapshot packet is used in a fresh schedule-switch replay. At the measured boundary, rolling and reversed suffix loops each take about 19 µs, while validation and resume take about 1 ms. The switch does not pay for this short suffix. A host checkpoint certificate is required and completion-only code proofs cannot authorize the switch. These routines do not claim continuous WASM hot swapping or transformer-native execution.

## Local replay

From visuaml-client/voxel-replay with the existing Python environment and C compiler:

    PYTHONPATH=.:../backend python tools/evaluate_histories_v68.py
    PYTHONPATH=.:../backend python tools/evaluate_histories_v68.py --import-bundle
    npm run dev

Open /dependency-fusion.html to inspect the history, switch planners, compare quoted and revealed profiles, and follow the branch reconvergence. Recorded evidence is used throughout. Use --write for fresh raw timings with the locked calibration, or --calibrate --write to generate a new calibration and a paired archive.
