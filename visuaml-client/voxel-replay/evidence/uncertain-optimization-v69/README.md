# v69 Uncertainty-aware verified optimization

The v64–v68 local replays passed without changing their archives. The v68 negative learned-policy finding remains intact. v69 uses empirical calibration and deterministic planning. It adds no learned model because the small held-out workload does not warrant one.

## Performance and correctness identities

The measurement archive covers 40 scopes on one AMD EPYC 9V74 host. It uses Cranelift speed and none, eight and sixteen lanes, short and full footprints, and uniform-byte and periodic input pages. The archive binds CPU, architecture, Wasmtime version, optimization level, method, binary identities, and initial page hash. Each curve uses 16 warmed randomized paired batches of 40 calls. Separate quick measurements use five batches of 16 calls per variant. Exact scalar results and full pages are checked on deployment. The independent v68 verifier imports all 20 code roots and rejects the rehashed attempt to elevate completion evidence to migration authority.

Proof identity is independent of compiler level. Performance quotes require their hardware and footprint key. A quote from the other Cranelift setting is refused on import into a decision. In an offline control that ignores this gate, the old interval would contain only 22 of 32 comparisons. In scope, the pooled calibration radius contains 31 of 32 held-out savings observations. These are descriptive counts over related layouts, not a coverage guarantee. The sample has disjoint train, calibration and holdout layouts. Proof and deployment upper quotes are empirical training maxima, exceeded on 3 and 12 of 32 held-out checks respectively. The rounded use-count scenario envelope covers 132 of 144 revisions; the compiler-shift family includes deliberate count outliers. Recurrence is quoted from recent same-scope history; the measured return frequencies are in the outcome archive. No evidence here establishes portability to another CPU.

## Conservative deployment

Twenty-four paired histories have six revisions each. Future footprints, compiler settings and exact use counts are hidden online. Each policy sees current scope, the noisy count hint and its prior observations. The economic replay predicts and charges per-call medians from locked measurement cohorts, including measured proof, retrieval, deployment and quick-benchmark costs. It does not execute every replayed workload count in the policy benchmark. Only the privileged clairvoyant DP receives future outcomes. Bounded DP and beam score foreseeable revisits using current quotes; MPC updates the count estimate with observed hint error. The Bayesian control computes discrete expected utility with a downside penalty. The robust policy prices the low savings estimate, lower favorable use count and upper cost quotes before deciding.

| Policy | Replayed net across 24 histories | Negative revisions | Benchmarks |
| --- | ---: | ---: | ---: |
| Greedy | 1,473 ms | 17 | 0 |
| Bayesian | 1,452 ms | 9 | 0 |
| Robust | 741 ms | 0 | 0 |
| Bounded DP | 1,499 ms | 31 | 0 |
| Beam | 1,492 ms | 40 | 0 |
| MPC | 1,485 ms | 34 | 0 |
| Fixed measurement | 405 ms | 50 | 38 |
| Adaptive value of information | 1,499 ms | 31 | 0 |
| Clairvoyant | 1,695 ms | 15 | 0 |

The robust policy deployed at 15 revisions versus 74 for greedy. It eliminated sampled downside and sacrificed substantial value. Bounded DP, beam and MPC have paired mean intervals against greedy spanning zero. Fixed benchmarking changed 15 deployment decisions, yet only one purchased benchmark repaid its own charge on the immediate matched counterfactual. Its sampled loss relative to greedy is clear. The adaptive policy declined all purchases at the measured quote level. The result does not establish a generally optimal benchmark rule.

## Measured payoff and persistence

A separate experiment runs every requested invocation on one held-out eight-lane full-footprint workload with Cranelift none. It charges fresh proof generation, independent validation, store lifecycle and deployment once. At 256 calls the pointer rewrite lost 40.54 ms median after its 41.36 ms upfront cost. At 16,000 calls it earned 12.47 ms median, with a paired mean interval of 11.26–19.91 ms. Seven paired repeats share the measured setup charge. This is one binary pair on one host. The Wasmtime engine already exists before the setup timer. The run uses separate warmed instances and identical input layout.

The 48-request two-branch fixture spends about 739 ms on the isolated proof path and 183 ms sharing verified premises. Fresh checks fall from 384 to 59. Both stores hold the same 279 content-addressed nodes and 80,607 bytes because identical files deduplicate independently of validation reuse. The shared memo grows over branch visits; scope and lineage stay distinct. Reopening the proof roots yields the same independently checked binaries. A rollback to a sibling is refused and an ancestor rollback leaves the sibling untouched.

Migration is measured at four logical snapshot-ready boundaries with 4, 32, 256 and 1,024 records remaining. Each run charges packet export to disk, file load, scope revalidation, restoration, schedule change and continued execution. Every resumed page matches the uninterrupted reference. The switch loses about 1.70–1.83 ms median across these sizes. No tested workload breaks even. The prior fresh-process importer replays independently; process launch is outside the resident measurement. The contract cannot authorize arbitrary WASM fuel-cut migration.

## Replay locally

From visuaml-client/voxel-replay in the research environment:

    PYTHONPATH=.:../backend python tools/uncertain_optimization_v69.py
    PYTHONPATH=.:../backend python tools/uncertain_optimization_v69.py --import-only
    PYTHONPATH=.:../backend python tools/bench_lifecycle_v69.py --measure
    npm run dev

Open /dependency-fusion.html to inspect uncertainty intervals, observed savings, benchmark decisions, and negative outcomes. The viewer uses archived evidence. For a new performance cohort run tools/measure_uncertainty_v69.py --measure and then tools/uncertain_optimization_v69.py --write. Measurement commands replace raw timings; use a clean research checkout to preserve the published cohort.
