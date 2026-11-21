# v62 Adaptive synthesis and hidden-regime allocation

The integration branch descends from both v61 experiments. The original recipe runner, fresh-process checkpoint, 12 specialization proofs, evidence-allocation benchmark and derived-equivalence import were replayed locally. Their artifacts are unchanged. The v62 experiments share content-addressed evidence and exact authority with v60/v61. Recipe registers and allocator belief vectors are separate typed state representations. Python drives both v62 loops.

## Active conditional synthesis

The learned controller sees a query budget, costs, proof availability and prior input/output observations. It chooses an input to sample, an interval predicate to submit, or abstention. It receives neither the target predicate nor an operation sequence. The target is an indicator over eight consecutive signed i32 values with zero outside that domain. The bounded synthesis grammar contains all contiguous ranges and the empty predicate. A submitted predicate is compiled from signed comparisons. A deterministic reference emits a sum of equality tests. One mismatch sends a concrete counterexample back into the observation state. An exact partition check alone authorizes completed proposals. Samples and model scores have no authority.

Three models were trained using complete trajectories from a finite version-space DP and frozen at 2667bc0 before the held-out generator. Recurrent attention retains its latent state across requests. Stateless attention and MLPs see the complete observations indexed by input. JSON boundaries exercise the same stateless attention with local serialization. Exact bounded DP and greedy split are programmatic synthesis baselines. The two Wasmtime binaries run through the same native optimizer. Equality saturation was outside this finite predicate synthesis experiment.

| Methods | Nominal approvals / 64 | Skewed cost approvals / 64 |
| --- | ---: | ---: |
| Recurrent attention | 35 | 0 |
| Recurrent weights, reset each query | 0 | 0 |
| Stateless attention / JSON boundary | 39 | 6 |
| MLP | 54 | 2 |
| Exact bounded DP | 56 | 15 |
| Greedy split | 55 | 5 |

No method authorized an out-of-grammar two-island predicate or a task with unavailable proof. The recurrent model made 63 denied proposals out of 64 shifted-cost tasks. Nominal medians were 0.74 ms for recurrent attention and 0.024 ms for exact bounded DP in this finite domain, including assembly overhead but excluding Wasmtime compilation and persistent proof work. Shifting the first live observation changed every one of the 64 recurrent nominal request traces, and the gate still accepted only correct programs. This measures causal dependence on observations, not useful recurrence by itself.

Eight selected approvals were lowered, proved universally with SMT and rechecked by an independent signed-partition importer. The existing exact CPU tensor executor and Wasmtime agreed on each recorded input and full page, including signed extremes, with packed fallback blocked. The narrow contract covers successful completion, i32 return and unchanged memory. Shortening code changes source-step and fuel accounting. This does not establish general WebAssembly equivalence. A forged, fully rehashed certificate was rejected, and input 1 returned 0 in the reference and 1 in the wrong range. The refinement witness permits equivalence under a narrower input assumption, preserves both scope and source identities, and rejects widening that assumption. Prior transitivity and relational witnesses remain independently replayable.

## Hidden regimes across tasks

This loop has a separate typed proof/skip interface. A series of tasks shares an unknown reliability regime. Each controller receives the same three initial observations and the current cost, benefit, budget, reuse status and previous result. Later proof outcomes inform subsequent tasks. The proof budget is shared across the 12 tasks. Models with recurrent state and full-history attention trained from complete trajectories on two seeds. The loss uses cumulative verified payoff less spend plus auxiliary validity, cost and reuse forecasts. Those forecasts never authorize anything. The four artifacts were frozen at fb27269 before held-out scenarios were generated.

The same 80 schedules per family go to all policies. The world and compiler accounting fixtures use exact but simulated outcomes drawn from the specified regime. Four live sequences separately exercise actual frozen forecasts and SMT obligations through scoped reuse. The Bayesian planner updates the two-regime posterior and computes a finite-horizon prove/stop policy. It knows the nominal model without access to the live regime, and plans over remaining slots without looking at future reuse. The robust planner takes a pessimistic adjustment, and risk-greedy penalizes uncertainty. A separately labeled privileged regime oracle sees the generating probability. The accounting archive also includes hindsight regret.

| Net modeled value | Stable | Weaker signal | Regime switch |
| --- | ---: | ---: | ---: |
| Bayesian DP | 3,413 | 2,432 | 3,176 |
| Robust DP | 3,493 | 2,151 | 2,997 |
| Recurrent 721 | 3,515 | 2,030 | 2,971 |
| Recurrent 722 | 3,072 | 2,232 | 3,462 |
| Full history 721 | 2,986 | 2,325 | 3,109 |
| Full history 722 | 3,286 | 2,321 | 3,604 |

There is no replicated advantage over the strong deterministic planners. The recurrent seed that led stable scores had a paired 95% bootstrap interval of [−3.01, 4.99] net units per sequence against Bayesian DP, while the other seed fell behind. The recurrent model's stable accounting loop took about 2.2 ms median per sequence, versus 0.17 ms for Bayesian DP. Replacing historical outcomes while the current descriptor and latest outcome remained fixed switched 9 of 136 proof choices on seed 721 and 18 on seed 722. Full-history attention switched 54 and 32. Shuffling a matched history changed zero and two choices for the recurrent seeds. The observed memory is causal, with no consistent economic edge demonstrated.

## Specialization economics

The eight audited binary pairs retired 10–35 source instructions in the reference and nine in the interval form. Native calls used preinstantiated Wasmtime modules with 32 measured calls per binary. The paired bootstrap interval for mean native saving spans zero. The proof-production break-even calculations use the observed per-pair native median savings and all recorded compilation, SMT, persistence and independent import charges. Positive estimates are recorded per pair, and pairs with no observed saving have no finite break-even. These exclude controller and deployment costs, so are lower estimates conditional on the observed saving. Workload rewards elsewhere are modeled and are not presented as speedups. Raw phase times include assembly/proposal, verification, lookup, persistent import and execution. The live regime sample has only two episodes per domain and is a path check.

## Replay

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_adaptive_synthesis_v62.py
    PYTHONPATH=.:../backend python tools/evaluate_adaptive_synthesis_v62.py --verify-bundle
    PYTHONPATH=.:../backend python tools/evaluate_hidden_regime_v62.py

Use --write to regenerate the deterministic archives and collect fresh timings. The verifier command opens the portability bundle independently. State, raw proposals, rejected computations, counterexamples, proofs and measurements are all in the evidence directory.
