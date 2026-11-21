# Recurrent evidence allocation

This is the allocation experiment on top of the prior recipe-assembly branch. The v60 importer, compiler and world replay, and prior stress results were replayed first. Their artifacts were preserved.

A shared-weight attention controller sees three typed candidate tokens and a recurrent state token. Each candidate exposes costs, modeled payoff, quoted validity and detectability, evidence status and remaining budget. The state token is carried through all requests. The model was trained from full trajectories using actor–critic updates on certified payoff less query charges. No deterministic policy labels were used. Training varies regime, costs, payoffs, and initially cached certificates. The model froze at c5b368c before the final fixture generator was added.

A proposal can stop, screen or ask for a strong check. An exact mask enforces availability and affordability. The journal preserves the raw choice and the scheduled one. An inconclusive sample cannot promote a candidate. A strong result alone can certify a previously unknown candidate. Each paid call consumes positive budget and advances an option's evidence state. Three options allow at most six queries; one final step permits stopping. The exact runner remains in Python and reconstructs the whole trace with the published weights.

## Held-out allocation results

Each family contains 96 paired schedules. Outcomes are generated independently of the frozen policy. Shared shift varies an underlying regime common to the candidates, rank reversal shifts the most promising candidates' validity, and warm cache supplies valid preexisting certificates. The calibrated DP uses the generating probabilities as a privileged reference. The robust DP uses a pessimistic interval endpoint with detectability held fixed. Risk-sensitive greedy, proof-first, replay-first, v59 MLP, and reset-state attention use the same quoted observations and budgets.

| Net modeled units | Recurrent | Reset attention | v59 MLP | Quoted DP | Calibrated DP | Robust DP | Risk-sensitive greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated | 856 | 918 | 918 | 1,175 | 1,175 | 966 | 1,160 |
| Shared shift | 287 | 280 | 381 | 446 | 810 | 362 | 436 |
| Rank reversal | 556 | 518 | 537 | 576 | 602 | 748 | 649 |
| Warm cache | 301 | 223 | 159 | 137 | 243 | 428 | 203 |

The recurrent policy spends fewer calls than quoted DP in each family. Robust DP yields higher realized net value in all four. Resetting the latent state hurts on three shifted families and helps on the calibrated one, but this is one training seed and a fixed-weight ablation. There is no reproduced recurrent advantage over the robust baseline. The recurrent policy missed 107 individually profitable opportunities and had 23 raw proposals changed by its deterministic mask. Zero unsupported candidates were accepted.

## Live execution and cost breakdown

The scheduler also executes six cases each against live compiler obligations, live Pocket forecasts, and a typed frozen-VM screen with a separate universal SMT query. The frozen module runs the existing CPU tensor graph with packed fallback blocked and verifies its observed sample with Wasmtime. A pair agreeing at input zero still fails the universal check. The tensor option is a sampled evidence producer under the same typed contract, while authorization remains separate.

The recurrent policy's live median decision time was 33.04 ms for the compiler, 0.85 ms for Pocket, and 0.81 ms for the tensor-screen workload. Those medians mix executions and abstentions, with six cases per domain and policy. The timing archive separates proposal assembly, scope, execution, verification, retrieval and complete decision time. Live policy ranking uses modeled tariffs so those comparisons still use identical resource caps. A separate measured payback witness charges one SMT verification against 12 paired frozen executions of a source and valid replacement. It saves approximately 197 ms net on this local replay API workload. This is an individual optimization witness, not a measured net win for the learned allocator.

## Evidence composition and validation placement

The derived equivalence node combines A≡B and B≡C only when the same canonical intermediate identity and observation semantics appear on both sides. The importer independently checks both v60 roots and binds both as parents. Replacing an endpoint rejects authorization. Changing the intermediate invalidates the derived root transitively. The related bounded-inference witness checks that a corridor's later target cannot be reached without passing its earlier target, then propagates the independently measured two-edge lower bound. The archive records the graph and initial intervals.

For one persistent affine proof, 24 resident Python validations took 0.507 ms median per request. Using an embedded C normalizer under the same Python scope and disk validator took 0.471 ms. A long-lived subprocess took 0.691 ms including IPC, with one 1.20 s startup measurement. The Python and C paths use the same validated inputs and proof contract. Native compilation happens before timing. Raw phase reports include filesystem lookup, assembly, proof publication and invalidation.

## Local reproduction

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_recurrent_evidence_v61.py
    PYTHONPATH=.:../backend python tools/bench_recurrent_evidence_v61.py
    PYTHONPATH=.:../backend python tools/compose_evidence_v61.py
    PYTHONPATH=.:../backend python tools/compose_evidence_v61.py --import-bundle evidence/recurrent-evidence-v61/composition.json

The last command independently rechecks the archived derived certificate in a new process. Timings live in separate raw files.
