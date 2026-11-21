# v60 Persistent evidence and prior stress

The persistent store is a content-addressed DAG with assumption, source, query, witness, certificate and decision nodes. A decision inherits the dependencies of its parents. Invalidation walks the descendant closure of changed inputs, while an import independently rebuilds the expected scope and rejects a stale root. Historical roots remain valid for their original inputs. Node files are installed atomically. The experiment exercises normal process restart.

The bundle contains a compiler root and a Pocket root. The producer commits the compiler root to disk before starting the reader. The reader independently checks the obligation against the caller's inputs. Tests reject a stale scope, damaged bytes, and a false certificate whose nodes were all rehashed. Packed fallback is blocked during the execution comparison.

## Equivalence and execution

The compiler benchmark uses nine generated WASM program pairs across repeated cancellation and constant distributivity, including two invalid rewrites. Each expression is a straight-line affine function of one i32 input. Source and candidate are generated from validated typed tokens and bound by binary hash. The claim covers return and unchanged linear memory at successful completion for every i32 input, when enough fuel is supplied. Rewriting changes source-step and fuel behavior.

The producer asks SMT whether any 32-bit input distinguishes the pair. The importer independently recomputes the two affine normal forms over the ring of integers modulo 2^32. Addition and subtraction combine coefficients; multiplication is allowed when one operand is constant. Equal coefficient pairs establish universal equivalence for this fragment. The seven approved pairs pass both checks. SMT produces a concrete counterexample for each rejected pair, and both the frozen reference and native Wasmtime reproduce the differing returns.

Sample execution is exposed as a typed completion query bound to input, page, fuel and both source identities. The witness records both engines, their outputs and source steps. Its authority is only that sample. The separate universal evidence grants adoption. The frozen reference runs its existing CPU tensor graph under a host harness. The timings compare a preinstantiated Wasmtime call against the frozen replay API, which includes loading the image and final page capture. Each binary has seven measured calls per engine after an initial run. On the 40-cancellation pair, native medians were 83.1 µs for source and 49.8 µs for candidate. The frozen replay medians were 17.45 ms and 0.83 ms. These are small local samples. They measure execution rather than making every accepted rewrite an economic win.

Pocket retains the v59 cardinal dependency cone. The importer rebuilds the cone and independently evolves it with the scalar rule. The archived forecast survives an off-cone edit, rejects a change to the target, and traces transitive invalidation back to the source node. This proves import correctness for that forecast. The small world fixture still requires a scalar replay, so it demonstrates scope reuse without a claim of wall-clock savings.

## Complete reuse measurements

The 40-cancellation proof was requested 16 times in each hit-rate cohort using an already running reader. Cold requests include fresh scope construction, SMT, publication, and filesystem work. Cross-process hits include request serialization, fresh scope construction, hashed disk lookup, algebraic validation, and IPC. Raw samples include startup, hash, invalidation, and execution timings separately.

| Hits of 16 | Total request time |
| ---: | ---: |
| 0 | 51.91 ms |
| 4 | 41.58 ms |
| 8 | 31.90 ms |
| 12 | 19.74 ms |
| 16 | 10.59 ms |

The remote median in the individual phase test was 0.707 ms, with 0.214 ms for rebuilding the scope and 0.356 ms for validation including 0.111 ms of file reads. In-process trusted lookup with scope reconstruction took 0.244 ms median. A single measured process launch took 1.27 seconds. At the observed all-hit versus cold-publication difference, that startup alone amortizes after about 493 requests. This projection applies to a resident reader and this proof size. Starting a fresh interpreter per request would erase the savings.

## Matched prior stress

The frozen v59 model receives the same 256 cost schedules in each regime. A common random draw per candidate generates possible evidence under accurate, optimistic, pessimistic and rank-shifted priors. The rank shift makes the highest-payoff candidate less likely to be correct. This is a separate accounting simulation of evidence outcomes. The persistent and native replay experiment above runs actual checks. The calibrated DP and greedy baselines receive the generating probabilities. The robust DP plans at the pessimistic endpoint of its declared independent validity ranges. The risk-sensitive greedy baseline penalizes uncertain gains.

| Regime | Learned | Quoted DP | Calibrated DP | Calibrated greedy | Robust DP | Risk-sensitive greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accurate | 3,449 | 3,811 | 3,811 | 3,872 | 3,792 | 3,758 |
| Optimistic | 1,586 | 1,533 | 2,168 | 2,148 | 2,168 | 1,778 |
| Pessimistic | 5,893 | 6,520 | 6,808 | 6,828 | 5,476 | 6,142 |
| Rank shift | 776 | 1,038 | 2,442 | 2,374 | 1,507 | 1,292 |

Scores are realized net modeled units. The learned policy beats quoted DP under optimism, and robust DP leads both there. Under rank shift the learned policy missed 46 profitable cases and had 2,954 units of regret against the hindsight oracle. Zero unsupported plans were adopted. The hard gate denied three learned requests across the four cohorts. The archive records raw proposals, affordability, charged cost and missed decisions.

## Replay locally

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_persistent_evidence_v60.py
    PYTHONPATH=.:../backend python tools/persistent_evidence_v60.py --bundle evidence/persistent-evidence-v60/dag.json
    PYTHONPATH=.:../backend python tools/stress_metareasoning_v60.py

The first and third commands compare deterministic replay outputs with the published archives. The bundle command stages the supplied nodes as untrusted bytes and independently checks both roots. Use --write with the evaluators to capture fresh raw measurements. Timings remain separate from deterministic outcomes.
