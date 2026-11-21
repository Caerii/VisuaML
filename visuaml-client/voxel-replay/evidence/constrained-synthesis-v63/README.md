# v63 Constrained neuro-symbolic synthesis

The v62 conditional and hidden-regime archives replayed unchanged. The routers were frozen at c6f2fbd before the v63 held-out generator was installed. The two recurrent models use seeds 631 and 632. The stateless attention and MLP controls use seed 631. All models saw the same 1,800 synthetic trajectories from an entropy-per-charge teacher, over widths 5–9 and one or two ranges.

## What failed in v62

The archived 63 denied cost-shift proposals break down into 40 repeated probes, 17 unaffordable requests, and six with both defects. Applying the legality mask to the same frozen v62 weights changed 125 proposals on those 64 tasks and still certified **0/64**. Masking repaired dispatch and left the policy's synthesis failure visible.

On those same archival inputs, the full v63 system, which also has exact filtering and counterexample-guided candidate generation, certified 64/64 for recurrent seed 631, 63/64 for seed 632, and 64/64 for both entropy and bounded lookahead. This diagnostic changes the search protocol as well as the model. It should not be read as an estimate of the isolated effect of learned weights.

## Typed search and exact authority

The version space is the canonical list of bounded range-union bit patterns consistent with two disjoint observation masks. A new verified example or counterexample is the only operation that updates those masks. The model sees ambiguity, split entropy, costs, remaining budget and action availability. It scores typed requests to probe a point, check one of four deterministically enumerated candidates, or stop. Candidate generation and the final proof checker stay deterministic. Every proof attempt and returned counterexample enters the content-addressed DAG with its predecessor. Identical traces can share nodes.

The dispatch comparison uses hard masking, rank-filtered constrained decoding, an inference-only soft invalid-action penalty, and a hierarchical class-then-operand heuristic over the same learned scores. Hard masking and rank filtering give identical traces with identical tie-breaking. The soft penalty and hierarchy are decoding ablations, rather than separately trained policies. The reserve-proof ablation rules out probes that would leave too little budget for a proof, and stops immediately if proof is unavailable. All variants pass through the same exact gate. No illegal query reached an executor.

The grammar has two semantic contracts. The predicate returns a Boolean combination of signed comparisons formed by a union of disjoint intervals. The affine patch returns x plus a fixed i32 delta when the predicate holds, with WASM wraparound; it returns x elsewhere. These are two distinct output contracts with the same latent interval search problem. A width-14, three-range space has 6,476 canonical candidates. The observer compares successful completion, return value and unchanged memory, with sufficient fuel. Loops, stateful writes, general equality saturation, and transformer-native execution remain outside this slice.

Certified candidates undergo independent partition validation, with emitted WASM hashes and exact scope bound into the root. The importer also checks that a decision names a candidate matching its certificate. Eight compositional pairs were additionally checked by SMT and executed in Wasmtime and the exact CPU tensor VM with packed fallback blocked. Their replay spans signed extremes and arithmetic wraparound. The forged rehash, wider domain, modified coefficient, and changed fuel-contract controls were rejected. The wrong patch gives 2,147,483,641 versus −2,147,483,624 at the archived counterexample. The certificate binds a completion-only fuel contract; a sample does not grant universal authority.

## Held-out results

Each new family has 32 tasks. The larger grammar uses three actual separated ranges, widths 11–14, and includes both output contracts. All methods see the same observations and query budgets. The bounded baseline uses one full action and a second-probe worst-case lookahead. The entropy and enumerative baselines expose the same grammar and validation protocol.

| Policy | Nominal certified | Cost shift certified | Three-range certified | Three-range credits spent |
| --- | ---: | ---: | ---: | ---: |
| Recurrent 631 | 9 | 19 | 9 | 762 |
| Recurrent 632 | 8 | 18 | 8 | 869 |
| Stateless attention | 11 | 19 | 9 | 847 |
| MLP | 7 | 20 | 11 | 878 |
| Entropy | 10 | 20 | 11 | 865 |
| Bounded lookahead | 8 | 18 | 9 | 856 |
| Enumerative | 7 | 17 | 7 | 926 |
| Recurrent 631 with proof reserve | 13 | 19 | 11 | 746 |

The reserve gate also reduced spend on the unavailable-proof family from 423 credits to zero. Its 32 filtered raw proposals are preserved. Across all evaluations, the checker accepted no unsupported result. Seed 631 had no raw illegal proposals under the ordinary gate and seed 632 had two on compositional tasks. The absence of illegal proposals with seed 631 explains why multiple decoding modes agree. The paired bootstrap interval for seed 631 minus bounded lookahead on the three-range family is [−0.125, 0.156] in approval fraction. The work savings of some configurations did not establish an advantage over the best symbolic or stateless baselines.

## Timing and economics

Three-range median search-and-check times included version-space construction and evidence writes. They were about 21.9 ms for recurrent 631, 17.8 ms for MLP, 16.6 ms for entropy and 104.6 ms for bounded lookahead. The archive separates model proposal, exact filtering, persistence and recheck. The eight selected binaries retired 24–54 instructions in the pointwise reference and 19–31 in the range form. The native mean saving bootstrap interval spans zero. Both binaries run through Wasmtime's same native engine, with 32 measured warmed calls per binary and eight paired cases. The per-pair amortization includes synthesis, recheck, lowering and an SMT audit. The finite observed estimates range from about 66,000 to 700,000 repetitions, with no observed finite break-even for two pairs. These are timing-based estimates and exclude unmeasured deployment costs.

## Local replay

From visuaml-client/voxel-replay in the research Python environment:

    PYTHONPATH=.:../backend python tools/evaluate_constrained_synthesis_v63.py
    PYTHONPATH=.:../backend python tools/evaluate_constrained_synthesis_v63.py --import-bundle

The first command regenerates all trajectories and checks their deterministic archive. The second stages the DAG and independently imports 114 distinct certificate roots in a fresh process. Use --write to regenerate the archives and raw timing samples.
