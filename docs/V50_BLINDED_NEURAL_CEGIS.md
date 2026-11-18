# v50 — Blinded neural-to-symbolic synthesis, CEGIS, and behavioral equivalence classes

**Research scope:** A deliberately bounded program-synthesis algorithm recovers compact, executable i32 functions from the *observed outputs of v49's constructed neural interpreter*, without receiving the original WASM source program, its program tokens, the neural weight matrices or the construction manifest. An independent Z3 verifier separately checks the recovered typed function against original source WASM semantics. This is not unique or arbitrary program decompilation, especially not an unknown learned transformer's semantics.

## Why this is a different question

v43 recovered one polynomial from a *known analytic ReLU architecture template*. v48 disassembled a *known explicit program-token format* and demonstrated source ambiguity. v49 now performs full-width modular i32 neural arithmetic across independently tested canonical WASM programs.

v50 attempts something more general, but with a **small declared hypothesis grammar**: *given only the neural execution oracle (f(x))* and a verifier that can return a signed-i32 counterexample, recover the simplest equivalent typed expression.

The synthesizer does not inspect the source, code history, weights, token encodings or internal traces. The external verifier does inspect the original typed source, but only to check its candidate; it does not transmit the source itself to the synthesizer.

## Bounded candidate language

- Modular signed-i32 polynomials (a x^2 + b x + c), with integer coefficients in [-16,16] and arithmetic modulo (2^{32}).
- Boolean-valued `i32.eqz(x)`, `i32.lt_s(x,k)` and `i32.gt_s(x,k)`, with signed thresholds k in [-16,16].
- No general circuits, indirect memory, host effects, program loops, variables, i64, floats or arbitrary learned network operations.
- Source programs must be admissible by the already strict v40 typed WASM IR.

The search builds candidates from output observations at seven preregistered signed integer inputs `[-3,-2,-1,0,1,2,3]`. Polynomials exploit the three-point modular relation `y(1)=a+b+c` to determine a candidate b for each bounded a; the other points reject spurious hypotheses. Predicate candidates are enumerated over declared thresholds. Candidates are ranked by explicit structural complexity rather than matching source-byte history.

## Actual counterexample-guided refinement

After selecting a candidate, the synthesizer calls an opaque verifier callback. That separate Z3 verifier symbolically executes the *original* typed WASM and asks whether any signed-i32 input produces different results from the candidate. If the outcome is:

- **UNSAT:** under this separately encoded closed-world i32 semantics, the candidate is equivalent to the original source function on all signed i32 inputs. The candidate is compiled into a new, canonical WASM module by the existing v40 backend.
- **SAT:** save the exact concrete signed-i32 counterexample, query the black-box neural oracle at that input, and refine the candidate set. The returned counterexample must actually falsify the proposed candidate; otherwise fail closed.
- **UNKNOWN:** refuse certification rather than silently declaring success.

The verifier's certificate is bound to the candidate's SHA-256. The synthesis code does **not** accept a generic `unsat` response without a matching candidate identity. Bounded rounds, bounded coefficients, input type checks and finite query records prevent the optimizer from becoming an unreviewed search process.

The independent SMT result applies to **source-versus-symbolic-program semantics**, not a machine-checked proof of all input values through the neural gate interpreter itself. That gate interpreter has separate finite Wasmtime state witnesses from v49.

## The constructive decompilation demonstration

The first experiment includes five different original canonical WASM source programs. One includes a quadratic computation expressed through several native WASM instructions, one deliberately computes identity with a redundant addition by zero, and others test signed predicates. The extractor gets no source code, only outputs from the v49 neural machine.

A full-width signed-i32 holdout set checks the recovered code in independent Wasmtime, the original source WASM, the source typed interpreter and v49 neural execution. The report retains source/target canonical WASM bytes, complete candidate histories, counterexamples, cryptographic identifiers, oracle queries and independent tool versions.

A separate cubic counterexample is deliberately outside the grammar and **must be rejected**, rather than spuriously reported as successfully decompiled.

## Why "arbitrary decompilation" is impossible as unique historical recovery

Two different programs can implement the same function over all possible inputs. For example,

```
local.get 0
end
```

and

```
local.get 0
i32.const 0
i32.add
end
```

have different source bytes, yet identical observable i32 behavior. v48 established this with Z3. No analysis of external behavior alone can distinguish which original source was historically used; nonidentifiability is inherent.

For arbitrary neural systems, internal activation bases can be permuted, rescaled, rotated where the architecture allows, or changed through training while retaining behavior. A reliable decompiler should aim for a **minimal equivalent, typed, explanatory mechanism under a declared observation/intervention contract**, and explicitly report multiple acceptable mechanisms.

v50 therefore claims **behavioral synthesis in a bounded i32 grammar**, not unique historical source recovery or arbitrary unknown-weight decompilation. True unknown-weight mechanism extraction requires blinded weight-space circuit recovery, activation interventions, graph decomposition, learned representation bases, and generalization tests across independent checkpoints.

## Next compounding gates

1. Extend the grammar to bitwise Boolean functions, shifts, modular comparisons, bounded piecewise functions and straight-line arithmetic combinations, with a cost-aware CEGIS search and Z3 counterexamples.
2. Bring real original WASM **structured block/loop/branch semantics** and memory/traps into the typed machine before treating corresponding behaviors as supported by neural execution.
3. Extract the fixed v49 operator gate topology **from weight tensors without its compiler manifest**, with active probes and gauge/permutation invariance checks. Do not pretend the current data-token disassembler is unknown-weight decompilation.
4. Build a hypothesis registry that stores explicit source, input/output/side-effect contracts, minimality cost, independent solver proof, model provenance, negative controls, and unresolved ambiguity.
5. For unknown pretrained networks, use CEGIS over candidate symbolic/KAN subcircuits with intervention oracles in the **original model**, and test separate checkpoints. Do not equate fitting logits with uncovering internal causal structure.
6. Track the official [WASM Core 3.0 specification](https://www.w3.org/TR/wasm-core-2/) and [WebAssembly/spec test suite](https://github.com/WebAssembly/spec) by semantic feature and supported effects, not an inflated percentage based on opcode counts.

## Reproduction

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_blinded_behavioral_synthesis_v50.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/probe_blinded_neural_synthesis_v50.py \
    --output /tmp/blinded-neural-cegis-v50.json
```

The pinned CI workflow separately installs Wasmtime and Z3; the original raw witness and independent CI-origin manifest must be archived before this experimental checkpoint is promoted. **This is research on a stacked draft PR and does not modify `main`.**
