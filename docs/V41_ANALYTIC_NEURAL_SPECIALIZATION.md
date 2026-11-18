# v41 — Exact bounded neural specialization and function recovery from weights

Scientific scope: a fixed, analytically constructed two-affine-layer ReLU network that reproduces one real validated WASM function for all 256 inputs in the declared signed 8-bit domain. This is NOT a transformer VM, general-purpose wasm interpreter, arbitrary trained model decompiler, or all-input i32 proof.

## Shared source of semantics

The source is the actual bounded one-function WASM module admitted by v40. The independent Wasmtime runtime executes the binary. Our typed state-transition interpreter independently executes the same I32Program. We compile the typed program's complete finite-domain output table into a ReLU circuit, then reconstruct that mathematical function from only its weight tensors (deliberately throwing away source-program identity).

The deliberately nonlinear example is the WASM expression f(x) = x*x - 3*x + 7. Its accepted inputs are the signed 8-bit integers -128 through 127. Results are signed i32, with no overflow in that domain.

## Analytically constructed weights

Let x_j = -128 + j and y_j = f(x_j), j=0..255. Define first slope a=y_1-y_0 and discrete second differences c_j = y_{j+1}-2y_j+y_{j-1} for j=1..254.

The exact finite-domain function is represented by:

    F(x) = y_0 + a ReLU(x + 128)
           + sum_{j=1}^{254} c_j ReLU(x - x_j)

This is a regular neural computation: W_out @ ReLU(W_in @ [x] + b_in) + b_out with 255 hidden ReLU neurons. No gradient descent, dataset fitting, or model-generated code. All weights are known-integer float64 and the absolute sum bound is less than 2^52, making integer arithmetic exact under binary64 for the admitted inputs. Every accepted integer is checked against the separate typed interpreter, and independently against actual Wasmtime.

## A restricted form of neural decompilation

The decoder is given only W_in, b_in, W_out and b_out. It does NOT read the original program, source hash, output lookup table or compilation receipt. It recognizes the known hinge-basis template, sorts neurons by bias to eliminate hidden-unit permutation ambiguity, and recovers every finite-domain output and discrete slope exactly. It refuses malformed hinge layouts, noninteger weights, nonfinite tensors and unsafe sum bounds.

This is recovery of a bounded mathematical *function* from a known representational family. It neither recovers the unique original program nor proves that the function is equivalent outside its admitted domain.

## Critical negative witness

At input x=128, the ReLU construction extrapolates linearly while the original quadratic continues with a different second difference. The numerical probe records that discrepancy as a genuine counterexample to extending the claim beyond the admitted domain.

The ability to preserve meaningful negative evidence is part of the verification contract, not a weakness to conceal.

## Reproduction

Run with the exact pinned NumPy, Wasmtime, Pytest, Ruff and package prerequisites in .github/workflows/analytic-neural-wasm-v41.yml:

    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_analytic_relu_specialization_v41.py

    PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/probe_analytic_neural_wasm_v41.py --output /tmp/neural-wasm-v41.json

The numerical JSON includes the complete analytic weight arrays, 256 independent Wasmtime/typed/ReLU/recovered output witnesses, source WASM bytes, source code digests, tensor hashes, and the explicit out-of-domain falsifier. An archive copy and CI-origin manifest must be committed before it is considered durable.

## Next research gate

Construct a *programmable* bounded neural state-transition interpreter rather than a specialized finite function. Its weights must implement state update, instruction decoding, and routing with explicit bounded-domain semantics across multiple programs. Only after independent per-instruction traces agree should a decompiler try to reconstruct the interpreter's instruction contracts without a construction manifest.

That would advance toward the fixed-weight transformer-VM approach of Percepta-Core and NarendraPatwardhan. This current experiment is a deliberately smaller, certified intermediary.
