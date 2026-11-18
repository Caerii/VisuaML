# v44 — Constructing a new neural architecture from a recovered symbolic mechanism

**Scope:** One analytically constructed two-ReGLU-gate quadratic circuit. This is a *different activation family*, not an inference speedup or general trained-model decompiler.

## From a 255-ReLU representation to a two-gate construction

v41 analytically constructed a 255-hidden-neuron **ordinary ReLU** hinge network whose finite-domain behavior equals a signed i32 quadratic. v43 extracted the compact symbolic identity P(x)=x²−3x+7 from the original v41 weight tensors, without exposing the old WASM source to the extractor. A separate Z3 bit-vector check established that the newly generated symbolic program and the original typed WASM agreed for all 2³² i32 inputs under the admitted pure modular arithmetic semantics—**not** that the old neural network did.

v44 now builds a **new kind of network** from the recovered mathematical rule, using two ReGLU-style gated units and a linear skip:

```
h_plus(x)  = ReLU( x) * ( x)
h_minus(x) = ReLU(-x) * (-x)
output(x)  = a*h_plus(x) + a*h_minus(x) + b*x + c
```

With the recovered coefficients (a,b,c)=(1,-3,7), this is exactly P(x) **over exact real arithmetic**.

The equality follows from the two sign cases: for x≥0, h_plus=x² and h_minus=0; for x<0, h_plus=0 and h_minus=x². The independent v44 SMT test encodes both sign cases in Z3's exact Real theory, insists on UNSAT for each possible deviation, and deliberately changes one gate sign to obtain SAT.

This is not backpropagation or training. All gate/readout weights are set analytically. Critically, a ReGLU gate multiplies two activation-dependent values, whereas a normal ReLU neuron only performs a linear affine threshold. **The 255-to-2 unit-count reduction is not a same-operation-cost comparison**. No runtime speedup is claimed.

## Actual serialized parameters

| Object | Architecture | Hidden units | Stored scalar coefficients |
| --- | --- | ---: | ---: |
| v41 original | Scalar two-affine-layer ReLU hinge network | 255 | 766 |
| v44 constructed | Two separate ReGLU-gated units plus linear skip | 2 | 10 |

The new circuit's tensors are explicitly retained: gate weights [1,-1], value weights [1,-1], two zero gate biases, two equal output weights [a,a], and scalar linear skip b and bias c. Its raw binary layout can be permuted between the two gates, and the inverse extractor recognizes the same symbolic function without a source manifest. All weights and the transformed raw arrays have content hashes.

## Numerical exactness and safety

Float64 is not mathematical exact arithmetic for arbitrary magnitudes. We reject malformed types, NaNs, nonintegral coefficients, invalid gate pairs, asymmetric output channels, wrong polynomial hashes and all inputs whose upper bound on integer products/partial sums approaches 2⁵². The bounded i8 domain has exact integer results in float64 for this circuit. Independent tests also check several additional safe integer values outside that domain.

The old v41 network **does not** regain generalization by decree: at x=128 it gives 16005 instead of the source program's 16007. The **new** ReGLU network computes 16007 at x=128, because it encodes the quadratic's real-algebraic identity rather than a finite-domain piecewise-linear extrapolation.

The v44 gating circuit is still not claimed to implement i32 modular wraparound for all signed 32-bit values. Only the separate reconstructed conventional symbolic WASM has the all-i32 SMT certificate. For large x, binary64 precision and WASM modular semantics diverge.

## Experiment and falsification protocol

The dedicated v44 workflow starts from the *original immutable v41 weight archive*, extracts the v43 symbolic rule, constructs the two-gate circuit, and recovers the same rule from the new circuit with provenance metadata absent. Only afterward does it load the original WASM to compare 256 in-domain inputs plus seven extended-input probes through the independent Wasmtime engine. It also checks the old x=128 neural counterexample and confirms that the new circuit does not share that failure.

The workflow includes negative tests for corrupted sign weights, invalid channel permutations, nonzero gate biases, nonfinite/float32 tensors and numeric overflow. CI must separately pass full repository regression.

Reproduce through `.github/workflows/constructed-reglu-v44.yml`:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_constructed_reglu_quadratic_v44.py
PYTHONPATH=visuaml-client/backend \
  python visuaml-client/backend/scripts/probe_constructed_reglu_v44.py \
    --output /tmp/constructed-gated-v44.json
```

Full raw weights, source digests, Z3 outcomes and Wasmtime witnesses must be archived separately with CI-origin hashes before the milestone is considered permanent. This archive must preserve the true claim boundaries rather than silently changing the historical v43 proof.

## Why this compounds

v43 was **recover a mathematical rule from weights, then compile into WASM/C++17**. v44 is **compile that recovered rule into a new, analytically more compact neural architecture**.

That creates the first end-to-end example of bidirectional representation engineering in VisuaML:

```
Original typed WASM
  → analytically constructed 255-ReLU weights (v41)
  → extract symbolic quadratic from weights (v43)
  → independently check against WASM with all-i32 SMT (v43)
  → synthesize different two-ReGLU-gate architecture (v44)
  → independently recover same symbolic rule from new weights (v44)
```

The next challenge is **generalizing this compiler**, not announcing universal construction: discover and validate other identities (low-degree polynomials, comparisons, bounded arithmetic, routing, state changes), implement typed capability and precision contracts for ReLU/ReGLU/attention primitives, synthesize mixtures of modules, and challenge them with adversarial inputs and independent trusted reference implementations.

The key scientific insight is that a verified *symbolic intermediate* permits a change in computational basis and architecture. It can reveal constructions that a weight-only rank reduction would never discover. That is precisely why v42's negative pretrained SVD result is worth preserving.
