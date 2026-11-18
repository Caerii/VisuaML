# v43 — Neural weight-to-symbolic recovery, cross-compilation, and all-i32 SMT verification

**Status:** Research milestone, first-party CI required. Not an arbitrary trained-network decompiler, programmable neural VM, or machine-checked all-input proof of a neural circuit.

## Scientific question

Can we recover a compact, interpretable symbolic computation from analytically constructed neural weights **without inspecting the original program**, and then independently verify the recovered computation in native C++, real WebAssembly, and an SMT bit-vector solver?

The input to the v43 extractor is ONLY the original immutable v41 neural weight tensor bundle, whose SHA-256 is `035a85ab65c9ed395e3dcb0824899609a14e20563113f0c6370731efe0a0675e`. We do not pass the original source WASM, compiler receipt, output lookup table, or provenance manifest to the extraction function. The original WASM is loaded *after extraction* for separate independent verification.

## Recovering a polynomial with a stated hypothesis class

v41 already recognizes a restricted 255-neuron two-affine-layer ReLU hinge basis and reconstructs its 256-output function for each integer x in [-128,127]. This does not decompile arbitrary black-box neural networks. v43 adds a **different, more compact hypothesis class**: integer polynomials of degree ≤2.

Let `y_i` denote the value recovered from the weights at x = -128 + i. The candidate is inferred from `y_0,y_1,y_2`:

```
a = (y2 - 2*y1 + y0) / 2
b = (y1 - y0) - a*(2*(-128)+1)
c = y0 - a*(-128)^2 - b*(-128)
P(x) = a*x^2 + b*x + c
```

We reject nonintegral coefficients, noncanonical neural basis/tensor geometries, coefficients outside signed i32, and **any mismatch at the other 253 domain points**. A high-index single-hinge perturbation that affects only x=127 MUST defeat the polynomial hypothesis despite fitting the first three points. Hidden-neuron order must not affect extraction: the v41 decoder first canonicalizes the hinge basis by bias.

For the original v41 weight archive, this derives **P(x)=x²−3x+7**. The initial neural circuit represents that function exactly on the admitted 256 integer inputs; the symbolic expression is much smaller than the original weight representation.

## Reverse compilation

From the recovered typed polynomial, independent emitters generate:

- **WASM i32 program** through the existing v40 `I32Program` and strict canonical WASM encoder. The function computes x*x*a + x*b + c under well-defined wrapping 32-bit arithmetic.
- **Native C++17** with the same C ABI, strictly integer-typed coefficients, unsigned 32-bit modular intermediate operations and a defined bit-pattern conversion using `memcpy` on the checked target. Avoid signed-overflow undefined behavior.
- A deterministic symbolic-expression string and SHA-256 certificate binding coefficients, recovered function identity and domain.

These are alternative *conventional backends* for the **recovered rule**. No model-supplied strings are copied into executable C++ syntax.

## SMT theorem, and its precise boundary

A separate Z3 bitvector interpreter symbolically executes every admitted opcode of the *original source WASM* as a 32-bit bitvector. It then poses:

```
exists x: original_wasm_i32(x) != recovered_polynomial_i32(x)
```

If Z3 returns UNSAT, then **under the encoded bounded straight-line WASM semantics**, no 32-bit input distinguishes the original symbolic program and recovered symbolic polynomial. If Z3 returns SAT, record a concrete counterexample; if UNKNOWN, treat the experiment as failed/inconclusive.

An intentionally wrong candidate with coefficient b=-2 MUST produce SAT and a witness.

This is **SMT-solver-checked equivalence**, not a Lean/Coq proof object. It is also **not a proof that the neural network implements the original program for all i32 inputs**. That latter claim is false: the v41 weight representation extrapolates linearly beyond x=127. The original counterexample at x=128, where the neural network returns 16,005 while WASM returns 16,007, remains preserved and is retested.

## External replay and durable evidence

The dedicated GitHub workflow installs pinned versions of NumPy, Z3, Wasmtime, and C++17 tooling. It checks all 256 admitted input values in four distinct forms: original WASM in Wasmtime, regenerated WASM in Wasmtime, generated C++17 shared library, and the detached neural weight circuit. Beyond that certified neural domain, it adds 70 boundary/random 32-bit inputs to check the two WASM implementations, the typed interpreter, and C++17 against the all-i32 SMT result. A full transcript must record both compiled binary and source hashes, all source and target raw WASM bytes, complete C++17 source, source-file identities, Z3 theorem outcome and falsifying negative-control witness.

The original raw CI artifact plus a separate digest/source-commit origin manifest must be archived permanently before the research milestone is considered complete. Historical artifacts may never be rewritten to match new source.

## Research constraints and next milestones

The extractor explicitly searches one hypothesis class (degree≤2 integer polynomial). It does not discover a general program hidden inside an arbitrary learned transformer, does not recover a uniquely original source, and does not imply that a polynomial works outside the neural domain unless separately proved against the source symbolic semantics.

The next high-leverage directions, after independent verification, are:

1. Extend the symbolic hypothesis search to piecewise-linear/rational integer functions, small finite-state recurrences, and algebraic circuits, with per-family falsification gates.
2. Canonicalize known reparameterization symmetries so equivalent weight layouts share a function identity.
3. Make an honest **Verified Primitive Registry**: only accepted candidates receive typed semantics, source-bound evidence, domain and effects, independent oracles, cost models, and compiler lowering.
4. Compare extracted candidate primitives from *genuinely distinct pretrained checkpoints* under controlled interventions and held-out data. The v42 rank-8 negative compression result is the baseline warning against accepting merely low-rank structure.
5. When a reconstructed rule passes exact and causal tests, compile it into novel analytical network structures, not just back into its original hinge representation.
6. For larger stateful interpreters, verify one explicit state transition at a time before attempting full constructed transformer-VM decompilation.

Related permanent evidence: [v41 original neural weight archive](../benchmarks/analytic_neural_wasm_v41.json), [v42 pretrained negative screen](../benchmarks/pretrained_primitive_screening_v42.json).
