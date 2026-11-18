# v45 — Typed KAN mechanisms, basis extensions and compositional symbolic evidence

**Status:** research branch. No pretrained interpretability, trained KAN, published speedup, or full Kolmogorov–Arnold theorem verification is claimed by this implementation.

## Motivation and relation to existing VisuaML work

The [original KAN approach](https://github.com/KindXiaoming/pykan) places learned **univariate functions on edges** rather than fixed activations only at nodes. A KAN layer commonly computes sums of incoming edge functions. Later extensions—including [MultKAN and kanpiler from KANs Meet Science](https://doi.org/10.1103/4t7t-v19l)—make products and symbolic formula compilation important for scientific discovery.

This directly complements v41–v44:
- v41 analytically constructed bounded neural functions, no training.
- v43 reconstructed a quadratic symbolic mechanism from **neural weights only** and separately used Z3 to prove original-vs-recovered **symbolic WASM** all-i32 equivalence.
- v44 synthesized the same rule in a different neural activation family using two exact-real ReGLU gates.
- **v45 adds an explicit language for composing individual univariate functions**, permitting local intervention and polynomial identity recovery without treating every edge function as an unexplained dense tensor.

The repository does **not** attempt to import all of pykan's checkpoint/version variants, claim general neural decompilation, or pretend a human-readable univariate curve is automatically causally meaningful. The v42 negative pretrained-SVD result is the reason for strict evidence separation.

## Implemented typed IR

`KANProgram` defines bounded named inputs, a DAG of nodes, and named outputs. Each `KANNode` combines incoming **edge functions** via `sum` (classic KAN) or `product` (MultKAN-style). The edge dictionary is interpreted exclusively as **univariate functions of a previously computed source**:

| Edge basis | Actual admitted parametrization | Inference / proof boundary |
|---|---|---|
| Polynomial | Monomial coefficients, degree ≤4 | Exact rational symbolic composition for polynomial-only DAGs, limited degree/graph size |
| Linear spline | Strictly increasing knots and tabulated values | Continuous piecewise-linear interpolation, **constant** left/right extrapolation; finite numerical witnesses |
| Gaussian RBF | Amplitudes, centers and positive widths | Gaussian radial basis, finite numerical witnesses |
| Fourier | Constant, cosine/sine harmonic pairs and bounded fundamental frequency | Periodic Fourier functions, finite numerical witnesses |

Limits: ≤8 inputs, ≤64 DAG nodes, ≤128 unique named edges, ≤8 inputs per node, finite parameters and activations, static declared input domain. The interpreter rejects cycles/use-before-definition, duplicate variables, unsupported operators, malformed knots, nonpositive RBF widths, invalid Fourier shapes, nonfinite values, wrong batch axes, and outputs outside the declared finite magnitude budget.

Two implementations must agree: an independent **scalar Python/math interpreter** and a **NumPy batched evaluator**. The exact rational polynomial normalizer computes the symbolic coefficient vector by composing polynomial edge maps through KAN sums and products. For any nonpolynomial edge it fails closed rather than generating invented exact symbolic identities. The exact identity is in rational arithmetic over the *declared decimal coefficients*, not a global floating-point bitwise theorem.

## First complete end-to-end experiment

A v45 front end receives the existing **v43 recovered polynomial** `x² − 3x + 7`; it receives no source WASM and no original archived v41 neural weights. It builds a generic graph:

```
x ─ identity(x) ┐
                PRODUCT -> square
x ─ identity(x) ┘

square ─ φ(z)=z ─┐
x ─ φ(x)=-3x ────┼─ SUM -> result
x ─ φ(x)=7 ──────┘
```

The graph is re-examined from its actual node and edge values to derive `[7,-3,1]` in exact rational symbolic arithmetic. It must not simply copy the original symbolic coefficient record. The complete original v41 archived analytic ReLU network, original WASM executed by Wasmtime, independently emitted symbolic WASM, and new KAN scalar/vector evaluators must agree over **all 256 signed-i8 inputs**. A separate SMT solver result applies to the original **symbolic** WASM and recovered v43 polynomial for all 2³² i32 values, not to the KAN graph outside its declared domain.

The old v41 out-of-domain counterexample at x=128 is kept in the report; v45 does **not** assert KAN semantics outside its own admitted interval.

The compiler also supports **edge ablations** by substituting one edge's function with zero, preserving the graph's source/fanin structure. A five-point controlled synthetic intervention records edge-wise maximum/mean absolute output displacement and the hash of the altered graph. This is a meaningful graph-local mechanism test, but not causal evidence about an unspecified pretrained model.

## Sources informing architecture choices

- Liu et al., [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756) (2024), [pykan](https://github.com/KindXiaoming/pykan)
- Liu et al., [Kolmogorov-Arnold Networks Meet Science](https://doi.org/10.1103/4t7t-v19l) (2025), multiplication nodes, symbolic-to-KAN compilation and scientific modularity
- [EfficientKAN](https://github.com/Blealtan/efficient-kan), basis expansion followed by matrix multiplication instead of naive per-edge activation tensor materialization
- Hoever & Schiele, [KANLib](https://arxiv.org/abs/2606.17927) (2026), modular basis and efficient implementations
- [A practitioner's guide to Kolmogorov–Arnold networks](https://doi.org/10.1016/j.cosrev.2026.100991) (2026), choice of basis and practical efficiency/interpretability tradeoffs

## Why KAN ≠ automatic mechanistic interpretability

An edge is explicit, but its function can still be arbitrarily complex. Reparameterizations, cancelling terms, correlated features, alternate decompositions, and unobserved input distributions all undermine unique interpretation. We must test the **causal effect of each proposed edge in the originating pretrained model**, with intervention and held-out witnesses, before calling it an extracted mechanism. A trained KAN surrogate that reproduces logits is not a proof that the teacher uses the same internal factorization.

## Critical next layers (not yet implemented)

1. **KAN Checkpoint Frontends:** versioned, safe import/export contracts for pykan and modular EfficientKAN/KANLib subsets. Reject unknown architecture metadata, arbitrary pickle files, and unsupported PyTorch Python class execution. Import *only* known tensor/state-dict formats and bind provenance.
2. **Analytic lowering:** compile polynomial basis expansions into common matrix kernels and fused CPU/C++17, then benchmark cold/warm throughput and real memory. Extend to robust spline, Gaussian, Fourier implementations with error bounds, not unsupported performance claims.
3. **Activation-conditioned extraction from pretrained networks:** compute a teacher response under controlled paths; fit low-capacity KAN candidate edge functions with a train/heldout split and intervention probes in the **original** trained model, using mismatched-control fits to expose ambiguity.
4. **MultKAN higher-order couplings:** allow named typed product/sum nodes with exact symbolic identities and verify the effects of nonlinear interactions. Compare against plain KAN and MLP under matched wall-clock and parameter budgets.
5. **Mechanistic library:** gate admission through reproducible source identities, causal interventions, independent oracles, held-out distribution tests, solver proof boundaries, and a clear label of analytical, fitted or learned coefficients.
6. **Scientific priors:** monotone/symmetric/conserved-function constraints, polynomial/Chebyshev, wavelet, rational and active-subspace basis families with explicit domain guards.

The goal is a **composable mathematical edge-function language**, not an indiscriminate KAN benchmark zoo.

## Independent reproduction

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_kan_mechanism_ir_v45.py \
  visuaml-client/backend/tests/test_kan_symbolic_frontend_v45.py

PYTHONPATH=visuaml-client/backend python \
  visuaml-client/backend/scripts/probe_kan_mechanisms_v45.py \
  --output /tmp/kan-mechanisms-v45.json
```

The dedicated workflow pins NumPy/Wasmtime/Z3/Pytest/Ruff and replays the original v41 source-bound weight archive. A green run proves only that the declared fixtures and evidence gates passed. Full repository regression, permanent original raw artifact archiving and independent integrity checks remain mandatory before the milestone is complete.

## Pretrained teacher response and fitted KAN controls (second, distinct experiment)

To connect the edge-function substrate to a *real trained network*, the v45 workflow also recovers and hashes the **same external pretrained MNIST-8 ONNX checkpoint** used by v37–v42, independently executing it with ONNX Runtime CPU. It evaluates 65 scalar perturbations of **one fixed, seeded 28×28 image**. The scalar parameter `t` scales the image from zero toward its full intensity over [0,1].

Thirty-three coordinates are used for **least-squares coefficient fitting** and 32 strictly disjoint interleaved coordinates for **held-out evaluation**. The selected family is determined by training error only. Each family becomes an actual typed KAN program with ten independent output edges—one for each teacher logit—then is evaluated via the separate graph evaluator:

- `linear`: degree-one monomial baseline
- `polynomial`: degree-four monomial basis
- `linear_spline`: nine fixed knots and linearly interpolated values
- `gaussian_rbf`: nine Gaussian centers and positive fixed widths
- `fourier`: four harmonics at fundamental period one

The fitted KAN parameters are **DATA-DERIVED**. This is not a weight-only recovered neural mechanism or an analytically compiled teacher model. It is a controlled one-dimensional functional surrogate, useful for comparing representations under equal input access. Independent scalar NumPy ONNX interpretation checks three teacher coordinates against ONNX Runtime under the source-bound v37 tolerance, and the whole record carries original model/input hashes, actual KAN graphs, their coefficients, training and heldout errors, and the selected family.

The resulting evidence cannot establish that any KAN edge exists in the teacher's internal network, that a KAN generalizes to real handwritten digits, or that a KAN is faster/better than a matched MLP in production. It *can* identify where different explicit basis families succeed or fail at approximating a measured nonlinear teacher response, without passing the holdout samples into coefficient construction.

This distinction is fundamental. An interpretable *surrogate* provides an inspectable approximation; mechanistic interpretability needs controlled interventions in the **original checkpoint** and tests across unobserved contexts. The v45 surrogate is a precursor to those experiments, not their conclusion.
