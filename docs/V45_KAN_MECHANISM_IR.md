# v45 — Typed KAN / MultKAN basis graphs for constructive mechanism research

**Research status:** draft compiler infrastructure, not a replacement for pykan, a complete Kolmogorov-Arnold representation-theorem implementation, or evidence that all KAN mechanisms are interpretable.

## Scientific motivation and claim boundaries

KANs replace ordinary scalar weights with trainable **univariate edge functions**. For a layer, the computational form is:

```
y_j = Σ_i φ_{ji}(x_i)
```

This is an attractive language for VisuaML because each edge's function can be inspected, fit, perturbed, symbolically represented, compiled independently, and reviewed with explicit domain and error contracts.

But the Kolmogorov–Arnold representation theorem does **not** imply that arbitrary pretrained models decompose uniquely into meaningful edges, that a KAN learned from data is mechanistically faithful, or that KANs are universally faster or more accurate. The original authors themselves emphasize small scientific-model settings; both theory and independent empirical assessments advise caution about efficiency and interpretability claims.

VisuaML v45 therefore implements a **separate typed research dialect** whose purpose is to make function-level hypotheses executable and falsifiable. It does not silently label ordinary ONNX/Fx dense layers as KANs or infer causal semantics from a fitted curve.

## Five admitted univariate edge basis families

Each edge is an immutable, typed `KANEdge` carrying source index, family, finite float64 coefficients, bounded input domain, and optional knots or scale. Only the following bounded forms are admitted:

| Family | Mathematical computation | Intended experiment |
| --- | --- | --- |
| **Power** | Σ a_k x^k | Exact-polynomial construction and v43 neural-symbol re-encoding |
| **Chebyshev** | Σ a_k T_k((2x−lo−hi)/(hi−lo)) | Smooth/oscillatory polynomial laws with numerically better basis conditioning |
| **Fourier** | a₀+Σ[a_k cos(2πk(x−lo)/period)+b_k sin(…)] | Periodic candidate mechanisms; no automatic polynomial certificate |
| **Gaussian/RBF** | Σ a_k exp(−((x−c_k)/σ)²) | FastKAN-inspired localized features, not a byte-compatible FastKAN import |
| **Linear spline (degree-one B-spline)** | Continuous linear interpolation of ordered knots and coefficients | Inspectable piecewise-local functions; **not** pykan's adaptive cubic B-splines |

All edge functions have explicit finite input domains and reject unsupported extrapolation. Bounded numerical input shapes, values, graph depth/width, basis length, and row counts are checked before execution.

The `KANNode` supports **sum** (ordinary edge-based KAN) and **product** (restricted MultKAN-style interaction) aggregators, with explicit additive node bias. Two stages can represent x*y by first transmitting each input through a univariate edge, then multiplying the results. This is a typed product mechanism, not a full reimplementation of pykan's MultKAN training and pruning API.

## What is actually implemented

- `visuaml/compiler_lab/typed_kan_v45.py`: immutable typed KAN/MultKAN graph; shape/basis/knots/coefficient guards; independent scalar and vectorized NumPy interpreters; symbolic extraction for **power + Chebyshev-only** graphs via exact rational arithmetic on the *actual binary64 coefficients* and SymPy; explicit importer from certified v43 polynomials.
- `visuaml/compiler_lab/torch_kan_v45.py`: independent PyTorch float64 oracle using real `nn.Parameter` coefficients, with optional gradient updates. Trained coefficients are not silently treated as the original immutable typed graph.
- `visuaml/compiler_lab/native_kan_codegen_v45.py`: **one graph-generic** C++17 source emitter, with local checked input buffer ABI, all admitted basis families, independent native compilation and output/source/binary receipts. No architecture-specific model name switch or arbitrary source-string interpolation.
- `scripts/probe_typed_kan_v45.py`: seven independent synthetic witnesses (five basis families, a two-layer product graph, and the original v41→v43 recovered quadratic imported as a KAN edge) executed by scalar Python, NumPy, PyTorch and native C++, with explicit numerical tolerances.
- `tests/test_typed_kan_v45.py`: adversarial malformed edge/graph tests, symbolic regression cross-checks, native-vs-framework parity, gradient-flow test, MultKAN interaction and provenance-bound v43 integration.

## Cross-representation example: v41→v43→v44→v45

The original v41 weights describe a bounded quadratic by 255 ordinary ReLU hinges. v43 recovered a symbolic polynomial from those actual weights without source-program metadata. v44 constructed an alternative two-ReGLU-gate representation. v45 now expresses the **same recovered symbolic function** as a single power-basis KAN edge:

```text
φ(x) = 7 − 3x + x²
```

The basis coefficients are exact integers and are independently evaluated and compiled into C++17 on the declared [-128,127] input interval. This shows **a third representational basis** for the same function, not a claim that an arbitrary learned KAN can be decompiled or that all three architectures have comparable runtime costs.

## Mechanistic interpretability: validation pipeline

A hypothetical learned function `φ` does not graduate into the verified primitive registry because its plot looks like a polynomial. It must first pass the following acceptance protocol:

1. **Representation identity:** exact source model/edge/activation pipeline and coefficient provenance; compensate or reject known gauge and permutation symmetries.
2. **Functional extraction:** attempt a degree-bounded symbolic, spline, RBF or Fourier explanation with a predetermined domain and error budget. Preserve counterexamples rather than clipping inputs or hiding residuals.
3. **Causal replacement:** replace precisely the hypothesized edge or module in the complete pretrained model; compare downstream behavior with a control, including matched parameter/compute-budget random controls.
4. **Generalization:** hold out both inputs and *genuinely independent* pretrained checkpoints/architectures before claiming transfer. The permanently archived v42 full-model rank-screening negative result demonstrates why weight-space similarity alone is not enough.
5. **Compositional proof:** report additive/product effects, numerical precision and error propagation; where possible, use independent bit-vector/real-arithmetic SMT, but keep solver-checked symbolic semantics separate from floating-point/neural executions.
6. **Execution economics:** report native CPU/GPU memory and latency under fair matched workloads. A KAN edge can be easier to inspect but more expensive to evaluate than a fused dense linear kernel.

## Relationship to the KAN ecosystem

- [Original pykan](https://github.com/KindXiaoming/pykan) and [KAN 2.0 / KANs Meet Science](https://journals.aps.org/prx/abstract/10.1103/4t7t-v19l): edge functions, multiplication nodes, sparsification, feature and modular structure discovery, scientific laws.
- [EfficientKAN](https://github.com/Blealtan/efficient-kan): basis expansion and matrix multiplication without expanding every [batch,out,in] activation; demonstrates an important compiler lowering strategy and a regularizer interpretation tradeoff.
- [FastKAN](https://github.com/ZiyaoLi/fast-kan): Gaussian RBF approximation replacing B-spline basis, with published author-reported forward-pass speed improvements. v45 adopts the *family of RBF edge functions*, not their full implementation/training conventions.
- [Research review (2026)](https://doi.org/10.1016/j.cosrev.2026.100991): basis-family choices, applicability conditions, memory/latency costs and cautions against universal performance assertions.
- [Critical assessment](https://doi.org/10.21203/rs.3.rs-7063327/v1): independent claims about computational overhead and potential mismatch between theoretical guarantees and empirical applicability. The results are contested and task-dependent.

## Explicitly deferred capabilities

These are **not** silently claimed to exist: adaptive cubic B-spline knot refinement, full pykan state_dict import, multidimensional tensor/channel feature maps, generic ONNX graph ingestion into a KAN representation, learned-edge extraction from real pretrained models, attribution patches, attention-KAN/ConvKAN variants, verified cross-checkpoint transfer, GPU-native fused kernels, or any native speedup relative to MLPs.

Their correct order is: certify this typed dialect and generic native execution; add cubic/adaptive B-spline contracts with independent pykan comparison; implement causal edge transplant into a real model; compare multiple pretrained checkpoints with held-out interventions; then optimize kernels and broaden adapters. **Interpretability is an experimental claim about function and causation, not a class name.**

## Reproduction

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q \
  visuaml-client/backend/tests/test_typed_kan_v45.py
PYTHONPATH=visuaml-client/backend \
  python visuaml-client/backend/scripts/probe_typed_kan_v45.py \
    --output /tmp/typed-kan-multibasis-v45.json
```

The dedicated workflow `.github/workflows/typed-kan-v45.yml` pins dependencies and archives full raw input and native source/binary identifiers and scalar/NumPy/Torch/C++ numerical witnesses. Full repository regression must be green before this can be reviewed. A passing experiment still needs permanent raw artifact plus source-bound origin manifest before the record is considered archival.
