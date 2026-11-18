# v42 — Mine pretrained weights for *candidates*, not imaginary circuits

**Scientific claim boundary:** static decompositions can propose computational primitives; only actual causal interventions, independent numerical oracles, and held-out tests can establish whether a candidate describes a behavior relevant to the original model. Neither SVD reconstruction accuracy nor synthetic input agreement proves a universal circuit, interpretability, image classification accuracy, or transfer across architectures.

## Hypothesis

Instead of constructing every network from a blank slate, use pretrained checkpoints as empirical discovery substrates. Search for structure that can be described compactly, manipulate it in a complete model, test independent counterexamples and ultimately graduate validated mechanisms into a typed, reusable standard library. This is **not** a claim that arbitrary learned models can be analytically decompiled.

Three evidence levels must not be conflated:

- **Algebraic:** exact or bounded numerical matrix identities, shape constraints and invertible gauge transformations.
- **Causal-functional:** a specific intervention changes/retains measurable full-model outputs under named inputs.
- **Transferable:** a candidate survives unseen inputs, checkpoints, architectures and domains under preregistered thresholds and preferably intervention tests.

All claims require independent oracles, negative controls and recorded failures. A pipeline that declares feature names merely because a singular vector looks Gabor-like is unsound.

## First concrete experiment: true MNIST-8 final projection

Use the exact pretrained ONNX Model Zoo MNIST-8 model already pinned by v37:
- Immutable upstream revision: `a19f9a8c2333de1df9b03f10f5739f468b699a1a`
- Complete ONNX SHA-256: `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`
- Exact reviewed pretrained weight: `Parameter193`, (16,4,4,10), interpreted by the original graph as a (256,10) final projection.

Extract deterministic truncated SVD proposals at ranks 2, 4, 8, 10. Reinsert each candidate into the *original complete model* without changing other initializers. The rank-10 path is a numerical full-rank reconstruction control. A rank-4 input-coordinate-scramble control retains the same singular spectrum as the rank-4 approximation but deliberately changes the represented function.

For each changed model:
1. Validate complete ONNX graph; hash the actual new full model bytes and candidate parameter bytes.
2. Execute **both** the independently coded scalar NumPy ONNX interpreter and ONNX Runtime CPU without graph optimizations, using exact input and output names.
3. Before considering any candidate, enforce the pre-existing v37 tolerance (absolute/relative 1e-4) between the two independent oracles.
4. Compare original and intervened final logits, reporting relative L2 change, maximum absolute logit change and top-1 identity agreement. These are not labeled accuracies.
5. Use five named **discovery** inputs, then five disjoint synthetic **held-out** inputs. Selection threshold is preregistered: all discovery cases retain top-1 and have relative logit L2 difference ≤0.05. Report held-out success or failure independently; do not tune after reading holdout results.
6. Record the negative control even if it happens to behave comparably, and never suppress failures to make a prettier story.

The intervention should falsify the naive claim that retaining a large fraction of weight spectral energy implies preserving original network behavior. Weight reconstruction error and functional error are different metrics.

## Next expansion after the first candidate survey

### 1 — Activation-aware mechanism proposal

Collect activation covariance and conditional Jacobian measurements, not just weight SVDs. Compare low-rank directions selected from weights alone against directions prioritized by actual activations. Require both SVD and randomized matched-budget baselines, and measure output sensitivity to a controlled intervention.

### 2 — Causal graph extraction

Implement feature ablation, patching, cross-layer path attribution and activation restoration; record original versus replacement-model residuals and calibrated completeness. An interpretable replacement circuit is a **hypothesis** until interventions in the original checkpoint support it. See [Anthropic's Circuit Tracing Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

### 3 — Symmetry / gauge canonicalization

Canonicalize hidden permutations, positive ReLU rescaling where the contract permits it, and basis alignment in linear subspaces before comparing features between models. The equivalence relation is function/observable dependent and cannot be assumed for LayerNorm, residual interactions or quantized variants. See [Git Re-Basin](https://arxiv.org/abs/2209.04836).

### 4 — Generalization and compatibility studies

Test candidate circuits across at least three genuinely independent trained checkpoints, not just variants of one checkpoint. Use held-out real data, controlled distribution shifts, model/seed metadata, causal interventions, and invariance-aware transfer rather than relying on similarity alone. Research demonstrates that successfully stitching models can hide significant representational differences: [Functional Alignment Can Mislead](https://proceedings.mlr.press/v267/smith25a.html), [Invariance-Aware Stitching](https://proceedings.mlr.press/v306/athanasiadis26a.html).

### 5 — Compounding constructive linker

After a candidate passes its declared gates, compile it into a versioned primitive with (a) a typed Semantic IR contract, (b) exact or bounded numerical references, (c) a source/weight provenance graph, (d) compiler backend support and (e) explicit conditions of transfer. Compose primitives with a linker that tracks output dimensions, effects, numerical error budgets and unresolved approximations. Learned glue is allowed, but recorded as a **learned component** instead of silently recategorized as analytically constructed.

### 6 — Mechanism-aware cost model

Evaluate total costs, not only inference FLOPs: discovery compute, evidence generation, compile latency, serialized bytes, actual wall-clock tokens/images per second, peak RAM and generalization across platforms. A sparse mathematical description may not be faster than a fused native kernel. Use matched hardware/inputs and rigorous cold/warm statistics.

## Reproduce

With CI-pinned Python, ONNX 1.18.0, ONNX Runtime 1.22.1 and NumPy 2.2.6:

    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_pretrained_primitive_candidates_v42.py

    PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/probe_pretrained_primitives_v42.py --model /path/to/exact/mnist-8.onnx --output /tmp/primitive-survey-v42.json

The [dedicated Actions workflow](../.github/workflows/pretrained-primitives-v42.yml) independently fetches and hashes the source model and publishes the full intervention matrix. The full CI suite remains mandatory. A green candidate survey is **not** a completed extraction/synthesis library. The original raw record must be archived with an independent CI-origin manifest before promoting permanent provenance status.

## Observed first-party CI result — original evidence preserved

The independently executed [CI run 44222189766](https://github.com/Caerii/VisuaML/actions/runs/44222189766) re-fetched the exact pinned pretrained MNIST-8 bytes and ran all candidate models through both scalar NumPy and ONNX Runtime. The original **54,727-byte JSON** is committed as [the full experiment](../benchmarks/pretrained_primitive_screening_v42.json), with a [separate CI-origin manifest](../benchmarks/pretrained_primitive_screening_v42.origin.json) binding the full record SHA-256, source commit, all relevant source-file hashes, and exact checkpoint/parameter identities. The archive is now guarded against overwrites. Its permanent integrity regression rejects reclassifying a negative result as a successful compressed model.

| Candidate rank | Weight spectral energy retained | Top-1 unchanged on discovery / 5 | Top-1 unchanged on heldout / 5 |
| --- | ---: | ---: | ---: |
| 2 of 10 | 33.75% | 3 | 3 |
| 4 of 10 | 59.11% | 4 | 1 |
| 8 of 10 | 92.33% | 5 | 3 |
| 10 of 10 (full-rank control) | 100.0% | 5 | 5 |
| rank-4 coordinate-scramble negative control | 59.11% | 1 | 0 |

**Predeclared acceptance outcome:** **No compressed candidate qualified.** Rank 10 is a valid complete-rank numerical reconstruction and explicitly **not** compression. In particular, although rank 8 retained over 92% of matrix spectral energy and kept all five discovery top-1 labels, it changed two of five held-out top-1 labels and also exceeded the separate relative-logit-error threshold. A rank-4 scrambled tensor with the *same singular spectrum* produced substantially different outputs.

These probes are deliberately synthetic, small and unlabeled. They are useful counterexamples to unwarranted inferences from singular-value energy, not evidence that rank-8 compression fails on the entire natural MNIST distribution or that any discovered feature transferred. The full record preserves per-input logit displacement and both independent implementation hashes. This is a **negative scientific result** with substantial compounding value for future candidate-review gates.
