# v46 — Internal pretrained activations, KAN edge surrogates and causal channel ablations

**Research status:** Independent numerical CI and permanent evidence are acceptance gates. This is not a claim of identifying a semantic feature, causally extracting an internal KAN edge, or validating cross-checkpoint transfer.

## Why v46 is different from v45

v45 admitted KAN/MultKAN mathematical edge functions, constructed exact polynomial DAGs from a previously recovered symbol, and fitted KAN surrogates to **final output logits** of the pinned pretrained CNN. Fitting logits provides an inspectable approximation but no information about whether the teacher has matching internal mechanisms.

v46 crosses the first necessary boundary: **instrument a real internal activation tensor of the original pretrained model and intervene on the same tensor inside the full graph**. The traced graph and every modified graph are checked independently, preserving the exact source SHA-256 and source revision.

- Pretrained checkpoint: original ONNX Model Zoo MNIST-8, SHA-256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`, revision `a19f9a8c2333de1df9b03f10f5739f468b699a1a`.
- Reviewed original node: `ReLU114_Output_0`, the second convolutional ReLU activation stage (16 channels, NCHW). The precise spatial dimensions are inferred and checked by the independent ONNX shape-inference engine.
- The original graph stays immutable. Tracing adds a typed diagnostic output, not new compute operations.
- Channel ablation renames precisely one activation output and inserts a single shape-broadcast **float32 Mul mask** restoring the original tensor name. Every other original node and downstream consumer remains intact; identity mask (all ones) must preserve logits at the original numerical tolerance.
- An independent **scalar NumPy interpreter** walks the unmodified original operators (reusing the reviewed v37 closed-world scalar kernels, not the ONNX Runtime operator kernels). It applies the channel mask at the same semantic boundary. Both stage activations and perturbed final logits must agree with ONNX Runtime independently.

## Data-derived KAN curves at the original hidden activation

For each scalar `t` in [0,1], feed `t × image` into the pinned CNN, measure the target activation tensor and reduce each of its 16 channels to its spatial mean. The data is **not** a learned semantic feature: spatial mean discards location, phase, and much of the activation structure.

Two independently seeded fixed random base images distinguish fitting a curve from transferring it across image manifolds:

- **Train input manifold, 17 coordinates:** evenly spaced `t` values for image seed 460.
- **Within-image heldout, 16 disjoint coordinates:** interleaved `t` values withheld from curve fitting.
- **Cross-image control, 16 coordinates:** the same withheld `t` values on an independently drawn image seed 461. These are out-of-manifold probes, not a representative image distribution.

Fit the five already-typed v45 edge families (linear, polynomial, piecewise-linear spline, Gaussian RBF, Fourier) **only to the 17 training-channel-means** using deterministic least squares. For each candidate, report its complete KAN edge graph, coefficient values, source-bound digest, within-image heldout activation error and cross-image activation error. Select the family using training error only, not holdout results.

These fitted edge curves represent one-dimensional descriptions of *measured activations*. They are not deduced from the CNN's actual weights, and similarity cannot establish a unique mechanistic decomposition.

## Actual channel causality

Evaluate all 16 single-channel zero masks directly in the **original full pretrained CNN** over five preregistered discovery intensity values. Choose the channel with the greatest discovery mean absolute final-logit change, and a negative-control channel with the smallest. Both selections are made without inspecting heldout effects.

Then evaluate the selected and control channels on each heldout coordinate for both the original and second image. Preserve source/modified full-checkpoint hashes, the exact channel masks, output hashes, logit movement, and top-1 identity changes. An all-one identity mask must agree numerically on every heldout input.

This is **causal evidence that modifying a specific original activation channel changes downstream outputs on those inputs**. It does not establish that the fitted KAN curve is a causal edge of the CNN. Nor do comparative output changes establish a channel's human-interpretable semantic label.

## Falsification and compounding research gates

The dangerous shortcut is to conflate a fitted channel curve with an extracted teacher mechanism. Every accepted record must explicitly distinguish:

1. Original pretrained hidden activation **observation**.
2. Independent channel **intervention** inside the original model.
3. KAN **functional surrogate** of one channel's spatial mean.
4. **Transfer** of that surrogate to a distinct image manifold, measured separately.
5. Unproven **semantic mechanism equivalence**, which requires substantially more experiments and causal circuit reconstruction.

The experiment cannot be promoted to real-model mechanistic interpretability without multiple independent real data distributions, pathway-specific intervention/patched-node comparisons, activation restoration tests and an architecture-matched control. For a distinct KAN edge to be admitted as a causal teacher mechanism, its changes would need to predict and reproduce interventions in the original checkpoint, not merely interpolate its activations.

## Reproduction

The dedicated CI workflow `.github/workflows/pretrained-causal-kan-v46.yml` pins the original checkpoint, ONNX 1.18.0, ONNX Runtime 1.22.1, NumPy 2.2.6, Pytest 8.4.2 and Ruff. The positive tests run only when the real source checkpoint is explicitly present; full repository CI skips the model-dependent cases rather than pretending they were run in that environment.

```bash
PYTHONPATH=visuaml-client/backend VISUAML_PINNED_MODEL=/path/to/exact/mnist-8.onnx \
  python -m pytest -q visuaml-client/backend/tests/test_onnx_activation_causality_v46.py

PYTHONPATH=visuaml-client/backend \
  python visuaml-client/backend/scripts/probe_pretrained_internal_kan_v46.py \
    --model /path/to/exact/mnist-8.onnx --output /tmp/internal-kan-v46.json
```

The experimental JSON includes exact training/heldout coordinates, activation means, complete fitted KAN graph coefficients, per-channel discovery effects, selected-control heldout interventions, independent scalar-vs-ONNX numerical witnesses and explicit epistemic limitations. Original raw run bytes and a separate digest/source-commit origin manifest should be permanently archived only after passing dedicated CI and full regression.

## Next experiments

The most consequential next stages are spatially resolved activation subspaces, activation patching between counterfactual image pairs, KAN edge interactions between channels rather than isolated pooled means, direct activation→output mechanistic surrogates with heldout causal intervention validation, and transfer studies across genuinely distinct trained checkpoints. This is how the KAN compiler can support **mechanistic** research rather than collecting attractive but potentially misleading mathematical plots.

## Observed results and immutable evidence (the original v46 experiment)

The original trusted-branch [independent CI run 44240406427](https://github.com/Caerii/VisuaML/actions/runs/44240406427) checked the exact model's hidden tensor `ReLU114_Output_0`, whose reviewed static shape is **[1, 16, 14, 14]**. Its full **145,852-byte original report** is checked in as [the permanent intervention/curve record](../benchmarks/pretrained_internal_kan_causality_v46.json), SHA-256 `180d50d2bc1d444848050a64437860479d657427dffd8dbc42694d9be6977d00`, with a [separate source-commit/CI origin manifest](../benchmarks/pretrained_internal_kan_causality_v46.origin.json). The raw original experiment was sourced at commit `272deae161f1c780c652b419fff8133676abf2a6`, not retroactively attributed to the later documentation commit.

### Hidden-activation KAN fitting

Each basis fitted the 16 per-channel spatial-mean activation curves from **17 training coordinates**, evaluated against **16 disjoint within-image heldout coordinates**, then on the same 16 coordinates of an independently seeded second image. All errors below are mean absolute errors across channels and probe coordinates.

| Basis | Training max absolute activation error | Same-image heldout MAE | Different-image MAE |
| --- | ---: | ---: | ---: |
| Linear | 0.02559 | 0.002297 | 0.008243 |
| Degree-4 polynomial | 0.004226 | 0.000348 | 0.007368 |
| Nine-knot linear spline | 0.002537 | **0.000107** | **0.007358** |
| Nine-center Gaussian RBF | **0.001671** | 0.000184 | 0.007370 |
| Four-harmonic Fourier | 0.06641 | 0.002330 | 0.007789 |

The RBF model was selected based only on training error; the spline achieved the best same-image heldout error. The spline's different-image error is approximately **69 times** its within-image heldout error. This is direct negative evidence against treating a fitted one-dimensional activation curve as an image-invariant mechanism.

### Actual hidden-channel causal intervention

All **16** channels were masked one at a time inside the original model under five preregistered discovery input intensities. Channel **5** had the largest measured mean logit effect; channel **3** had the smallest, with zero measured effect in those discovery probes. Both were selected before consulting heldout inputs.

| Mean absolute change across final logits | Channel 5 (selected) | Channel 3 (control) |
| --- | ---: | ---: |
| Five discovery intensity probes | 0.16385 | 0 |
| 16 heldout intensities, original image | 0.13967 | 0 |
| 16 heldout intensities, different image | 0.09270 | 0.00411 |

**Neither intervention changed the top predicted class on any of the 32 heldout probes.** This illustrates another critical distinction: a channel can be causally consequential for output logits without changing the final argmax decision on a given sample.

The identity-mask control preserved original logits on all heldout inputs. The independent scalar oracle matched both the traced internal tensor and the perturbed final logits on three separately named probes; the maximum absolute difference in the original same-image full-scale case was below 1.1×10⁻⁶. The archived report includes both implementations' output hashes, modified-model digests and exact input hashes.

The evidence establishes a **real, tested causal activation-channel effect** in the original pretrained network. It does not establish that the KAN-fitted channel-mean curves *caused* those output changes, represent equivalent internal teacher modules, generalize to other checkpoints, or capture named semantic features.
