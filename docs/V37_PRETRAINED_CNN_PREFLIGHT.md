# v37 — The second pretrained architecture starts at the byte boundary

The v36.1 source-tree correction and independent fixed-Tiny-Llama replay are
preserved unchanged. This branch begins the **distinct, graph-generated**
pretrained-model milestone in [issue #14](https://github.com/Caerii/VisuaML/issues/14).

The candidate is the ONNX Model Zoo `mnist-8.onnx` trained CNN. The initial
objective is **not** to declare pretrained CNN compilation. It is to establish
a reproducible, machine-readable source-to-operator ledger that no subsequent
lowering can silently relabel as successful inference.

## Fixed external identity

- Repository: `onnxmodelzoo/mnist-8` (the upstream model zoo's Hugging Face mirror)
- Immutable model repository revision: `a19f9a8c2333de1df9b03f10f5739f468b699a1a`
- File: `mnist-8.onnx`
- Expected whole-file SHA-256: `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`
- Expected whole-file size: **26,454 bytes**

All three source identities are verified **before** the passive ONNX parser
receives the checkpoint. The CI workflow downloads the immutable revision, not
`main`, then checks its exact SHA and byte length.

## What the inventory checks

The byte audit uses the pinned `onnx` package, rejects external data references,
unsupported initializer types, nonfinite float constants, custom domains,
control-flow and effects, repeated SSA/initializer definitions, unfounded graph
inputs and unbounded resource sizes. It checks the graph against ONNX's checker
and emits each node's operator, attributes and producer/consumer names, plus
source-bound initializer fingerprints. The `candidate_native_op` field is
**only a spelling hint**, not a native admission statement. Even `Conv` requires
separate shape, bias, padding, dilation, and memory admission before compilation.

The output's `status` remains `preflight_only`; `executed_native` and
`numerical_oracle_executed` are explicitly false. No existing roadmap acceptance
status should change because this check passes.

## Next acceptance gates

1. Archive the actual verified ONNX operator ledger from CI as a permanent
   reviewed, byte-identified manifest; include any failed preflights.
2. Extend the shared `SemanticProgram` native dialect only for precisely
   characterized needed operators and attributes; never hide model-specific
   branches inside C++ emission.
3. Admit the real checkpoint through a **data-only** importer with actual
   constant hashes and guarded geometric/ABI contracts.
4. Run generated C++ against an independent ONNX Runtime oracle on multiple
   random and boundary inputs, then test adversarial weight and graph changes.
5. Archive source, IR, compiler, binary, raw numerical differences, toolchain,
   experimental geometry and cold/warm samples. Add a second *distinct* real
   pretrained architecture before marking issue #14 accepted.

## Reproduce static stage

```bash
python -m pip install onnx==1.18.0 numpy==2.2.6 pytest==8.4.2
cd visuaml-client/backend
python -m pytest -q tests/test_pretrained_onnx_preflight_v37.py
python scripts/audit_pretrained_onnx_v37.py \
  --model /path/to/sha-verified/mnist-8.onnx \
  --sha256 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf \
  --revision a19f9a8c2333de1df9b03f10f5739f468b699a1a \
  --output /tmp/mnist-8-preflight.json
```

**Scope:** a research preflight, not inference, trained architecture
certification, ONNX Runtime correctness, or native performance evidence.

## Independently observed CI checkpoint ledger

The [first exact-byte CI audit](https://github.com/Caerii/VisuaML/actions/runs/44207675149) passed its seven adversarial tests and pinned-source download gate. Its full unmodified [8,044-byte graph record](../benchmarks/pretrained_mnist8_onnx_preflight_v37.json) and separate [origin manifest](../benchmarks/pretrained_mnist8_onnx_preflight_v37.origin.json) are now checked in. The original artifact ZIP's SHA-256 matched the digest reported by GitHub Actions. The workflow uses `GITHUB_SHA` at the PR's synthetic **merge** commit (`b5a9396636d7e011434d81dd15fbb11d7299eaf3`), whose actual parents are `main` `6fb0371c...` and PR head `d4b90100...`; this is deliberately not mislabeled as the PR head SHA.

| Observed ONNX operator | Count |
| --- | ---: |
| `Conv` | 2 |
| `Add` (broadcasted learned biases) | 3 |
| `Relu` | 2 |
| `MaxPool` | 2 |
| `Reshape` | 2 |
| `MatMul` | 1 |

The checkpoint has **eight exact named initializers**, takes `[1,1,28,28]` float32 and produces `[1,10]` float32. Both convolutions specify `auto_pad=SAME_UPPER` and 5×5 kernels; the pools have 2×2 stride-2 and 3×3 stride-3 kernels. `Add` is **broadcasting**, not v32's equal-shape addition. One weight tensor is reshaped *before* matrix multiplication; another reshapes activations. Thus the true missing contracts are static reshape, max pooling, bounded broadcasting, and ONNX `SAME_UPPER` normalization, with exact buffer and constant provenance. Recognizing `Conv` and `Add` by name alone would be scientifically misleading.

Subsequent CI regenerates the record from the verified external bytes and **compares the generated JSON byte-for-byte** with the durable original. Source or toolchain changes must record a new observation rather than quietly changing this one. The archive test binds those immutable bytes to its separately reported origin and explicitly rejects treating this preflight as numerical execution.

## v37.2 — externally released pretrained CNN independent numerical oracles

The exact source-preflight checkpoint was reused, with no change to its archived 8,044-byte audit. On [fresh GitHub Actions run 44209109664](https://github.com/Caerii/VisuaML/actions/runs/44209109664), the original **trained MNIST-8 weights** executed in two mathematically separate ways:

- **ONNX Runtime 1.22.1 CPU** runs the real checkpoint, with graph optimizations disabled.
- A **closed-world scalar NumPy interpreter** consumes the same parsed ONNX node sequence and initializers, independently implementing NCHW 5×5 convolution with `SAME_UPPER` padding, bias broadcasts, ReLU, max pooling, static reshape and rank-2 matrix multiplication. This does not use ONNX Runtime or PyTorch to compute its convolution or pooling results.

Five input witnesses (all-zero, all-one, centered impulse, checkerboard, and seeded-random image) passed the original preregistered **atol = rtol = 1e-4** elementwise acceptance. The largest observed absolute difference was **7.152557373046875e-7**. This result is *real pretrained numerical execution* with independent oracle agreement, **not** an execution of the generated C++ Semantic IR compiler. It is also not independent evidence that the trained MNIST-8 checkpoint is accurate on real MNIST labels.

The unmodified [2,972-byte numerical report](../benchmarks/pretrained_mnist8_onnx_numpy_oracles_v37.json) and [origin manifest](../benchmarks/pretrained_mnist8_onnx_numpy_oracles_v37.origin.json) are permanently committed. The original artifact ZIP SHA-256 matches GitHub's recorded digest; the actual PR merge checkout commit `fe164cd7ea006a456eb8b8b9e760afc1396e4393` was checked to have main `6fb0371...` and PR head `abb0e5b...` as parents. The source audit preserved its previous exact SHA-256. A separate archive-verifier unit test checks the numerical record bytes, origin linkage, accepted witness matrix and explicitly absent native execution claim.

Reproduction from the repository root with pinned Python dependencies plus `onnxruntime==1.22.1`:

```bash
PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/probe_pretrained_mnist8_oracles_v37.py \
  --model /path/to/pinned/mnist-8.onnx \
  --sha256 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf \
  --revision a19f9a8c2333de1df9b03f10f5739f468b699a1a \
  --output /tmp/mnist8-numerical-v37.json
```

Do **not** compare numerical report bytes across different CPUs or BLAS/compiler libraries as a reproducibility criterion; floating-point accumulation may differ. Compare bounded numerical outputs and preserve each observation separately. The historical record is immutable.

## Next acceptance barrier

The original v32 Semantic IR native emitter is still restricted to equal-shape `add`, limited per-axis dimensions, symmetric scalar convolution padding, and has no `maxpool2d` or `reshape`. For this real checkpoint, introduce precise separate contracts for ONNX broadcast add, `SAME_UPPER` convolution including even/asymmetric cases, max pool, and static reshape (both parameters and activations), with SSA proofs of element counts and signed-shape constraints before lowering. A graph-derived native implementation must take weight tensors from this pinned model and pass the same five oracle witnesses before a second pretrained compiled-family certificate can be claimed. Do not route MNIST-8 around Semantic IR or compile a hand-coded CNN.
