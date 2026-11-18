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
