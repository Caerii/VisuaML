# v38 — Checked ONNX-to-Semantic IR and graph-generated native CPU replay

**Status: draft research, under CI verification.** This milestone depends on the independently reviewed v37 pretrained MNIST-8 source and two-oracle experiment. A green source audit is **not** a green C++ numerical replay.

## Why the architecture-independent path matters

The v28 native CPU decoder interpreted a fixed Tiny Llama program. The v32 work introduced a separate *graph-derived* C++ emitter but only produced synthetic residual MLP and convolution cases. v37 recovered the exact ONNX Model Zoo MNIST-8 pretrained graph and independently compared ONNX Runtime with an independently written scalar NumPy interpreter over five inputs. This patch attempts a genuinely different pretrained family through the **same** v32 graph-generated C++ pathway, never a handwritten classifier implementation.

**One typed SemanticProgram → one emitter → C++17 kernel → two independent numerical references.**

The ONNX adapter is a *closed-world* front end. It verifies exact external checkpoint SHA-256 before decoding, rejects arbitrary model code/pickle/custom domains/external tensors, creates SSA nodes and explicit float32 weight buffers, and binds source hashes and compile-time reshape constants to the SemanticProgram source digest. No name-based MNIST-8 cases are present in its lowering.

## Exact admitted shape/effect contracts

The shared native emitter continues to enforce a single pure float32 effect-free region, strict finite tensor input/output boundaries, ≤16 runtime inputs, ≤64 SSA operations, ≤65,536 elements per value, 8MiB total retained bytes, and 50M scalar multiply budget.

New *strict* operations and variants:

- **reshape:** only static shape, exact element-count preservation, C-contiguous logical layout; integer ONNX shape operands are validated compile-time constants and remain in checkpoint/graph digests, never dynamic float inputs.
- **broadcast_add:** right-aligned NumPy/ONNX broadcasting, reject unequal nonunit dimensions, and emit indices from the admitted shapes, not from user-controlled strings.
- **maxpool2d:** finite NCHW, square bounded kernel/stride, exact explicit top/left/bottom/right padding, no empty windows, no dilation, no ceil mode.
- **conv2d:** prior symmetric integer padding preserved; new explicit asymmetric padding used when ONNX SAME_UPPER is normalized. Grouped/dilated convolutions remain unadmitted.
- **matmul, relu, add:** existing v32 operations preserved, including strict equal-shape add; ONNX Add imports as the separate checked broadcast contract.

Dims may now be ≤512 per axis to accommodate the observed 256-dimensional flattened trained FC weight, while element, byte and compute limits still hold.

The compiler never emits unescaped ONNX symbol strings into C++; source uses internally generated integer slots. The native pointer ABI is process-owned, not hostile-code isolation, and the Python wrapper validates input names/shapes/dtypes/finiteness and copies buffers before invocation. All source and binary hashes are recorded per generated implementation.

## Reproduce independently

With the pinned dependencies from `.github/workflows/pretrained-onnx-native-v38.yml` and trusted system C++17 compiler:

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_native_semantic_codegen_v32.py backend/tests/test_native_fx_frontend_v32.py backend/tests/test_onnx_semantic_native_v38.py

# Download externally-published model only from immutable revision and verify:
# SHA256 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf
cd ..
PYTHONPATH=visuaml-client/backend \
  python visuaml-client/backend/scripts/probe_pretrained_mnist8_native_v38.py \
    --model /absolute/path/to/pinned/mnist-8.onnx \
    --output /tmp/mnist8-native-v38.json
```

Predefined witnesses: zeros, ones, centered impulse, checkerboard and seeded-random float32 inputs. The exact same original checkpoint must run in compiled native C++, ONNX Runtime CPU and the v37 closed-world scalar NumPy interpreter. Every output is compared elementwise using the **previously fixed** atol=rtol=1e-4, recording maxima, source and tensor hashes and the full generated C++ code. A changed tolerance would require a new report and explicit rationale.

No benchmark success, independent codegen parity or second pretrained-family claim may be made until the dedicated CI workflow passes and its original unmodified artifact is archived with the origin manifest. Even success will be finite-checkpoint, finite-input evidence, not all-input equivalence.

## Separate next research track — reversible virtual machines and WASM

See `V38_NEURAL_VM_RESEARCH.md` for an independent plan. Computer-in-transformer systems using analytical fixed weights are compilation results, not decompilation results. WASM rewriting requires its own typed verifier, index remapping, control-flow/relocation rules, differential execution and byte-level evidence. Do not count a reordered binary hash as equivalent program behavior.

## Sources

- [v37 numerical and origin manifest](../benchmarks/pretrained_mnist8_onnx_numpy_oracles_v37.json)
- [v32 graph-generated compiler](V32_NATIVE_SEMANTIC_CODEGEN.md)
- [v36.1 forensic provenance correction](V36_PRETRAINED_REPLAY.md)
- [Cross-architecture acceptance issue #14](https://github.com/Caerii/VisuaML/issues/14)
