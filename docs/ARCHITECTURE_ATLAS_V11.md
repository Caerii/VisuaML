# VisuaML v11 — Architecture Atlas, microprobes, and exotic reference kernels

**Scope:** versioned architecture taxonomy, operator gap accounting, fixed-fixture execution probes, and restricted mathematical reference kernels. This is **not** a claim of checkpoint compatibility, full training, universal graph capture, or device execution.

## Why we built this

Model **architecture names are not compiler capabilities**. A single family such as Transformer may use dense, grouped-query or latent attention, various rotary/positional schemes, dense or sparse expert layers, multiple KV cache formats, quantization, and multiple training/decode modes. A checkpoint cannot be certified because its architecture name appears in an allowlist.

The atlas separates six evidence levels:

1. **Catalogued**: named family and required computational motifs, no executable claim.
2. **Operator candidate**: representative required op spellings and feature contracts appear available, not yet verified.
3. **Fixed microprobe**: a trusted, tiny module was traced; capture and supported AOT lowering were actually run.
4. **Real model audit**: named checkpoint configuration, actual operator census, shapes, dynamic guards, and unsupported operations recorded.
5. **Executable parity**: portable/hybrid backend execution compared against the reference with varied shapes, seeds, and outputs; training additionally compares gradients and state.
6. **Physical calibration**: hardware measurements, provenance, and separate validation. No such claim exists in v11.

Versioned snapshot: `visuaml-client/public/architecture_atlas.v1.json`; generator: `visuaml-client/backend/scripts/generate_architecture_atlas.py`; React route: `/architectures`.

## Catalog coverage

59 representative families across 10 domains: classical, language, vision, generative, audio, multimodal, graphs/3D, scientific, biological, and robotics.

The top-priority families include dense/encoder/decoder Transformers, GQA/MQA/MLA, fused attention, sparse MoE, SSM/Mamba/RWKV/hybrids, quantized Transformers, ViT/CNN, U-Net/DiT/flow, acoustic encoders, GNNs, and neural assembly calculus. Less conventional research motifs include xLSTM, KAN, Fourier neural operators, neural ODEs, modern Hopfield networks, spiking LIF, 3D Gaussian splatting, NeRF, world models, capsule networks, and reservoir computing.

**Operator-gap planning** is weighted incidence across catalog entries. It is not empirical adoption, benchmark performance, incremental gain, or implementation cost. A missing critical semantic contract may block a whole model even if most of its tensor operators are present.

## Real portable runtime expansion

Added tested, deliberately narrow NumPy-AOT contracts:

| Operator | Admitted subset | Caveat |
| --- | --- | --- |
| `sin` | NumPy elementwise | Differentiation not covered |
| `stack` | Bounded lists of tensors | Shape/dtype checks delegated to NumPy + output verification |
| `where`, `ge` | Pure selection/comparison | No GPU/autograd guarantee |
| `index_add` | Functional index addition, axis 0, 1D indices | Duplicate-index summation order may affect floating-point error |
| `avg_pool2d` | 2×2, stride 2, NCHW, zero padding | Rejects other kernels/padding |
| `interpolate_nearest` | NCHW nearest-neighbor, 2× scale | Rejects arbitrary shape, interpolation mode or alignment |
| `topk_attr` | `.indices`/`.values` of `torch.topk` | General `getattr` forbidden; top-k ties rejected as non-portable |

These extend existing primitive execution, but do not establish complete U-Net, MoE, GNN, SSM, or spiking-family support. All inputs are concretely shape/dtype-guarded.

## Fixed microarchitecture probes

Run with:

```bash
cd visuaml-client
python -m pytest -q backend/tests/test_architecture_atlas_v11.py backend/tests/test_exotic_references_v11.py
```

As tested locally on PyTorch 2.10 CPU, nine deliberately tiny fixtures executed through the independent NumPy-AOT interpreter and compared against PyTorch: MLP, residual convolution, attention-core, graph scatter, unrolled recurrence, spiking gate, U-Net-like nearest upsampling, sine feature map, and MoE top-k indices. The GRUCell fixture captures, but does **not** lower to portable NumPy-AOT. These are witness-level results, not release-wide or checkpoint-wide promises. GitHub CI determines branch-level acceptance.

## Exotic executable reference kernels, not compiled operators

`exotic_references.py` provides NumPy and Torch CPU implementations of four **restricted mathematical primitives**:

- Diagonal selective scan: `state_t = a_t * state_(t−1) + b_t * input_t`, output `c_t * state_t`; **not Mamba**.
- Stable top-k linear expert routing: selected-softmax mixture of linear experts, no capacity management or distributed dispatch; **not DeepSeek-MoE**.
- Directed additive graph scatter: includes duplicate edges and explicit degree counts; **not a complete Graph Transformer**.
- Discrete LIF: fixed-timestep membrane, reset-to-zero, optional absolute refractory interval; **not a neuromorphic device simulator**.

No arbitrary third-party model code is downloaded or executed by the coverage probes. For full checkpoints, use a dedicated sandbox with resource caps, no credentials, no writable host volumes, explicit trust policy, and independent numerical validation.

## Critical missing architecture capabilities (v12+)

1. **Sequence state**: full scan, recurrent cells, RWKV/DeltaNet, mixed cache/SSM state, checkpoint serialization, dynamic time dimensions.
2. **Generation execution**: causal/attention masks, KV and latent caches, rotary transformations, sampling loops, speculative and beam decode.
3. **Sparse MoE semantics**: routing, expert capacity, token (un)permutation, grouped GEMM, fallback, expert parallel collectives.
4. **Graph, 3D, and geometry**: scatter/gather/segment reductions, ragged graphs, equivariant SO(3)/SE(3), neighbor search, sparse 3D.
5. **Diffusion & flow**: timestep conditioning, adaptive ODE solvers, scheduler state, U-Net/DiT blocks, 3D/video layouts.
6. **Training**: reverse-mode autodiff, fused optimizers, gradient checkpointing, dropout/RNG, sharding and distributed collectives; forward-only benchmarks are insufficient.
7. **Numerical formats**: BF16, FP8, MXFP8, INT8/INT4, 1.58-bit, quantization scales, accumulators, complex FFT and interpolation.
8. **Real ecosystem adapters**: ONNX/StableHLO/MLIR, Hugging Face Transformers/Diffusers/timm/PyG metadata adapters, checkpoints in safetensors, runtime plugins (IREE/TVM/Triton), guarded remote model execution.
9. **Hardware mapping**: explicit optical operator semantics; measured A/D and D/A conversions, calibration drift, laser power, nonlinearities, interconnect and scheduling constraints.

## Acceptance rules for new architecture claims

A new `Architecture` entry is informational. Mark it *working* only with: captured real model identity and commit, weights/license, capture mode, supported dtype/shape guards, operator and effect coverage, reference output, independently executed artifact, per-layer and final numeric error across multiple inputs, latency/memory provenance, and negative tests. A new operation must add a runtime interpreter, static contract checks, frontend normalization, witness tests, failure tests and documented caveats.

## Related primary documentation

- PyTorch `torch.export` and functional ATen/ONNX: https://docs.pytorch.org/docs/stable/onnx_export.html
- Hugging Face Transformers: https://huggingface.co/docs/transformers/main/index
- Hugging Face Diffusers: https://huggingface.co/docs/diffusers/en/api/pipelines/overview
- NVIDIA Megatron Core MoE semantics: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- LLVM MLIR: https://mlir.llvm.org/
- IREE compiler/runtime: https://iree.dev/
- Torch-MLIR architecture: https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
