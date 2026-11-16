# v12 — Neural family-block parity and certification ladder

This stage implements **six small, trusted CPU reference workloads**, each in two separately coded implementations, NumPy and PyTorch. The static report under `visuaml-client/public/family_block_certificates.v12.json` records test inputs, reference fingerprints, seeds, and numerical parity. The `ArchitectureAtlas` React page displays these results without treating them as checkpoint certifications.

## What is now executable

1. **Tiny autoregressive decoder** — token embeddings, RMSNorm, grouped-query attention, interleaved RoPE, causal masks, append-only KV cache, SwiGLU-style MLP, residual connections, vocabulary logits, chunked prefill and incremental greedy token generation. This is a **small, random-weight one-layer fixture**, not a trained Llama checkpoint, tokenizer-integrated application, production decode engine or quantized KV cache.
2. **Convolutional diagonal state-space block** — causal depthwise Conv1D, diagonal recurrent state updates, gated outputs, chunking, and explicit persistent state. Not the fused selective-scan CUDA kernel or any released Mamba checkpoint.
3. **Sparse MoE** — stable top-k routing, renormalized expert gating, deterministic per-expert capacity, dropped overflow contributions, expert MLPs, and weighted recombination. Not load-balancing training loss or distributed all-to-all expert parallelism.
4. **Diffusion / flow matching** — deterministic time-conditioned vector field with Euler and Heun ODE updates. Not a trained U-Net/DiT checkpoint or full image/video/noise schedule.
5. **Graph attention** — COO graph aggregation, directed multigraph edges including duplicates, segment-wise softmax and index-add. Not a production GAT or graph-batch dynamic-shape compiler.
6. **Quantization and training** — symmetric per-output-channel int8 weight quantization, float dequantization, SGD with PyTorch autograd compared against independent analytic NumPy gradients. Not production mixed-precision training, integer-matmul lowering, fake quantization or optimizer-state checkpointing.

## Trust ladder

`catalogued` → `operator identified` → `synthetic reference-block parity` → `compiler AOT executable` → `pinned real checkpoint parity` → `physical-device measurements`.

**The six v12 certificates stop at the third rung.** Previously successful primitive-level AOT probes do not upgrade the six representative blocks to the fourth rung. No new physical hardware access, full checkpoint import, or model-family coverage is claimed.

The inputs are fixed, bounded `float32`/`int64` arrays. Programs are not loaded from arbitrary user-supplied Python. The certificates are generated from a trusted local script. The reported source fingerprint and input digests are reproducibility aids, not cryptographic attestation of who performed the execution.

## Reproduction

From `visuaml-client` with PyTorch, NumPy and pytest installed:

```bash
python -m pytest -q backend/tests/test_family_blocks_v12.py backend/tests/test_family_certification_v12.py
PYTHONPATH=backend python backend/scripts/generate_family_certificates.py
```

Versioned JSON can be served by the existing Vite public-file mechanism. Generated snapshot reports **the specific PyTorch and NumPy versions used**; differences across execution environments require regeneration and comparison. The numerical tolerances are per tiny block and cannot establish universal FP32 equivalence.

## Next compiler upgrades

Connect each block to a *real* staged lowering: exported PyTorch ATen → typed VisuaML semantic IR → explicit state/memory dialect → independent NumPy-AOT bytecode → hybrid runtime. A complete decoder prefill/decode is the initial acceptance target; a second target should be an actual, independently version-pinned small Hugging Face model in a controlled test environment. Continue with runtime-aware state-space scans, MoE dispatch and irregular graph semantics; measure memory, precision, and scheduling before optimizing for experimental optical hardware.
