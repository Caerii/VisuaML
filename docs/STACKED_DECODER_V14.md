# VisuaML v14 — a bounded multi-layer decoder artifact with canonical checkpoint ABI

## What is executable

The v14 compiler accepts **one to four** fixed-weight, GQA/RoPE/SwiGLU residual blocks with shared token embedding and vocabulary projection. The FX frontend flattens a per-layer KV-cache tuple into distinct, typed tensor inputs, preserving the closed-world NumPy-AOT manifest. It never evaluates generated Python code, and the portable runtime has no PyTorch dependency.

A compiled invocation accepts `(token_ids, k0, v0, ..., kN, vN)` and yields `(logits, updated_k0, updated_v0, ..., updated_kN, updated_vN)`. Each `(batch, query_length, past_length)` requires its own concrete compiled artifact. The host driver maintains **immutable, independently typed KV tensors per layer**, implements greedy autoregressive generation, checks vocabulary bounds in both the driver and the portable manifest, and rejects silently mutated weights.

The independent reference is based on `family_blocks_v12.decoder_step_numpy` and a separate NumPy residual/SwiGLU stack. Compiled and reference versions compare logits and **every KV tensor after each prefill/decode chunk**.

## Canonical checkpoint contract

`write_canonical_checkpoint` and `load_canonical_checkpoint` accept a small, specifically named schema with only finite `float32` arrays, 1–4 layers, matching attention topology, and a weight-table SHA-256 digest. The archive uses NPZ without pickle, validates its declared members and uncompressed-byte budget before loading, rejects duplicate archive member names, and rejects unexpected weight keys. The provenance string is **user-declared**, not cryptographically authenticated. Checksums establish content identity, not model origin or authorization.

This is an *intermediate canonical checkpoint format*. It does **not** automatically import GPT-2, Llama, Gemma, Qwen, Mistral, Hugging Face Transformers or safetensors layouts. In particular, the current tiny model has no final global model-wide RMSNorm or tokenizer, and normalizations/effect semantics must match a target architecture before real checkpoint conversion can be certified.

## Local commands

From the repository root:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_stacked_decoder_v14.py visuaml-client/backend/tests/test_stacked_certification_v14.py
PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/generate_stacked_certificate_v14.py --output visuaml-client/public/stacked_decoder_certificate.v14.json
```

Minimal Python usage:

```python
import numpy as np
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights, empty_stack, CompiledStackedDecoder

weights = StackedWeights.seeded(19, layers=3)
model = CompiledStackedDecoder(weights)
logits, cache = model.step(np.array([[1, 2, 3]], dtype=np.int64), empty_stack(weights, 1))
generated, cache = model.generate(np.array([[1, 2, 3]], dtype=np.int64), steps=3)
```

## Measured and unmeasured boundaries

- Actual: NumPy-AOT software execution, generated manifest, typed state, multiple seed/depth witnesses, NumPy-only subprocess execution, synthetic checkpoint round-trip.
- Not established: real pretrained checkpoint compatibility, tokenizer correctness, per-token latency advantage, arbitrary layer depth, mixed precision, dynamic-shape single-artifact inference, GPU/optical kernels, model-specific generation modes, physical Ayo measurements.
- Each passing numerical witness covers only its recorded inputs and execution environment, not a universal all-input proof.

## Next research and engineering gates

1. Map a version-pinned, openly licensed real checkpoint into a formally specified importer, preserving RMSNorm, RoPE variant, tied weights, all original config guards, and tokenizer semantics. Validate original checkpoint hashes and licensing, with an independent framework output oracle.
2. Separate learned state, static weight tensors and dynamic KV buffers in the canonical IR, avoiding recompiling graphs solely because past length changes.
3. Implement safe paging/sliding-window cache policies with numerical witnesses and explicit memory-residency accounting.
4. Compile the stack to a real backend adapter only after operator legality, memory layout, precision and transfer semantics have been established.
