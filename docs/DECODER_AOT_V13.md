# VisuaML v13 — complete tiny-decoder NumPy-AOT execution

## Precisely what is implemented

The existing v12 synthetic GQA/RoPE decoder can now compile a full **shape-specialized chunk** into a closed-world `visuaml.numpy-aot.v1` manifest and constants archive. The compiled program consumes **int64 token IDs, float32 key cache, float32 value cache**, and returns **float32 vocabulary logits and the two updated cache tensors**. A trusted driver performs chunked prefill, incremental greedy generation, cache-capacity enforcement and artifact shape caching.

The graph is captured through PyTorch FX, compiled via `compile_numpy_aot`, and executed through the NumPy-only interpreter. The actual standalone NumPy artifact runs without importing PyTorch; a subprocess regression test verifies this. The current implementation specializes the graph to concrete batch, query length, and past-cache length. This is *not* dynamic-shape universal compilation or a physical accelerator.

## Portable extensions

- Two closed-world operators: `cat` and `repeat_interleave` (bounded, explicit-axis, non-mutating NumPy semantics).
- PyTorch FX method `__getitem__` normalized to existing `getitem` operator. This does **not** enable arbitrary Python method calls.
- `input_constraints`: optional, integrity-hashed half-open integer input bounds. The raw artifact rejects negative/out-of-vocabulary token indices even when the driver isn't present.
- Cached context append is purely functional: caller-owned key/value arrays are not mutated.

## Verification

- Multiple seeds, chunk shapes and nonzero-cache prefill cases.
- Per-chunk logits, keys and values compared against **both** NumPy and PyTorch reference implementations.
- Greedy generated tokens and final cache checked against both references.
- Round-trip `manifest.json` / `constants.npz` and integrity tampering rejection.
- PyTorch-forbidden subprocess execution of the standalone portable runtime.
- Invalid input shapes, dtypes, token IDs, cache overflow, cache nonfiniteness and resource admission tested.
- Evidence: `public/decoder_compiler_certificate.v13.json`. Synthetic, witnessed, **not** a proof or hardware performance claim.

### Local execution

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_decoder_aot_v13.py
PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/generate_decoder_certificate_v13.py
```

```python
import numpy as np
from visuaml.compiler_lab.tiny_decoder_v12 import TinyDecoderWeights
from visuaml.compiler_lab.decoder_aot_v13 import CompiledTinyDecoder

weights = TinyDecoderWeights.seeded(seed=19)
executor = CompiledTinyDecoder(weights)
tokens, cache = executor.generate(np.array([[1, 2, 3]], dtype=np.int64), steps=3)
print(tokens, cache.keys.shape)
```

The artifact is **not a sandbox** for arbitrary uploaded Python models or untrusted NPZ files. Source kinds and integrity digests are not signatures or authenticated device measurements.

## Still missing

A version-pinned licensed pretrained checkpoint; multiple real decoder layers; general RoPE variants and attention biases; paged/sliding-window cache management; fp16/bf16 quantization; dynamic generation in-device; cross-vendor executable lowering; any photonic hardware execution; measured systems-level latency/energy. Do not label the whole Transformer family as checkpoint-certified.
