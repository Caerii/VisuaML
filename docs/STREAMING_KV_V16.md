# VisuaML v16 — resident KV streaming CPU runtime

## What actually changed

The v14/v15 compiled stacked decoder currently creates a new NumPy-AOT shape specialization for each distinct `(batch, query, past)` tuple. Its functional attention graph concatenates old and new K/V arrays and repeats grouped-query KV heads. Those operations are semantically appropriate for portable functional execution but expensive in an autoregressive Python interpreter.

`streaming_decoder_v16.py` implements an **opt-in alternative NumPy CPU runtime** for the same strict synthetic `StackedWeights` ABI. Its `StreamingStackedDecoder`:

- freezes a content-addressed, immutable snapshot of the supplied weights at construction;
- preallocates key and value storage of shape `[batch, max_context, kv_heads, head_dim]` for each layer;
- maintains a committed cache cursor that changes only after all layers successfully execute;
- appends directly to uncommitted cache slots and discards new slots if a later operator fails;
- reshapes queries to `[batch, kv_heads, query_head_groups, query_len, head_dim]`, broadcasting K/V instead of `repeat_interleave`-materializing copies;
- computes RoPE trigonometry in advance for the entire bounded context;
- exposes snapshots and validated, atomic restoration, detached from its owned state;
- enforces a 1–4 layer, 1–8 batch, <=32 chunk, <=64 context, float32 weights, int64 bounded-vocabulary input contract.

No Python source loading, arbitrary checkpoint parser, dynamic Python callbacks, or photonic hardware invocation is involved. The module imports a legacy StackedWeights definition that depends on PyTorch, so **this adapter is not a standalone PyTorch-free AOT artifact**. Use the existing NumPy-AOT path when standalone artifact portability matters.

## Evidence

The new `test_streaming_decoder_v16.py` checks multi-chunk logits **and every layer's full KV contents** against the independently written v14 NumPy reference across seeds and layer counts; also tests generation tokens, snapshot aliasing, checkpoint mutation isolation, overflow, invalid shape/dtype/vocabulary input, prevalidated atomic restore, and late-stage injected failures. The benchmark separately verifies generated-token parity with the warm v15 prepared AOT driver.

The matched benchmark in `benchmarks/streaming_kv_v16.json` uses a **synthetic** batch-one 4-token prompt and generates 4 tokens, single-threaded OpenBLAS, 9 timed warm trials per depth. It excludes first-time compilation. Results are machine- and Python/BLAS-version-specific, and show no physical hardware or general pretrained-model throughput. The v15 AOT comparator performs the same token-generation task but has a different runtime boundary; the performance delta is not evidence that the AOT compiler emits equivalently fast code.

## Reproduce

From repo root after installing its Python dependencies:

```sh
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_streaming_decoder_v16.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=visuaml-client/backend \
  python visuaml-client/backend/examples/streaming_kv_bench_v16.py --layers 1 2 4 --steps 4 --trials 9 \
  --output benchmarks/streaming_kv_v16.json
```

## Current non-goals and next performance work

- This is **not** a general Llama/Hugging Face checkpoint executor, CUDA kernel, FlashAttention, numerical proof, optical hardware implementation, or a dynamically shaped NumPy-AOT artifact.
- The work is bounded to small, fixed-profile GQA/RoPE/SwiGLU fixtures. Long sequence performance, multi-request concurrency, and cancellation need independent engineering.
- The runtime uses owned mutable state and is **not thread-safe**; one independent session should own each instance.
- Snapshotting/restore copies data; that is deliberate to prevent aliasing. A production KV cache would benefit from paged allocation, batched scheduling, continuous batching, eviction and quantization with separate correctness gates.
- v17 priority: make the common Semantic IR lower to a compiled dynamic-length KV primitive and compare a version-pinned pretrained checkpoint across multiple context lengths. Avoid conflating this CPU session adapter with such future compiler work.
