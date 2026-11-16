# VisuaML v18 — typed decoder mathematics and transactional KV execution

## What is executable

The v17 SemanticProgram now drives the actual forward math for the bounded
synthetic v14 stacked decoder, in addition to the v17 transactional KV cache.

The new module semantic_decoder_v18.py generates an explicit typed SSA
operation graph containing embedding lookup; RMSNorm; Q, K, V projections;
head splitting; rotary embeddings; per-layer state write/read effects;
causal GQA attention; residual connections; SwiGLU; and final vocabulary
projection. It lowers that graph to an **interpreted NumPy CPU execution plan**
with owned KV buffers. The graph is canonical and checkpoint-bound: unknown
operations, wiring changes, reordered effects, parameter or shape changes,
and different model weights are rejected at lowering.

This makes the computation **actually driven by the graph's operators and
operands**, unlike the v16 Python loop or the v17 state-only IR. A session
interprets each operation, checks concrete dynamic tensor shapes, dtype,
finiteness and resource admission, and frees SSA values after their final use.
Cache updates become visible only after all layers and logits succeed.
The same compiled plan admits multiple query/past lengths up to the fixed
capacity, without compiling one artifact for each past length.

## What is NOT established

- Not machine-code AOT or GPU optimization. The interpreter may be slower than
  the existing v16 specialized CPU loop, especially for tiny token counts.
- Not a complete Hugging Face or Llama importer. The supported checkpoint is
  still the explicitly bounded synthetic GQA/RoPE/SwiGLU ABI.
- Not general dynamic tensor shape compilation, distributed serving, fused
  kernels, all-input numerical proof, optical execution or physical energy data.
- This stage intentionally keeps v16 as the independent reference.
  Do not claim checkpoint-wide coverage or speedups from correctness tests.

## Reproduce

From repository root:

    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_semantic_decoder_v18.py
    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests

The first suite exercises multiple seeds, layer counts and batches, irregular
chunking, whole-context cache parity, generation, malformed graph rejection,
checkpoint binding, transactional failure rollback, snapshot alias isolation
and manifest integrity. The CI suite must pass before promoting this stage.

## Next work

1. Make the full stateful graph a versioned serializable artifact with
   independently validated loading, separate from the execution environment.
2. Implement a real pinned checkpoint importer and tokenizer compatibility,
   preserving the architecture's actual norm/position and attention semantics.
3. Add shape-polymorphic and memory-planned CPU lowering with matched latency,
   peak-memory, and compilation-overhead benchmarks against established engines.
4. Extend the same semantic contracts to a genuinely independent compiled
   backend (IREE/MLIR, Triton or another existing runtime).
5. Generalize mathematical operations and explicit effects to SSM, MoE, graph,
   diffusion and neural-assembly execution without claiming that these families
   already work end to end.
