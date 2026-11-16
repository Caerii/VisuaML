# VisuaML v17: dynamic resident-KV semantic lowering (stage 1)

## Precisely what shipped

Version 17 introduces an executable and closed-world **KV append dialect**
represented by the shared typed SemanticProgram. A compiler validates the exact
state-effect sequence and lowers it into an independently instantiated NumPy
execution session. Unlike the existing v13-v15 AOT artifact interface, the
program's token-chunk extent is **symbolic** and the same plan executes different
chunk lengths without shape-specific recompilation.

This is a first-stage compiler primitive, **not** a compiled Transformer,
general dynamic-shape NumPy-AOT artifact, JIT-generated native kernel, GPU
backend, full semantic-to-decoder lowering, or production attention engine.
The CPU operation dispatch is interpreted NumPy; the word "compiled" denotes
verified lowering to a sealed schedule, not machine-code generation.

The admitted dialect has per-layer K and V float32 cells, a symbolic
per-request 'step' extent, fixed batch/head/capacity dimensions, ordered
'kv_append' then state 'write' operators, and an effect-token chain.
It is closed-world: unknown operations, changed ordering, forged output/state
wiring, bad source digests and over-budget dimensions are rejected.
Data-plane append validates every layer and copies input chunks *before*
publishing the new cache cursor. It supports up to 16 layers, 8 batch,
16 KV heads, 128 head dimensions, and capacity 4096, subject to a global
64 MiB reservation cap and <=256 tokens per append.

Sessions own their cache storage and are not thread-safe. Snapshots are
detached; restore checks all cells before allocating replacement buffers.
Program hashes and serialized manifest hashes detect accidental changes;
they are **not** digital signatures, model checkpoint verification, or
proofs of mathematical equivalence.

## Why this matters

Before v17, 'semantic_ir.py' was predominantly an inspectable description,
while the v16 decoder owned K/V management directly. This change demonstrates
an **actual** Semantic IR -> validation -> lowering -> execution path for the
first reusable stateful primitive. The v16 decoder remains the numerical oracle,
and is **not yet rewritten** to consume this backend: the next integration
step is to use the lowered session (or a zero-copy staged variant) for the
decoder's native KV writes without regressing v16 latency.

## Reproduce

From repository root with Python dependencies installed:

    PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_kv_lowering_v17.py

Included tests cover irregular token chunks, dynamic lengths with one compiled
plan, full cache parity against the independent v16 decoder, snapshot/restore,
failure-atomic validation, manifest tampering, strict effect ordering, source
fingerprints and memory admission.

## Next integration gates

1. Refactor v16's layer loop to use a compiled KV session with transactional
   *staging* (no extra per-layer materialization in the fast path); preserve
   full v14/v15 reference parity and single-token performance.
2. Extend the typed IR beyond append into per-layer RMSNorm, RoPE, GQA, causal
   attention, SwiGLU and logits with explicit shape/effect/aliasing contracts.
   The resulting decoder must be generated from validated IR rather than from
   a fixed Python reference loop.
3. Import a separately version-pinned, licensed *real* small checkpoint and
   tokenizer using a strictly validated mapping (no arbitrary uploaded code).
   Compare layer-wise logits and states to an independent framework oracle.
4. Establish matched prefill/decode performance tests with multiple context
   lengths, cold and warm latencies, p50/p95, peak memory and compile costs.
5. Only after a real model is certified should CUDA/Triton, CPU fusion,
   quantized cache formats and external photonic backends be promoted above
   experimental capability claims.

## Stage 2: decoder execution using compiled KV state effects

The existing v16 decoder now has an opt-in `semantic_kv=True` execution mode. Its actual layer loop stages new K/V data in a session whose buffer allocation, per-layer write ordering and atomic cursor publication come from the validated v17 SemanticProgram. The default v16 path remains unchanged and is the independent comparison.

New tests compare logits and every K/V tensor against v16 across irregular chunk lengths through the complete 64-token context. They also inject an exception in a late decoder layer and verify the prefix remains unchanged, then test snapshot/restore and generation.

This is **the first integrated compiler-driven state schedule**, not a full decoder graph lowering: RMSNorm, RoPE, attention, MLP and logits are still computed by the original NumPy math. Exact throughput parity with v16 is not yet measured and must not be assumed.
