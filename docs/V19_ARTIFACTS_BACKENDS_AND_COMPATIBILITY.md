# VisuaML v19 — artifact, memory, portability and model-intake checkpoint

## Verified scope

Version 19 adds a strict, on-disk source-bound v18 decoder artifact;
conservative symbolic memory liveness; live fixed-fixture IR inspection;
TorchScript-compiled CPU primitive witnesses; a restricted local Hugging Face
Llama safetensors intake; and a matched NumPy CPU timing harness.

**Do not infer pretrained-checkpoint parity or faster inference from these tests.**

### Artifact portability

The semantic_bundle_v19 module writes semantic.json, checkpoint.json,
weights.npz and bundle.json. It binds the canonical typed graph to the
checkpoint digest, rejects tampering and unknown JSON fields, and independently
reloads only the same closed-world compiler-approved program. SHA-256 gives
content identity, not proof of the claimed source or digital signing.
Loading still uses the v14 bounded synthetic checkpoint ABI.

### Memory

The memory_planner_v19 module derives last-use intervals and worst admitted
logical SSA tensor bytes for a bounded query/context length. The v18 interpreter
already discards intermediate values after their final use. This planner is
*analysis only*: it has not introduced a reusable allocation arena and does not
measure actual retained host RSS, temporary NumPy allocations or device memory.

### Second backend

torchscript_primitives_v19 has strict float32 TorchScript CPU compilation for
matmul, add, multiply, SiLU and RMSNorm. It validates input shapes/dtypes,
bounds allocation and compares outputs to independently implemented NumPy
reference arithmetic. This is real compiled execution of *primitive subsets*,
not complete lowering of the stateful decoder into TorchScript or IREE/Triton.

### Pretrained Llama intake

hf_llama_checkpoint_v19 reads only local config.json and model.safetensors,
validates exact named tensors, head/group dimensions, parameter dtypes,
resource limits, SHA-256 and a user-declared 40-character revision. Its
dedicated NumPy full-prefill executor reproduces conventional half-RoPE,
GQA, RMSNorm, SwiGLU and final model normalization. Independent PyTorch eager
tests run against locally generated HF-shaped safetensors fixtures.

The pinned TinyStories-trained model manojredhat/tiny-llama at revision
756e9746eb801752d69771ef3e9bc433b21d853d passed **real pretrained
full-prefill logits parity** against the Hugging Face PyTorch CPU reference
on 4-token and 6-token sequences. Recorded worst absolute logit errors were
1.13249e-6 and 1.37091e-6. See benchmarks/pretrained_llama_parity_v19.json
and the completed pretrained parity GitHub Actions run 43804918553.
This certifies those exact witnesses, not tokenizer behavior, all prompts,
incremental KV decoding, or compiler IR lowering for pretrained weights. Its revision is user-declared provenance,
not cryptographic evidence of a remote repository identity. Scaling RoPE,
arbitrary head sizes, attention biases, sharded models and unsupported configs
are deliberately rejected. It does not yet perform incremental token streaming,
tokenizer parity or lower the Llama checkpoint into v18's canonical IR.

The synthetic v18 decoder uses interleaved rotary pairs and lacks final global
RMSNorm; a normal Llama checkpoint generally cannot be blindly mapped into it.
Merging these paths without preserving the Llama-specific semantics would
silently invalidate model results.

### Inspection

The Architecture Atlas contains a live IR evidence panel for the decoder
family. Run the existing local server with VISUAML_COMPILER_LAB_ENABLED=1;
the guarded endpoint is /api/compiler-lab/semantic-inspect. It accepts only
seeded local fixtures, not arbitrary Python source, uploads or checkpoints.
It reports executed parity against v16, the typed operation graph, state
effects, hashes and a conservative memory estimate.

### Matched benchmark

From the project root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=visuaml-client/backend \
      python visuaml-client/backend/examples/semantic_decoder_bench_v19.py \
      --layers 1 2 4 --prompt 4 --steps 4 --trials 25

The report records source hashes, dependency versions, compilation-plan setup,
warm-run medians, p95, median absolute deviation and generated-output parity.
It compares two software CPU paths, not full model serving or GPU execution.

## Test commands

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_semantic_bundle_v19.py \
      visuaml-client/backend/tests/test_semantic_inspection_v19.py \
      visuaml-client/backend/tests/test_torchscript_primitives_v19.py \
      visuaml-client/backend/tests/test_hf_llama_checkpoint_v19.py \
      visuaml-client/backend/tests/test_semantic_decoder_bench_v19.py

## Remaining acceptance gates for v20

1. Expand the pinned real Llama witness beyond two integer-token prompts:
   validate its tokenizer, generation, incremental K/V state, checkpoint
   licensing and a broader numerical/precision error suite.
2. Lower actual Llama rotary, final normalization and checkpoint contracts
   through a common, more general semantic IR.
3. Turn symbolic lifetimes into alias-safe allocation reuse, and measure actual
   peak memory, cold and warm performance at more realistic context lengths.
4. Compile the same full stateful model into an independently executable
   native backend using IREE/MLIR, Triton or another well-maintained system.
5. Verify rewrite legality, error budgets, memory effects, numerical precision
   and benchmark provenance before automated optimization search.
