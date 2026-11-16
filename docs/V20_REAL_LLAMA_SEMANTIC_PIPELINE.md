# VisuaML v20 — Real pretrained Llama through typed Semantic IR

## What changed

v19 could load a restricted, real Hugging Face Llama checkpoint and verify
full-prefill logits, but this local reference was separate from the v18 typed
compiler. v20 closes that architecture gap for one admitted model family.

The pipeline is:

    local config.json + model.safetensors
      → strict v19 Llama config/tensor admission
      → immutable source-bound weight snapshots
      → v20 typed SemanticProgram with explicit state effects
      → bounded KVSession from v17
      → NumPy CPU or TorchScript-hybrid CPU operation interpreter
      → independent full-context Hugging Face/NumPy parity witnesses

The canonical v20 graph contains actual half-RoPE (Llama rotate_half), RMSNorm
with the model's epsilon, grouped-query attention with causal cache reads,
transposed linear projection weight contracts, SwiGLU, residual connections,
and final model-wide RMSNorm. Its output is vocabulary logits.

No model Python modules are imported from the checkpoint; no pickle,
trust_remote_code, remote code execution, or arbitrary IR is accepted.
The importer checks exact tensor names, dtypes and shapes, and the lowering
reconstructs a strict canonical graph from the actual model identity.

## Independent evidence

The pinned trained model manojredhat/tiny-llama was tested at immutable
revision 756e9746eb801752d69771ef3e9bc433b21d853d,
with weights SHA-256 d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03.

In a completed GitHub Actions run, its IR-based NumPy execution matched
independent Hugging Face PyTorch logits on two inputs, with worst absolute
error 1.1324882507324219e-6. Its TorchScript-hybrid run reached a worst
absolute error 1.1920928955078125e-6 and executed 67 compiled arithmetic
primitive calls. Split-prefill decoding and greedy token generation were
also compared against full-context Hugging Face recomputation.

Source workflow: https://github.com/Caerii/VisuaML/actions/runs/43806383413

These are exact witness cases, not all-prompt formal equivalence. The model
revision label is verified in the pinned external workflow, but provenance
claims for other local bundles are user-declared and not signed.

## New runtime limits

- Only the admitted v19 conventional Llama family; scaled RoPE, biases,
  alternative norm placement, arbitrary attention architecture, sharded
  checkpoints, and model-specific control flow remain unsupported.
- NumPy CPU dispatch executes all operators. The TorchScript-hybrid backend
  compiles selected matmul, SiLU, residual/add and elementwise multiply
  operations; RoPE, normalization, GQA attention and KV effects still
  execute with NumPy. It is not a native compiled full decoder.
- Maximum two sessions per batch, one-to-32 token request chunks, maximum
  128 resident positions (or checkpoint maximum, whichever is smaller),
  and at most 64 MiB resident KV memory. The importer also enforces a
  separate 128 MiB model file limit and strict numerical finiteness.
- Tokenizer parity is not established. Tests supply integer tokens directly.
  Sampling strategies beyond greedy and production serving are unimplemented.

## Real Llama bundle (strict software artifact)

llama_bundle_v20.py records the loaded checkpoint bytes, canonical typed
SemanticProgram and digest-bound manifest in a single folder. Loading
recomputes the graph from the validated model config and weights; simply
changing the stored graph and rehashing the manifest cannot authorize
unsupported operations. It rejects symlinks, oversize members, noncanonical
manifest JSON, altered source bytes and invalid source labels.

SHA-256 checks establish content integrity, not publisher authentication.
This artifact format is distinct from the synthetic v19 decoder bundle.

## Performance experiments

examples/llama_semantic_bench_v20.py compares actual admitted local Llama
weights through the old v19 NumPy full-prefill reference, the v20 graph
interpreter and the TorchScript hybrid. It verifies output parity before
measurement and reports source identities, median, p95 and MAD. All numbers
are CPU software results and require compatible CPU/BLAS settings. No
hardware performance or token throughput advantage is claimed.

From repo root after installing dependencies:

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_llama_semantic_v20.py \
      visuaml-client/backend/tests/test_llama_bundle_v20.py \
      visuaml-client/backend/tests/test_llama_bench_v20.py

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=visuaml-client/backend \
      python visuaml-client/backend/examples/llama_semantic_bench_v20.py \
      /path/to/pinned/checkpoint --revision <40-hex-revision> \
      --lengths 4 8 16 --trials 9

The dedicated GitHub workflow probes the pinned public checkpoint without
trusting remote source files.

## Next critical engineering gates

1. Establish tokenizer identity and text-to-token parity with actual published
   tokenizer files and licensed model provenance.
2. Broaden numerical witnesses to many prompts, context lengths, batches,
   incremental KV snapshots and adversarial edge cases, with error tolerances
   sensitive to precision and model architecture.
3. Generalize Llama-specific lowerings into a reusable semantic dialect
   and common operator backend, instead of copying architecture-specific
   loops or treating one model family as universal compatibility.
4. Compile the **complete** decoder graph through a separate native backend
   (e.g., an MLIR/IREE or Triton target), then compare end-to-end output,
   memory, compile time, and latency against appropriate baselines.
5. Replace conservative liveness estimates with actual memory arena planning,
   then measure physical peak memory and long-context/per-request performance.
6. Add constrained rewrite admission (effect preservation, numerical budget
   and independent witnesses), enabling reliable cross-hardware optimization.


## Evidence-driven fused feed-forward optimization

After the initial real-model benchmark showed excessive per-operator
TorchScript dispatch overhead, v20 introduced an optional
torchscript_fused backend. A strictly admitted pure SwiGLU feed-forward
region is traced as a single CPU TorchScript function. The canonical
SemanticProgram remains unchanged; the runtime's scheduler recognizes
only the known RMSNorm/gate/SiLU/up/multiply/down pattern and cannot
cross the KV state-effect barriers.

On the pinned TinyStories model (same revision and weight hash), the
new backend was numerically checked against independent Hugging Face
CPU execution; worst absolute logit error was 9.5367431640625e-7.
Five-trial warm CPU medians in milliseconds:

| Input | v19 NumPy reference | v20 NumPy IR | v20 per-op TorchScript | v20 fused FFN |
| --- | ---: | ---: | ---: | ---: |
| 4 tokens | 6.242 | 7.709 | 17.786 | 8.453 |
| 8 tokens | 6.954 | 8.382 | 19.239 | 9.483 |
| 16 tokens | 8.052 | 9.713 | 20.020 | 10.215 |

This is ~2x faster than *per-operator TorchScript*, not faster than the
reference NumPy runtime. It supports the research thesis that reducing
backend transfer boundaries and fusing regions is a higher priority than
compiling more tiny operators independently.

Full source-bound measurements and environment metadata:
benchmarks/pretrained_llama_fused_ffn_v20.json; GitHub Actions run
https://github.com/Caerii/VisuaML/actions/runs/43807443577

## Tokenizer compatibility gate

The pinned model repository publishes tokenizer.model and
tokenizer_config.json. sentencepiece_tokenizer_v20.py now admits
bounded local files and exposes a deterministic text/ID API. Unit tests
compare the bridge against a separately loaded SentencePieceProcessor
and reject invalid vocabulary, revisions, corrupt metadata, symlinks
and unsupported input lengths.

A separate pinned external workflow tests the same tokenizer files
against Hugging Face AutoTokenizer. Only promote text-to-tokenizer parity
after that external comparison completes successfully. No arbitrary
language-model tokenizer family is implied.
