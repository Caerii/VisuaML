# Verification and performance evidence

This is the evidence ledger for VisuaML's compiler research. Each number belongs to a specific model, source revision, environment, workload and reference—not an unrestricted compiler-wide claim.

## What evidence means

1. **Static admission:** tensor shapes, dtypes, allowed operations, state-effect ordering and checkpoint identity.
2. **Synthetic parity:** independently implemented NumPy/PyTorch oracles for known seeds, shapes, layers and KV states.
3. **Real-checkpoint witnesses:** an immutable source revision, file hashes, tokenizer witnesses and reference logits for explicit inputs.
4. **Compilation and performance:** separate compile time, warm execution time and source-bound model/backend configurations.
5. **Resource ownership:** negative tests for tampering, failure-atomic KV, session isolation, quotas and eviction lifetime.
6. **Physical hardware measurements:** real execution on a target device with actual latency/power/RSS instrumentation. **Not established for photonics.**

A test that passes one level does not establish the next. Numerical agreement is not an all-input formal proof. Logical buffer size accounting is not resident memory usage.

## Reproduce local CI

With Node 20, pnpm 10, Python 3.12 and the pinned backend requirements:

~~~bash
pnpm install
pnpm run install-python-deps
cd visuaml-client
python -m pytest -q backend/tests
cd ..
pnpm --filter visuaml-client run lint
pnpm --filter visuaml-client run test -- --run
pnpm --filter visuaml-client run build
pnpm --dir visuaml-client exec prettier --check .
~~~

The [standard CI workflow](../.github/workflows/ci.yml) executes the corresponding checks plus Ruff. The independent [pretrained-model workflow](../.github/workflows/v19-pretrained-probe.yml) downloads a specific public model revision and checks tokenizer and CPU model oracles. Its historical filename does not imply its capabilities stop at v19.

## v36: fresh independent pretrained C++ numerical replay

The [v36 protocol](V36_PRETRAINED_REPLAY.md) recovers the three exact v28 C++/Python source files from **different verified Git commits** because no single historical commit contains their archived combination. The [dedicated pinned replay workflow](../.github/workflows/pretrained-replay-v36.yml) executes a newly reconstructed, hash-verified source bundle against the pinned Tiny Llama checkpoint and independent Hugging Face PyTorch plus NumPy Semantic IR references. A passing CI run [44193012326](https://github.com/Caerii/VisuaML/actions/runs/44193012326) recorded max absolute errors of `1.52e-6` (HF) and `1.19e-6` (NumPy) on two pinned witnesses, full chunked KV/greedy parity and a 35.429 ms warm native median. The [complete new record](../benchmarks/pretrained_llama_independent_replay_v36.json) and [exact origin](../benchmarks/pretrained_llama_independent_replay_v36.origin.json) are checked byte-for-byte in separate lightweight CI. The [v36 graph](../visuaml-client/public/evidence/knowledge-v36.json) preserves all old v35 study IDs and its preflight-only plan, adding one distinct new experiment on the **same model family**. No formal proof, authenticated signer or dedicated-server throughput is claimed.

## v35: pinned-checkpoint provenance and conservative replay admission

The [v35 protocol](V35_REPLAYABLE_KNOWLEDGE_GRAPH.md) validates the checked-in historical v28 pretrained Llama witness against exact model revision, checkpoint/configuration hashes, semantic IR/native binary/source hashes, finite numerical tolerances and original CI provenance. A deterministic knowledge graph overlays v34 without changing prior IDs; its exact checkpoint edge links v28 to the pinned v31 attention observations. A separate standard-library-only preflight checks historical code hashes against the checkout and optionally verifies local safetensors/config bytes, reporting explicit blockers rather than executing a model. Historical CI evidence is **not** a freshly rerun numerical certificate.

## v34: historical research evidence indexing

The [v34 compounding library](V34_COMPOUNDING_EVIDENCE_LIBRARY.md) uses a lightweight standard-library Python package to derive a deterministic, hash-bound catalog from three reviewed historical JSON artifacts. It verifies the complete v33 original CI bytes and nested rewrite records, preserves separate direct-certification and optimization-time receipts, and exposes exact search/lineage APIs without inferring untested correctness. The [dedicated v34 CI workflow](../.github/workflows/evidence-library-v34.yml) tests tampering, unknown schemas, proposals, catalog reproduction and immutable historical source identity. The read-only browser route `/research/library` structurally validates the published catalog and presents negative observations; its link to an artifact is not a fresh numerical replay.

## v33: exported graph and reusable transformation evidence

The [v33 protocol](V33_EXPORT_AND_EVIDENCE.md) adds fail-closed static `torch.export` admission to the bounded generated native compiler. The [dedicated CI workflow](../.github/workflows/native-evidence-v33.yml) checks synthetic MLP/Conv2D exported graphs against eager PyTorch on multiple inputs, with strict parameter digests and shape guards. A ReLU-idempotence rewrite records independent NumPy/PyTorch finite witnesses and bounded concrete counterexamples on failure. The record SHA-256 is an integrity checksum, **not** an authenticated signer or formal proof. The public research percentage is unchanged.

## v32: executable Semantic IR-to-native research

The [permanent v32 numerical observation](../benchmarks/native_semantic_codegen_observation_v32.json) pins the synthetic graph and trusted-FX witness, generated C++ and executable hashes, compiler measurements and independent CI run. The full temporary CI artifact contains raw timings and generated source. Reproduction code and acceptance scope are documented below.

The [v32 graph-generated CPU report](V32_NATIVE_SEMANTIC_CODEGEN.md) documents the first bounded **SemanticProgram → generated C++17** execution path, distinct from the fixed native Llama decoder. The [native codegen workflow](../.github/workflows/native-semantic-v32.yml) verifies generated residual MLP and convolution programs against both NumPy and eager PyTorch, checks negative/fail-closed admission cases, and publishes source/binary/IR fingerprints and warm execution samples. A separate **restricted trusted FX** import path derives the same graph dialect from local eval-mode PyTorch modules, including explicit frozen parameter hashes.

These are *synthetic* graph and FX witnesses, **not two independently verified pretrained model families**. The trusted FX importer executes Python for tracing and shape propagation; it must never handle untrusted uploads. Compiler-generated source is allowed only from the closed-world validated operator subset and is not a native sandbox.

From `visuaml-client`:

```bash
python -m pytest -q backend/tests/test_native_semantic_codegen_v32.py backend/tests/test_native_fx_frontend_v32.py
PYTHONPATH=backend python backend/examples/native_semantic_codegen_demo_v32.py --output /tmp/native-semantic-v32.json
PYTHONPATH=backend python backend/examples/native_fx_codegen_demo_v32.py --output /tmp/native-fx-v32.json
```

Do not update the roadmap's verified native-family acceptance criterion until real independently pinned architecture-family witnesses exist.

## Pinned Tiny Llama model

The research reference is **manojredhat/tiny-llama**, source revision:

~~~text
756e9746eb801752d69771ef3e9bc433b21d853d
~~~

The source safetensors SHA-256 is:

~~~text
d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03
~~~

The reference workflow normalizes a missing safetensors format metadata field in a tensor-identical *temporary reference copy* so pinned Transformers can load it; the original downloaded file hash remains recorded.

## Primary source-bound evidence

| Evidence file | Supported conclusion |
| --- | --- |
| [v19 reference](../benchmarks/pretrained_llama_parity_v19.json) | Finite real-checkpoint logits parity |
| [v20 semantic IR](../benchmarks/pretrained_llama_semantic_v20.json) | The trained model executes through typed Llama IR |
| [v20 fusion](../benchmarks/pretrained_llama_fused_ffn_v20.json) | Fused SwiGLU versus per-operator TorchScript overhead |
| [v20 tokenizer](../benchmarks/pretrained_llama_text_pipeline_v20.json) | Two text-to-token witnesses and logits |
| [v21 whole graph](../benchmarks/pretrained_llama_whole_graph_v21.json) | Whole decoder CPU TorchScript tracing and warm prefill |
| [v22 dynamic cache](../benchmarks/pretrained_llama_polymorphic_kv_v22.json) | One trace for different queries and positions; KV byte accounting |
| [v23 graph reuse](../benchmarks/pretrained_llama_shared_template_v23.json) | Four-session graph compilation reuse and cold latency |
| [v24 shared weights](../benchmarks/pretrained_llama_shared_weights_v24.json) | Four-session module sharing, logical weight bytes and cold startup |
| [v25 cache ownership](../benchmarks/pretrained_llama_verified_registry_v25.json) | Eviction and live session survival |
| [v26 KV leases](../benchmarks/pretrained_llama_leased_registry_v26.json) | Active KV quota, release and eviction survival |
| [v28 native CPU](../benchmarks/pretrained_llama_native_cpu_v28.json) | Independent static C++17 whole-decoder CPU implementation versus NumPy/Hugging Face references |
| [v27 active model lifetime](../benchmarks/pretrained_llama_active_models_v27.json) | Real-checkpoint module identity across eviction/readmission and last-lease cleanup |

The v27 active-model lifetime also passed the pinned trained-checkpoint workflow on main. The evidence records distinct active modules across eviction/readmission, correct independent logits, and release of the active model-weight budget after closing both leases.

## Results with the important caveats

**v21:** Whole-graph TorchScript was faster than the project's NumPy reference for short pinned-model CPU prefills. It was not evaluated against the strongest inference engines.

**v22:** In a source-bound six-chunk CPU experiment, one polymorphic trace replaced six shape-specialized traces: approximately 40 ms versus 238 ms total tracing, with warm execution approximately 11.83 ms versus 11.78 ms. Returned logical K/V tensor bytes declined from around 3.65 MB to 1.00 MB. These were small, host-sensitive microbenchmarks.

**v23:** One verified graph shared across four clients reduced cold-start median from about 793 ms to 662 ms in one matched three-trial experiment.

**v24:** Reusing the weight-bearing module as well reduced four-client cold-start median from about 532 ms to 256 ms. Aggregate construction time for additional clients fell from about 243 ms to 11 ms, and counted module-buffer bytes dropped from about 100.5 MB to 25.1 MB. None of those bytes establish actual process RSS.

**v25–v27:** LRU, KV leases and active module lifetimes are software ownership invariants. They do not demonstrate OS-enforced physical-memory limits or serving throughput.

**v28:** A separately compiled scalar C++17 CPU kernel has passed pinned real Tiny Llama full-decoder, chunked K/V and generation witnesses. Its backend mathematics are independent of NumPy and TorchScript, but it is a fixed Llama ABI implementation, not arbitrary IR-to-native generation or an optimized production backend. See the [native compiler report](V28_NATIVE_CPU_COMPILER.md).

Avoid adding compile time to a wall-clock duration that already includes it. Benchmark every candidate on identical inputs, comparable threading, hardware and warm/cold conditions. Report medians, tails, hashes, tolerances and failure cases.

## Public evidence-weighted roadmap

The [public roadmap](ROADMAP.md) derives percentages from weighted acceptance criteria, counting only individually verified gates with source links. The reviewed v28 snapshot is **72.5%** under the published rubric, not a scientific prediction of the project's ultimate completeness. Changes to the rubric require versioned review and reproducible acceptance evidence.

## Reproduce a model benchmark

With a separately verified local checkpoint:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=visuaml-client/backend \
python visuaml-client/backend/examples/llama_polymorphic_bench_v22.py \
  /path/to/model --revision YOUR_40_HEX_REVISION \
  --chunks 4 1 1 1 3 2 --trials 25
~~~

A multi-session model weight benchmark:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=visuaml-client/backend \
python visuaml-client/backend/examples/shared_weights_bench_v24.py \
  /path/to/model --revision YOUR_40_HEX_REVISION \
  --clients 4 --trials 25
~~~

For the full requirements of accepting a new backend or model family see [Contributing](CONTRIBUTING.md). For trust assumptions see [Security](SECURITY_AND_BOUNDARIES.md).
