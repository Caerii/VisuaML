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
| [v27 active model lifetime](../benchmarks/pretrained_llama_active_models_v27.json) | Real-checkpoint module identity across eviction/readmission and last-lease cleanup |

The v27 active-model lifetime also passed the pinned trained-checkpoint workflow on main. The evidence records distinct active modules across eviction/readmission, correct independent logits, and release of the active model-weight budget after closing both leases.

## Results with the important caveats

**v21:** Whole-graph TorchScript was faster than the project's NumPy reference for short pinned-model CPU prefills. It was not evaluated against the strongest inference engines.

**v22:** In a source-bound six-chunk CPU experiment, one polymorphic trace replaced six shape-specialized traces: approximately 40 ms versus 238 ms total tracing, with warm execution approximately 11.83 ms versus 11.78 ms. Returned logical K/V tensor bytes declined from around 3.65 MB to 1.00 MB. These were small, host-sensitive microbenchmarks.

**v23:** One verified graph shared across four clients reduced cold-start median from about 793 ms to 662 ms in one matched three-trial experiment.

**v24:** Reusing the weight-bearing module as well reduced four-client cold-start median from about 532 ms to 256 ms. Aggregate construction time for additional clients fell from about 243 ms to 11 ms, and counted module-buffer bytes dropped from about 100.5 MB to 25.1 MB. None of those bytes establish actual process RSS.

**v25–v27:** LRU, KV leases and active module lifetimes are software ownership invariants. They do not demonstrate OS-enforced physical-memory limits or serving throughput.

Avoid adding compile time to a wall-clock duration that already includes it. Benchmark every candidate on identical inputs, comparable threading, hardware and warm/cold conditions. Report medians, tails, hashes, tolerances and failure cases.

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
