# The Compiler Lab: typed computation, explicit state, checked execution

The objective is to understand and transform computation without conflating **its mathematical meaning** with its first implementation. VisuaML's experimental Compiler Lab builds typed semantic programs with explicit tensor shapes and effects, then runs admitted versions on reference and compiled backends.

It does **not** compile arbitrary Python, all PyTorch operations, all Hugging Face architectures, or physical optical hardware. V28 adds an independent native **scalar C++17 implementation** for one admitted Llama dialect; it is not yet general source-generation from arbitrary SemanticProgram operations.

## The working end-to-end pipeline

~~~text
Trusted local safetensors + config + SentencePiece
                          |
               Strict model admission
                          |
        Immutable checkpoint + semantic SHA-256
                          |
           Canonical typed SemanticProgram
                          |
          Type, shape and effect verification
                     /           \
              NumPy oracle      CPU TorchScript
                     \           /
                  Numerical witnesses
                          |
             Verified whole-graph template
                          |
         Shared compiled code and model weights
                          |
        Independent resident KV per client lease
                          |
          Cache ownership / lifetime accounting
~~~

The most developed real-model example is a conventional, deliberately *small* Llama decoder supporting admitted RMSNorm, rotate-half RoPE, causal grouped-query attention, SwiGLU, final normalization, prefill and incremental KV. Nonstandard RoPE variants, arbitrary attention geometries, custom Python modules, sharded checkpoints and huge weights are rejected.

## Start the tests

Install the Python environment using the [project README](../README.md), then from the client directory:

~~~bash
cd visuaml-client
python -m pytest -q backend/tests
~~~

Focused real-model-shaped correctness/ownership suites:

~~~bash
python -m pytest -q \
  backend/tests/test_llama_semantic_v20.py \
  backend/tests/test_torchscript_graph_v21.py \
  backend/tests/test_torchscript_poly_v22.py \
  backend/tests/test_shared_weights_v24.py \
  backend/tests/test_template_registry_v25.py \
  backend/tests/test_session_leases_v26.py \
  backend/tests/test_active_models_v27.py
~~~

The [pretrained integration workflow](../.github/workflows/v19-pretrained-probe.yml) independently downloads a pinned real Tiny Llama revision, checks hashes and finite tokenizer/logit/state witnesses against Hugging Face and PyTorch, and exercises the model/session lifecycle.

## Example: verified local model with leased inference

Install the backend dependencies, obtain an allowed local safetensors checkpoint from a source you independently trust, and pin its revision and file SHA-256.

~~~python
import numpy as np
from visuaml.compiler_lab.active_models_v27 import ActiveModelLlamaRegistry
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    make_llama_semantic_program,
    lower_llama_semantic_program,
)

weights = load_hf_llama_local(
    "/path/to/trusted/llama",
    declared_revision="YOUR_VERIFIED_40_HEX_REVISION",
    # expected_weights_sha256="YOUR_INDEPENDENTLY_PINNED_SHA256",
)
program = make_llama_semantic_program(weights, batch=1)
plan = lower_llama_semantic_program(program, weights, batch=1)
registry = ActiveModelLlamaRegistry(
    max_cached_weight_bytes=256 * 1024 * 1024,
    max_active_model_weight_bytes=256 * 1024 * 1024,
    max_active_kv_bytes=32 * 1024 * 1024,
    max_active_sessions=4,
)
identity = registry.register(plan)
with registry.new_session(identity) as session:
    logits = session.step(np.array([[1, 2, 3]], dtype=np.int64))
    print(logits.shape, session.cursor)
print(registry.report())
~~~

Run a script from the client directory, or set PYTHONPATH=visuaml-client/backend from the root. Token IDs in this example are placeholders; generate meaningful IDs with the matching admitted SentencePiece files. The registry is **trusted, process-local software**; it does not sandbox arbitrary model code.

## Execution backends and what they mean

| Target | Implemented role | Key limitation |
| --- | --- | --- |
| NumPy interpreter | Checks canonical IR, operator results, explicit KV state | Python/NumPy dispatch overhead |
| Tiny NumPy AOT | Standalone bounded examples from earlier compiler experiments | Not general model compatibility |
| TorchScript primitives | Independently checked selected arithmetic operators | Excessive per-operator call boundaries |
| TorchScript fused FFN | One checked pure SwiGLU region | Not a complete native compiled model |
| TorchScript whole graph | All admitted Llama decoder math in one CPU traced graph | Concrete-shape specialization in v21 |
| Native C++17 research backend | Independently compiled Llama mathematical and KV decoder behind a narrow C ABI | Fixed admitted Llama dialect; not general native code generation, GPU, or tuned BLAS |
| Polymorphic TorchScript | One model/batch graph handles changing query length and KV cursor | CPU-only and bounded model/context |
| Verified shared templates | Reuse executable and immutable weights across clients | Per-client KV and process-local ownership |
| LRU/leases | Bounded cached weight, active KV and active model-module *logical* byte accounting | Not actual process RSS or OS isolation |

Under the polymorphic backend, the compiled graph reads only the live KV prefix and returns new K/V suffixes. The separate resident state validates and atomically commits outputs; failures cannot publish a half-finished cursor.

## Independent C++17 CPU target (v28)

For the same admitted model and canonical plan, a new [native CPU compiler](../visuaml-client/backend/visuaml/compiler_lab/native_cpu_v28.py) builds a trusted fixed C++17 kernel locally and checks the entire mathematical execution against an independently implemented NumPy oracle before allowing the executable to be used. Session K/V suffixes remain validated and transactional. The [pinned trained Tiny Llama evidence](../benchmarks/pretrained_llama_native_cpu_v28.json) compares the same independent C++ implementation with Hugging Face/PyTorch and the semantic interpreter.

~~~python
from visuaml.compiler_lab.native_cpu_v28 import NativeCpuLlamaCompiler

native = NativeCpuLlamaCompiler(plan)  # trusted system C++17 compiler required
with_oracle_checked = native.new_session()
native_logits = with_oracle_checked.step(np.array([[1, 2, 3]], dtype=np.int64))
print(native_logits.shape, native.report())
~~~

This is a **model-specific native implementation**, not automatic arbitrary-program IR-to-C++ code generation; it has no demonstrated production inference speed advantage. See [v28 native CPU research](V28_NATIVE_CPU_COMPILER.md) and the [public roadmap](ROADMAP.md).

## Interpreting evidence

A numeric match against an independent reference is evidence **for the tested model and inputs within a declared tolerance**, not a formal proof for all inputs. Different CPU/GPU precision modes can produce legitimate error changes. Hardware performance must be measured against a competent baseline, with compilation separated from warm decode.

The [Verification guide](VERIFICATION.md) records model and source hashes, measured CPU examples, pinned CI workflows, provenance and known uncertainties. The [Security guide](SECURITY_AND_BOUNDARIES.md) explains untrusted Python and logical-versus-physical memory boundaries.

## Next real milestones

A general, independently deployable native IR-to-CPU/GPU AOT generator; measured peak RSS and allocator lifetime; long-context prefill/decode scheduling; safe memory pooling; additional real pretrained architectures only after numerical certification; eventually hardware-algorithm co-design. The optical placement infrastructure remains **simulation**, not accelerator execution.
