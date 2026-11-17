# VisuaML architecture

This is the architectural source of truth for the collaborative graph application and the research compiler as of the **v28 development milestone (November 2026)**. Milestone labels are not published package versions.

## Two cooperating systems

~~~text
                Browser — React / TypeScript
            ┌────────────────────────────────┐
            │ React Flow · tensor inspection │
            │ architecture atlas · Yjs       │
            └──────────┬─────────────────────┘
                       │ HTTP / WebSocket
              ┌────────┴────────┐
              │                 │
           Fastify          Yjs server
         guarded API       collaboration
              │
      trusted Python tools
              │
   ┌──────────┴────────────────────────────────┐
   │         Python visuaml backend            │
   │  PyTorch FX export / graph analysis       │
   │  Typed Semantic IR / shape and effect     │
   │  Restricted Llama checkpoint admission    │
   └──────────┬────────────────────────────────┘
              │
    ┌─────────┴───────────────────────────┐
    │                                     │
 independent NumPy                 TorchScript CPU
 reference                          whole decoder
    │                                     │
    └────────────────┬────────────────────┘
                     │
           verified shared graph
              + model weights
                     │
         bounded model-template cache
                     │
           leased private KV per client
                     │
       active weight/lease accounting
~~~

The optical/assembly research components model or simulate hypothetical hardware behavior. They **do not** execute on a verified physical photonic accelerator.

## Web application

The React application lives under [client source](../visuaml-client/src/), with its API server under [server](../visuaml-client/server/) and Python under [backend](../visuaml-client/backend/). The web UI is built with Vite, TypeScript, React Flow and Three.js. Yjs and WebSockets support shared editing and presence; Clerk integration is optional for developer login flows.

Trusted shipped PyTorch models may be traced and exported by FX. Dynamic Python control flow is not generally capturable. The API limits model imports to allowlisted modules by default. Untrusted Python file upload execution is disabled unless unsafe developer options are explicitly enabled; **those options are not a sandbox**.

The separate Compiler Lab endpoints are local opt-in, restricted to fixed known scenarios and bounded numeric fixtures. Set the API environment variable **VISUAML_COMPILER_LAB_ENABLED=1** for POST /api/compiler-lab/analyze or GET /api/compiler-lab/semantic-inspect. These routes do not authorize arbitrary user Python.

See [Security and Boundaries](SECURITY_AND_BOUNDARIES.md) before running any supplied code.

## Compiler architecture

The modules are under [compiler_lab](../visuaml-client/backend/visuaml/compiler_lab/). Their separation is intentional: syntax, executable semantics, physical scheduling and numerical evidence should not be conflated.

| Layer | Main implementation | Contract |
| --- | --- | --- |
| FX analysis and tiny AOT | numpy_runtime.py, aot_compile.py | Restricted operators and shapes, independently compared |
| Baseline decoder | stacked_decoder_v14.py, streaming_decoder_v16.py | Trusted small reference and resident KV state |
| State and effects | semantic_ir.py, kv_lowering_v17.py | Typed SSA, ordered KV writes, atomic publication and rollback |
| Mathematical IR execution | semantic_decoder_v18.py | Interpreter driven by actual typed operations |
| Local pretrained intake | hf_llama_checkpoint_v19.py, sentencepiece_tokenizer_v20.py | Allowed Llama config/tensor schema and explicit hashes |
| Real-model semantics | llama_semantic_v20.py | Rotate-half RoPE, norm epsilon, GQA, SwiGLU, final norm |
| Compiled CPU | torchscript_graph_v21.py | Whole forward graph with valid changing query/cursor inputs |
| Independent native CPU | native_cpu_v28.cpp, native_cpu_v28.py | Statically maintained C++17 decoder with checked model ABI, independent scalar math and transactional KV |
| Execution reuse | compiled_template_v23.py, shared_weights_v24.py | One graph and weight module, many private KV sessions |
| Model ownership | template_registry_v25.py | Canonical semantic identity, bounded LRU and explicit eviction |
| Active KV | session_leases_v26.py | Logical KV byte quotas, release, best-effort abandoned-lease cleanup |
| Live model weights | active_models_v27.py | Exact template-instance accounting until the final lease ends |
| Artifacts and evidence | semantic_bundle_v19.py, llama_bundle_v20.py, memory_planner_v19.py | Canonical reload, hashes, symbolic lifetimes |

### From a verified checkpoint to decoding

The model loader checks local config.json and model.safetensors against a restricted Llama-style model schema. It rejects unknown architectures, unsupported shapes and unexpected weights. The lowerer regenerates a canonical typed SemanticProgram, binding operator order and model content to an identity.

The NumPy reference and independent Hugging Face/PyTorch CPU path supply finite numerical witnesses. CPU TorchScript traces the whole mathematical program; v22 accepts runtime token length and cache cursor rather than recompiling for every position. The graph consumes live KV prefixes and returns only the newly produced suffix tensors. The resident state validates these outputs, then atomically publishes a new cursor.

The v28 compiler also admits a second execution implementation in [independent C++17 native CPU code](V28_NATIVE_CPU_COMPILER.md), checks its outputs against NumPy and a pinned Hugging Face checkpoint, and publishes K/V through the same transactional runtime. This is a **fixed closed-world Llama backend**, not general IR-to-C++ program synthesis, a signed portable artifact, or a GPU accelerator.

A verified template can be shared across client sessions. Session K/V storage remains private. Model eviction removes **registry ownership**, not a model already leased by an active session. Separate quotas account for cached model buffers, active private KV buffers and active weight modules. These quotas are **logical accounting**, not physical RSS limits.

### Core state invariants

1. Only compiler-admitted, checkpoint-bound graphs reach the trusted compiled runtime.
2. Stateful KV updates commit after complete validated execution; a failure does not publish a partial cursor.
3. Shared executable and immutable weight modules are distinct from every client's private mutable KV.
4. An active client may survive model-cache eviction; new acquisition of an evicted model fails until readmission.
5. Distinct module instances can coexist after eviction/readmission even with an identical semantic digest. V27 counts both while leased.
6. Session access is single-owner; closing and stepping simultaneously is not supported.
7. Numeric comparison is established for explicit witnesses and tolerances, **not** for all possible network inputs.

## Verification and deployment

The [main CI workflow](../.github/workflows/ci.yml) runs Python pytest and Ruff plus frontend lint, tests, typecheck, build and Prettier. The [pinned-model workflow](../.github/workflows/v19-pretrained-probe.yml) independently verifies trained Tiny Llama weights and tokenizer against Hugging Face. Evidence lives in [benchmarks](../benchmarks/).

This is an experimental CPU inference runtime, not a public multi-tenant model-hosting service, a general native IR code generator, a signed portable AOT deployment system, or a physical optical processor. Deploying arbitrary models requires hardened OS-level isolation, resource limits, and separate threat modeling.

See the [Compiler Guide](COMPILER_GUIDE.md), [Verification](VERIFICATION.md), and [historical milestones](README.md).

## Public research roadmap

The browser route **/roadmap** is an evidence-weighted public ledger, with the reviewed snapshot in [roadmap.json](../visuaml-client/public/roadmap.json). Each acceptance gate declares a weight, status and evidence; percentages are computed from verified gates. Python and Vitest validate the method. This is a **versioned deployment snapshot**, not a live background monitor or universal percentage of scientific completeness. See [roadmap methodology](ROADMAP.md).
