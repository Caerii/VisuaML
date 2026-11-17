# VisuaML

![VisuaML logo](visuaml-client/public/visuaml_logo.png)

**Explore neural networks together. Understand their semantics. Compile and verify their execution.**

VisuaML is an open-source research platform combining interactive, collaborative PyTorch graph visualization with an experimental typed-intermediate-representation compiler and inference runtime. The web application helps researchers inspect model structures and tensor relationships. The Python Compiler Lab investigates how to represent, transform, execute, compare, and safely reuse neural computation across different execution strategies.

**Current research milestone: v29 (November 2026).** The v28 independent native CPU backend and redesigned client are integrated. The v29 research adds an independently compared scalar/four-row execution path, optional CPU-guarded AVX2 acceleration, source-bound paired benchmarks and process-memory evidence. These remain bounded Llama experiments, not a production inference backend or general IR-to-native compiler. The client package still reports version 0.0.0; research milestones are *not* published semantic-version releases.

[v29 native performance research](docs/V29_NATIVE_CPU_PERFORMANCE.md) · [Public research roadmap](docs/ROADMAP.md) · [Documentation](docs/README.md) · [Compiler guide](docs/COMPILER_GUIDE.md) · [Architecture](docs/ARCHITECTURE.md) · [Verification](docs/VERIFICATION.md) · [Security and limitations](docs/SECURITY_AND_BOUNDARIES.md)

[![CI](https://github.com/Caerii/VisuaML/actions/workflows/ci.yml/badge.svg)](https://github.com/Caerii/VisuaML/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](docs/LICENSE)

## What works today

### Explore neural networks as a team

- **Interactive graph inspection:** inspect layers, edges, inferred tensor shapes and spatial representations using React, TypeScript, React Flow and Three.js.
- **Real-time collaboration:** Yjs/WebSockets synchronize shared graph state, cursor presence and editing.
- **Model export:** use the trusted PyTorch FX processing backend and supported JSON, analysis and experimental hypergraph formats.
- **Architecture Atlas:** inspect catalogued architecture families and their required operators, including a live, bounded semantic-decoder evidence panel.

![Interactive graph](docs/media/01-interactive-graph.gif)

The browser can show bundled demo networks without Python; custom Python-backed model analysis requires the API. **Arbitrary uploaded Python is not sandboxed**: untrusted upload execution is disabled by default and the API restricts imported models to trusted allowlists. See [security](docs/SECURITY_AND_BOUNDARIES.md) before enabling developer import flags.

### Compile, inspect and verify bounded neural computation

The Compiler Lab represents a supported computation in a typed Semantic IR, explicitly recording tensor types, operations, state effects, and checkpoint identity. The most developed validated real-checkpoint workload is a restricted small Llama decoder: local safetensors, SentencePiece, independent numerical oracles, CPU TorchScript and a second independent scalar C++17 CPU execution implementation. The native implementation is model-specific, not general IR code generation or production-competitive inference.

| Subsystem | Implemented scope |
| --- | --- |
| Architecture Atlas | 59 **catalogued** families; not 59 fully runnable model families |
| Semantic representation | Tensor/value types, closed-world operators, ordered state effects and admission guards |
| Checkpoints | Strict local unsharded Llama safetensors and compatible tokenizer settings |
| CPU execution | NumPy reference and TorchScript, plus independently compiled native C++17 scalar, four-row and opt-in AVX2 Llama kernels with numerical/KV parity gates |
| Dynamic decoding | One traced Llama graph reused across different valid chunk lengths and KV cursor values |
| Model reuse | Verified graph/weight sharing between sessions; disjoint transactional K/V buffers |
| Model and session lifecycle | LRU template ownership, explicit KV leases and logical active-model weight accounting |
| Evidence | Negative tests, independent numerical witnesses, pinned model revisions, source-bound CPU benchmarks |
| Public research roadmap | Eight weighted workstreams; percentages are derived from verified, source-linked acceptance gates |
| Optical research | **Simulated** placement, calibration assumptions and noise models—not measured physical hardware |

~~~text
Local safetensors + configuration + tokenizer
                     |
           Strict model admission
                     |
       Checkpoint-bound typed Semantic IR
                     |
       Shape, math and effect validation
             /                  \
        NumPy oracle       TorchScript CPU graph
             \                  /
            Numerical comparisons
                     |
        Verified shared model template
                     |
       Private transactional KV sessions
                     |
         LRU ownership and lease quotas
~~~

The compiler's actual limitations matter as much as its capability: no arbitrary Hugging Face family support, no general IR-to-native code generator, no signed portable AOT package, no production GPU or optical backend, and no formal proof of all-input model equivalence. Read the [Compiler Guide](docs/COMPILER_GUIDE.md).

## Run the web application

**Requirements:** Node.js 20, pnpm 10, and Python 3.12 for the optional API/backend (matching the CI baseline).

~~~bash
git clone https://github.com/Caerii/VisuaML.git
cd VisuaML
pnpm install
pnpm --filter visuaml-client run dev
~~~

Open **http://localhost:5173**. The pnpm installation does not install Python dependencies by default.

For backend-backed model analysis and the Compiler Lab:

~~~bash
pnpm run install-python-deps
cd visuaml-client
cp env.example .env.local
pnpm run check-python
cd ..
~~~

Start three separate terminals from the repository root:

~~~bash
pnpm --filter visuaml-client run api
pnpm --filter visuaml-client run ws
pnpm --filter visuaml-client run dev
~~~

For local fixed-fixture compiler inspection only, start the API process with the environment variable **VISUAML_COMPILER_LAB_ENABLED=1**. The guarded endpoints are POST /api/compiler-lab/analyze and GET /api/compiler-lab/semantic-inspect. That opt-in does *not* allow arbitrary Python uploads.

See [client setup](docs/CLIENT_README.md) for details.

## Run compiler tests

After installing the pinned Python dependencies:

~~~bash
cd visuaml-client
python -m pytest -q backend/tests
~~~

Or exercise the end-to-end Llama IR and ownership path:

~~~bash
python -m pytest -q \
  backend/tests/test_llama_semantic_v20.py \
  backend/tests/test_torchscript_poly_v22.py \
  backend/tests/test_shared_weights_v24.py \
  backend/tests/test_template_registry_v25.py \
  backend/tests/test_session_leases_v26.py \
  backend/tests/test_active_models_v27.py
~~~

The separate [pinned pretrained-model workflow](.github/workflows/v19-pretrained-probe.yml) checks a named Tiny Llama source revision, hashes, tokenizer equivalence for specified prompts, PyTorch logits, streaming state and session ownership. Passing finite model witnesses does not imply correctness on arbitrary inputs.

## An open research ledger

The [public roadmap](docs/ROADMAP.md) gives anyone a shared, versioned view of eight workstreams, their weighted acceptance criteria, direct evidence links and unfinished obligations. **Its current reviewed snapshot is 72.5% weighted acceptance** (shown as 73% on the page)—not 72.5% of all possible research, a forecast, or a launch readiness score. The route will become publicly reachable at **visuaml.com/roadmap** only after the updated client is actually deployed with SPA route support. The GitHub merge does not by itself prove deployment.

## Why the compiler research matters

The goal is a computational research environment that can separate **what an algorithm means** from **how it runs**. Once that boundary is explicit and testable, researchers can search execution strategies, use independent numerical verification, and investigate hardware-specific transformations without casually weakening semantics.

A few measured milestones illustrate the approach, each on a narrowly defined, short CPU workload:

- **v21–v22:** whole-decoder TorchScript graphs and dynamic KV cursors reduced retracing; compilation costs were measured separately from warm execution.
- **v23–v24:** sharing a verified graph and its weight-bearing module reduced measured four-session cold startup from approximately 532 ms to 256 ms in one three-trial CI benchmark.
- **v25–v27:** model-template LRU eviction, private KV leases, and active model lifetime accounting make resource ownership more explicit; these are **logical**, not OS-enforced physical memory budgets.
- **v28:** a closed-world, independently compiled scalar C++17 CPU decoder has passed a pinned trained Llama numerical witness. It is a distinct implementation, not a claim of faster native inference.
- **v29:** RoPE angle reuse, per-invocation scratch reuse, four-row linear traversal and CPU-guarded AVX2 have source-bound numerical witnesses. A same-runner paired v28→v29 comparison measured a median **5.45×** AVX2 warm-latency ratio across two pairs on one AMD EPYC 7763 runner; a different shared runner measured a different ratio. See [v29 methodology](docs/V29_NATIVE_CPU_PERFORMANCE.md). Neither result demonstrates general large-model throughput or reduced physical peak memory.

Benchmark sources and complete caveats are indexed in [Verification](docs/VERIFICATION.md) and committed as JSON in [benchmarks](benchmarks/). The project is **not** claiming to outperform mature production inference engines, support all catalogued architectures, or deliver real photonic acceleration.

## Project layout

| Path | Purpose |
| --- | --- |
| [visuaml-client/src](visuaml-client/src/) | React frontend, architecture atlas, graph UI |
| [visuaml-client/server](visuaml-client/server/) | Fastify API and guarded Python orchestration |
| [visuaml-client/backend/visuaml](visuaml-client/backend/visuaml/) | PyTorch analysis/export package |
| [compiler_lab](visuaml-client/backend/visuaml/compiler_lab/) | Typed IR, verified model paths, backends and runtime |
| [benchmarks](benchmarks/) | Source-bound CPU/model numerical evidence |
| [docs](docs/README.md) | Architecture, security, user guides and research history |

## Research directions, contributions and license

Planned research includes *general* independent IR-to-native code generation, optimized CPU/GPU targets, physical memory and long-context profiling, model-family generalization under explicit semantics, safe hardware co-design and collaborative interpretability. These are plans, **not shipped features**. See [research directions](docs/FUTURE_DIRECTIONS.md).

Contributions involving model support, optimizations or new backends should include independent oracles, failed-input tests, provenance, and honest performance boundaries. See [Contributing](docs/CONTRIBUTING.md).

Licensed under the [MIT License](docs/LICENSE).
