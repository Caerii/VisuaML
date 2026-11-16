# VisuaML

![VisuaML logo](visuaml-client/public/visuaml_logo.png)

**Explore neural networks together. Understand their semantics. Compile and verify their execution.**

VisuaML is an open-source research platform combining interactive, collaborative PyTorch graph visualization with an experimental typed-intermediate-representation compiler and inference runtime. The web application helps researchers inspect model structures and tensor relationships. The Python Compiler Lab investigates how to represent, transform, execute, compare, and safely reuse neural computation across different execution strategies.

**Latest research milestone: v27 (November 2026).** The client package still reports version 0.0.0; these milestones are *not* published semantic-version releases.

[Documentation](docs/README.md) · [Compiler guide](docs/COMPILER_GUIDE.md) · [Architecture](docs/ARCHITECTURE.md) · [Verification](docs/VERIFICATION.md) · [Security and limitations](docs/SECURITY_AND_BOUNDARIES.md)

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

The Compiler Lab represents a supported computation in a typed Semantic IR, explicitly recording tensor types, operations, state effects, and checkpoint identity. The most advanced validated real-checkpoint workload is a restricted small Llama decoder: local safetensors, SentencePiece, independent numerical oracles, and a CPU TorchScript whole-decoder execution path.

| Subsystem | Implemented scope |
| --- | --- |
| Architecture Atlas | 59 **catalogued** families; not 59 fully runnable model families |
| Semantic representation | Tensor/value types, closed-world operators, ordered state effects and admission guards |
| Checkpoints | Strict local unsharded Llama safetensors and compatible tokenizer settings |
| CPU execution | NumPy reference, selected TorchScript primitives, fused regions and whole-decoder tracing |
| Dynamic decoding | One traced Llama graph reused across different valid chunk lengths and KV cursor values |
| Model reuse | Verified graph/weight sharing between sessions; disjoint transactional K/V buffers |
| Model and session lifecycle | LRU template ownership, explicit KV leases and logical active-model weight accounting |
| Evidence | Negative tests, independent numerical witnesses, pinned model revisions, source-bound CPU benchmarks |
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

The compiler's actual limitations matter as much as its capability: no arbitrary Hugging Face family support, no native standalone AOT binary, no production GPU or optical backend, and no formal proof of all-input model equivalence. Read the [Compiler Guide](docs/COMPILER_GUIDE.md).

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

## Why the compiler research matters

The goal is a computational research environment that can separate **what an algorithm means** from **how it runs**. Once that boundary is explicit and testable, researchers can search execution strategies, use independent numerical verification, and investigate hardware-specific transformations without casually weakening semantics.

A few measured milestones illustrate the approach, each on a narrowly defined, short CPU workload:

- **v21–v22:** whole-decoder TorchScript graphs and dynamic KV cursors reduced retracing; compilation costs were measured separately from warm execution.
- **v23–v24:** sharing a verified graph and its weight-bearing module reduced measured four-session cold startup from approximately 532 ms to 256 ms in one three-trial CI benchmark.
- **v25–v27:** model-template LRU eviction, private KV leases, and active model lifetime accounting make resource ownership more explicit; these are **logical**, not OS-enforced physical memory budgets.

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

Planned research includes independent native CPU/GPU AOT targets, physical memory and long-context profiling, model-family generalization under explicit semantics, safe hardware co-design and collaborative interpretability. These are plans, **not shipped features**. See [research directions](docs/FUTURE_DIRECTIONS.md).

Contributions involving model support, optimizations or new backends should include independent oracles, failed-input tests, provenance, and honest performance boundaries. See [Contributing](docs/CONTRIBUTING.md).

Licensed under the [MIT License](docs/LICENSE).
