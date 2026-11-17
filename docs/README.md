# VisuaML documentation

VisuaML combines a collaborative neural-network graph application with a separate experimental typed-IR compiler and verified runtime. This index distinguishes **implemented, numerically exercised capabilities** from **historical designs, simulated hardware and research proposals**.

## Essential guides

| Start here | When to read it |
| --- | --- |
| [Project README](../README.md) | Product overview, quick start and accurate current scope |
| [Public Research Roadmap](ROADMAP.md) | Auditable progress percentages, acceptance gates, global access and review methodology |
| [Compiler Guide](COMPILER_GUIDE.md) | Actual model compatibility, typed IR, backends and executable examples |
| [Architecture](ARCHITECTURE.md) | Full-stack data flow, compiler modules, state and ownership |
| [Verification](VERIFICATION.md) | Tests, pinned real-checkpoint witnesses, benchmarks and numerical evidence |
| [Security and Boundaries](SECURITY_AND_BOUNDARIES.md) | Untrusted model imports, hash limits, memory and isolation boundaries |
| [Client Setup](CLIENT_README.md) | Running the web app, Fastify API, Yjs service and Python backend |
| [Contributing](CONTRIBUTING.md) | CI gates, benchmark acceptance and maintaining these documents |

## Compiler research history

Version labels are **research milestones**, not published package versions.

| Milestones | Subjects |
| --- | --- |
| [v2](COMPILER_LAB_V2.md), [v3](COMPILER_LAB_V3.md), [v4](COMPILER_LAB_V4.md), [v5](COMPILER_LAB_V5.md) | Early compiler, graph and verification contracts |
| [v6 hardware](HARDWARE_CALIBRATION_V6.md), [v6 state](STATEFUL_ASSEMBLY_V6.md), [v7](EFFECT_PROGRAM_V7.md), [v8](MULTI_AREA_AND_EVIDENCE_V8.md), [v9](MULTIFIBER_AND_EMPIRICAL_V9.md), [v10](COMPILER_LAB_V10.md) | Effectful computation and **simulated** hardware/assembly placement |
| [v11](ARCHITECTURE_ATLAS_V11.md), [v12](ARCHITECTURE_BLOCK_CERTIFICATION_V12.md) | Architecture catalog, operator audit and bounded reference probes |
| [v13](DECODER_AOT_V13.md), [v14](STACKED_DECODER_V14.md), [v15](PERFORMANCE_V15.md), [v16](STREAMING_KV_V16.md) | CPU AOT examples, stacked decoders and bounded KV |
| [v17](KV_SEMANTIC_LOWERING_V17.md), [v18](SEMANTIC_DECODER_V18.md) | State-effect IR and executable decoder mathematics |
| [v19](V19_ARTIFACTS_BACKENDS_AND_COMPATIBILITY.md), [v20](V20_REAL_LLAMA_SEMANTIC_PIPELINE.md) | Real safetensors, SentencePiece, numerical witnesses and typed Llama IR |
| [v22](V22_POLYMORPHIC_LLAMA_COMPILATION.md), [v23](V23_SHARED_COMPILED_LLAMA_TEMPLATE.md) | Whole decoder graph, polymorphic cursor and shared executable |
| [v24](V24_SHARED_WEIGHTS_RUNTIME.md), [v25](V25_VERIFIED_TEMPLATE_REGISTRY.md), [v26](V26_LEASED_KV_SESSION_ACCOUNTING.md), [v27](V27_ACTIVE_MODEL_LIFETIME.md) | Shared model weights, registry eviction and explicit lifetime accounting |
| [v28 native CPU](V28_NATIVE_CPU_COMPILER.md) | Independent scalar C++17 model execution and pinned Llama numerical certification |

For complete source-bound numeric evidence, visit [benchmarks](../benchmarks/) and read [Verification](VERIFICATION.md). The Architecture Atlas catalogues 59 families; this **does not** certify 59 pretrained model families.

## Web app, research vision and collaboration

- [Multiplayer](MULTIPLAYER.md): Yjs/WebSocket collaboration and presence.
- [Research directions](FUTURE_DIRECTIONS.md): proposals, including interpretability and categorical computing, not a shipped feature list.
- [Future-directions index](future-directions/README.md): research thesis documents.
- [Integration summary](INTEGRATION_SUMMARY.md) and [bidirectional integration](BIDIRECTIONAL_INTEGRATION_SUMMARY.md): historical engineering notes.
- [Contributors](CONTRIBUTORS.md) and [MIT license](LICENSE).

## Evidence levels

Catalogued family → executable microprobe → independent reference parity → pinned real checkpoint → compiled backend execution → physical device calibration. Each rung must be established separately. Simulated optical results are *not* physical hardware measurements, and finite numerical witnesses are *not* all-input formal proofs.
