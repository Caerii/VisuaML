# Security and capability boundaries

**VisuaML is research software. Do not treat Python model imports as safe merely because they are part of a visualization workflow.**

## Trusted versus untrusted code

The default Fastify API restricts Python imports to shipped model allowlists. Untrusted model-upload execution is disabled. The developer flag **VISUAML_ENABLE_UNSANDBOXED_PYTHON_IMPORTS=1** explicitly opts into unsandboxed Python imports in local development; it does not confine filesystem, network, process, or credential access. Do not use it on an internet-facing service.

The Python FX pipeline can execute model code during import and tracing. A future arbitrary-model service would require isolated workers, resource controls, network/file restrictions, timeouts and auditable capabilities. Those protections are **not** part of the current web app.

The Compiler Lab fixed-fixture endpoints accept allowlisted scenarios and bounded seed/layer inputs. Enabling **VISUAML_COMPILER_LAB_ENABLED=1** does not confer permission to execute arbitrary uploads.

## Models, hashes and trusted provenance

The experimental Llama importer reads *local* config.json and model.safetensors, rejects unsupported architecture variants, limits tensor names, sizes, dtypes and numerical finiteness, and avoids pickle or model-supplied Python. SentencePiece import similarly checks bounded local files and explicit tokenizer parameters.

A SHA-256 hash establishes that two pieces of content match. It is **not** a signature or proof that an author, repo, or model license is trustworthy. A supplied revision SHA is a claim unless compared with independently obtained source metadata. Canonical artifact admission rejects unapproved graph rewrites even when an attacker rehashes an untrusted manifest; this is still not a formal proof of semantic equivalence for all inputs.

## Resource ownership

- The compiler's KV state is transactional and belongs to one session.
- A shared model template may serve multiple sessions without sharing their mutable KV.
- Evicting a cached model does not invalidate a previously acquired session.
- V26 tracks logical private KV bytes under session leases; V27 additionally tracks distinct weight-bearing module objects retained by live leases, including after eviction and readmission.
- These counters do **not** cover TorchScript graph memory, temporary activations, PyTorch native allocations, physical process RSS, or resources created outside the registry.
- Single-session step/close races are not supported. Hostile in-process Python can inspect objects; this is not a multi-tenant security boundary.

## Accuracy, hardware and public claims

A model family appearing in the Architecture Atlas means it was **catalogued**, not that pretrained weights, training, or inference for that family have passed. Numerical parity tests establish agreement only for the pinned models, shapes and tolerances exercised. Small CPU measurements are not evidence of production inference superiority.

Optical placement, attenuation and noise modules are research **simulations**, not physical photonic accelerator benchmarks. Native GPU and standalone CPU AOT execution remain future work.

Do not commit secrets or personal information. Environment variables prefixed with VITE_ are sent to the browser build, so never store secrets in them. Handle security reports privately rather than publishing exploitable details or tokens in GitHub issues.

For implementation scope, see [Architecture](ARCHITECTURE.md); for exact evidence, see [Verification](VERIFICATION.md).
