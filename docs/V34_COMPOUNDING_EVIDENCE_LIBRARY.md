# v34 — A compounding library of computational evidence

*The real unit of progress is not a new kernel. It is a finding whose assumptions, independent witnesses and implementation identity remain intelligible and reusable after its original experiment.*

VisuaML now has several historically distinct kinds of compiler evidence: bounded native Llama/attention observations, source-generated MLP and convolution computation witnesses, guarded PyTorch-exported programs, and an initial evidence-gated transformation. Before v34 these existed as separately curated JSON documents, tests and CI artifacts. v34 makes them **discoverable in a single versioned evidence corpus** without pretending that observations are mathematical proofs or universally transferable performance claims.

## 1. The initial, reviewed corpus

The deterministic v34 catalog indexes **12 observations from three source artifacts**, not twelve independent model families:

| Source | Indexed observations | Provenance and limitation |
| --- | ---: | --- |
| [v31 online attention](../benchmarks/native_online_attention_observation_v31.json) | 4 context-sweep observations | One pinned Llama and three synthetic shape studies; historically curated summary, mixed results |
| [v32 native semantic code generation](../benchmarks/native_semantic_codegen_observation_v32.json) | 4 generated-graph observations | Synthetic declared MLP/Conv and restricted trusted FX counterparts; curated CI summary |
| [v33 export and rewrite experiment](../benchmarks/export_evidence_observation_v33.json) | 4 records: 2 exported models, direct rewrite witness and independent optimization-time observation | Full source-bound original CI JSON, generated C++ and compiler fingerprints, audited [origin byte hash](../benchmarks/export_evidence_observation_v33.origin.json) |

The v33 rewrite and optimizer numerical receipts are **different records** because they were produced by different compiler invocations. Their canonical semantic, generated source and native-binary identities agree; their record digests differ legitimately with compilation measurements. They are intentionally indexed separately, with exact receipt identifiers, rather than collapsed or falsified.

## 2. Versioned scientific identities

The standard-library-only `visuaml_evidence.library_v34` package exposes immutable `EvidenceEntry` records. Each includes:

- A content hash of the exact source JSON bytes and deterministic study identifier
- Evidence schema, kind, source file, CI workflow and (where present) source commit
- Semantic IR, generated C++ and native binary hashes **as distinct identities**
- Checkpoint fingerprint when the historical record actually provides one
- Operator sequence, model/geometry/target/compiler context where available
- Finite observed results, evidence class, status and original limitations

**None of those identities substitutes for another.** Two compilations can have the same generated native binary while relying on different input data, parameter hashes, numerical tolerances or evaluator contexts. A shared hash is a lineage edge: it makes related studies findable; it does not establish the correctness of one on the other's inputs.

The exact-match `query()` API supports semantic digest, native binary hash, checkpoint hash, operations, evidence kind and context. Missing metadata never acts as a wildcard when an explicit value is requested. `neighbors()` reports exact native binary or source identity and no inferred performance portability.

## 3. Trust and admission tiers

| Tier | Meaning | Automatic promotion? |
| --- | --- | --- |
| Reviewed full source-bound artifact | Original archived CI bytes and constituent receipts checked against their recorded hashes | No |
| Curated historical observation | Reviewable source and workflow links, with acknowledged limits on raw evidence retention | No |
| Unreviewed proposal | New append-only SHA-addressed bounded JSON, stored outside the catalog | Never |

The initial library has only **three explicitly approved source paths**. Unknown schemas fail admission. The v33 complete artifact must match its independently checked-in historical byte digest, each generated C++ source must match its compiler receipt, and the direct and optimization-time rewrite receipts must independently satisfy their canonical digests and exact transformation identity.

Unreviewed proposals are stored by `sha256(data)` under a separate `proposals/` directory. Every proposal also receives an immutable `*.origin.json` sidecar binding its exact artifact digest to an approved-format source CI URL and explicit `unreviewed` status. A missing, altered or mismatched sidecar fails verification. Duplicate bytes with conflicting origin are not silently merged. Interrupted partial writes remain invalid until deliberately reconciled. They cannot replace the source-controlled reviewed catalog, and submitting a proposal cannot change the public roadmap. This is an append-only research staging mechanism, **not an authenticated review system, remote upload service, signature scheme or hostile-input sandbox**.

The library does not import PyTorch, compile or execute code, run dynamic Python or load pickle. It is intentionally isolated from the application package's heavyweight ML imports so the evidence index can be audited cheaply and in constrained environments.

## 4. Reproduction and queries

From the repository root:

```bash
PYTHONPATH=visuaml-client/backend python -m pytest -q visuaml-client/backend/tests/test_evidence_library_v34.py

PYTHONPATH=visuaml-client/backend python visuaml-client/backend/scripts/build_evidence_library_v34.py --check
```

The build command without `--check` generates the exact public `visuaml-client/public/evidence/library-v34.json` from reviewed source files. This file is **generated, not manually edited**. Its catalog digest covers ordered studies, source hashes, acceptance classes and limitations. CI checks that the checked-in bytes match regeneration. Source changes, numerical claims or benchmark outcomes that were not reviewed cannot be quietly introduced as additional accepted observations.

Search the historical corpus programmatically:

```python
from pathlib import Path
from visuaml_evidence import EvidenceLibrary

library = EvidenceLibrary.from_repo(Path("."))
for study in library.query(
    kind="native_generated_graph",
    op_sequence=("matmul", "relu", "matmul", "add"),
):
    print(study.label, study.native_binary_sha256, study.workflow_url)

# A shared binary digest is a lineage relation, not a proof.
print(library.neighbors(library.entries[0].entry_id))
```

## 5. The browser is a research index, not a black box

The new read-only `/research/library` route is accessible from `/research` and Compiler Lab. It loads the static catalog with **no Python backend**, validates a restricted source-bound schema, and supports searching by computation family, operator, evidence kind, and recorded context. Each observation exposes its exact provenance, source artifact, finite numerical results, native identity, limitations and other records that share a binary fingerprint.

A selected study's failed or non-winning observation remains visible. The default attention backend is not changed by discovering that online attention won on one narrow synthetic geometry. The ReLU rewrite's finite numerical success is also distinguished from its initially non-winning latency comparison.

The client parser checks that the published source paths and workflow links are within the known VisuaML set, and its tests reject malformed digests, inflated counts and unsafe metadata. This is **structural validation**, not cryptographic authentication. The deterministic Python catalog generator plus reviewed source control is the stronger provenance boundary.

The source merge is distinct from website deployment. `visuaml.com/research/library` and `visuaml.com/roadmap` need independent production host verification, still tracked in [issue #8](https://github.com/Caerii/VisuaML/issues/8).

## 6. How this should compound

A genuinely useful next generation needs a **model and experiment knowledge graph**, not just a bigger index:

1. Admit a second pinned real pretrained architecture through the same IR-to-native pathway, with independent model and checkpoint provenance. This remains [issue #14](https://github.com/Caerii/VisuaML/issues/14).
2. Add versioned proof obligations to each transformation family: admissible shape/effect guards, finite witnesses, failing counterexamples and—where feasible—independent translation validation. Do not pretend finite experiments are formal proofs.
3. Build a reusable schedule/cost search over **exactly matched** tensor geometries, device ISA, numeric contract and compiler flags. Keep raw samples, compile costs and regressions; never blindly generalize a CI-runner speedup.
4. Introduce authenticated reviewer approval separately from content hashing. A stable digest verifies bytes; it does not identify or authorize an author.
5. Let Compiler Lab drill from graph node to typed operation, generated implementation, source-bound evidence, counterexample and historical alternatives. Every link should reuse a known identity instead of copying unverifiable prose.

**This v34 milestone intentionally does not change the roadmap's 72.5% reviewed snapshot.** Catalog size counts observations, not overall research completion, percent of supported architectures or scientific proof strength.
