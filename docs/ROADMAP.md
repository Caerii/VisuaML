# The public VisuaML research roadmap

*A legible record of computational progress is more useful than a reassuring percentage.*

The open [research roadmap](https://visuaml.com/roadmap) is designed to let collaborators anywhere examine the same accepted results, the remaining uncertainty, and the next scientific or engineering proof obligations. It belongs to the open-source repository, not to a private dashboard.

**Deployment status:** the route is implemented in the client. Its presence on visuaml.com depends on successfully deploying the commit containing it and on the host supporting direct client-side routing. A GitHub merge alone does not establish that production is deployed.

## What a percentage means

Each research field contains specific acceptance criteria. Each criterion has a name, weight, description, status, and—if verified—at least one link to supporting evidence. Criteria inside each field must total 100 points. The eight fields also receive weights that total 100 portfolio points.

Only a criterion marked **verified** contributes its weight to completion. Work that is merely *in development* or *planned* earns zero points. This prevents a polished interface or an ongoing experiment from masquerading as a demonstrated scientific result.

The calculations are:

~~~text
Field completion = sum(weight of its verified acceptance criteria)

Total completion =
  sum(field portfolio weight × field completion) / 100
~~~

The v28-reviewed snapshot is **72.5% weighted acceptance** (rounded to 73% on screen). This number is **not** a probability of success, percentage of total research completed in any absolute sense, estimate of developer hours remaining, or promise of a delivery date. It describes one versioned rubric that may itself grow as new scientific obligations are discovered.

The public [roadmap.json](../visuaml-client/public/roadmap.json) is the canonical snapshot. Percentages are computed by the web client, not duplicated in a second, unreviewed source. The JSON schema is checked both by the TypeScript runtime and Python/Vitest tests.

## The eight workstreams

| Research domain | Portfolio weight | Core question |
| --- | ---: | --- |
| Collaborative research workbench | 15% | Can people jointly examine computational structure and trustworthy evidence? |
| Executable semantic representation | 20% | Is mathematical and stateful meaning explicit enough to validate and transform? |
| Real-model and tokenizer fidelity | 15% | Can an actual trained checkpoint preserve its behavior through the compiler? |
| Independent compilation and hardware | 20% | Can the same admitted semantics execute on genuinely different software/hardware targets? |
| Search, fusion and execution economics | 10% | Which transformations improve costs without silently weakening correctness? |
| Evidence, provenance and safety | 10% | What exactly was tested, against which independent oracle, and with which limitations? |
| Shared execution and resource ownership | 5% | Who owns weights and state through admission, eviction, cancellation, and release? |
| Hardware–algorithm co-design | 5% | Do physical measurements actually validate our hardware models and placement choices? |

The roadmap intentionally assigns lower progress to independent GPU and physical-hardware support than to earlier semantic and numerical work. Architecture family names, simulated optical effects, and general interest do not earn the same status as executable numerical witnesses.

## Active research without a percentage increase

The [merged v29 native-performance milestone](https://github.com/Caerii/VisuaML/pull/7) established RoPE reuse, four-row and runtime-guarded AVX2 kernels, repeatable CPU timings and physical-memory probes. Its [methodology](V29_NATIVE_CPU_PERFORMANCE.md) and [source-bound pretrained CI witness](https://github.com/Caerii/VisuaML/actions/runs/44124083185) are independently inspectable. Initial short-workload AVX2 results are promising, but do **not** by themselves satisfy the roadmap's unverified GPU, portable AOT, representative serving, physical-energy or model-family criteria. No existing status or portfolio percentage should change merely because a draft experiment is fast.

The [merged v30 opt-in batched prefill experiment](https://github.com/Caerii/VisuaML/pull/10) extends this evidence with CPU-guarded, shape-aware two-token AVX2 weight reuse. Its [methodology](V30_NATIVE_PREFILL.md) records failed scalar and early SIMD variants, a five-geometry numerical and latency sweep, and a same-runner v29→v30 default-path comparison. Early improvements are limited to wider **synthetic** geometries, while a pinned real-model witness verifies numerical compatibility. These findings do **not** satisfy GPU, portable AOT, additional pretrained-family, representative serving, or physical-energy criteria. Reviewed roadmap acceptance remains unchanged.

The [v31 optional online-attention experiment](https://github.com/Caerii/VisuaML/pull/11) adds stable streaming softmax/value accumulation alongside the existing score-materializing baseline. The [bounded methodology](V31_ONLINE_ATTENTION.md) and [durable empirical observation](../benchmarks/native_online_attention_observation_v31.json) document independent numerical/KV witnesses, a pinned real-model experiment and three synthetic geometries. The results are mixed; the default attention kernel is unchanged, and nothing here proves long-context production performance or lower physical peak memory. **No public roadmap acceptance weight or status changes as a result of these experiments.**

The [v32 graph-generated native compiler research](V32_NATIVE_SEMANTIC_CODEGEN.md) begins lowering a strict subset of the shared Semantic IR into independently compiled C++17, rather than calling a fixed model-specific decoder. Its [research implementation PR](https://github.com/Caerii/VisuaML/pull/13) contains separate synthetic residual MLP and Conv2D graphs, a restricted trusted FX importer, independent NumPy/PyTorch checks, and source-bound generated-binary receipts. These are **not independently admitted pretrained architecture families**, general native code generation or signed portable AOT packages. The public acceptance percentage and all underlying criteria are unchanged while the broader v32 scientific obligations remain open.

The [v33 guarded export and evidence investigation](V33_EXPORT_AND_EVIDENCE.md) connects functional exported ATen graphs, lifted parameter digests and exact static guards to the graph-generated native Semantic IR compiler. Its first pure ReLU-idempotence rewrite is independently tested against NumPy and PyTorch, preserving failed concrete witnesses and digest-linked (not signer-authenticated) records. These are finite synthetic witnesses, **not** proof of all-input equivalence or independently certified pretrained families. No roadmap acceptance criterion or weighted percentage changes from this draft work.

The [v34 compounding evidence library](V34_COMPOUNDING_EVIDENCE_LIBRARY.md) indexes historical attention, generated native graph, exported-model and transformation experiments through exact source/IR/binary identities. The [research PR](https://github.com/Caerii/VisuaML/pull/16) distinguishes reviewed source-bound records from curated historical summaries and append-only unreviewed proposals. Its catalog counts **observations**, not independent architectures, proven transformations, transferable speedups or formal proof obligations. **No public roadmap acceptance status or percentage changes from creating this index.**

The [v35 replayable computational knowledge graph research](V35_REPLAYABLE_KNOWLEDGE_GRAPH.md) connects the historical v28 **real pinned Tiny Llama** numerical witness to the v31 attention study through exact checkpoint SHA-256, without modifying existing v34 study IDs. Its replay plan explicitly reports code/checkpoint drift and does not perform inference or claim a new independent numerical reproduction. The reviewed roadmap weights and percentages remain unchanged: an additional historical entry is not new compiler coverage or all-input proof.

## The evidence review process

A proposed change to the public roadmap should be reviewed like a compiler change.

1. **State an acceptance condition.** The requirement must be specific enough that someone other than its implementer can decide whether it passed.
2. **Attach a stable source.** Prefer code, independent numerical tests, source-bound benchmark JSON, model hashes, or a CI run. Marketing copy is not evidence.
3. **Distinguish scope.** A fixed synthetic microprobe, a pinned real model, an independent native implementation, and a physical-hardware measurement are separate milestones.
4. **Update only when verified.** Once a requirement actually passes, change its status to verified. The recorded weight remains the basis of arithmetic.
5. **Validate the data.** Run the Python roadmap evidence tests, the TypeScript calculation tests, and the frontend build. The schema refuses verified criteria with missing evidence or weights that fail to sum to 100.
6. **Merge and deploy.** The public JSON is version-controlled. The browser fetches the published snapshot; only after deployment can a viewer at visuaml.com see the update.
7. **Preserve the history.** Keep old commits and source-bound experiment files accessible. Do not rewrite an obsolete result as though it never happened.

A criterion can be revised when the original rubric is no longer scientifically adequate. Such a change should explain what new obligation was discovered and why the percentage moved; percentages are not a monotonic score to optimize.

## The difference between version control and a live monitor

The roadmap is a *reviewed public snapshot*. It is designed to be globally viewable through the production site, but it does not currently stream live GitHub job status or background research results. A future system could publish signed build provenance or visualize CI status while keeping accepted research criteria separate from transient test runs.

A new scientific result becomes part of the public acceptance ledger only when it has been checked and the corresponding JSON change is merged and deployed. This protects the public record from treating every failed CI run, open experiment or auto-generated claim as progress.

## Reproduce

From the project root, after installing the pinned backend and frontend dependencies:

~~~bash
cd visuaml-client
python -m pytest -q backend/tests/test_public_roadmap_v28.py
pnpm exec vitest run src/ui/Roadmap/roadmapModel.test.ts
pnpm exec prettier --check src/ui/Roadmap
pnpm run build
~~~

To inspect the methods, see [Verification](VERIFICATION.md), the [Compiler Guide](COMPILER_GUIDE.md), and the [v28 native backend report](V28_NATIVE_CPU_COMPILER.md). The underlying research work should remain intelligible even if the website changes its design.
