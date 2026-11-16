# VisuaML v23 — Reuse one validated Llama executable across independent sessions

## Problem

V22 demonstrated a single TorchScript graph that admits changing token-chunk
lengths and KV-cache cursor values without retracing. However, directly
constructing a new polymorphic decoder session still creates and traces its
own graph, even when serving another conversation with the same frozen model.

V23 introduces an explicit CompiledLlamaTemplate, scoped to a verified,
checkpoint-bound Llama SemanticProgram and batch contract. It compiles once,
validates the resulting logits and all KV states against the independent
NumPy semantic executor, and retains a read-only TorchScript executable in
process memory.

Subsequent new_session() calls reuse the graph object while obtaining a
**new KVSession**, new cursor, and independent snapshot/restore state.
The first compilation and publication of the shared executable occur under
a reentrant lock to avoid racing multiple template creators. Distinct
sessions may then run without sharing their mutable KV state.

This is source-bound **process-local graph sharing**, not a persistent
on-disk native AOT artifact, remote execution service or cache of arbitrary
untrusted Python functions. A session remains single-owner, even though
multiple sessions can share a read-only executable.

## API

    checkpoint = load_hf_llama_local(
        "/path/to/local/checkpoint", declared_revision=PINNED_40_CHAR_SHA
    )
    program = make_llama_semantic_program(checkpoint, batch=1)
    plan = lower_llama_semantic_program(program, checkpoint, batch=1)
    template = CompiledLlamaTemplate(plan)
    session_a = template.new_session()  # traces and validates once
    session_b = template.new_session()  # reuses identical graph
    assert template.report()["compiled_graphs"] == 1

The template enforces checkpoint and semantic identity at creation and
uses the same v22 cursor-polymorphic interface for all sessions.

## Tests

- 1-, 2-, and 4-layer admitted local Llama fixtures.
- Three independent KV sessions sharing exactly the same graph object.
- Separate histories and unrelated reset behavior across conversations.
- Source checkpoint mutation after template construction cannot silently
  alter the frozen compiled model.
- Concurrent template session creation under a lock produces only one trace.
- An actual pinned trained Tiny Llama integration workflow compares template
  sessions against the same independent Hugging Face CPU reference.

Reproduce from repository root:

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_compiled_template_v23.py

## Important remaining costs

V23 reuses compiled graphs within one process, but each new stateful session
still runs canonical plan validation, constructs its own resident KV cache,
and incurs some model-side bookkeeping. It does **not** establish zero-copy
model weight sharing across session objects, resource-safe multi-process
inference, eviction of compiled templates, or persistent offline graph
serialization. The model's memory cost and physical RSS have not been
measured under realistic server concurrency.

Next work should introduce an explicit ownership/type-safe model-weight
arena, measured cache/graph lifetime policies, and separation of cold
compilation, warm inference and session setup costs.


## Pinned real-checkpoint cold-session measurement

Measured for manojredhat/tiny-llama at revision
756e9746eb801752d69771ef3e9bc433b21d853d and weights
SHA-256 d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03.

An independently completed GitHub Actions run compared four distinct
sessions executing the same pinned trained model. Across three trials
(all with identical input and full-logit parity checks):

| Cold-session measurement | Four separate traces | One shared compiled template |
| --- | ---: | ---: |
| Compiled graphs | 4 | 1 |
| Median trace construction, ms | 277.326 | 67.448 |
| Median wall time, ms | 793.483 | 662.101 |
| Observed wall p95, ms | 871.408 | 665.495 |

The approximately 16.6% reduction in observed total cold-session wall
time is smaller than the approximately 75.7% reduction in tracing
time because source admission, per-client plan/weight handling and private
KV storage allocation continue to incur costs in each new session.
Trace time is part of wall time and MUST NOT be added to it.

Source-bound evidence:
benchmarks/pretrained_llama_shared_template_v23.json
and workflow https://github.com/Caerii/VisuaML/actions/runs/43914346114.

These experiments do not establish throughput under concurrent serving,
true physical memory savings, or an offline persistable AOT executable.
