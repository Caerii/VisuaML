# VisuaML v25 — Bounded, verified multi-model executable registry

## Problem solved

V23 could reuse one compiled graph across sessions. V24 additionally reuses
one trusted, weight-bearing Torch module, eliminating repeated model snapshots
and per-session module construction. A system managing multiple model
checkpoints still needs an explicit policy for admission, lookup, bounded
cache ownership, eviction and concurrent requests.

VerifiedLlamaRegistry in template_registry_v25.py is a synchronous,
process-local registry of SharedWeightsLlamaTemplate objects keyed by the
canonical semantic SHA-256 digest. The digest binds the exact checkpoint,
model profile, batch contract and typed SemanticProgram.

## Admission and lookup

The registry's register(plan) path:

1. Rejects noncanonical or forged model programs using the established
   Llama Semantic IR lowerer. Invalid model claims cannot enter the registry.
2. Checks a pre-admission logical model-weight byte estimate against the
   requested cached-weight budget.
3. Builds and independently certifies a v24 shared-weight template before
   changing registry ownership. Failure leaves existing cached models intact.
4. Rechecks the verified Torch module's logical weight-buffer bytes.
5. Evicts least-recently-used entries until both the maximum cached model
   count and logical cached-weight byte constraints are satisfied.
6. Atomically admits the new verified template and returns its canonical
   semantic SHA-256 key.

new_session(key) retrieves the already certified template and creates an
isolated, transactional resident KV state without retracing or rebuilding
the weight-bearing model module. A single process-wide reentrant lock
serializes template admission and session-factory selection, avoiding races
and duplicate compilation under concurrent registrations.

The explicit evict(key) operation removes the registry's ownership but does
**not** invalidate previously acquired sessions. Their graph references,
model buffers and KV histories remain valid until those sessions are
released. This distinction prevents cancellation or unexpected model
substitution in the middle of an inference stream.

## Byte budget semantics: do not confuse this with RSS

A registry budget counts logical tensor bytes owned by weight-bearing Torch
modules in the *cached templates*. It does NOT bound:

- PyTorch allocator overhead and other native allocations;
- memory held by TorchScript compiled graphs;
- per-client resident KV cache bytes or temporary activations;
- models evicted from the registry but still referenced by active sessions;
- virtual memory, physical RSS or multi-process allocations.

Consequently this is **bounded cache residency**, not a hard memory safety
quota for the process. Production support requires active-session leases,
lifetime accounting and measured physical peak memory before making such
claims.

## Tests and verified limitations

Tests exercise multiple genuinely different local Llama checkpoints, source
identity, concurrent factory lookup, mismatched model admission, oversized
checkpoint rejection, deterministic LRU eviction, explicit removal, and
continued numerical correctness of previously acquired sessions after
eviction. They compare logits to an independently implemented NumPy Llama
reference and check that private K/V buffers are never shared between
clients.

The separate pinned pretrained Tiny Llama workflow additionally verifies
registry admission, cache eviction, lookup rejection after removal and
continued correct incremental decoding in an already acquired session.

Run the focused tests:

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_template_registry_v25.py \
      visuaml-client/backend/tests/test_shared_weights_v24.py

This work does not provide an API server, multi-process isolation, automatic
graph unloading, tensor paging, CUDA kernels or a portable native compiled
artifact. The model-specific CPU TorchScript execution restrictions and
numeric error budgets remain in force.

## Next critical engineering gates

1. Implement explicit session leases and active-model accounting; track
   evicted-but-live templates until their last session is released.
2. Enforce physical RSS limits using process isolation and measured native
   allocation behavior, not logical estimates alone.
3. Add cost-aware prefill/decode scheduling, batching and state paging with
   strict rollback/aliasing guarantees.
4. Demonstrate a second genuinely independent native AOT or GPU backend
   running the same validated semantic graph, with parity and compile costs.
5. Validate more realistic pretrained checkpoints and longer context windows
   before promoting this registry into a general inference service.
