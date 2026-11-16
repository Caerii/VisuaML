# VisuaML v26 — Explicit live-session leases and logical KV accounting

## The next coordination boundary

V23 enabled one compiled Llama graph to serve multiple private KV sessions.
V24 eliminated redundant model-weight module creation; v25 added a bounded
LRU registry for multiple models. V25's cache weight budget does not capture
private KV memory held by active inference sessions, especially after
their model has been evicted from the registry.

V26 introduces LeaseManagedLlamaRegistry and LeasedLlamaSession, with a
separate logical KV byte budget and explicit maximum concurrently held
session count. The subclass overrides the unleased session factory, so
its intended API does not silently bypass quota accounting.

Before admitting a new session the registry calculates the exact bound
for each client's resident FP32 K and V buffers:

    batch × layers × max_context × kv_heads × head_dim × 2 × 4 bytes

Under the same registry lock it checks remaining capacity and active
session count, then allocates only the new client's private cache.
The actual allocated NumPy buffer sum must agree with this formula
before accounting is committed. If a request exceeds a logical budget,
it is rejected without allocating another live model session.

The resulting lease provides step, generate, snapshot, restore and reset,
and acts as a Python context manager. close() is idempotent. Closing
clears the private resident KV storage and releases its logical byte
reservation, even when an exception escapes a with-statement.
A weakref finalizer provides best-effort cleanup for abandoned leases,
but callers should use explicit context managers or close because GC
scheduling is nondeterministic.

Eviction of a cached *model* and closure of an acquired *session* are
distinct operations. Removing a model's registry entry does not abort
ongoing inference. The live lease retains its verified executable,
may finish and commit additional KV effects, then releases its own
logical reservation when closed.

## Invariants and tests

- Active session count and logical KV byte budgets are checked before
  each allocation, using a synchronous lock to avoid concurrent races.
- Multiple sessions using one verified model preserve independent KV
  cursors and numeric parity with the independent NumPy Llama reference.
- The second or third lease is denied when a quota is full. Releasing
  one lease makes capacity available without retracing the model.
- An already acquired lease can continue executing after its model's
  registry entry is evicted. New cache lookups for the evicted model
  are rejected.
- Context-manager exception handling, idempotent explicit close and
  a best-effort weakref finalizer clear KV contents and release
  registry accounting.
- Any further step, generate or snapshot on a released lease is rejected.
- A dedicated workflow tests quota enforcement, trained Tiny Llama
  output equivalence and eviction-survival on the pinned checkpoint.

## Usage

    registry = LeaseManagedLlamaRegistry(
        max_cached_weight_bytes=256 * 1024 * 1024,
        max_active_kv_bytes=64 * 1024 * 1024,
        max_active_sessions=16,
    )
    key = registry.register(verified_llama_plan)

    with registry.new_session(key) as session:
        logits = session.step(token_ids)
        print(session.cursor)

    assert registry.report()["active_sessions"] == 0

The checkpoint and verified Llama SemanticProgram are admitted via the
existing v19/v20 signed-by-content-hash (not cryptographically signed)
contract. No arbitrary checkpoint code is executed.

Run the focused tests:

    PYTHONPATH=visuaml-client/backend python -m pytest -q \
      visuaml-client/backend/tests/test_session_leases_v26.py \
      visuaml-client/backend/tests/test_template_registry_v25.py

## What this does not guarantee

The live KV budget counts only *logical capacity* of leased NumPy KV buffers.
It is NOT a hard physical RSS limit and does not account for:

- Temporary activations, PyTorch allocator overhead and traced graph buffers.
- Active weights whose cache entries were evicted but which live leases retain.
- Unleased sessions constructed through other developer APIs in-process.
- Concurrent mutation of one lease from several threads, or close racing step.
- Snapshots copied out of the lease prior to close, which are owned by callers.

For a production, multi-tenant system, true OS process isolation, separate
memory budget enforcement, runtime cancellation contracts and a host-side
capability broker are still needed. The software remains a CPU TorchScript
research runtime, not a deployed inference server.

## Next research/engineering steps

1. Associate tracked model-weight ownership with all active leases. Maintain
   an active-template budget independent from cache LRU residency; count
   evicted-but-live model modules until their final lease closes.
2. Measure actual allocator and process RSS across clients, prefill, decode,
   errors and cancellations, enforcing hard limits via process isolation.
3. Implement a typed KV page allocator or a reusable bounded session pool
   with strict aliasing and transactional rollback contracts.
4. Build independently deployable native AOT CPU/GPU execution artifacts
   with deterministic provenance and full-model parity evidence.
5. Use these foundations to support a governed multi-model scheduler with
   batch-aware prefill/decode and explicit cost-based backend selection.
