# VisuaML v27 — Accounting for active model modules after eviction

The v25 LRU cache counts model templates it currently owns. V26 separately accounts for each active inference session's private KV buffers. Yet a model evicted from the LRU can remain alive because an existing session still holds the compiled executable and its weight-bearing module.

**The missing quantity is active model-module lifetime.**

V27 introduces ActiveModelLlamaRegistry. When the first live lease acquires a verified model module, its logical weight-buffer bytes enter the active-model budget. Later leases sharing that same template do not duplicate the count. When the final lease closes, the module leaves the active account.

Importantly, this tracks **actual template object identity**, not merely semantic SHA-256. If a model is evicted but still referenced by one client, then the *same* checkpoint is admitted into a *second* compiled module, both module instances count until their own leases close. Content identity does not imply a single in-process object.

A model-weight budget can reject a new module while allowing additional leases against an already counted shared model. Registry cache removal does not cancel a live lease, and releasing the final lease removes its corresponding active-weight charge.

## Example

~~~python
registry = ActiveModelLlamaRegistry(
    max_cached_weight_bytes=256 * 1024 * 1024,
    max_active_model_weight_bytes=256 * 1024 * 1024,
    max_active_kv_bytes=64 * 1024 * 1024,
    max_active_sessions=16,
)
key = registry.register(verified_llama_plan)
with registry.new_session(key) as session:
    logits = session.step(token_ids)
assert registry.report()["active_sessions"] == 0
~~~

See [active_models_v27.py](../visuaml-client/backend/visuaml/compiler_lab/active_models_v27.py) and [its tests](../visuaml-client/backend/tests/test_active_models_v27.py). The tests cover shared-model reuse, eviction-survival, re-admission of identical weights as a distinct module, quota failure, and last-lease release.

This is **logical model-module buffer accounting**, not process RSS, a native allocator quota, a cross-process cache or protection from hostile in-process Python. See [Security](SECURITY_AND_BOUNDARIES.md) and [Architecture](ARCHITECTURE.md).
