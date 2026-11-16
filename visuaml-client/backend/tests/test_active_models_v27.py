"""v27: logical active model ownership survives cache eviction until final release."""

from __future__ import annotations

import numpy as np
import pytest

from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.active_models_v27 import (
    ActiveModelBudgetExceeded,
    ActiveModelLlamaRegistry,
)
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.template_registry_v25 import TemplateNotCached


def build(tmp_path, layers):
    directory = tmp_path / str(layers)
    directory.mkdir()
    fixture(directory, layers=layers)
    model = load_hf_llama_local(directory, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    estimate = (
        sum(a.nbytes for a in plan.weights.arrays.values())
        + 2 * plan.weights.profile.head_dim * plan.capacity * 4
    )
    kv = (
        plan.batch * plan.weights.profile.layers * plan.capacity
        * plan.weights.profile.kv_heads * plan.weights.profile.head_dim * 8
    )
    return model, plan, estimate, kv


def test_duplicate_leases_one_weight_module_until_final_release(tmp_path):
    model, plan, cost, kv = build(tmp_path, 1)
    registry = ActiveModelLlamaRegistry(
        max_cached_weight_bytes=cost + 4096,
        max_active_model_weight_bytes=cost + 4096,
        max_active_kv_bytes=max(4096, kv * 3),
        max_active_sessions=3,
    )
    key = registry.register(plan)
    a = registry.new_session(key)
    b = registry.new_session(key)
    assert registry.report()["distinct_active_model_modules"] == 1
    assert registry.report()["active_model_weight_logical_bytes"] == cost
    assert registry.report()["active_sessions"] == 2
    entries = list(registry.report()["active_model_lease_counts"].values())
    assert entries[0]["leases"] == 2 and entries[0]["still_cached"]
    assert registry.evict(key)
    assert registry.report()["active_model_weight_logical_bytes"] == cost
    assert not list(registry.report()["active_model_lease_counts"].values())[0]["still_cached"]
    with pytest.raises(TemplateNotCached):
        registry.new_session(key)
    ids = np.array([[1, 2]], np.int64)
    np.testing.assert_allclose(
        a.step(ids), model.forward(ids), atol=5e-5, rtol=5e-5
    )
    a.close()
    assert registry.report()["active_model_weight_logical_bytes"] == cost
    b.close()
    assert registry.report()["active_model_weight_logical_bytes"] == 0
    assert registry.report()["active_sessions"] == 0


def test_evicted_and_readmitted_identical_digest_count_as_two_modules(tmp_path):
    model, plan, cost, kv = build(tmp_path, 1)
    registry = ActiveModelLlamaRegistry(
        max_cached_weight_bytes=cost + 4096,
        max_active_model_weight_bytes=cost * 2 + 4096,
        max_active_kv_bytes=max(4096, kv * 3),
        max_active_sessions=3,
    )
    key = registry.register(plan)
    old = registry.new_session(key)
    registry.evict(key)
    assert registry.register(plan) == key
    fresh = registry.new_session(key)
    assert old._owned()._torch is not fresh._owned()._torch
    assert registry.report()["distinct_active_model_modules"] == 2
    assert registry.report()["active_model_weight_logical_bytes"] == cost * 2
    ids = np.array([[1, 3, 4]], np.int64)
    for lease in (old, fresh):
        np.testing.assert_allclose(
            lease.step(ids), model.forward(ids), atol=5e-5, rtol=5e-5
        )
    old.close()
    assert registry.report()["active_model_weight_logical_bytes"] == cost
    fresh.close()
    assert registry.report()["active_model_weight_logical_bytes"] == 0


def test_active_weight_budget_blocks_new_module_but_not_same_model_leases(tmp_path):
    model_a, plan_a, cost_a, kv_a = build(tmp_path, 1)
    _, plan_b, cost_b, kv_b = build(tmp_path, 2)
    registry = ActiveModelLlamaRegistry(
        max_cached_weight_bytes=max(cost_a, cost_b) + 4096,
        max_active_model_weight_bytes=max(cost_a, cost_b) + 4096,
        max_active_kv_bytes=max(4096, kv_a * 3 + kv_b),
        max_active_sessions=4,
    )
    a_id = registry.register(plan_a)
    first = registry.new_session(a_id)
    second = registry.new_session(a_id)
    assert registry.report()["active_model_weight_logical_bytes"] == cost_a
    b_id = registry.register(plan_b)
    with pytest.raises(ActiveModelBudgetExceeded):
        registry.new_session(b_id)
    assert registry.report()["rejected_weight_leases"] == 1
    assert registry.report()["active_sessions"] == 2
    first.close()
    second.close()
    with registry.new_session(b_id) as remaining:
        assert remaining.cursor == 0
        assert registry.report()["active_model_weight_logical_bytes"] == cost_b
    assert registry.report()["active_model_weight_logical_bytes"] == 0


@pytest.mark.parametrize("budget", (0, True, 3 * 1024 * 1024 * 1024))
def test_invalid_model_budget_rejected(budget):
    with pytest.raises(ValueError):
        ActiveModelLlamaRegistry(max_active_model_weight_bytes=budget)
