"""v26 quota-reserved Llama sessions, explicit release and isolated live state."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.session_leases_v26 import (
    LeaseBudgetExceeded,
    LeaseClosed,
    LeaseManagedLlamaRegistry,
)
from visuaml.compiler_lab.template_registry_v25 import TemplateNotCached, TemplateRegistryError


def setup(tmp_path, *, layers=2, batch=1, slots=2):
    fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch), model, batch=batch
    )
    p = model.profile
    bytes_per_session = (
        batch * p.layers * plan.capacity * p.kv_heads * p.head_dim * 8
    )
    template_bytes = (
        sum(a.nbytes for a in model.arrays.values())
        + 2 * p.head_dim * plan.capacity * 4
    )
    registry = LeaseManagedLlamaRegistry(
        max_cached_weight_bytes=template_bytes + 4096,
        max_models=2,
        max_active_kv_bytes=max(4096, slots * bytes_per_session),
        max_active_sessions=slots,
    )
    key = registry.register(plan)
    return model, registry, key, bytes_per_session


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_lease_budget_is_reserved_and_released_idempotently(tmp_path, layers):
    model, registry, key, cost = setup(tmp_path, layers=layers, slots=2)
    first = registry.new_session(key)
    second = registry.new_session(key)
    report = registry.report()
    assert report["active_sessions"] == 2
    assert report["active_kv_logical_bytes"] == 2 * cost
    assert report["admitted_leases"] == 2
    assert first.report()["private_kv_bytes"] == cost
    with pytest.raises(LeaseBudgetExceeded):
        registry.new_session(key)
    assert registry.report()["rejected_leases"] == 1
    first.close()
    first.close()
    assert registry.report()["active_kv_logical_bytes"] == cost
    assert registry.report()["active_sessions"] == 1
    with pytest.raises(LeaseClosed):
        first.step(np.array([[1]], np.int64))
    with pytest.raises(LeaseClosed):
        first.snapshot()
    with registry.new_session(key) as third:
        ids = np.array([[1, 2, 3]], np.int64)
        np.testing.assert_allclose(
            third.step(ids), model.forward(ids), atol=5e-5, rtol=5e-5
        )
        assert third.cursor == 3
        assert registry.report()["active_sessions"] == 2
    second.close()
    assert registry.report()["active_sessions"] == 0
    assert registry.report()["active_kv_logical_bytes"] == 0
    assert registry.report()["released_leases"] == 3


def test_evicted_model_session_remains_valid_until_lease_close(tmp_path):
    model, registry, key, cost = setup(tmp_path, layers=1, slots=1)
    lease = registry.new_session(key)
    assert registry.evict(key)
    assert registry.report()["models"] == 0
    assert registry.report()["active_sessions"] == 1
    assert registry.report()["active_kv_logical_bytes"] == cost
    with pytest.raises(TemplateNotCached):
        registry.new_session(key)
    ids = np.array([[1, 2]], np.int64)
    np.testing.assert_allclose(
        lease.step(ids), model.forward(ids), atol=5e-5, rtol=5e-5
    )
    lease.close()
    assert registry.report()["active_sessions"] == 0
    assert registry.report()["cached_weight_module_logical_bytes"] == 0


def test_exceptional_exit_resets_private_kv_and_releases_quota(tmp_path):
    _, registry, key, _ = setup(tmp_path, layers=1, slots=1)
    with pytest.raises(RuntimeError, match="injected failure"):
        with registry.new_session(key) as lease:
            lease.step(np.array([[1, 2]], np.int64))
            assert lease.cursor == 2
            raise RuntimeError("injected failure")
    assert registry.report()["active_sessions"] == 0
    assert registry.report()["active_kv_logical_bytes"] == 0
    with registry.new_session(key) as another:
        assert another.cursor == 0
        assert another.report()["cursor"] == 0


def test_unclosed_lease_finalizer_releases_quota(tmp_path):
    _, registry, key, _ = setup(tmp_path, layers=1, slots=1)
    lease = registry.new_session(key)
    lease.step(np.array([[1]], np.int64))
    assert registry.report()["active_sessions"] == 1
    del lease
    gc.collect()
    assert registry.report()["active_sessions"] == 0
    assert registry.report()["active_kv_logical_bytes"] == 0
    assert registry.report()["released_leases"] == 1


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_active_sessions": 0},
        {"max_active_sessions": True},
        {"max_active_kv_bytes": 0},
        {"max_active_kv_bytes": True},
    ),
)
def test_invalid_active_quota_configuration(kwargs):
    with pytest.raises(TemplateRegistryError):
        LeaseManagedLlamaRegistry(**kwargs)


def test_unknown_digest_and_released_lease_contract(tmp_path):
    _, registry, key, _ = setup(tmp_path, layers=1, slots=1)
    with pytest.raises(TemplateNotCached):
        registry.new_session("0" * 64)
    lease = registry.new_session(key)
    lease.close()
    with pytest.raises(LeaseClosed):
        lease.generate(np.array([[1]], np.int64), steps=1)
    with pytest.raises(LeaseClosed):
        lease.report()
