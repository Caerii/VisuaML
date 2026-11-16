"""v25 admission, eviction, identity and concurrency for verified model templates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.template_registry_v25 import (
    TemplateNotCached,
    TemplateRegistryError,
    VerifiedLlamaRegistry,
)


def prepare(tmp_path, layers):
    directory = tmp_path / f"llama_{layers}"
    directory.mkdir()
    fixture(directory, layers=layers)
    model = load_hf_llama_local(directory, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    cap = min(plan.weights.profile.context, 128)
    bytes_ = (
        sum(a.nbytes for a in plan.weights.arrays.values())
        + 2 * plan.weights.profile.head_dim * cap * 4
    )
    return model, plan, bytes_


def test_registry_hits_share_graph_and_isolate_conversations(tmp_path):
    model, plan, budget = prepare(tmp_path, 2)
    registry = VerifiedLlamaRegistry(
        max_cached_weight_bytes=budget + 1024, max_models=2
    )
    key = registry.register(plan)
    a, b = registry.new_session(key), registry.new_session(key)
    assert a._compiled[(1, -1, -1)] is b._compiled[(1, -1, -1)]
    assert a._torch is b._torch
    assert a.cache._arrays["k0"] is not b.cache._arrays["k0"]
    tokens_a = np.array([[1, 2, 3]], np.int64)
    tokens_b = np.array([[4, 5]], np.int64)
    np.testing.assert_allclose(
        a.step(tokens_a), model.forward(tokens_a), atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(
        b.step(tokens_b), model.forward(tokens_b), atol=5e-5, rtol=5e-5
    )
    assert a.cursor == 3 and b.cursor == 2
    report = registry.report()
    assert report["models"] == 1
    assert report["admissions"] == 1
    assert report["cached_weight_module_logical_bytes"] == budget
    assert report["cache_hits"] == 2
    assert registry.register(plan) == key
    assert registry.report()["cache_hits"] == 3
    assert registry.report()["entries_lru_to_mru"][0]["compiled_graphs"] == 1


def test_lru_eviction_does_not_kill_active_session(tmp_path):
    first_model, first_plan, first_bytes = prepare(tmp_path, 1)
    second_model, second_plan, second_bytes = prepare(tmp_path, 2)
    cap = max(first_bytes, second_bytes)
    assert cap < first_bytes + second_bytes
    registry = VerifiedLlamaRegistry(
        max_cached_weight_bytes=cap, max_models=2
    )
    first_key = registry.register(first_plan)
    old_session = registry.new_session(first_key)
    second_key = registry.register(second_plan)
    assert first_key != second_key
    assert registry.report()["models"] == 1
    assert registry.report()["evictions"] == 1
    assert registry.report()["entries_lru_to_mru"][0]["semantic_sha256"] == second_key
    with pytest.raises(TemplateNotCached):
        registry.new_session(first_key)
    assert registry.report()["cache_misses"] == 1
    # Eviction changes registry ownership only; a live executor retains the
    # correct graph and independent KV state for its pinned model.
    tokens = np.array([[1, 3, 4]], np.int64)
    np.testing.assert_allclose(
        old_session.step(tokens), first_model.forward(tokens),
        atol=5e-5, rtol=5e-5
    )
    second_session = registry.new_session(second_key)
    np.testing.assert_allclose(
        second_session.step(tokens), second_model.forward(tokens),
        atol=5e-5, rtol=5e-5
    )
    assert old_session._torch is not second_session._torch


def test_oversized_admission_does_not_evict_existing_model(tmp_path):
    small, small_plan, small_bytes = prepare(tmp_path, 1)
    _, big_plan, big_bytes = prepare(tmp_path, 4)
    assert big_bytes > small_bytes
    registry = VerifiedLlamaRegistry(
        max_cached_weight_bytes=small_bytes + 256, max_models=1
    )
    key = registry.register(small_plan)
    with pytest.raises(TemplateRegistryError, match="budget"):
        registry.register(big_plan)
    assert registry.report()["models"] == 1
    assert registry.report()["evictions"] == 0
    tokens = np.array([[1, 2]], dtype=np.int64)
    np.testing.assert_allclose(
        registry.new_session(key).step(tokens),
        small.forward(tokens), atol=5e-5, rtol=5e-5
    )


def test_manual_evict_is_idempotent_and_sessions_survive(tmp_path):
    model, plan, weight_bytes = prepare(tmp_path, 1)
    registry = VerifiedLlamaRegistry(
        max_cached_weight_bytes=weight_bytes + 256
    )
    key = registry.register(plan)
    session = registry.new_session(key)
    assert registry.evict(key)
    assert not registry.evict(key)
    assert registry.report()["models"] == 0
    assert registry.report()["cached_weight_module_logical_bytes"] == 0
    assert session.report()["specializations"] == 1
    np.testing.assert_allclose(
        session.step(np.array([[1]], np.int64)),
        model.forward(np.array([[1]], np.int64)),
        atol=5e-5, rtol=5e-5,
    )


def test_concurrent_access_never_retraces_or_shares_kv(tmp_path):
    model, plan, weight_bytes = prepare(tmp_path, 2)
    registry = VerifiedLlamaRegistry(
        max_cached_weight_bytes=weight_bytes + 256
    )
    key = registry.register(plan)

    def run(index):
        session = registry.new_session(key)
        tokens = np.array([[index + 1, 3, 2]], dtype=np.int64)
        np.testing.assert_allclose(
            session.step(tokens), model.forward(tokens), atol=5e-5, rtol=5e-5
        )
        return session._torch, session.cache._arrays["k0"], session.cursor

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(4)))
    assert all(item[0] is results[0][0] for item in results)
    assert len({id(item[1]) for item in results}) == 4
    assert all(item[2] == 3 for item in results)
    assert registry.report()["admissions"] == 1
    assert registry.report()["cache_hits"] == 4


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_cached_weight_bytes": 0},
        {"max_cached_weight_bytes": True},
        {"max_models": 0},
        {"max_models": 65},
        {"max_models": True},
    ),
)
def test_invalid_budget_contracts_rejected(kwargs):
    with pytest.raises(TemplateRegistryError):
        VerifiedLlamaRegistry(**kwargs)


def test_unknown_and_malformed_digest_rejected():
    registry = VerifiedLlamaRegistry()
    with pytest.raises(TemplateRegistryError):
        registry.new_session("../escape")
    with pytest.raises(TemplateNotCached):
        registry.new_session("a" * 64)
    assert registry.report()["cache_misses"] == 1
