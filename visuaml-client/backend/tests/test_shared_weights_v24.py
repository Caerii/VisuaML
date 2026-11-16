"""v24: verified model weights and graph are shared, private KV remains isolated."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.compiled_template_v23 import CompiledTemplateError
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.shared_weights_v24 import SharedWeightsLlamaTemplate


def prepare(tmp_path, layers=2, batch=1):
    fixture(tmp_path, layers=layers)
    weights = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(weights, batch=batch), weights, batch=batch
    )
    return weights, plan


@pytest.mark.parametrize("layers", (1, 2, 4))
@pytest.mark.parametrize("batch", (1, 2))
def test_many_clients_reuse_graph_and_weight_module(tmp_path, layers, batch):
    weights, plan = prepare(tmp_path, layers, batch)
    template = SharedWeightsLlamaTemplate(plan)
    a, b, c = (template.new_session() for _ in range(3))
    signature = (batch, -1, -1)
    assert a._compiled[signature] is b._compiled[signature] is c._compiled[signature]
    assert a._torch is b._torch is c._torch
    assert a.cache._arrays["k0"] is not b.cache._arrays["k0"]
    assert a.cache._arrays["k0"] is not c.cache._arrays["k0"]
    assert template.report()["fast_sessions_without_model_snapshot"] == 2
    assert template.report()["verified_torch_module_instances"] == 1
    assert template.report()["weight_module_logical_bytes"] > 0
    assert template.report()["compiled_graphs"] == 1
    one = np.full((batch, 3), 1, dtype=np.int64)
    two = np.full((batch, 2), 3, dtype=np.int64)
    np.testing.assert_allclose(
        a.step(one), weights.forward(one), atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(
        b.step(two), weights.forward(two), atol=5e-5, rtol=5e-5
    )
    assert a.cursor == 3 and b.cursor == 2 and c.cursor == 0
    after_a = a.step(np.full((batch, 1), 4, dtype=np.int64))
    np.testing.assert_allclose(
        after_a,
        weights.forward(
            np.concatenate((one, np.full((batch, 1), 4, dtype=np.int64)), axis=1)
        )[:, -1:],
        atol=5e-5, rtol=5e-5,
    )
    assert b.cursor == 2 and c.cursor == 0
    c.step(np.full((batch, 1), 5, dtype=np.int64))
    a.reset()
    assert a.cursor == 0 and b.cursor == 2 and c.cursor == 1


def test_no_additional_model_lowering_or_weight_module_constructed(tmp_path, monkeypatch):
    weights, plan = prepare(tmp_path, layers=1)
    template = SharedWeightsLlamaTemplate(plan)
    original = template.new_session()
    assert original.report()["specializations"] == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("Model lowering/weight snapshot must not recur")

    monkeypatch.setattr(
        "visuaml.compiler_lab.torchscript_graph_v21.lower_llama_semantic_program",
        forbidden,
    )
    monkeypatch.setattr(
        "visuaml.compiler_lab.llama_semantic_v20._freeze_checkpoint", forbidden,
    )
    new = template.new_session()
    assert new._torch is original._torch
    ids = np.array([[1, 2, 3]], np.int64)
    np.testing.assert_allclose(
        new.step(ids), weights.forward(ids), atol=5e-5, rtol=5e-5
    )
    assert template.report()["compiled_graphs"] == 1


def test_mutation_of_original_checkpoint_does_not_change_shared_model(tmp_path):
    weights, plan = prepare(tmp_path)
    template = SharedWeightsLlamaTemplate(plan)
    first = template.new_session()
    ids = np.array([[1, 2, 3]], np.int64)
    expected = first.step(ids)
    weights.arrays["model.norm.weight"] = np.zeros(
        (weights.profile.hidden,), dtype=np.float32
    )
    later = template.new_session()
    np.testing.assert_allclose(
        later.step(ids), expected, atol=0, rtol=0
    )
    assert first.cursor == later.cursor == 3


def test_shared_runtime_rejects_forged_model_plan(tmp_path):
    _, plan = prepare(tmp_path)
    with pytest.raises(CompiledTemplateError):
        SharedWeightsLlamaTemplate("arbitrary executable")
    with pytest.raises(ValueError):
        SharedWeightsLlamaTemplate(replace(plan, batch=2))


def test_concurrent_creation_and_inference_isolated(tmp_path):
    weights, plan = prepare(tmp_path, layers=2)
    template = SharedWeightsLlamaTemplate(plan)

    def worker(i: int):
        session = template.new_session()
        inputs = np.array([[i + 1, 2, 3]], dtype=np.int64)
        result = session.step(inputs)
        np.testing.assert_allclose(
            result, weights.forward(inputs), atol=5e-5, rtol=5e-5
        )
        return session._torch, session.cursor

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(worker, range(4)))
    assert all(cursor == 3 for _, cursor in outputs)
    assert all(module is outputs[0][0] for module, _ in outputs)
    assert template.report()["compiled_graphs"] == 1
    assert template.report()["fast_sessions_without_model_snapshot"] == 3
