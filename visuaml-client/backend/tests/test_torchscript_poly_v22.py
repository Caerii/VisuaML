"""v22 compile-once, dynamically varied query length and KV cursor witnesses."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_hf_llama_checkpoint_v19 import fixture, torch_oracle
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.torchscript_graph_v21 import WholeGraphError


def prepare(tmp_path, *, layers=2, batch=1):
    weights, config = fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch), model, batch=batch
    )
    return weights, config, model, plan


@pytest.mark.parametrize("layers", (1, 2, 4))
@pytest.mark.parametrize("batch", (1, 2))
def test_single_torchscript_graph_reuses_arbitrary_token_and_cursor_shapes(
    tmp_path, layers, batch
):
    weights, config, model, plan = prepare(tmp_path, layers=layers, batch=batch)
    poly = plan.new_session(backend="torchscript_poly")
    v21 = plan.new_session(backend="torchscript_graph")
    numpy_ir = plan.new_session(backend="numpy")
    rng = np.random.default_rng(119 + layers + batch)
    offset = 0
    for length in (5, 1, 3, 7, 2, 4):
        ids = rng.integers(0, model.profile.vocabulary, size=(batch, length), dtype=np.int64)
        got = poly.step(ids)
        from_v21 = v21.step(ids)
        from_ir = numpy_ir.step(ids)
        np.testing.assert_allclose(got, from_v21, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(got, from_ir, atol=5e-5, rtol=5e-5)
        for got_layer, expected_layer in zip(poly.snapshot().layers, numpy_ir.snapshot().layers):
            np.testing.assert_allclose(
                got_layer.keys, expected_layer.keys, atol=5e-5, rtol=5e-5
            )
            np.testing.assert_allclose(
                got_layer.values, expected_layer.values, atol=5e-5, rtol=5e-5
            )
        offset += length
        assert poly.cursor == offset
        assert poly.report()["specializations"] == 1
    assert v21.report()["specializations"] == 6
    report = poly.report()
    assert report["dynamic_query_and_cursor"] is True
    assert report["traces"][0]["query"] == -1
    assert report["traces"][0]["past"] == -1
    assert report["traces"][0]["graph_nodes"] > 20
    assert len(report["semantic_sha256"]) == 64
    full = poly.snapshot()
    assert all(layer.keys.shape[1] == offset for layer in full.layers)


def test_full_context_and_reset_uses_same_compiled_graph(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=2)
    poly = plan.new_session(backend="torchscript_poly")
    rng = np.random.default_rng(93)
    history = []
    for length in (1, 3, 16, 12, 7, 25):
        tokens = rng.integers(0, model.profile.vocabulary, size=(1, length), dtype=np.int64)
        history.append(tokens)
        result = poly.step(tokens)
        full = np.concatenate(history, axis=1)
        np.testing.assert_allclose(
            result, model.forward(full)[:, -length:], atol=5e-5, rtol=5e-5
        )
    assert poly.cursor == 64
    assert poly.report()["specializations"] == 1
    with pytest.raises(WholeGraphError):
        poly.step(np.array([[1]], dtype=np.int64))
    poly.reset()
    fresh = np.array([[7, 6, 5, 4]], np.int64)
    np.testing.assert_allclose(
        poly.step(fresh), model.forward(fresh), atol=5e-5, rtol=5e-5
    )
    assert poly.report()["specializations"] == 1


def test_polymorphic_graph_matches_independent_torch_oracle(tmp_path):
    weights, config, model, plan = prepare(tmp_path, layers=2, batch=2)
    poly = plan.new_session(backend="torchscript_poly")
    for tokens in (
        np.array([[1, 2, 3], [3, 1, 2]], np.int64),
        np.array([[4, 5], [5, 4]], np.int64),
    ):
        got = poly.step(tokens)
        # Include all committed context in the independent PyTorch reference.
        snapshot = poly.snapshot()
        if snapshot.layers[0].keys.shape[1] == 3:
            history = tokens.copy()
        else:
            history = np.concatenate((history, tokens), axis=1)
        expected = torch_oracle(torch.from_numpy(history), weights, config)
        np.testing.assert_allclose(
            got, expected.numpy()[:, -tokens.shape[1]:], atol=5e-5, rtol=5e-5
        )
    assert poly.report()["specializations"] == 1


def test_polymorphic_restore_and_greedy_generation_match_numpy(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=1)
    poly = plan.new_session(backend="torchscript_poly")
    numpy_ir = plan.new_session()
    prompt = np.array([[1, 2, 3, 4]], np.int64)
    poly.step(prompt)
    checkpoint = poly.snapshot()
    poly.step(np.array([[5, 6]], np.int64))
    poly.restore(checkpoint)
    assert poly.cursor == 4
    np.testing.assert_allclose(
        poly.step(np.array([[7]], np.int64)),
        model.forward(np.array([[1, 2, 3, 4, 7]], np.int64))[:, -1:],
        atol=5e-5, rtol=5e-5,
    )
    poly.reset()
    got, state = poly.generate(prompt, steps=5)
    wanted, oracle_state = numpy_ir.generate(prompt, steps=5)
    np.testing.assert_array_equal(got, wanted)
    for x, y in zip(state.layers, oracle_state.layers):
        np.testing.assert_allclose(x.keys, y.keys, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(x.values, y.values, atol=5e-5, rtol=5e-5)
    assert poly.report()["specializations"] == 1


def test_polymorphic_inputs_are_validated_before_tracing(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=1)
    poly = plan.new_session(backend="torchscript_poly")
    for bad in (
        np.array([[-1]], np.int64),
        np.array([[model.profile.vocabulary]], np.int64),
        np.array([[1]], np.int32),
        np.ones((2, 1), np.int64),
        np.ones((1, 33), np.int64),
    ):
        with pytest.raises(WholeGraphError):
            poly.step(bad)
    assert poly.cursor == 0
    assert poly.report()["specializations"] == 0


def test_polymorphic_compiled_graph_cannot_publish_modified_kv_prefix(
    tmp_path, monkeypatch
):
    _, _, _, plan = prepare(tmp_path, layers=1)
    poly = plan.new_session(backend="torchscript_poly")
    poly.step(np.array([[1, 2, 3]], np.int64))
    before = poly.snapshot()
    signature = (1, -1, -1)
    original = poly._compiled[signature]

    def corrupt(*args):
        result = original(*args)
        replaced = list(result)
        keys = replaced[1].clone()
        keys[:, 0] += 1
        replaced[1] = keys
        return tuple(replaced)

    monkeypatch.setitem(poly._compiled, signature, corrupt)
    with pytest.raises(WholeGraphError, match="previously committed"):
        poly.step(np.array([[4]], np.int64))
    assert poly.cursor == 3
    for a, b in zip(before.layers, poly.snapshot().layers):
        np.testing.assert_array_equal(a.keys, b.keys)
        np.testing.assert_array_equal(a.values, b.values)
