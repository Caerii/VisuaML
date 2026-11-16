"""v21 entire stateful Llama forward lowers into one native-executed TorchScript graph."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture, torch_oracle
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    LlamaSemanticError,
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.torchscript_graph_v21 import WholeGraphError


def prepare(tmp_path, *, layers=1, batch=1):
    originals, config = fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch), model, batch=batch
    )
    return originals, config, model, plan


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_whole_graph_prefill_matches_two_independent_oracles(tmp_path, layers):
    originals, config, model, plan = prepare(tmp_path, layers=layers, batch=2)
    target = plan.new_session(backend="torchscript_graph")
    direct = plan.new_session(backend="numpy")
    ids = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], np.int64)
    observed = target.step(ids)
    np.testing.assert_allclose(
        observed, direct.step(ids), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        observed, model.forward(ids), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        observed, torch_oracle(
            __import__("torch").from_numpy(ids), originals, config
        ).numpy(), rtol=5e-5, atol=5e-5,
    )
    for got, expected in zip(target.snapshot().layers, direct.snapshot().layers):
        np.testing.assert_allclose(got.keys, expected.keys, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(got.values, expected.values, atol=5e-5, rtol=5e-5)
    report = target.report()
    assert report["specializations"] == 1
    assert report["traces"][0]["graph_nodes"] > 20
    assert report["traces"][0]["past"] == 0
    assert report["traces"][0]["query"] == 5
    assert len(report["semantic_sha256"]) == 64


def test_stateful_chunks_match_full_context_and_snapshot_restore(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=2)
    target = plan.new_session(backend="torchscript_graph")
    direct = plan.new_session()
    tokens = np.array([[1, 3, 5, 7, 2, 4, 6, 1]], np.int64)
    for lo, hi in ((0, 3), (3, 4), (4, 8)):
        x = target.step(tokens[:, lo:hi])
        y = direct.step(tokens[:, lo:hi])
        np.testing.assert_allclose(x, y, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(
            x, model.forward(tokens[:, :hi])[:, lo:hi],
            atol=5e-5, rtol=5e-5,
        )
        for got, expected in zip(target.snapshot().layers, direct.snapshot().layers):
            np.testing.assert_allclose(
                got.keys, expected.keys, atol=5e-5, rtol=5e-5
            )
            np.testing.assert_allclose(
                got.values, expected.values, atol=5e-5, rtol=5e-5
            )
    assert target.report()["specializations"] == 3
    snapshot = target.snapshot()
    target.step(np.array([[2]], np.int64))
    target.restore(snapshot)
    assert target.cursor == 8
    target.reset()
    assert target.cursor == 0
    # Same shape reuses a traced graph; token values are dynamic inputs.
    again = target.step(np.array([[7, 6, 5]], np.int64))
    np.testing.assert_allclose(
        again, model.forward(np.array([[7, 6, 5]], np.int64)),
        atol=5e-5, rtol=5e-5,
    )
    assert target.report()["specializations"] == 4


def test_trace_is_not_specialized_to_original_token_values(tmp_path):
    _, _, model, plan = prepare(tmp_path)
    engine = plan.new_session(backend="torchscript_graph")
    for ids in (
        np.array([[1, 2, 3]], np.int64),
        np.array([[5, 7, 1]], np.int64),
    ):
        engine.reset()
        np.testing.assert_allclose(
            engine.step(ids), model.forward(ids), atol=5e-5, rtol=5e-5
        )
    assert engine.report()["specializations"] == 1


def test_rejects_invalid_tokens_before_state_mutation(tmp_path):
    _, _, model, plan = prepare(tmp_path)
    engine = plan.new_session(backend="torchscript_graph")
    for bad in (
        np.array([[-1]], np.int64),
        np.array([[model.profile.vocabulary]], np.int64),
        np.array([[1]], np.int32),
        np.ones((2, 1), np.int64),
        np.ones((1, 33), np.int64),
    ):
        with pytest.raises(WholeGraphError):
            engine.step(bad)
        assert engine.cursor == 0
    assert engine.report()["specializations"] == 0


def test_ir_wiring_is_validated_before_compilation(tmp_path):
    _, _, model, plan = prepare(tmp_path)
    first = plan.program.regions[0].operations[0]
    altered = replace(first, op="untrusted_eval")
    region = replace(
        plan.program.regions[0],
        operations=(altered, *plan.program.regions[0].operations[1:]),
    )
    with pytest.raises(LlamaSemanticError):
        lower_llama_semantic_program(
            replace(plan.program, regions=(region,)), model
        )


def test_compiled_generation_matches_numpy_recompute(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=1)
    original = plan.new_session()
    compiled = plan.new_session(backend="torchscript_graph")
    prompt = np.array([[1, 3, 4]], np.int64)
    ids_a, state_a = original.generate(prompt, steps=2)
    ids_b, state_b = compiled.generate(prompt, steps=2)
    np.testing.assert_array_equal(ids_a, ids_b)
    for a, b in zip(state_a.layers, state_b.layers):
        np.testing.assert_allclose(a.keys, b.keys, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(a.values, b.values, atol=5e-5, rtol=5e-5)
    assert compiled.report()["specializations"] == 3
