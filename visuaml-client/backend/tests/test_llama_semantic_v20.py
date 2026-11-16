"""Real Llama-shaped checkpoints lower into an executable typed stateful IR."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from test_hf_llama_checkpoint_v19 import fixture, torch_oracle
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    LlamaSemanticError,
    lower_llama_semantic_program,
    make_llama_semantic_program,
)


def prepare(tmp_path, *, layers=2):
    originals, config = fixture(tmp_path, layers=layers)
    imported = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    graph = make_llama_semantic_program(imported, batch=2)
    compiled = lower_llama_semantic_program(graph, imported, batch=2)
    return originals, config, imported, graph, compiled


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_real_weight_semantic_ir_matches_numpy_and_torch_oracles(tmp_path, layers):
    originals, config, imported, graph, plan = prepare(tmp_path, layers=layers)
    assert plan.semantic_sha256 == graph.digest()
    assert plan.explain()["operator_histogram"]["rope_half"] == 2 * layers
    assert plan.explain()["operator_histogram"]["rmsnorm_eps"] == 2 * layers + 1
    assert plan.explain()["operator_histogram"]["gqa_attention"] == layers
    assert len(plan.explain()["operator_histogram"]) > 7
    session = plan.new_session()
    tokens = np.array([[1, 2, 3, 4, 5, 6], [5, 2, 4, 7, 3, 1]], dtype=np.int64)
    chunked = []
    for lo, hi in ((0, 2), (2, 3), (3, 6)):
        observed = session.step(tokens[:, lo:hi])
        chunked.append(observed)
        expected_numpy = imported.forward(tokens[:, :hi])[:, lo:hi]
        with torch.no_grad():
            expected_torch = torch_oracle(
                torch.from_numpy(tokens[:, :hi]), originals, config
            ).numpy()[:, lo:hi]
        np.testing.assert_allclose(observed, expected_numpy, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(observed, expected_torch, rtol=4e-5, atol=4e-5)
    all_logits = np.concatenate(chunked, axis=1)
    np.testing.assert_allclose(all_logits, imported.forward(tokens), rtol=4e-5, atol=4e-5)
    assert session.cursor == 6
    assert len(session.snapshot().layers) == layers
    assert all(item.keys.shape[1] == 6 for item in session.snapshot().layers)


def test_same_ir_executes_torchscript_compiled_hybrid_primitives(tmp_path):
    _, _, _, _, plan = prepare(tmp_path, layers=1)
    baseline = plan.new_session(backend="numpy")
    hybrid = plan.new_session(backend="torchscript_hybrid")
    tokens = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
    for part in (tokens[:, :2], tokens[:, 2:]):
        a = baseline.step(part)
        b = hybrid.step(part)
        np.testing.assert_allclose(a, b, rtol=6e-5, atol=6e-5)
        for left, right in zip(baseline.snapshot().layers, hybrid.snapshot().layers):
            np.testing.assert_allclose(left.keys, right.keys, rtol=6e-5, atol=6e-5)
            np.testing.assert_allclose(left.values, right.values, rtol=6e-5, atol=6e-5)
    report = hybrid.report()
    assert report["compiled_primitive_calls"] >= 8
    assert report["compiled_primitive_specializations"] > 0
    assert "hybrid" in report["backend"]


def test_greedy_generation_uses_state_and_matches_full_recompute(tmp_path):
    _, _, imported, _, plan = prepare(tmp_path, layers=2)
    engine = plan.new_session()
    prompt = np.array([[1, 2, 3], [2, 3, 4]], np.int64)
    ids, state = engine.generate(prompt, steps=5)
    assert ids.shape == (2, 5)
    assert state.layers[0].keys.shape[1] == 8
    full = prompt.copy()
    expected = []
    for _ in range(5):
        next_token = np.argmax(
            imported.forward(full)[:, -1], axis=-1
        ).astype(np.int64)[:, None]
        expected.append(next_token)
        full = np.concatenate((full, next_token), axis=1)
    np.testing.assert_array_equal(ids, np.concatenate(expected, axis=1))


def test_rejects_forged_operations_effects_and_model_weights(tmp_path):
    _, _, imported, graph, plan = prepare(tmp_path)
    operations = graph.regions[0].operations
    altered = replace(
        operations[0], op="untrusted_dynamic_python",
    )
    region = replace(graph.regions[0], operations=(altered, *operations[1:]))
    with pytest.raises(LlamaSemanticError):
        lower_llama_semantic_program(replace(graph, regions=(region,)), imported, batch=2)
    effect = next(i for i, op in enumerate(operations) if op.effect == "write")
    modified = replace(operations[effect], effect="pure", token_in=None, token_out=None)
    region = replace(graph.regions[0], operations=(
        *operations[:effect], modified, *operations[effect + 1:]
    ))
    with pytest.raises(LlamaSemanticError):
        lower_llama_semantic_program(replace(graph, regions=(region,)), imported, batch=2)
    with pytest.raises(LlamaSemanticError):
        lower_llama_semantic_program(graph, imported, batch=1)
    mutated = dict(imported.arrays)
    mutated["model.norm.weight"] = np.zeros_like(mutated["model.norm.weight"])
    modified_weights = replace(imported, arrays=mutated)
    with pytest.raises(LlamaSemanticError, match="Noncanonical"):
        lower_llama_semantic_program(graph, modified_weights, batch=2)
    with pytest.raises(LlamaSemanticError):
        plan.new_session(backend="execute_remote_code")


def test_forged_initial_plan_cannot_override_compiler_contract(tmp_path):
    _, _, imported, _, plan = prepare(tmp_path)
    with pytest.raises(LlamaSemanticError):
        replace(plan, batch=1).new_session()
    original_id = plan.semantic_sha256
    original = plan.new_session()
    # Source dict replacement after planning must not mutate frozen compiled weights.
    imported.arrays["model.norm.weight"] = np.zeros((imported.profile.hidden,), np.float32)
    assert plan.new_session().plan.semantic_sha256 == original_id
    assert original.plan.semantic_sha256 == original_id


def test_failed_midgraph_effects_leave_previous_cache_untouched(tmp_path, monkeypatch):
    _, _, _, _, plan = prepare(tmp_path)
    executor = plan.new_session()
    executor.step(np.array([[1, 2], [2, 3]], np.int64))
    previous = executor.snapshot()
    method = executor._execute

    def failing(op, *args):
        if op.name == "l1_attention":
            raise RuntimeError("injected failed attention")
        return method(op, *args)

    monkeypatch.setattr(executor, "_execute", failing)
    with pytest.raises(RuntimeError, match="injected failed"):
        executor.step(np.array([[4], [5]], np.int64))
    assert executor.cursor == 2
    now = executor.snapshot()
    for before, after in zip(previous.layers, now.layers):
        np.testing.assert_array_equal(before.keys, after.keys)
        np.testing.assert_array_equal(before.values, after.values)
    monkeypatch.setattr(executor, "_execute", method)
    executor.step(np.array([[4], [5]], np.int64))
    executor.restore(previous)
    assert executor.cursor == 2


def test_tied_embeddings_model_has_explicit_head_alias(tmp_path):
    originals, config = fixture(tmp_path, layers=1)
    config["tie_word_embeddings"] = True
    (tmp_path / "config.json").write_text(__import__("json").dumps(config))
    del originals["lm_head.weight"]
    save_file(originals, str(tmp_path / "model.safetensors"))
    imported = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    program = make_llama_semantic_program(imported, batch=1)
    plan = lower_llama_semantic_program(program, imported)
    assert program.regions[0].operations[-1].operands[-1] == "model.embed_tokens.weight"
    assert plan.new_session().step(np.array([[1, 2]], dtype=np.int64)).shape == (1, 2, 64)
