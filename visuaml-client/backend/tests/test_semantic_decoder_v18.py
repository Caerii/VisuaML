"""v18: typed SemanticProgram drives all decoder math with stateful KV execution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.semantic_decoder_v18 import (
    DecoderLoweringError,
    lower_decoder_semantic_program,
    make_decoder_semantic_program,
)
from visuaml.compiler_lab.stacked_decoder_v14 import (
    StackedWeights,
    empty_stack,
    reference_stack_step,
)
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


@pytest.mark.parametrize("seed", (0, 19))
@pytest.mark.parametrize("layers", (1, 2, 4))
@pytest.mark.parametrize("batch", (1, 2))
def test_full_decoder_ir_matches_independent_v14_and_v16(seed, layers, batch):
    weights = StackedWeights.seeded(seed, layers=layers)
    program = make_decoder_semantic_program(weights, batch=batch)
    compiled = lower_decoder_semantic_program(program, weights, batch=batch)
    session = compiled.new_session()
    handwritten = StreamingStackedDecoder(weights, batch=batch)
    reference = empty_stack(weights, batch)
    rng = np.random.default_rng(800 + seed)
    for n in (5, 1, 7, 3, 8):
        tokens = rng.integers(0, weights.embedding.shape[0], (batch, n), dtype=np.int64)
        got = session.step(tokens)
        old = handwritten.step(tokens)
        independent, reference = reference_stack_step(weights, tokens, reference)
        np.testing.assert_allclose(got, old, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(got, independent, atol=3e-5, rtol=3e-5)
        snapshot = session.snapshot()
        for actual, expected in zip(snapshot.layers, reference.layers):
            np.testing.assert_allclose(actual.keys, expected.keys, atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(actual.values, expected.values, atol=3e-5, rtol=3e-5)
    assert session.cursor == 24
    assert session.report()["ops_per_chunk"] > 20


def test_full_64_token_context_ir_cache_parity():
    weights = StackedWeights.seeded(73, layers=4)
    session = lower_decoder_semantic_program(
        make_decoder_semantic_program(weights, batch=2), weights, batch=2
    ).new_session()
    reference = empty_stack(weights, 2)
    rng = np.random.default_rng(19)
    for n in (7, 11, 19, 9, 18):
        tokens = rng.integers(0, 19, (2, n), dtype=np.int64)
        actual = session.step(tokens)
        expected, reference = reference_stack_step(weights, tokens, reference)
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert session.cursor == 64
    for actual, expected in zip(session.snapshot().layers, reference.layers):
        np.testing.assert_allclose(actual.keys, expected.keys, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(actual.values, expected.values, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("layers", (1, 3, 4))
def test_generation_and_state_restore_match_handwritten_decoder(layers):
    weights = StackedWeights.seeded(7, layers=layers)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    session = plan.new_session()
    baseline = StreamingStackedDecoder(weights)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    a, cached_a = session.generate(prompt, steps=7)
    b, cached_b = baseline.generate(prompt, steps=7)
    np.testing.assert_array_equal(a, b)
    for x, y in zip(cached_a.layers, cached_b.layers):
        np.testing.assert_allclose(x.keys, y.keys, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(x.values, y.values, atol=3e-5, rtol=3e-5)
    previous = session.snapshot()
    session.step(np.array([[3]], dtype=np.int64))
    session.restore(previous)
    assert session.cursor == 11
    session.reset()
    assert session.cursor == 0


def test_semantic_program_rejects_wiring_effect_and_checkpoint_changes():
    weights = StackedWeights.seeded(17, layers=2)
    program = make_decoder_semantic_program(weights)
    region = program.regions[0]
    op = region.operations[0]
    altered = replace(region, operations=(replace(op, op="execute_untrusted_python"), *region.operations[1:]))
    with pytest.raises(DecoderLoweringError):
        lower_decoder_semantic_program(replace(program, regions=(altered,)), weights)
    altered = replace(region, operations=(replace(op, operands=("embedding", "tokens")), *region.operations[1:]))
    with pytest.raises(DecoderLoweringError):
        lower_decoder_semantic_program(replace(program, regions=(altered,)), weights)
    idx = next(i for i, item in enumerate(region.operations) if item.effect == "write")
    changed_effect = replace(region.operations[idx], effect="pure", token_in=None, token_out=None)
    altered = replace(
        region, operations=(
            *region.operations[:idx], changed_effect, *region.operations[idx + 1:]
        ),
    )
    with pytest.raises(DecoderLoweringError):
        lower_decoder_semantic_program(replace(program, regions=(altered,)), weights)
    with pytest.raises(DecoderLoweringError, match="Noncanonical"):
        lower_decoder_semantic_program(program, StackedWeights.seeded(18, layers=2))
    with pytest.raises(DecoderLoweringError):
        lower_decoder_semantic_program(program, weights, batch=2)


def test_runtime_failures_are_atomic_and_snapshots_are_detached(monkeypatch):
    weights = StackedWeights.seeded(11, layers=3)
    session = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights).new_session()
    first = np.array([[1, 2]], dtype=np.int64)
    session.step(first)
    previous = session.snapshot()
    first_copy = previous.layers[0].keys.copy()
    operation = session._execute

    def fail_at_second_layer(op, *args):
        if op.name == "l1_attention":
            raise RuntimeError("injected mid-graph failure")
        return operation(op, *args)

    monkeypatch.setattr(session, "_execute", fail_at_second_layer)
    with pytest.raises(RuntimeError, match="mid-graph"):
        session.step(np.array([[3]], dtype=np.int64))
    assert session.cursor == 2
    np.testing.assert_array_equal(session.snapshot().layers[0].keys, first_copy)
    assert not previous.layers[0].keys.flags.writeable
    monkeypatch.setattr(session, "_execute", operation)
    session.step(np.array([[3]], dtype=np.int64))
    session.restore(previous)
    assert session.cursor == 2
    np.testing.assert_array_equal(session.snapshot().layers[0].keys, first_copy)


def test_invalid_tokens_and_context_fail_without_state_mutation():
    weights = StackedWeights.seeded(3, layers=1)
    session = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights).new_session()
    for tokens in (
        np.array([[1, 2]], dtype=np.int32),
        np.array([[-1]], dtype=np.int64),
        np.array([[weights.embedding.shape[0]]], dtype=np.int64),
        np.ones((1, 33), dtype=np.int64),
        np.ones((2, 1), dtype=np.int64),
    ):
        with pytest.raises(DecoderLoweringError):
            session.step(tokens)
        assert session.cursor == 0
    session.step(np.ones((1, 32), dtype=np.int64))
    session.step(np.ones((1, 32), dtype=np.int64))
    with pytest.raises(DecoderLoweringError):
        session.step(np.ones((1, 1), dtype=np.int64))
    assert session.cursor == 64


def test_manifest_and_state_transitions_are_inspectable_and_bound():
    weights = StackedWeights.seeded(19, layers=2)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    manifest, explanation = plan.manifest(), plan.explain()
    assert len(manifest["checkpoint_sha256"]) == 64
    assert len(manifest["semantic_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["operations"] == explanation["operation_count"]
    assert explanation["operator_histogram"]["rmsnorm"] == 4
    assert explanation["operator_histogram"]["gqa_attention"] == 2
    assert len(explanation["state_effect_order"]) == 8
    assert plan.new_session().plan.digest == plan.digest
    with pytest.raises(DecoderLoweringError):
        lower_decoder_semantic_program(replace(plan.semantic, outputs=("missing",)), weights)
