"""v16: resident, length-polymorphic KV execution; numerical and isolation gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.stacked_decoder_v14 import (
    StackContractError,
    StackedWeights,
    empty_stack,
    reference_stack_step,
)
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


@pytest.mark.parametrize("layers", (1, 2, 4))
@pytest.mark.parametrize("seed", (0, 19))
def test_streaming_matches_reference_logits_and_every_cache(layers, seed):
    weights = StackedWeights.seeded(seed, layers=layers)
    rng = np.random.default_rng(seed + 901)
    tokens = rng.integers(0, 19, (2, 13), dtype=np.int64)
    streaming = StreamingStackedDecoder(weights, batch=2)
    baseline = empty_stack(weights, 2)
    offset = 0
    for n in (5, 1, 3, 1, 3):
        part = tokens[:, offset : offset + n]
        got = streaming.step(part)
        expected, baseline = reference_stack_step(weights, part, baseline)
        np.testing.assert_allclose(got, expected, atol=3e-5, rtol=3e-5)
        snapshot = streaming.snapshot()
        assert streaming.cursor == offset + n
        for actual, wanted in zip(snapshot.layers, baseline.layers):
            np.testing.assert_allclose(actual.keys, wanted.keys, atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(actual.values, wanted.values, atol=3e-5, rtol=3e-5)
        offset += n
    details = streaming.report()
    assert details.live_bytes * streaming.capacity == details.reserved_bytes * streaming.cursor
    assert details.capacity_tokens == 64


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_generates_same_tokens_as_reference(layers):
    weights = StackedWeights.seeded(43, layers=layers)
    initial = np.array([[1, 2, 3], [5, 6, 7]], np.int64)
    streamed = StreamingStackedDecoder(weights, batch=2)
    ids, snapshot = streamed.generate(initial, steps=6)
    baseline = empty_stack(weights, 2)
    expected, baseline = reference_stack_step(weights, initial, baseline)
    tokens = []
    for _ in range(6):
        next_ids = np.argmax(expected[:, -1], axis=-1).astype(np.int64)[:, None]
        tokens.append(next_ids)
        expected, baseline = reference_stack_step(weights, next_ids, baseline)
    np.testing.assert_array_equal(ids, np.concatenate(tokens, axis=1))
    assert snapshot.layers[0].keys.shape[1] == 9
    for current, original in zip(snapshot.layers, baseline.layers):
        np.testing.assert_allclose(current.keys, original.keys, atol=3e-5, rtol=3e-5)


def test_snapshot_and_restore_detached_and_immutable():
    weights = StackedWeights.seeded(7, layers=2)
    decoder = StreamingStackedDecoder(weights)
    decoder.step(np.array([[1, 2, 3]], np.int64))
    previous = decoder.snapshot()
    assert not previous.layers[0].keys.flags.writeable
    previous_bits = previous.layers[0].keys.copy()
    _ = decoder.step(np.array([[4, 5]], np.int64))
    np.testing.assert_array_equal(previous.layers[0].keys, previous_bits)
    decoder.restore(previous)
    assert decoder.cursor == 3
    np.testing.assert_array_equal(decoder.snapshot().layers[0].keys, previous_bits)
    decoder.reset()
    assert decoder.cursor == 0
    assert decoder.snapshot().layers[0].keys.shape[1] == 0


def test_restore_rejects_corruption_before_mutating_any_state():
    from visuaml.compiler_lab.stacked_decoder_v14 import StackState

    decoder = StreamingStackedDecoder(StackedWeights.seeded(9, layers=2))
    decoder.step(np.array([[1, 4]], np.int64))
    original = decoder.snapshot()
    broken = tuple(original.layers)
    mutated = broken[1].keys.copy()
    mutated[0, 0, 0, 0] = np.nan
    invalid = StackState((broken[0], replace(broken[1], keys=mutated)))
    with pytest.raises(StackContractError, match="restore"):
        decoder.restore(invalid)
    assert decoder.cursor == 2
    for now, before in zip(decoder.snapshot().layers, original.layers):
        np.testing.assert_array_equal(now.keys, before.keys)
        np.testing.assert_array_equal(now.values, before.values)


def test_input_guards_and_context_limit_are_fail_closed():
    decoder = StreamingStackedDecoder(StackedWeights.seeded(31, layers=1))
    for bad in (
        np.array([1, 2], dtype=np.int64),
        np.array([[1]], dtype=np.int32),
        np.array([[-1]], dtype=np.int64),
        np.array([[19]], dtype=np.int64),
        np.ones((1, 33), dtype=np.int64),
        np.ones((2, 1), dtype=np.int64),
    ):
        with pytest.raises(StackContractError):
            decoder.step(bad)
        assert decoder.cursor == 0
    decoder.step(np.ones((1, 32), dtype=np.int64))
    decoder.step(np.ones((1, 32), dtype=np.int64))
    with pytest.raises(StackContractError, match="outside admitted domain"):
        decoder.step(np.ones((1, 1), dtype=np.int64))
    assert decoder.cursor == 64


def test_snapshotted_checkpoint_does_not_follow_mutated_original_weights():
    weights = StackedWeights.seeded(19, layers=2)
    runtime = StreamingStackedDecoder(weights)
    original_sha = runtime.checkpoint_sha256
    weights.embedding[0, 0] = 42
    assert runtime.checkpoint_sha256 == original_sha
    expected = StackedWeights.seeded(19, layers=2)
    baseline = empty_stack(expected, 1)
    token = np.array([[0, 1, 3]], np.int64)
    got = runtime.step(token)
    reference, _ = reference_stack_step(expected, token, baseline)
    np.testing.assert_allclose(got, reference, atol=3e-5, rtol=3e-5)


def test_late_execution_failure_leaves_previous_cache_untouched(monkeypatch):
    weights = StackedWeights.seeded(11, layers=3)
    runtime = StreamingStackedDecoder(weights)
    runtime.step(np.array([[1, 2]], np.int64))
    prior = runtime.snapshot()
    call_count = 0
    old_rope = runtime._rope

    def failing_rope(value, offset):
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise RuntimeError("injected layer-three failure")
        return old_rope(value, offset)

    monkeypatch.setattr(runtime, "_rope", failing_rope)
    with pytest.raises(RuntimeError, match="layer-three"):
        runtime.step(np.array([[3]], np.int64))
    assert runtime.cursor == 2
    for current, original in zip(runtime.snapshot().layers, prior.layers):
        np.testing.assert_array_equal(current.keys, original.keys)
        np.testing.assert_array_equal(current.values, original.values)


def test_no_kv_head_replication_in_resident_storage():
    weights = StackedWeights.seeded(3, layers=4)
    runtime = StreamingStackedDecoder(weights, batch=2)
    report = runtime.report()
    _, _, depth, kv_heads = weights.validate()
    assert report.reserved_bytes == 2 * 4 * 2 * 64 * kv_heads * depth * 4
    assert all(not arr.flags.writeable for arr in runtime._weights.arrays().values())
