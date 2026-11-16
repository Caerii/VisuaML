"""V17: semantic-to-executable dynamic KV lowering and state correctness."""

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.kv_lowering_v17 import (
    KVLoweringError,
    KVSnapshot,
    load_kv_manifest,
    lower_kv_semantic_program,
    make_kv_semantic_program,
)
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


def plan(*, batch=2, layers=2, kv_heads=2, head_dim=4, capacity=64):
    return lower_kv_semantic_program(
        make_kv_semantic_program(
            batch=batch, layers=layers, kv_heads=kv_heads,
            head_dim=head_dim, capacity=capacity,
        )
    )


def chunks(compiled, n, *, seed=19):
    rng = np.random.default_rng(seed)
    return {
        slot: rng.standard_normal(
            (compiled.batch, n, compiled.kv_heads, compiled.head_dim)
        ).astype(np.float32)
        for slot in compiled.slots
    }


@pytest.mark.parametrize("sizes", [(1, 3, 5, 1), (7, 11, 19, 9, 18), (32, 32)])
def test_one_program_handles_varying_lengths_without_recompilation(sizes):
    compiled = plan()
    session = compiled.new_session()
    digest = compiled.semantic_sha256
    expected = {slot: [] for slot in compiled.slots}
    for i, n in enumerate(sizes):
        batch = chunks(compiled, n, seed=i + 8)
        for name, arr in batch.items():
            expected[name].append(arr)
        assert session.append(batch) == sum(sizes[:i + 1])
        current = dict(session.snapshot().buffers)
        for name in compiled.slots:
            np.testing.assert_array_equal(current[name], np.concatenate(expected[name], axis=1))
        assert session.plan.semantic_sha256 == digest
    report = session.report()
    assert report["live_bytes"] * compiled.capacity == report["reserved_bytes"] * session.cursor


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_dynamic_kv_append_matches_actual_v16_decoder(layers):
    weights = StackedWeights.seeded(41, layers=layers)
    _, _, depth, heads = weights.validate()
    compiled = plan(batch=2, layers=layers, kv_heads=heads, head_dim=depth)
    session = compiled.new_session()
    decoder = StreamingStackedDecoder(weights, batch=2)
    rng = np.random.default_rng(412)
    start = 0
    for n in (7, 1, 5, 10, 3):
        tokens = rng.integers(0, weights.embedding.shape[0], (2, n), dtype=np.int64)
        decoder.step(tokens)
        reference = decoder.snapshot()
        staged = {}
        for i, cache in enumerate(reference.layers):
            staged[f"k{i}"] = cache.keys[:, start:start + n].copy()
            staged[f"v{i}"] = cache.values[:, start:start + n].copy()
        session.append(staged)
        actual = dict(session.snapshot().buffers)
        for i, cache in enumerate(reference.layers):
            np.testing.assert_allclose(actual[f"k{i}"], cache.keys, rtol=0, atol=0)
            np.testing.assert_allclose(actual[f"v{i}"], cache.values, rtol=0, atol=0)
        start += n


def test_manifest_round_trip_and_closed_world_rejection():
    compiled = plan()
    manifest = compiled.manifest()
    assert load_kv_manifest(manifest).semantic_sha256 == compiled.semantic_sha256
    for mutation in (
        {**manifest, "capacity": 63},
        {**manifest, "schema": "unknown"},
        {**manifest, "evil_code": "import os"},
        {**manifest, "manifest_sha256": "0" * 64},
    ):
        with pytest.raises(KVLoweringError):
            load_kv_manifest(mutation)


def test_semantic_effect_reordering_and_shape_mutations_are_rejected():
    original = make_kv_semantic_program(
        batch=2, layers=2, kv_heads=2, head_dim=4, capacity=64
    )
    region = original.regions[0]
    reordered = replace(region, operations=(region.operations[1], region.operations[0], *region.operations[2:]))
    with pytest.raises(KVLoweringError):
        lower_kv_semantic_program(replace(original, regions=(reordered,)))
    changed = replace(region.operations[0], op="arbitrary_python")
    bad = replace(region, operations=(changed, *region.operations[1:]))
    with pytest.raises(KVLoweringError, match="Noncanonical"):
        lower_kv_semantic_program(replace(original, regions=(bad,)))
    forged = replace(original, source_sha256="0" * 64)
    with pytest.raises(KVLoweringError, match="Noncanonical"):
        lower_kv_semantic_program(forged)


def test_input_validation_is_atomic_and_does_not_alias():
    compiled = plan()
    session = compiled.new_session()
    first = chunks(compiled, 3)
    session.append(first)
    snapshot = session.snapshot()
    first["k0"][:] = 123
    np.testing.assert_array_equal(dict(session.snapshot().buffers)["k0"], dict(snapshot.buffers)["k0"])
    bad = chunks(compiled, 2)
    bad["v1"][0, 0, 0, 0] = np.nan
    with pytest.raises(KVLoweringError):
        session.append(bad)
    assert session.cursor == 3
    for name, arr in snapshot.buffers:
        np.testing.assert_array_equal(dict(session.snapshot().buffers)[name], arr)
    bad = chunks(compiled, 4)
    bad["k0"] = bad["k0"].astype(np.float64)
    with pytest.raises(KVLoweringError):
        session.append(bad)
    with pytest.raises(KVLoweringError):
        session.append({"k0": np.zeros((2, 1, 2, 4), np.float32)})
    assert session.cursor == 3


def test_snapshot_restore_is_detached_and_fails_closed():
    compiled = plan()
    session = compiled.new_session()
    session.append(chunks(compiled, 5))
    previous = session.snapshot()
    assert not dict(previous.buffers)["k0"].flags.writeable
    session.append(chunks(compiled, 3))
    session.restore(previous)
    assert session.cursor == 5
    for name, arr in previous.buffers:
        np.testing.assert_array_equal(dict(session.snapshot().buffers)[name], arr)
    damaged = list(previous.buffers)
    altered = damaged[-1][1].copy()
    altered[0, 0, 0, 0] = np.inf
    damaged[-1] = (damaged[-1][0], altered)
    with pytest.raises(KVLoweringError):
        session.restore(KVSnapshot(previous.semantic_sha256, 5, tuple(damaged)))
    assert session.cursor == 5
    with pytest.raises(KVLoweringError):
        session.restore(KVSnapshot("0" * 64, 5, previous.buffers))
    assert session.cursor == 5
    session.reset()
    assert session.cursor == 0


def test_capacity_and_resource_admission():
    with pytest.raises(KVLoweringError):
        make_kv_semantic_program(batch=8, layers=16, kv_heads=16, head_dim=128, capacity=4096)
    with pytest.raises(KVLoweringError):
        make_kv_semantic_program(batch=True, layers=1, kv_heads=1, head_dim=4, capacity=64)
    compiled = plan(capacity=8)
    session = compiled.new_session()
    session.append(chunks(compiled, 8))
    with pytest.raises(KVLoweringError):
        session.append(chunks(compiled, 1))
    assert session.cursor == 8


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_v17_semantic_decoder_matches_v16_logits_and_full_cache(layers):
    weights = StackedWeights.seeded(107, layers=layers)
    baseline = StreamingStackedDecoder(weights, batch=2)
    compiled = StreamingStackedDecoder(weights, batch=2, semantic_kv=True)
    rng = np.random.default_rng(101)
    for n in (7, 1, 3, 19, 8, 2, 5, 19):
        tokens = rng.integers(0, weights.embedding.shape[0], (2, n), dtype=np.int64)
        a = baseline.step(tokens)
        b = compiled.step(tokens)
        np.testing.assert_allclose(b, a, atol=3e-5, rtol=3e-5)
        for old, new in zip(baseline.snapshot().layers, compiled.snapshot().layers):
            np.testing.assert_array_equal(new.keys, old.keys)
            np.testing.assert_array_equal(new.values, old.values)
        assert compiled._kv_session.cursor == compiled.cursor
    assert compiled.cursor == 64


def test_v17_decoder_failure_restore_and_generation():
    weights = StackedWeights.seeded(39, layers=3)
    baseline = StreamingStackedDecoder(weights)
    compiled = StreamingStackedDecoder(weights, semantic_kv=True)
    prompt = np.array([[1, 2, 3]], np.int64)
    np.testing.assert_allclose(
        compiled.step(prompt), baseline.step(prompt), atol=3e-5, rtol=3e-5
    )
    snapshot = compiled.snapshot()
    rope = compiled._rope
    calls = 0

    def fail_on_late_layer(value, start):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected layer-three failure")
        return rope(value, start)

    compiled._rope = fail_on_late_layer
    with pytest.raises(RuntimeError, match="layer-three"):
        compiled.step(np.array([[4]], np.int64))
    assert compiled.cursor == compiled._kv_session.cursor == 3
    for before, after in zip(snapshot.layers, compiled.snapshot().layers):
        np.testing.assert_array_equal(before.keys, after.keys)
        np.testing.assert_array_equal(before.values, after.values)
    compiled._rope = rope
    compiled.step(np.array([[4]], np.int64))
    compiled.restore(snapshot)
    assert compiled.cursor == compiled._kv_session.cursor == 3
    compiled.reset()
    assert compiled.cursor == compiled._kv_session.cursor == 0
    a, _ = baseline.generate(prompt, steps=5)
    b, _ = compiled.generate(prompt, steps=5)
    np.testing.assert_array_equal(a, b)


def test_v17_effect_schedule_staging_commits_only_complete_layers():
    compiled = plan(layers=2)
    session = compiled.new_session()
    arrays = chunks(compiled, 3)
    with session.begin_append(3) as tx:
        tx.stage(0, arrays["k0"], arrays["v0"])
        with pytest.raises(KVLoweringError, match="incomplete"):
            tx.commit()
        with pytest.raises(KVLoweringError, match="order"):
            tx.stage(0, arrays["k0"], arrays["v0"])
        with pytest.raises(KVLoweringError, match="uncommitted"):
            session.snapshot()
    assert session.cursor == 0
    assert session._active is None
    with session.begin_append(3) as tx:
        tx.stage(0, arrays["k0"], arrays["v0"])
        tx.stage(1, arrays["k1"], arrays["v1"])
        assert tx.commit() == 3
    assert session.cursor == 3
    with pytest.raises(KVLoweringError, match="no longer active"):
        tx.stage(0, arrays["k0"], arrays["v0"])


def test_v17_session_canonicalizes_forged_plan_fields():
    from visuaml.compiler_lab.kv_lowering_v17 import KVSession

    compiled = plan()
    forged = replace(compiled, batch=7, slots=("evil",))
    session = KVSession(forged)
    assert session.plan == compiled
    assert set(session._arrays) == set(compiled.slots)
