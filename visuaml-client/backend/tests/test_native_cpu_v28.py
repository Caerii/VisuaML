"""Independent native C++17 Llama backend versus NumPy/PyTorch references."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_hf_llama_checkpoint_v19 import fixture, torch_oracle
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)
from visuaml.compiler_lab.native_cpu_v28 import (
    NativeCpuError,
    NativeCpuLlamaCompiler,
    compiler_available,
)

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="No C++17 compiler installed"
)


def prepare(tmp_path, *, layers=2, batch=1):
    originals, config = fixture(tmp_path, layers=layers)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch), model, batch=batch
    )
    return originals, config, model, plan


@pytest.mark.parametrize("layers,batch", ((1, 1), (2, 2), (4, 1)))
def test_native_whole_decoder_matches_independent_numpy_and_torch(
    tmp_path, layers, batch
):
    weights, config, model, plan = prepare(tmp_path, layers=layers, batch=batch)
    compiler = NativeCpuLlamaCompiler(plan)
    native = compiler.new_session()
    oracle = plan.new_session(backend="numpy")
    rng = np.random.default_rng(200 + layers)
    history = []
    for chunk in (3, 1, 5, 2):
        tokens = rng.integers(
            0, model.profile.vocabulary, size=(batch, chunk), dtype=np.int64
        )
        history.append(tokens)
        full = np.concatenate(history, axis=1)
        got = native.step(tokens)
        np.testing.assert_allclose(
            got, oracle.step(tokens), atol=2e-4, rtol=2e-4
        )
        np.testing.assert_allclose(
            got, model.forward(full)[:, -chunk:], atol=2e-4, rtol=2e-4
        )
        with torch.no_grad():
            expected = torch_oracle(
                torch.from_numpy(full), weights, config
            ).numpy()[:, -chunk:]
        np.testing.assert_allclose(
            got, expected, atol=2e-4, rtol=2e-4
        )
        for left, right in zip(native.snapshot().layers, oracle.snapshot().layers):
            np.testing.assert_allclose(
                left.keys, right.keys, atol=2e-4, rtol=2e-4
            )
            np.testing.assert_allclose(
                left.values, right.values, atol=2e-4, rtol=2e-4
            )
    assert native.cursor == 11
    report = compiler.report()
    assert report["schema"] == "visuaml.native-cpu-llama.v28"
    assert len(report["cpp_source_sha256"]) == 64
    assert len(report["executable_sha256"]) == 64
    assert report["compilation_ms"] > 0
    assert report["independent_max_abs_error"] < 2e-4


def test_native_generation_and_checkpoint_restore(tmp_path):
    _, _, _, plan = prepare(tmp_path, layers=2, batch=1)
    native = NativeCpuLlamaCompiler(plan).new_session()
    semantic = plan.new_session()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    native.step(prompt)
    saved = native.snapshot()
    native.step(np.array([[4]], dtype=np.int64))
    native.restore(saved)
    assert native.cursor == 3
    native.reset()
    ids_a, state_a = native.generate(prompt, steps=5)
    ids_b, state_b = semantic.generate(prompt, steps=5)
    np.testing.assert_array_equal(ids_a, ids_b)
    for a, b in zip(state_a.layers, state_b.layers):
        np.testing.assert_allclose(a.keys, b.keys, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(a.values, b.values, atol=2e-4, rtol=2e-4)
    assert ids_a.shape == (1, 5)


def test_invalid_tokens_and_failed_native_call_never_publish_partial_cache(
    tmp_path, monkeypatch
):
    _, _, model, plan = prepare(tmp_path, layers=1, batch=1)
    compiler = NativeCpuLlamaCompiler(plan)
    native = compiler.new_session()
    native.step(np.array([[1, 2]], np.int64))
    state = native.snapshot()
    for tokens in (
        np.array([[-1]], dtype=np.int64),
        np.array([[model.profile.vocabulary]], dtype=np.int64),
        np.array([[1]], dtype=np.int32),
        np.ones((2, 1), np.int64),
        np.ones((1, 33), np.int64),
    ):
        with pytest.raises(NativeCpuError):
            native.step(tokens)
        assert native.cursor == 2

    original = compiler._decode
    monkeypatch.setattr(compiler, "_decode", lambda *args: -99)
    with pytest.raises(NativeCpuError, match="failed"):
        native.step(np.array([[3]], np.int64))
    monkeypatch.setattr(compiler, "_decode", original)
    assert native.cursor == 2
    for before, after in zip(state.layers, native.snapshot().layers):
        np.testing.assert_array_equal(before.keys, after.keys)
        np.testing.assert_array_equal(before.values, after.values)


def test_native_admission_rejects_forged_semantic_graph(tmp_path):
    _, _, _, plan = prepare(tmp_path, layers=1)
    with pytest.raises(NativeCpuError):
        NativeCpuLlamaCompiler("untrusted-native-source")
    operation = plan.program.regions[0].operations[0]
    changed = replace(operation, op="remote_eval")
    region = replace(
        plan.program.regions[0],
        operations=(changed, *plan.program.regions[0].operations[1:]),
    )
    with pytest.raises(ValueError):
        NativeCpuLlamaCompiler(
            replace(plan, program=replace(plan.program, regions=(region,)))
        )


def test_native_sessions_share_binary_but_not_mutable_cache(tmp_path):
    _, _, model, plan = prepare(tmp_path, layers=1)
    compiler = NativeCpuLlamaCompiler(plan)
    a, b = compiler.new_session(), compiler.new_session()
    assert a._compiler is b._compiler
    assert a.cache._arrays["k0"] is not b.cache._arrays["k0"]
    first = np.array([[1, 2]], np.int64)
    second = np.array([[3, 4, 5]], np.int64)
    np.testing.assert_allclose(
        a.step(first), model.forward(first), atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(
        b.step(second), model.forward(second), atol=2e-4, rtol=2e-4
    )
    assert a.cursor == 2 and b.cursor == 3


@pytest.mark.parametrize(
    "chunks",
    (
        (1, 1, 1, 1, 1, 1),
        (2, 4),
        (4, 2),
        (6,),
    ),
)
def test_native_rope_reuse_is_chunk_partition_invariant(tmp_path, chunks):
    """Q/K angle reuse must not depend on prefill/decode chunk boundaries."""
    _, _, model, plan = prepare(tmp_path, layers=1, batch=2)
    compiler = NativeCpuLlamaCompiler(plan)
    native = compiler.new_session()
    oracle = plan.new_session(backend="numpy")
    rng = np.random.default_rng(629)
    tokens = rng.integers(
        0, model.profile.vocabulary, size=(2, 6), dtype=np.int64
    )
    cursor = 0
    actual_parts = []
    expected_parts = []
    for width in chunks:
        part = tokens[:, cursor:cursor + width]
        actual_parts.append(native.step(part))
        expected_parts.append(oracle.step(part))
        cursor += width
    assert cursor == tokens.shape[1]
    np.testing.assert_allclose(
        np.concatenate(actual_parts, axis=1),
        np.concatenate(expected_parts, axis=1),
        atol=2e-4, rtol=2e-4,
    )
    for actual, expected in zip(
        native.snapshot().layers, oracle.snapshot().layers
    ):
        np.testing.assert_allclose(
            actual.keys, expected.keys, atol=2e-4, rtol=2e-4
        )
        np.testing.assert_allclose(
            actual.values, expected.values, atol=2e-4, rtol=2e-4
        )


@pytest.mark.parametrize("layers,batch", ((1, 1), (2, 2)))
def test_native_four_row_linear_matches_separate_scalar_binary(
    tmp_path, layers, batch
):
    """Two independently compiled kernel variants must agree, including KV."""
    _, _, model, plan = prepare(tmp_path, layers=layers, batch=batch)
    fast_compiler = NativeCpuLlamaCompiler(plan)
    reference_compiler = NativeCpuLlamaCompiler(plan, reference_kernels=True)
    assert fast_compiler.report()["linear_kernel"] == "four_row"
    assert reference_compiler.report()["linear_kernel"] == "scalar_reference"
    assert (
        fast_compiler.report()["executable_sha256"]
        != reference_compiler.report()["executable_sha256"]
    )
    fast, reference = fast_compiler.new_session(), reference_compiler.new_session()
    semantic = plan.new_session(backend="numpy")
    rng = np.random.default_rng(831 + layers)
    for width in (3, 1, 4, 2):
        tokens = rng.integers(
            0, model.profile.vocabulary, size=(batch, width), dtype=np.int64
        )
        a, b, c = fast.step(tokens), reference.step(tokens), semantic.step(tokens)
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(a, c, rtol=2e-4, atol=2e-4)
        for af, bf, cf in zip(
            fast.snapshot().layers,
            reference.snapshot().layers,
            semantic.snapshot().layers,
        ):
            np.testing.assert_allclose(af.keys, bf.keys, rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(af.values, bf.values, rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(af.keys, cf.keys, rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(af.values, cf.values, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    "shape",
    (
        {"hidden": 32, "heads": 4, "kv": 2, "middle": 80, "vocab": 64},
        {"hidden": 48, "heads": 6, "kv": 3, "middle": 101, "vocab": 67},
    ),
)
def test_optional_native_avx2_matches_scalar_four_row_and_numpy(tmp_path, shape):
    """Runtime SIMD dispatch must preserve full decoding and KV state.

    The second geometry exercises non-multiple-of-eight output widths and
    input tails, without requiring that the test machine actually has AVX2.
    """
    fixture(tmp_path, layers=2, **shape)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=2), model, batch=2
    )
    simd = NativeCpuLlamaCompiler(plan, enable_avx2=True)
    four_row = NativeCpuLlamaCompiler(plan)
    scalar = NativeCpuLlamaCompiler(plan, reference_kernels=True)
    assert simd.report()["avx2_requested"] is True
    assert four_row.report()["avx2_requested"] is False
    if simd.report()["avx2_active"]:
        assert simd.report()["linear_kernel"].startswith("avx2_runtime")
    else:
        assert simd.report()["linear_kernel"] == "four_row"
    sessions = [
        simd.new_session(), four_row.new_session(), scalar.new_session(),
        plan.new_session(backend="numpy"),
    ]
    rng = np.random.default_rng(2929)
    for width in (3, 1, 5):
        tokens = rng.integers(
            0, model.profile.vocabulary, size=(2, width), dtype=np.int64
        )
        outputs = [session.step(tokens) for session in sessions]
        for output in outputs[:-1]:
            np.testing.assert_allclose(
                output, outputs[-1], atol=2e-4, rtol=2e-4
            )
        states = [session.snapshot() for session in sessions]
        for state in states[:-1]:
            for actual, reference in zip(state.layers, states[-1].layers):
                np.testing.assert_allclose(
                    actual.keys, reference.keys, atol=2e-4, rtol=2e-4
                )
                np.testing.assert_allclose(
                    actual.values, reference.values, atol=2e-4, rtol=2e-4
                )


def test_native_avx2_selection_rejects_conflicting_modes(tmp_path):
    _, _, _, plan = prepare(tmp_path, layers=1)
    with pytest.raises(NativeCpuError, match="Scalar reference"):
        NativeCpuLlamaCompiler(plan, reference_kernels=True, enable_avx2=True)
    with pytest.raises(NativeCpuError, match="booleans"):
        NativeCpuLlamaCompiler(plan, enable_avx2="true")


@pytest.mark.parametrize(
    "shape,batch",
    (
        ({"hidden": 32, "heads": 4, "kv": 2, "middle": 80, "vocab": 64}, 1),
        ({"hidden": 48, "heads": 6, "kv": 3, "middle": 101, "vocab": 67}, 2),
        ({"hidden": 128, "heads": 8, "kv": 4, "middle": 384, "vocab": 257}, 1),
    ),
)
def test_native_batched_prefill_matches_avx2_scalar_and_numpy(
    tmp_path, shape, batch
):
    """All six projection sites preserve chunked logits and every KV layer."""
    fixture(tmp_path, layers=2, **shape)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch), model, batch=batch
    )
    baseline = NativeCpuLlamaCompiler(plan, enable_avx2=True)
    batched = NativeCpuLlamaCompiler(
        plan, enable_avx2=True, enable_batched_prefill=True
    )
    scalar = NativeCpuLlamaCompiler(plan, reference_kernels=True)
    assert batched.report()["batched_prefill_requested"] is True
    assert baseline.report()["batched_prefill_requested"] is False
    expected_gate = (
        bool(batched.report()["avx2_active"]) and shape["hidden"] >= 128
    )
    assert batched.prefill_dispatch(7)["native_batched_selected"] == expected_gate
    assert batched.prefill_dispatch(1)["native_batched_selected"] is False
    assert baseline.prefill_dispatch(7)["native_batched_selected"] is False
    assert (
        baseline.report()["executable_sha256"]
        != batched.report()["executable_sha256"]
    )
    sessions = (
        baseline.new_session(),
        batched.new_session(),
        scalar.new_session(),
        plan.new_session(backend="numpy"),
    )
    rng = np.random.default_rng(3001)
    # Prefill, decode, a longer continuation, and another decode.
    for length in (7, 1, 4, 1):
        tokens = rng.integers(
            0, model.profile.vocabulary, size=(batch, length),
            dtype=np.int64,
        )
        outputs = [session.step(tokens) for session in sessions]
        for actual in outputs[:-1]:
            np.testing.assert_allclose(
                actual, outputs[-1], atol=2e-4, rtol=2e-4
            )
        states = [session.snapshot() for session in sessions]
        for state in states[:-1]:
            for actual, expected in zip(state.layers, states[-1].layers):
                np.testing.assert_allclose(
                    actual.keys, expected.keys, atol=2e-4, rtol=2e-4
                )
                np.testing.assert_allclose(
                    actual.values, expected.values, atol=2e-4, rtol=2e-4
                )
    assert sessions[1].cursor == 13


def test_native_batched_prefill_rejects_unsafe_selection(tmp_path):
    _, _, _, plan = prepare(tmp_path, layers=1)
    with pytest.raises(NativeCpuError, match="Scalar reference"):
        NativeCpuLlamaCompiler(
            plan, reference_kernels=True, enable_batched_prefill=True
        )
    with pytest.raises(NativeCpuError, match="booleans"):
        NativeCpuLlamaCompiler(plan, enable_batched_prefill="yes")
    with pytest.raises(NativeCpuError, match="requires guarded AVX2"):
        NativeCpuLlamaCompiler(plan, enable_batched_prefill=True)
    compiler = NativeCpuLlamaCompiler(plan, enable_avx2=True)
    with pytest.raises(NativeCpuError, match="Invalid prefill"):
        compiler.prefill_dispatch(0)
    with pytest.raises(NativeCpuError, match="Invalid prefill"):
        compiler.prefill_dispatch(33)
