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
    _, _, model, plan = prepare(tmp_path, layers=2, batch=1)
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
