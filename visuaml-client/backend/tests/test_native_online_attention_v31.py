"""Differential bounded-context streaming softmax vs score-materializing CPU decoder."""

from __future__ import annotations

import numpy as np
import pytest
from safetensors.torch import save_file
from test_hf_llama_checkpoint_v19 import fixture
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
    not compiler_available(), reason="Native C++17 compiler not installed"
)


@pytest.mark.parametrize(
    "shape,batch,chunks",
    (
        (
            {"hidden": 32, "heads": 4, "kv": 2, "middle": 80, "vocab": 64},
            1, (1, 7, 16, 16, 16),
        ),
        (
            {"hidden": 48, "heads": 6, "kv": 3, "middle": 101, "vocab": 67},
            2, (16, 16, 16, 8),
        ),
        (
            {"hidden": 96, "heads": 8, "kv": 2, "middle": 160, "vocab": 73},
            1, (32, 16, 15),
        ),
    ),
)
def test_online_attention_matches_independent_numpy_and_kv(
    tmp_path, shape, batch, chunks
):
    """Verify logit and final K/V parity near the admitted context boundary."""
    fixture(tmp_path, layers=2, **shape)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model, batch=batch),
        model, batch=batch,
    )
    baseline = NativeCpuLlamaCompiler(plan, enable_avx2=True)
    online = NativeCpuLlamaCompiler(
        plan, enable_avx2=True, enable_online_attention=True
    )
    assert baseline.report()["attention_kernel"] == "materialized_scores"
    assert online.report()["attention_kernel"] == "online_stable_softmax"
    assert online.report()["online_attention_requested"] is True
    assert (
        baseline.report()["executable_sha256"]
        != online.report()["executable_sha256"]
    )
    engines = (
        baseline.new_session(),
        online.new_session(),
        plan.new_session(backend="numpy"),
    )
    rng = np.random.default_rng(3117)
    for width in chunks:
        tokens = rng.integers(
            0, model.profile.vocabulary, size=(batch, width), dtype=np.int64
        )
        actual, candidate, expected = (
            session.step(tokens) for session in engines
        )
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(candidate, expected, atol=3e-4, rtol=3e-4)
        for observed in (engines[0].snapshot(), engines[1].snapshot()):
            for state, reference in zip(observed.layers, engines[2].snapshot().layers):
                np.testing.assert_allclose(
                    state.keys, reference.keys, atol=3e-4, rtol=3e-4
                )
                np.testing.assert_allclose(
                    state.values, reference.values, atol=3e-4, rtol=3e-4
                )
    assert engines[1].cursor == sum(chunks)
    final_snapshot = engines[1].snapshot()
    engines[1].reset()
    assert engines[1].cursor == 0
    engines[1].restore(final_snapshot)
    assert engines[1].cursor == sum(chunks)


def test_online_attention_selection_is_explicit(tmp_path):
    fixture(tmp_path, layers=1)
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    compiler = NativeCpuLlamaCompiler(plan, enable_online_attention=True)
    report = compiler.report()
    assert report["attention_kernel"] == "online_stable_softmax"
    assert "-DVISUAML_NATIVE_ONLINE_ATTENTION=1" in report["kernel_compile_flags"]
    assert len(report["cpp_source_sha256"]) == 64
    with pytest.raises(NativeCpuError, match="booleans"):
        NativeCpuLlamaCompiler(plan, enable_online_attention="true")


def test_online_attention_extreme_finite_score_separation(tmp_path):
    """Sharp but finite scores must preserve stable-softmax model behavior.

    Multiply only Q/K projection weights, preserving an otherwise admitted
    Llama graph; independent NumPy execution remains the numerical oracle.
    """
    tensors, _ = fixture(tmp_path, layers=1)
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ):
        tensors[name] = tensors[name] * 120.0
    save_file(tensors, str(tmp_path / "model.safetensors"))
    model = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(model), model
    )
    candidate = NativeCpuLlamaCompiler(
        plan, enable_online_attention=True
    ).new_session()
    reference = plan.new_session(backend="numpy")
    baseline = NativeCpuLlamaCompiler(plan).new_session()
    sequence = np.array(
        [[1, 2, 7, 3, 5, 11, 9, 4, 13, 2, 8, 6, 1, 7, 12, 5]],
        dtype=np.int64,
    )
    for piece in (sequence[:, :8], sequence[:, 8:12], sequence[:, 12:]):
        expected = reference.step(piece)
        for engine in (candidate, baseline):
            actual = engine.step(piece)
            np.testing.assert_allclose(
                actual, expected, atol=3e-4, rtol=3e-4
            )
            for layer, verified in zip(
                engine.snapshot().layers, reference.snapshot().layers
            ):
                np.testing.assert_allclose(
                    layer.keys, verified.keys, atol=3e-4, rtol=3e-4
                )
                np.testing.assert_allclose(
                    layer.values, verified.values, atol=3e-4, rtol=3e-4
                )
