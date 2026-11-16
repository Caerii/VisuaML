"""Pure feed-forward region fusion must preserve the original Llama SSA meaning."""

from __future__ import annotations

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.fused_ffn_v20 import FusedFFNError, TorchScriptFusedFFN
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)


@pytest.mark.parametrize("seed", (2, 13))
def test_fused_ffn_matches_independent_numpy(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 5, 24)).astype(np.float32)
    norm = rng.normal(size=(24,)).astype(np.float32)
    gate = rng.normal(size=(48, 24)).astype(np.float32)
    up = rng.normal(size=(48, 24)).astype(np.float32)
    down = rng.normal(size=(24, 48)).astype(np.float32)
    eps = 1e-5
    n = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * norm
    g = n @ gate.T
    u = n @ up.T
    result = (g / (1.0 + np.exp(-g)) * u) @ down.T
    compiler = TorchScriptFusedFFN()
    got, report = compiler.run(x, norm, gate, up, down, eps=eps)
    np.testing.assert_allclose(got, result, atol=4e-5, rtol=4e-5)
    assert report.schema == "visuaml.fused-llama-ffn.v20"
    assert report.graph_nodes > 2
    assert compiler.specializations == 1


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_fused_backend_matches_actual_graph_and_reference(tmp_path, layers):
    fixture(tmp_path, layers=layers)
    imported = load_hf_llama_local(tmp_path, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(imported), imported
    )
    reference = plan.new_session()
    fused = plan.new_session(backend="torchscript_fused")
    tokens = np.array([[1, 3, 5, 7, 4, 2]], np.int64)
    for first, end in ((0, 3), (3, 4), (4, 6)):
        a = reference.step(tokens[:, first:end])
        b = fused.step(tokens[:, first:end])
        np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(
            b, imported.forward(tokens[:, :end])[:, first:end],
            atol=5e-5, rtol=5e-5,
        )
        for x, y in zip(reference.snapshot().layers, fused.snapshot().layers):
            np.testing.assert_allclose(x.keys, y.keys, atol=5e-5, rtol=5e-5)
            np.testing.assert_allclose(x.values, y.values, atol=5e-5, rtol=5e-5)
    assert fused.report()["fused_ffn_region_calls"] == layers * 3
    assert fused.report()["fused_ffn_specializations"] > 0
    assert fused.report()["compiled_primitive_calls"] == 0


def test_fused_ffn_rejects_wrong_shape_and_nonfinite_weights():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1, 3, 16)).astype(np.float32)
    norm = np.ones((16,), np.float32)
    gate = np.ones((32, 16), np.float32)
    up = gate.copy()
    down = np.ones((16, 32), np.float32)
    compiler = TorchScriptFusedFFN()
    with pytest.raises(FusedFFNError):
        compiler.run(x, norm, gate, up, down, eps=-1e-5)
    with pytest.raises(FusedFFNError):
        compiler.run(x, norm, gate, up, down[:, :-1], eps=1e-5)
    bad = gate.copy()
    bad[0, 0] = np.inf
    with pytest.raises(FusedFFNError):
        compiler.run(x, norm, bad, up, down, eps=1e-5)
    assert compiler.specializations == 0
