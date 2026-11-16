"""Differential witnesses for strict TorchScript CPU primitive lowering."""

import numpy as np
import pytest
from visuaml.compiler_lab.torchscript_primitives_v19 import (
    TorchScriptBackendError,
    TorchScriptPrimitives,
)


@pytest.mark.parametrize("operator", ("matmul", "rmsnorm", "silu", "add", "multiply"))
@pytest.mark.parametrize("seed", (0, 19))
def test_compiled_torchscript_matches_independent_numpy(operator, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 5, 12)).astype(np.float32)
    w = rng.standard_normal((12, 16)).astype(np.float32)
    b = rng.standard_normal((12,)).astype(np.float32)
    if operator == "matmul":
        args, expected = (x, w), x @ w
    elif operator == "rmsnorm":
        args = (x, b)
        expected = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * b
    elif operator == "silu":
        args, expected = (x,), x / (1.0 + np.exp(-x))
    elif operator == "add":
        args, expected = (x, x), x + x
    else:
        args, expected = (x, x), x * x
    compiler = TorchScriptPrimitives()
    result, report = compiler.run(operator, *args)
    np.testing.assert_allclose(result, expected, atol=5e-6, rtol=5e-6)
    assert report.executable == "TorchScript compiled CPU graph"
    assert report.graph_nodes > 0
    assert compiler.specializations == 1
    result2, _ = compiler.run(operator, *args)
    np.testing.assert_array_equal(result2, result)
    assert compiler.specializations == 1


def test_closed_world_inputs_and_shape_guards():
    compiler = TorchScriptPrimitives()
    x = np.ones((1, 3, 8), np.float32)
    with pytest.raises(TorchScriptBackendError):
        compiler.run("execute_python", x)
    with pytest.raises(TorchScriptBackendError):
        compiler.run("silu", x.astype(np.float64))
    with pytest.raises(TorchScriptBackendError):
        compiler.run("silu", np.full_like(x, np.nan))
    with pytest.raises(TorchScriptBackendError):
        compiler.run("matmul", x, np.ones((4, 8), np.float32))
    with pytest.raises(TorchScriptBackendError):
        compiler.run("rmsnorm", x, np.ones((7,), np.float32))
    assert compiler.specializations == 0
