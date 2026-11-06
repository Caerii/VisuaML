"""Run with: PYTHONPATH=visuaml-client/backend pytest -q visuaml-client/backend/tests/test_compiler_spike.py"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from visuaml.compiler_spike import (
    NumpyBackend,
    OpticalSurrogateBackend,
    UnsupportedOperation,
    capture,
    compare_outputs,
    decompose_linear,
    fuse_linear_pattern,
    lower,
    plan,
    register_backend,
    to_visuaml_graph,
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))


class WinnerTakeAll(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, state, weights):
        drive = state @ weights
        _, indices = torch.topk(drive, self.k, dim=-1)
        return torch.zeros_like(drive).scatter(-1, indices, 1.0)


def test_analysis_contains_shapes_and_decompositions():
    program = capture(MLP(), torch.randn(2, 4))
    assert program.ir.input_names == ("x",)
    assert [n.op for n in program.ir.nodes].count("linear") == 2
    assert program.ir.summary()["estimated_flops"] > 0
    assert any(n.decomposition for n in program.ir.nodes)
    assert program.ir.nodes[-2].tensors[0].shape == ("2", "3")
    json.dumps(program.ir.as_dict())


def test_linear_decomposition_executes_and_approximates():
    torch.manual_seed(1)
    x = torch.randn(5, 4)
    captured = capture(MLP(), x)
    expanded = decompose_linear(captured, x)
    assert "linear" not in [n.op for n in expanded.ir.nodes]
    assert [n.op for n in expanded.ir.nodes].count("matmul") == 2
    expected = captured.graph_module(x)
    actual = expanded.graph_module(x)
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-6)


def test_cross_backend_mlp():
    torch.manual_seed(2)
    x = torch.randn(6, 4)
    program = capture(MLP(), x)
    result = lower(program, "numpy")
    assert not result.plan.unsupported
    reference = program.graph_module(x)
    candidate = result.run(x)
    assert compare_outputs(reference, candidate)["max_abs_error"] < 1e-5
    expanded = decompose_linear(program, x)
    result2 = lower(expanded, "numpy")
    assert compare_outputs(reference, result2.run(x))["max_abs_error"] < 1e-5


def test_sparse_wta_torch_numpy_surrogate():
    torch.manual_seed(11)
    state, weights = torch.randn(3, 5), torch.randn(5, 12)
    captured = capture(WinnerTakeAll(), state, weights)
    assert {"topk", "scatter", "matmul"}.issubset({n.op for n in captured.ir.nodes})
    reference = captured.graph_module(state, weights)
    result = lower(captured, "numpy")
    np.testing.assert_array_equal(reference.numpy(), result.run(state, weights))
    analog = lower(captured, "optical-surrogate")
    assert any(x.reason == "explicit digital fallback" for x in analog.plan.placements)
    assert analog.run(state, weights).shape == (3, 12)


def test_unsupported_operation_fails_closed():
    class Unsupported(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    captured = capture(Unsupported(), torch.randn(2, 3))
    with pytest.raises(UnsupportedOperation, match="unsupported nodes"):
        lower(captured, "numpy")
    assert plan(captured.ir, NumpyBackend()).unsupported


def test_noise_surrogate_is_seeded_not_device_claim():
    torch.manual_seed(10)
    inp = torch.randn(3, 4)
    captured = capture(MLP(), inp)
    simulator = OpticalSurrogateBackend(quant_bits=5, noise_std=0.01, seed=10)
    register_backend(simulator)
    a = lower(captured, "optical-surrogate").run(inp)
    b = lower(captured, "optical-surrogate").run(inp)
    np.testing.assert_array_equal(a, b)
    assert compare_outputs(captured.graph_module(inp), a)["max_abs_error"] > 0


def test_custom_backend_extensible():
    class CustomBackend:
        name = "my-new-backend"
        native_ops = frozenset({"linear", "relu"})
        fallback_ops = frozenset()

        def run(self, gm, *args):
            return gm(*args)

    register_backend(CustomBackend())
    x = torch.randn(2, 4)
    captured = capture(MLP(), x)
    candidate = lower(captured, "my-new-backend")
    assert candidate.plan.backend == "my-new-backend"
    assert torch.allclose(candidate.run(x), captured.graph_module(x))


def test_visuaml_canvas_adapter_uses_ir_identity():
    x = torch.randn(2, 4)
    captured = capture(MLP(), x)
    overlay = plan(captured.ir, OpticalSurrogateBackend())
    graph = to_visuaml_graph(captured.ir, overlay)
    assert len(graph["nodes"]) == len(captured.ir.nodes)
    assert graph["nodes"][1]["data"]["backend"] == "optical-surrogate"
    assert graph["edges"][0]["source"] == "x"
    json.dumps(graph)


def test_biasless_linear_decomposition():
    class Biasless(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2, bias=False)

        def forward(self, x):
            return self.linear(x)

    x = torch.randn(4, 3)
    base = capture(Biasless(), x)
    rewritten = decompose_linear(base, x)
    assert (
        compare_outputs(base.graph_module(x), rewritten.graph_module(x))[
            "max_abs_error"
        ]
        < 1e-6
    )
    assert not any(x.op == "add" for x in rewritten.ir.nodes)


def test_decompose_then_fuse_linear():
    torch.manual_seed(3)
    x = torch.randn(3, 4)
    source = capture(MLP(), x)
    lowered = decompose_linear(source, x)
    re_fused = fuse_linear_pattern(lowered, x)
    assert [n.op for n in re_fused.ir.nodes].count("linear") == 2
    assert [n.op for n in re_fused.ir.nodes].count("matmul") == 0
    assert (
        compare_outputs(source.graph_module(x), re_fused.graph_module(x))[
            "max_abs_error"
        ]
        < 1e-5
    )
    assert (
        compare_outputs(source.graph_module(x), lower(re_fused, "numpy").run(x))[
            "max_abs_error"
        ]
        < 1e-5
    )
