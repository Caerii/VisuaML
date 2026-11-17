"""Static torch.export -> Semantic IR -> native C++ numerical witnesses.

Export is performed only on known local test modules; the importer does not
execute model Python, and never claims unknown remote modules are safe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from visuaml.compiler_lab.native_export_frontend_v33 import lower_static_export
from visuaml.compiler_lab.native_semantic_codegen_v32 import (
    NativeIRCodegenError,
    NativeSemanticCompiler,
)


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 7, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(7, 4, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) + x


class ResidualConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x)) + x


@pytest.mark.parametrize(
    "kind,shape,expected",
    (
        (ResidualMLP, (3, 4), ("matmul", "relu", "matmul", "add")),
        (ResidualConv, (1, 2, 5, 5), ("conv2d", "relu", "add")),
    ),
)
@pytest.mark.parametrize("seed", (17, 38))
def test_static_export_compiles_and_matches_eager_and_numpy(
    kind, shape, expected, seed
):
    torch.manual_seed(seed)
    module = kind().eval()
    sample = torch.randn(*shape)
    exported = torch.export.export(module, (sample,), strict=True)
    bound = lower_static_export(exported, sample, trusted=True)
    assert tuple(x.op for x in bound.program.regions[0].operations) == expected
    compiler = NativeSemanticCompiler(bound.program)
    actual, receipt = compiler.run_with_receipt(**bound.validated_inputs())
    with torch.no_grad():
        torch_expected = module(sample).detach().numpy()
    if kind is ResidualMLP:
        w1 = module.fc1.weight.detach().numpy().T.copy()
        w2 = module.fc2.weight.detach().numpy().T.copy()
        np_expected = (
            np.maximum(sample.numpy() @ w1, 0) @ w2 + sample.numpy()
        )
    else:
        # A separate independent NumPy spatial convolution oracle.
        x = sample.numpy()
        w = module.conv.weight.detach().numpy()
        reference = np.zeros_like(x)
        for channel in range(2):
            for iy in range(5):
                for ix in range(5):
                    total = np.float32(0)
                    for incoming in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                yy, xx = iy + ky - 1, ix + kx - 1
                                if 0 <= yy < 5 and 0 <= xx < 5:
                                    total += x[0, incoming, yy, xx] * w[
                                        channel, incoming, ky, kx
                                    ]
                    reference[0, channel, iy, ix] = total
        np_expected = np.maximum(reference, 0) + x
    np.testing.assert_allclose(actual, torch_expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual, np_expected, atol=2e-5, rtol=2e-5)
    assert receipt["source_program_sha256"] == bound.export_contract_sha256
    assert receipt["semantic_sha256"] == bound.program.digest()
    assert len(bound.parameter_sha256) == (2 if kind is ResidualMLP else 1)
    assert len(receipt["generated_cpp_sha256"]) == 64

    # Static guards allow new content under the *same* exact geometry.
    fresh = torch.randn(*shape)
    fresh_inputs = bound.validated_inputs(fresh)
    changed = compiler.run(**fresh_inputs)
    with torch.no_grad():
        expected_fresh = module(fresh).numpy()
    np.testing.assert_allclose(changed, expected_fresh, atol=2e-5, rtol=2e-5)
    assert not np.array_equal(changed, actual)


def test_static_export_rejects_untrusted_and_bad_input_contract():
    module = ResidualMLP().eval()
    sample = torch.randn(3, 4)
    exported = torch.export.export(module, (sample,), strict=True)
    with pytest.raises(NativeIRCodegenError, match="trusted=True"):
        lower_static_export(exported, sample)
    with pytest.raises(NativeIRCodegenError, match="shape"):
        lower_static_export(exported, torch.randn(4, 4), trusted=True)
    bound = lower_static_export(exported, sample, trusted=True)
    with pytest.raises(NativeIRCodegenError, match="fixed input guard"):
        bound.validated_inputs(torch.randn(4, 4))
    with pytest.raises(NativeIRCodegenError, match="fixed input guard"):
        bound.validated_inputs(torch.randn(3, 4, dtype=torch.float64))
    with pytest.raises(NativeIRCodegenError, match="fixed input guard"):
        bound.validated_inputs(torch.full((3, 4), float("nan")))


def test_static_export_checks_lifted_parameter_digest():
    module = ResidualMLP().eval()
    sample = torch.randn(3, 4)
    bound = lower_static_export(
        torch.export.export(module, (sample,), strict=True),
        sample, trusted=True,
    )
    weight = next(iter(bound.parameter_sha256))
    original = bound.inputs[weight].copy()
    bound.inputs[weight].flat[0] += np.float32(0.5)
    with pytest.raises(NativeIRCodegenError, match="provenance mismatch"):
        bound.validated_inputs()
    np.testing.assert_array_equal(original[1:], bound.inputs[weight][1:])


class BiasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=True)

    def forward(self, x):
        return self.fc(x)


class UnsupportedActivation(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def test_static_export_rejects_unadmitted_ops_and_bias():
    x = torch.randn(3, 4)
    for model in (BiasedModel().eval(), UnsupportedActivation().eval()):
        exported = torch.export.export(model, (x,), strict=True)
        with pytest.raises(NativeIRCodegenError):
            lower_static_export(exported, x, trusted=True)


def test_static_export_rejects_symbolic_range_guards():
    module = ResidualMLP().eval()
    sample = torch.randn(3, 4)
    symbolic = torch.export.export(
        module, (sample,), strict=True,
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2, max=8)}},
    )
    with pytest.raises(NativeIRCodegenError, match="Symbolic"):
        lower_static_export(symbolic, sample, trusted=True)


def test_static_export_does_not_mutate_original_input():
    sample = torch.tensor(
        [[-1.0, 3.0, -0.5, 2.0],
         [4.0, -2.0, 1.0, 1.0],
         [0.1, 0.3, 0.7, 0.9]], dtype=torch.float32,
    )
    original = sample.clone()
    module = ResidualMLP().eval()
    export = torch.export.export(module, (sample,), strict=True)
    bound = lower_static_export(export, sample, trusted=True)
    NativeSemanticCompiler(bound.program).run(**bound.validated_inputs())
    torch.testing.assert_close(sample, original)


class ParameterMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3) * 0.1)

    def forward(self, x):
        return x @ self.weight


def test_exported_rank_two_parameter_matmul_is_bound_and_generated():
    torch.manual_seed(333)
    model = ParameterMatMul().eval()
    first = torch.randn(2, 4)
    program = torch.export.export(model, (first,), strict=True)
    bound = lower_static_export(program, first, trusted=True)
    assert [op.op for op in bound.program.regions[0].operations] == ["matmul"]
    assert len(bound.parameter_sha256) == 1
    compiler = NativeSemanticCompiler(bound.program)
    native = compiler.run(**bound.validated_inputs())
    with torch.no_grad():
        baseline = model(first).numpy()
    np.testing.assert_allclose(native, baseline, atol=2e-5, rtol=2e-5)
    second = torch.randn(2, 4)
    alternative = compiler.run(**bound.validated_inputs(second))
    with torch.no_grad():
        torch_alternative = model(second).numpy()
    np.testing.assert_allclose(
        alternative, torch_alternative, atol=2e-5, rtol=2e-5
    )
