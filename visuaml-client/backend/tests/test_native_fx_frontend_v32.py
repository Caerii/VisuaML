"""End-to-end trusted FX graphs lowered into generated native C++17.

These are real traced PyTorch modules with synthetic parameter values. They
are NOT a second independently verified pretrained architecture family.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from visuaml.compiler_lab.native_cpu_v28 import compiler_available
from visuaml.compiler_lab.native_fx_frontend_v32 import lower_trusted_fx
from visuaml.compiler_lab.native_semantic_codegen_v32 import (
    NativeIRCodegenError,
    NativeSemanticCompiler,
)

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Native C++17 compiler unavailable"
)


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 7, bias=False)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(7, 4, bias=False)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x))) + x


class ResidualConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x)) + x


@pytest.mark.parametrize(
    "model_type,shape,expected_ops",
    (
        (ResidualMLP, (3, 4), ["matmul", "relu", "matmul", "add"]),
        (ResidualConv, (1, 2, 5, 5), ["conv2d", "relu", "add"]),
    ),
)
def test_restricted_trusted_fx_to_generated_cpp_against_torch(
    model_type, shape, expected_ops
):
    torch.manual_seed(132)
    model = model_type().eval()
    example = torch.randn(*shape)
    binding = lower_trusted_fx(model, example, trusted=True)
    assert binding.program.source_sha256
    compiler = NativeSemanticCompiler(binding.program)
    result, evidence = compiler.run_with_receipt(
        **binding.validated_inputs()
    )
    with torch.no_grad():
        oracle = model(example).numpy()
    np.testing.assert_allclose(result, oracle, atol=2e-5, rtol=2e-5)
    assert evidence["op_sequence"] == expected_ops
    assert len(binding.source_tensor_hashes) == (
        2 if model_type is ResidualMLP else 1
    )
    assert evidence["source_program_sha256"] == binding.program.source_sha256
    assert evidence["native_binary_sha256"] != evidence["generated_cpp_sha256"]
    # Changing only weights must change the originating model fingerprint.
    changed = model_type().eval()
    changed_binding = lower_trusted_fx(changed, example, trusted=True)
    assert binding.program.source_sha256 != changed_binding.program.source_sha256


def test_restricted_fx_requires_explicit_trusted_eval_model():
    model = ResidualMLP()
    example = torch.randn(3, 4)
    with pytest.raises(NativeIRCodegenError, match="trusted=True"):
        lower_trusted_fx(model, example)
    with pytest.raises(NativeIRCodegenError, match="eval-mode"):
        lower_trusted_fx(model, example, trusted=True)
    with pytest.raises(NativeIRCodegenError, match="bounded"):
        lower_trusted_fx(model.eval(), example.double(), trusted=True)


class BiasedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=True)

    def forward(self, x):
        return self.fc(x)


class GroupedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize(
    "model,shape",
    ((BiasedMLP, (3, 4)), (GroupedConv, (1, 2, 5, 5))),
)
def test_restricted_fx_rejects_unsupported_real_module_attributes(
    model, shape
):
    with pytest.raises(NativeIRCodegenError, match="Unsupported|Biased"):
        lower_trusted_fx(model().eval(), torch.randn(*shape), trusted=True)


def test_restricted_fx_contract_rejects_unsupported_effect_mutation():
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x)

    with pytest.raises(NativeIRCodegenError, match="In-place"):
        lower_trusted_fx(Inplace().eval(), torch.randn(3, 4), trusted=True)


def test_restricted_fx_rejects_shape_and_malformed_program_before_codegen():
    model = ResidualMLP().eval()
    binding = lower_trusted_fx(model, torch.randn(3, 4), trusted=True)
    malformed = replace(binding.program, outputs=("nonexistent_output",))
    with pytest.raises(NativeIRCodegenError):
        NativeSemanticCompiler(malformed)
    with pytest.raises(NativeIRCodegenError, match="bounded"):
        lower_trusted_fx(model, torch.full((3, 4), float("nan")), trusted=True)


def test_fx_parameter_binding_rejects_mutation_after_provenance_capture():
    model = ResidualMLP().eval()
    original = torch.randn(3, 4)
    binding = lower_trusted_fx(model, original, trusted=True)
    assert set(binding.validated_inputs()) == set(binding.inputs)
    name = next(iter(binding.source_tensor_hashes))
    binding.inputs[name][0, 0] += np.float32(0.1)
    with pytest.raises(NativeIRCodegenError, match="digest changed"):
        binding.validated_inputs()


def test_rejected_inplace_fx_module_does_not_mutate_original_sample():
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(x)

    sample = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float32)
    original = sample.clone()
    with pytest.raises(NativeIRCodegenError, match="In-place"):
        lower_trusted_fx(Inplace().eval(), sample, trusted=True)
    torch.testing.assert_close(sample, original)
