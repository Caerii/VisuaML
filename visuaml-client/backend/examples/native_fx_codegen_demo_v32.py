"""Pinned-seed trusted FX -> Semantic IR -> generated C++ evidence.

These are synthetic PyTorch model fixtures, NOT named pretrained checkpoints.
The trusted FX importer executes local Python; never use on untrusted uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from examples.native_semantic_codegen_demo_v32 import numpy_conv
from torch import nn
from visuaml.compiler_lab.native_fx_frontend_v32 import lower_trusted_fx
from visuaml.compiler_lab.native_semantic_codegen_v32 import NativeSemanticCompiler


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


def run(seed: int = 32) -> dict:
    if type(seed) is not int or not 0 <= seed < 1000000:
        raise ValueError("Expected bounded integer fixture seed")
    torch.manual_seed(seed)
    fixtures = (
        ("trusted_fx_mlp", ResidualMLP().eval(), torch.randn(3, 4)),
        ("trusted_fx_conv", ResidualConv().eval(), torch.randn(1, 2, 5, 5)),
    )
    results = []
    for family, model, sample in fixtures:
        binding = lower_trusted_fx(model, sample, trusted=True)
        compiler = NativeSemanticCompiler(binding.program)
        generated, receipt = compiler.run_with_receipt(
            **binding.validated_inputs()
        )
        with torch.no_grad():
            torch_expected = model(sample).numpy()
        x = sample.numpy()
        if family == "trusted_fx_mlp":
            w1 = model.fc1.weight.detach().numpy().T.copy()
            w2 = model.fc2.weight.detach().numpy().T.copy()
            numpy_expected = np.maximum(x @ w1, 0) @ w2 + x
        else:
            w = model.conv.weight.detach().numpy().copy()
            numpy_expected = np.maximum(numpy_conv(x, w), 0) + x
        np.testing.assert_allclose(
            generated, torch_expected, atol=2e-5, rtol=2e-5,
        )
        np.testing.assert_allclose(
            generated, numpy_expected, atol=2e-5, rtol=2e-5,
        )
        assert hashlib.sha256(compiler.source.encode()).hexdigest() == (
            receipt["generated_cpp_sha256"]
        )
        results.append({
            "family": family,
            "model_provenance_sha256": binding.program.source_sha256,
            "weight_sha256": binding.source_tensor_hashes,
            "max_abs_error_numpy": float(
                np.max(np.abs(generated - numpy_expected))
            ),
            "max_abs_error_torch": float(
                np.max(np.abs(generated - torch_expected))
            ),
            "compiler": receipt,
            "generated_cpp": compiler.source,
        })
    root = Path(__file__).resolve().parents[1]
    paths = (
        "examples/native_fx_codegen_demo_v32.py",
        "visuaml/compiler_lab/native_fx_frontend_v32.py",
        "visuaml/compiler_lab/native_semantic_codegen_v32.py",
    )
    return {
        "schema": "visuaml.trusted-fx-generated-native.v32",
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "seed": seed,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "fixtures": results,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths
        },
        "limitations": (
            "Trusted synthetic FX modules only; shape propagation executes "
            "Python, no untrusted module tracing, pretrained family claim, "
            "symbolic shapes, model-bias support, or portable offline AOT."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed)
    wire = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
