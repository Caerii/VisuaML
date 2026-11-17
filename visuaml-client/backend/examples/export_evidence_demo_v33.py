"""Reproducible v33 exported-model and transformation-evidence demonstration.

Uses only trusted, locally declared synthetic modules.  A passing record is
finite numerical evidence; it does not authenticate a signer or prove that
an optimization is correct on every input or architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from visuaml.compiler_lab.native_export_frontend_v33 import lower_static_export
from visuaml.compiler_lab.native_semantic_codegen_v32 import NativeSemanticCompiler
from visuaml.compiler_lab.optimization_search_v33 import evaluate_relu_candidate
from visuaml.compiler_lab.transformation_evidence_v33 import (
    certify_native_rewrite,
    eliminate_idempotent_relu,
    verify_record_integrity,
)


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 7, bias=False)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(7, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x))) + x


class ResidualConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x)) + x


class DoubleReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.relu(x))


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def run(*, seed: int = 3303, trials: int = 7) -> dict:
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError("Bounded deterministic seed required")
    if type(trials) is not int or not 2 <= trials <= 25:
        raise ValueError("Expected 2..25 benchmark repetitions")
    torch.manual_seed(seed)
    models = (
        ("exported_residual_mlp", ResidualMLP().eval(), torch.randn(3, 4)),
        ("exported_residual_conv", ResidualConv().eval(), torch.randn(1, 2, 5, 5)),
    )
    entries = []
    for name, module, example in models:
        exported = torch.export.export(module, (example,), strict=True)
        bound = lower_static_export(exported, example, trusted=True)
        compiler = NativeSemanticCompiler(bound.program)
        actual, receipt = compiler.run_with_receipt(
            **bound.validated_inputs()
        )
        with torch.no_grad():
            independent = module(example).detach().numpy().copy()
        np.testing.assert_allclose(
            actual, independent, atol=2e-5, rtol=2e-5,
        )
        fresh = torch.randn_like(example)
        observed = compiler.run(**bound.validated_inputs(fresh))
        with torch.no_grad():
            expected = module(fresh).detach().numpy().copy()
        np.testing.assert_allclose(
            observed, expected, atol=2e-5, rtol=2e-5,
        )
        samples = []
        for _ in range(trials):
            start = time.perf_counter_ns()
            compiler.run(**bound.validated_inputs())
            samples.append((time.perf_counter_ns() - start) / 1e6)
        entries.append({
            "family": name,
            "op_sequence": receipt["op_sequence"],
            "source_export_sha256": bound.export_contract_sha256,
            "parameter_sha256": bound.parameter_sha256,
            "input_sha256": receipt["input_sha256"],
            "output_sha256": _sha(actual),
            "max_abs_error_eager_torch": float(
                np.max(np.abs(actual - independent))
            ),
            "max_abs_error_second_input": float(
                np.max(np.abs(observed - expected))
            ),
            "generated_cpp": compiler.source,
            "compiler": compiler.report(),
            "warm_native_ms": {
                "median": round(statistics.median(samples), 6),
                "raw": [round(x, 6) for x in samples],
            },
        })
    sample = torch.tensor(
        [[-3.0, 0.5, 0.0, 1.3],
         [2.0, -4.0, 0.2, 3.0],
         [0.1, 0.6, -0.2, 1.0]], dtype=torch.float32,
    )
    exported = torch.export.export(DoubleReLU().eval(), (sample,), strict=True)
    bound = lower_static_export(exported, sample, trusted=True)
    before = bound.program
    after = eliminate_idempotent_relu(before)

    def numpy_oracle(*, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.maximum(x, 0), 0).astype(np.float32)

    def torch_oracle(*, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.relu(
                torch.relu(torch.from_numpy(x.copy()))
            ).numpy().copy()

    witnesses = [
        {"x": sample.numpy().copy()},
        {"x": torch.randn_like(sample).numpy().copy()},
    ]
    rewrite = certify_native_rewrite(
        before, after, rule="relu-idempotence-v1",
        witnesses=witnesses,
        numpy_oracle=numpy_oracle,
        torch_oracle=torch_oracle,
    )
    if rewrite["status"] != "passed_finite_witnesses":
        raise RuntimeError("The experimental rewrite did not pass its witnesses")
    if not verify_record_integrity(rewrite):
        raise RuntimeError("Evidence record integrity verification failed")
    optimization = evaluate_relu_candidate(
        before, witnesses=witnesses,
        numpy_oracle=numpy_oracle, torch_oracle=torch_oracle,
        trials=trials,
    )
    if optimization["status"] != "exploratory_timing_after_finite_numerical_witnesses":
        raise RuntimeError("Rejected candidate cannot enter experimental search")
    root = Path(__file__).resolve().parents[1]
    files = (
        "examples/export_evidence_demo_v33.py",
        "visuaml/compiler_lab/native_export_frontend_v33.py",
        "visuaml/compiler_lab/transformation_evidence_v33.py",
        "visuaml/compiler_lab/optimization_search_v33.py",
        "visuaml/compiler_lab/native_semantic_codegen_v32.py",
    )
    return {
        "schema": "visuaml.export-evidence-demonstration.v33",
        "source": "trusted synthetic locally defined PyTorch modules",
        "seed": seed,
        "trials": trials,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "model_witnesses": entries,
        "rewrite_witness": rewrite,
        "optimization_experiment": optimization,
        "source_sha256": {
            file: hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in files
        },
        "limitations": [
            "Synthetic model weights, not two pinned pretrained architectures.",
            "Strict static shapes only, no symbolic guard translation.",
            "Finitely tested outputs, not formal equivalence.",
            "Untrusted modules must never be traced by the caller.",
            "Digest integrity does not authenticate an evidence signer.",
            "No universal throughput, GPU/optical or energy claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3303)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run(seed=args.seed, trials=args.trials)
    wire = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(json.dumps({
        "schema": record["schema"],
        "families": [item["family"] for item in record["model_witnesses"]],
        "errors": [item["max_abs_error_eager_torch"] for item in record["model_witnesses"]],
        "rewrite_status": record["rewrite_witness"]["status"],
        "rewrite_digest": record["rewrite_witness"]["record_sha256"],
        "optimization_status": record["optimization_experiment"]["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
