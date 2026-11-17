"""Run and record two genuinely graph-generated native CPU fixture families.

This demonstrates pure, bounded Semantic IR -> C++17 generation with
independent NumPy and eager PyTorch numerical oracles. Fixtures are random,
NOT pretrained models, serving workloads, or universal native coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from math import prod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from visuaml.compiler_lab.native_semantic_codegen_v32 import NativeSemanticCompiler
from visuaml.compiler_lab.semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
)


def vt(*dims: int) -> ValueType:
    return ValueType(tuple(dims), "float32", 4 * prod(dims))


def graph(name: str, input_types, instructions, output: str) -> SemanticProgram:
    operations = tuple(
        SemanticOperation(
            label, "tensor", op, tuple(args), label, result_type,
            parameters=params,
        )
        for label, op, args, result_type, params in instructions
    )
    return SemanticProgram(
        "torch-fx", tuple(input_types), (),
        (SemanticRegion("forward", 1, operations),), (output,),
        hashlib.sha256(("declared-fixture:" + name).encode()).hexdigest(),
    )


def mlp_program() -> SemanticProgram:
    return graph(
        "residual-mlp",
        (("x", vt(3, 4)), ("w1", vt(4, 7)), ("w2", vt(7, 4))),
        (
            ("mm1", "matmul", ("x", "w1"), vt(3, 7), ()),
            ("relu1", "relu", ("mm1",), vt(3, 7), ()),
            ("mm2", "matmul", ("relu1", "w2"), vt(3, 4), ()),
            ("out", "add", ("mm2", "x"), vt(3, 4), ()),
        ), "out",
    )


def conv_program() -> SemanticProgram:
    return graph(
        "residual-convolution",
        (("x", vt(1, 2, 5, 5)), ("weight", vt(2, 2, 3, 3))),
        (
            ("conv", "conv2d", ("x", "weight"), vt(1, 2, 5, 5),
             (("stride", 1), ("padding", 1))),
            ("activated", "relu", ("conv",), vt(1, 2, 5, 5), ()),
            ("out", "add", ("activated", "x"), vt(1, 2, 5, 5), ()),
        ), "out",
    )


def numpy_conv(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    output = np.zeros_like(x)
    for n in range(1):
        for out_channel in range(2):
            for y in range(5):
                for z in range(5):
                    total = np.float32(0.0)
                    for channel in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = y + ky - 1, z + kx - 1
                                if 0 <= iy < 5 and 0 <= ix < 5:
                                    total += (
                                        x[n, channel, iy, ix]
                                        * weight[out_channel, channel, ky, kx]
                                    )
                    output[n, out_channel, y, z] = total
    return output


def benchmark(trials: int = 9, seed: int = 832) -> dict:
    if type(trials) is not int or not 2 <= trials <= 100:
        raise ValueError("Need 2..100 finite benchmark repetitions")
    rng = np.random.default_rng(seed)
    fixtures = (
        (
            "residual-mlp", mlp_program(),
            {
                "x": rng.standard_normal((3, 4)).astype(np.float32),
                "w1": rng.standard_normal((4, 7)).astype(np.float32) * 0.2,
                "w2": rng.standard_normal((7, 4)).astype(np.float32) * 0.2,
            },
        ),
        (
            "convolutional-residual", conv_program(),
            {
                "x": rng.standard_normal((1, 2, 5, 5)).astype(np.float32) * 0.2,
                "weight": rng.standard_normal((2, 2, 3, 3)).astype(np.float32) * 0.1,
            },
        ),
    )
    entries = []
    for name, semantic, arrays in fixtures:
        compiled = NativeSemanticCompiler(semantic)
        if name == "residual-mlp":
            expected_numpy = (
                np.maximum(arrays["x"] @ arrays["w1"], 0)
                @ arrays["w2"] + arrays["x"]
            )
            with torch.no_grad():
                expected_torch = (
                    torch.relu(
                        torch.from_numpy(arrays["x"])
                        @ torch.from_numpy(arrays["w1"])
                    )
                    @ torch.from_numpy(arrays["w2"])
                    + torch.from_numpy(arrays["x"])
                ).numpy()
        else:
            expected_numpy = (
                np.maximum(numpy_conv(arrays["x"], arrays["weight"]), 0)
                + arrays["x"]
            )
            with torch.no_grad():
                expected_torch = (
                    F.relu(F.conv2d(
                        torch.from_numpy(arrays["x"]),
                        torch.from_numpy(arrays["weight"]),
                        padding=1,
                    )) + torch.from_numpy(arrays["x"])
                ).numpy()
        observed, receipt = compiled.run_with_receipt(**arrays)
        np.testing.assert_allclose(
            observed, expected_numpy, atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            observed, expected_torch, atol=2e-5, rtol=2e-5
        )
        times = []
        for _ in range(trials):
            start = time.perf_counter_ns()
            compiled.run(**arrays)
            times.append((time.perf_counter_ns() - start) / 1e6)
        entries.append({
            "family": name,
            "op_sequence": receipt["op_sequence"],
            "max_abs_error_numpy": float(
                np.max(np.abs(observed - expected_numpy))
            ),
            "max_abs_error_torch": float(
                np.max(np.abs(observed - expected_torch))
            ),
            "compiler": receipt,
            "warm_single_run_ms": {
                "median": round(statistics.median(times), 6),
                "raw_samples": [round(value, 6) for value in times],
            },
        })
    root = Path(__file__).resolve().parents[1]
    paths = (
        "examples/native_semantic_codegen_demo_v32.py",
        "visuaml/compiler_lab/native_semantic_codegen_v32.py",
        "visuaml/compiler_lab/semantic_ir.py",
    )
    return {
        "schema": "visuaml.semantic-native-fixtures.v32",
        "seed": seed,
        "trials": trials,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "fixtures": entries,
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in paths
        },
        "limitations": [
            "Two synthetic fixture graph families, not two pretrained model families.",
            "Only closed-world pure static float32 SemanticProgram operations.",
            "Warm Python/native invocation includes ctypes and array validation.",
            "Locally compiled native code is not a sandbox or a portable AOT package.",
            "No claim of speedup over PyTorch or mature CPU inference engines.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = benchmark(trials=args.trials)
    wire = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
