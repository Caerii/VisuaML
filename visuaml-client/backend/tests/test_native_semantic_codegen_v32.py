"""v32 graph-generated C++17 tests: two independent structural families.

All tests use the SAME typed SemanticProgram compiler and ABI. The networks
are synthetic fixtures, NOT evidence of a second pretrained model family.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from math import prod

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from visuaml.compiler_lab.native_cpu_v28 import compiler_available
from visuaml.compiler_lab.native_semantic_codegen_v32 import (
    NativeIRCodegenError,
    NativeSemanticCompiler,
)
from visuaml.compiler_lab.semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
)

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Trusted local C++17 compiler unavailable"
)


def ty(*dims: int) -> ValueType:
    return ValueType(tuple(dims), "float32", 4 * prod(dims))


def operation(
    name: str, code: str, operands: tuple[str, ...],
    result_type: ValueType,
    parameters: tuple[tuple[str, int], ...] = (),
) -> SemanticOperation:
    return SemanticOperation(
        name, "tensor", code, operands, name, result_type,
        parameters=parameters,
    )


def program(
    origin: str,
    inputs: tuple[tuple[str, ValueType], ...],
    operations: tuple[SemanticOperation, ...],
    output: str,
) -> SemanticProgram:
    return SemanticProgram(
        "torch-fx", inputs, (),
        (SemanticRegion("forward", 1, operations),),
        (output,), hashlib.sha256(origin.encode()).hexdigest(),
    )


def mlp() -> SemanticProgram:
    return program(
        "v32 fixture residual MLP",
        (("x", ty(3, 4)), ("w1", ty(4, 7)), ("w2", ty(7, 4))),
        (
            operation("mm1", "matmul", ("x", "w1"), ty(3, 7)),
            operation("act", "relu", ("mm1",), ty(3, 7)),
            operation("mm2", "matmul", ("act", "w2"), ty(3, 4)),
            operation("sum", "add", ("mm2", "x"), ty(3, 4)),
        ),
        "sum",
    )


def conv() -> SemanticProgram:
    return program(
        "v32 fixture NCHW convolution",
        (("x", ty(1, 2, 5, 5)), ("weight", ty(2, 2, 3, 3))),
        (
            operation(
                "conv", "conv2d", ("x", "weight"), ty(1, 2, 5, 5),
                (("stride", 1), ("padding", 1)),
            ),
            operation("act", "relu", ("conv",), ty(1, 2, 5, 5)),
            operation("out", "add", ("act", "x"), ty(1, 2, 5, 5)),
        ),
        "out",
    )


def numpy_conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Separate, deliberately scalar convolution oracle with zero padding."""
    output = np.zeros((1, 2, 5, 5), dtype=np.float32)
    for n in range(1):
        for oc in range(2):
            for oy in range(5):
                for ox in range(5):
                    total = np.float32(0)
                    for ic in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = oy + ky - 1, ox + kx - 1
                                if 0 <= iy < 5 and 0 <= ix < 5:
                                    total += x[n, ic, iy, ix] * w[oc, ic, ky, kx]
                    output[n, oc, oy, ox] = total
    return output


@pytest.mark.parametrize("seed", (1, 37, 117))
def test_generated_mlp_matches_numpy_and_eager_torch(seed):
    rng = np.random.default_rng(seed)
    # Shape 3x4, intentionally noncontiguous; accepted via explicit copy.
    x = rng.standard_normal((4, 3)).astype(np.float32).T
    assert not x.flags.c_contiguous
    w1 = rng.standard_normal((4, 7)).astype(np.float32) * 0.2
    w2 = rng.standard_normal((7, 4)).astype(np.float32) * 0.2
    compiler = NativeSemanticCompiler(mlp())
    native, receipt = compiler.run_with_receipt(x=x, w1=w1, w2=w2)
    from_numpy = (np.maximum(x @ w1, 0) @ w2) + x
    with torch.no_grad():
        from_torch = (
            torch.relu(torch.from_numpy(x) @ torch.from_numpy(w1))
            @ torch.from_numpy(w2) + torch.from_numpy(x)
        )
    np.testing.assert_allclose(native, from_numpy, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(native, from_torch.numpy(), atol=2e-5, rtol=2e-5)
    assert receipt["schema"] == "visuaml.semantic-native-cpu.v32"
    assert receipt["op_sequence"] == ["matmul", "relu", "matmul", "add"]
    assert receipt["output_sha256"] == hashlib.sha256(native.tobytes()).hexdigest()
    assert receipt["input_sha256"]["x"] == hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()
    assert compiler.source == NativeSemanticCompiler(mlp()).source


@pytest.mark.parametrize("seed", (2, 16, 103))
def test_generated_conv_residual_matches_two_independent_oracles(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 2, 5, 5)).astype(np.float32) * 0.2
    weight = rng.standard_normal((2, 2, 3, 3)).astype(np.float32) * 0.15
    original = x.copy()
    compiler = NativeSemanticCompiler(conv())
    got, receipt = compiler.run_with_receipt(x=x, weight=weight)
    numpy_result = np.maximum(numpy_conv(x, weight), 0) + x
    with torch.no_grad():
        torch_result = (
            F.relu(F.conv2d(torch.from_numpy(x), torch.from_numpy(weight),
                           padding=1)) + torch.from_numpy(x)
        ).numpy()
    np.testing.assert_allclose(got, numpy_result, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(got, torch_result, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(x, original)
    assert receipt["op_sequence"] == ["conv2d", "relu", "add"]
    assert receipt["retained_bytes_upper_bound"] <= 8 * 1024 * 1024
    assert receipt["scalar_multiply_upper_bound"] > 0


def test_codegen_rejects_unknown_ops_and_bad_shapes_before_compile():
    base = mlp()
    ops = list(base.regions[0].operations)
    ops[0] = replace(ops[0], op="remote_eval")
    invalid = replace(base, regions=(replace(base.regions[0],
                                             operations=tuple(ops)),))
    with pytest.raises(NativeIRCodegenError):
        NativeSemanticCompiler(invalid)
    ops = list(base.regions[0].operations)
    ops[0] = replace(ops[0], result_type=ty(3, 8))
    invalid = replace(base, regions=(replace(base.regions[0],
                                             operations=tuple(ops)),))
    with pytest.raises(NativeIRCodegenError, match="matmul"):
        NativeSemanticCompiler(invalid)
    inputs = (("x", ValueType(("B", 4), "float32", None)),
              *base.input_types[1:])
    with pytest.raises(NativeIRCodegenError, match="concrete"):
        NativeSemanticCompiler(replace(base, input_types=inputs))


def test_codegen_rejects_malformed_conv_parameters_and_effects():
    base = conv()
    original = base.regions[0].operations
    for params in (
        (("stride", 0), ("padding", 1)),
        (("stride", 1), ("padding", True)),
        (("stride", 1), ("stride", 2)),
    ):
        op = replace(original[0], parameters=params)
        invalid = replace(base, regions=(replace(
            base.regions[0], operations=(op, *original[1:])
        ),))
        with pytest.raises(NativeIRCodegenError):
            NativeSemanticCompiler(invalid)
    effect = replace(original[0], effect="read", state_cell="secret")
    invalid = replace(base, regions=(replace(
        base.regions[0], operations=(effect, *original[1:])
    ),))
    with pytest.raises(NativeIRCodegenError):
        NativeSemanticCompiler(invalid)


def test_codegen_rejects_bad_inputs_and_nonfinite_native_result():
    compiler = NativeSemanticCompiler(mlp())
    args = {
        "x": np.ones((3, 4), dtype=np.float32),
        "w1": np.ones((4, 7), dtype=np.float32),
        "w2": np.ones((7, 4), dtype=np.float32),
    }
    with pytest.raises(NativeIRCodegenError, match="Exact named"):
        compiler.run(x=args["x"])
    with pytest.raises(NativeIRCodegenError, match="dtype/shape"):
        compiler.run(**{**args, "x": args["x"].astype(np.float64)})
    with pytest.raises(NativeIRCodegenError, match="dtype/shape"):
        compiler.run(**{**args, "x": np.full((3, 4), np.nan, dtype=np.float32)})
    with pytest.raises(NativeIRCodegenError, match="failed"):
        compiler.run(**{
            **args,
            "x": np.full((3, 4), 1e30, dtype=np.float32),
            "w1": np.full((4, 7), 1e30, dtype=np.float32),
        })


def test_codegen_is_bounded_and_emits_only_fixed_cpp_identifiers():
    base = mlp()
    inputs = (("edge.name-with-dashes", ty(3, 4)), *base.input_types[1:])
    ops = tuple(replace(
        op,
        operands=tuple(
            "edge.name-with-dashes" if name == "x" else name
            for name in op.operands
        ),
    ) for op in base.regions[0].operations)
    renamed = replace(base, input_types=inputs, regions=(
        replace(base.regions[0], operations=ops),
    ))
    compiler = NativeSemanticCompiler(renamed)
    assert "edge.name-with-dashes" not in compiler.source
    assert "v0" in compiler.source
    with pytest.raises(NativeIRCodegenError, match="budget"):
        NativeSemanticCompiler(replace(base, input_types=(
            ("x", ty(128, 128, 128, 128)), *base.input_types[1:]
        )))
