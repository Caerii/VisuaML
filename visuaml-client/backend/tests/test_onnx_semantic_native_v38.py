"""v38 graph-native admission tests: no model-name checks or handwritten CNN kernel."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.native_cpu_v28 import compiler_available
from visuaml.compiler_lab.native_semantic_codegen_v32 import (
    NativeIRCodegenError,
    NativeSemanticCompiler,
)
from visuaml.compiler_lab.onnx_semantic_frontend_v38 import lower_checked_onnx
from visuaml.compiler_lab.semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
)


def _fixture():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = {
        "conv_w": np.ones((1, 1, 1, 1), dtype=np.float32) * 2,
        "dense_w": np.array([[1, -1], [2, 1], [-1, 2], [3, 0]], dtype=np.float32),
        "bias": np.array([[0.5, -0.25]], dtype=np.float32),
        "shape": np.array([1, 4], dtype=np.int64),
    }
    nodes = [
        helper.make_node("Conv", ["x", "conv_w"], ["c"],
                         kernel_shape=[1, 1], strides=[1, 1],
                         auto_pad="SAME_UPPER"),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2],
                         strides=[2, 2], auto_pad="VALID"),
        helper.make_node("Reshape", ["p", "shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "dense_w"], ["dense"]),
        helper.make_node("Add", ["dense", "bias"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "small_nonmodel_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(v, name=k) for k, v in weights.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString(), weights


def _lower(data):
    return lower_checked_onnx(data, expected_sha256=hashlib.sha256(data).hexdigest(),
                              source_revision="a" * 40)


@pytest.mark.skipif(not compiler_available(), reason="C++17 compiler unavailable")
def test_model_agnostic_onnx_graph_native_and_parameter_immutability():
    data, weights = _fixture()
    binding = _lower(data)
    assert binding.program.origin == "onnx"
    assert tuple(op.op for op in binding.program.regions[0].operations) == (
        "conv2d", "relu", "maxpool2d", "reshape", "matmul", "broadcast_add"
    )
    compiler = NativeSemanticCompiler(binding.program)
    for seed in (0, 37):
        sample = np.random.default_rng(seed).uniform(-2, 2, (1, 1, 4, 4)).astype(np.float32)
        got, receipt = compiler.run_with_receipt(**binding.validated_inputs(sample))
        x = np.maximum(sample * 2, 0)
        expected = np.zeros((1, 1, 2, 2), dtype=np.float32)
        for y in range(2):
            for z in range(2):
                expected[0, 0, y, z] = x[0, 0, y*2:y*2+2, z*2:z*2+2].max()
        expected = expected.reshape(1, 4) @ weights["dense_w"] + weights["bias"]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
        assert receipt["generated_cpp_sha256"] == hashlib.sha256(compiler.source.encode()).hexdigest()
    binding.weight_inputs["conv_w"][0, 0, 0, 0] += 1
    with pytest.raises(NativeIRCodegenError, match="mutated"):
        binding.validated_inputs(np.zeros((1, 1, 4, 4), dtype=np.float32))


def test_reject_model_source_mismatch_before_onnx_decode():
    data, _ = _fixture()
    with pytest.raises(NativeIRCodegenError, match="preflight"):
        lower_checked_onnx(data + b"x", expected_sha256=hashlib.sha256(data).hexdigest(),
                           source_revision="a" * 40)


def test_reject_unsupported_onnx_opcode_and_invalid_reshape():
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    data, _ = _fixture()
    model = onnx.load_model_from_string(data)
    model.graph.node[1].op_type = "Sigmoid"
    with pytest.raises(NativeIRCodegenError, match="Unknown ONNX graph operator"):
        _lower(model.SerializeToString())
    model = onnx.load_model_from_string(data)
    for tensor in model.graph.initializer:
        if tensor.name == "shape":
            tensor.CopyFrom(numpy_helper.from_array(np.array([1, 3], np.int64), name="shape"))
    with pytest.raises(NativeIRCodegenError, match="preflight"):
        # ONNX full shape inference rejects the impossible MatMul first.
        _lower(model.SerializeToString())


@pytest.mark.skipif(not compiler_available(), reason="C++17 compiler unavailable")
def test_native_maxpool_negative_values_and_broadcast_guard():
    ty = lambda *shape: ValueType(shape, "float32", 4 * int(np.prod(shape)))
    op = SemanticOperation(
        "pool", "tensor", "maxpool2d", ("x",), "pool", ty(1, 1, 2, 2),
        parameters=(("kernel", 2), ("stride", 1), ("pad_top", 0),
                    ("pad_left", 0), ("pad_bottom", 0), ("pad_right", 0)),
    )
    bad = SemanticProgram("onnx", (("x", ty(1, 1, 3, 3)),), (),
                          (SemanticRegion("forward", 1, (op,)),), ("pool",), "a" * 64)
    compiler = NativeSemanticCompiler(bad)
    x = -np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    result = compiler.run(x=x)
    np.testing.assert_array_equal(result, np.array([[[[-1, -2], [-4, -5]]]], dtype=np.float32))
    malformed = replace(op, parameters=op.parameters[:-1])
    with pytest.raises(NativeIRCodegenError, match="MaxPool"):
        NativeSemanticCompiler(replace(bad, regions=(SemanticRegion("forward", 1, (malformed,)),)))
