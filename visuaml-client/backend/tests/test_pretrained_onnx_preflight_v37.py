"""Negative and positive evidence gates for the untrusted ONNX byte boundary."""

from __future__ import annotations

import hashlib

import pytest
from scripts.audit_pretrained_onnx_v37 import (
    CheckpointPreflightError,
    audit_bytes,
)

REVISION = "a19f9a8c2333de1df9b03f10f5739f468b699a1a"


def test_mismatched_sha_rejected_before_model_parser():
    with pytest.raises(CheckpointPreflightError, match="mismatch"):
        audit_bytes(b"not an ONNX model", expected_sha256="0" * 64, source_revision=REVISION)


def test_noncanonical_sha_and_malformed_revision_rejected():
    for digest in ("A" * 64, "a" * 63, "z" * 64):
        with pytest.raises(CheckpointPreflightError, match="canonical"):
            audit_bytes(b"model", expected_sha256=digest, source_revision=REVISION)
    with pytest.raises(CheckpointPreflightError, match="revision"):
        audit_bytes(b"model", expected_sha256=hashlib.sha256(b"model").hexdigest(),
                    source_revision="main")


def test_oversized_checkpoint_rejected_before_decoding():
    with pytest.raises(CheckpointPreflightError, match="oversized"):
        audit_bytes(b"\x00" * (16 * 1024 * 1024 + 1),
                    expected_sha256="0" * 64, source_revision=REVISION)


def _synthetic_model():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    tensor = helper.make_tensor("w", TensorProto.FLOAT, [1, 3], [0.25, -1.0, 2.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["out"], name="test_add")],
        "minimal-pure", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 8)])
    model.ir_version = 8
    return model


def _audit(model):
    data = model.SerializeToString(deterministic=True)
    return audit_bytes(data, expected_sha256=hashlib.sha256(data).hexdigest(),
                       source_revision=REVISION)


def test_static_model_inventory_is_exact_byte_bound_and_not_native_evidence():
    record = _audit(_synthetic_model())
    assert record["schema"] == "visuaml.pretrained-onnx-preflight.v37"
    assert record["status"] == "preflight_only"
    assert record["executed_native"] is False
    assert record["numerical_oracle_executed"] is False
    assert record["op_histogram"] == {"Add": 1}
    assert record["nodes"][0]["candidate_native_op"] == "add"
    assert len(record["initializers"]) == 1
    assert record["initializers"][0]["bytes"] == 12
    assert record["input"]["shape"] == [1, 3]


def test_checksum_cannot_authorize_external_initializer_reference():
    model = _synthetic_model()
    onnx = pytest.importorskip("onnx")
    tensor = model.graph.initializer[0]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../outside")
    with pytest.raises(CheckpointPreflightError):
        _audit(model)


def test_valid_graph_with_unsupported_operator_is_inventory_not_compilation():
    model = _synthetic_model()
    model.graph.node[0].op_type = "Mul"
    report = _audit(model)
    assert report["operators_requiring_extension_or_independent_admission"] == ["Mul"]
    assert report["nodes"][0]["candidate_native_op"] is None
    assert report["executed_native"] is False


def test_repeated_initializer_names_rejected_with_matching_checksum():
    model = _synthetic_model()
    model.graph.initializer.add().CopyFrom(model.graph.initializer[0])
    with pytest.raises(CheckpointPreflightError):
        _audit(model)
