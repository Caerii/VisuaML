"""Fail-closed, source-bound ONNX checkpoint inventory for native compilation.

This preflight never executes model code, never compiles a kernel, and never
calls this checkpoint a pretrained-native success. A later numerical replay
must independently compare the admitted Semantic IR against ONNX Runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = "visuaml.pretrained-onnx-preflight.v37"
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_INITIALIZER_BYTES = 8 * 1024 * 1024
MAX_GRAPH_NODES = 256
MAX_TENSOR_ELEMENTS = 4_000_000
# This inventory deliberately distinguishes possible lowerings from admitted
# SemanticProgram operations. Geometry, attribute and alias checks still apply.
NATIVE_V32_CANDIDATE_OPS = {"MatMul": "matmul", "Add": "add", "Relu": "relu", "Conv": "conv2d"}


class CheckpointPreflightError(ValueError):
    """The model is not the exact bounded, independently checkable artifact."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hex_digest(expected_sha256: str) -> None:
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise CheckpointPreflightError("Expected canonical lowercase SHA-256")


def _shape(value) -> list[int | str]:
    typ = value.type
    if not typ.HasField("tensor_type"):
        raise CheckpointPreflightError("Non-tensor graph input/output")
    if typ.tensor_type.elem_type <= 0:
        raise CheckpointPreflightError("Missing tensor dtype")
    dims: list[int | str] = []
    for dimension in typ.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            if dimension.dim_value < 0:
                raise CheckpointPreflightError("Negative tensor dimension")
            dims.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dims.append(str(dimension.dim_param))
        else:
            raise CheckpointPreflightError("Unknown tensor dimension")
    if len(dims) > 8:
        raise CheckpointPreflightError("Excessive tensor rank")
    return dims


def _attrs(node, onnx) -> dict:
    ledger: dict[str, object] = {}
    for attr in node.attribute:
        if attr.name in ledger:
            raise CheckpointPreflightError("Duplicate ONNX attribute")
        # Never recursively load graph attributes, reference functions or
        # sparse/external tensors into a claimed pure static graph.
        if attr.type not in (
            onnx.AttributeProto.FLOAT, onnx.AttributeProto.INT,
            onnx.AttributeProto.STRING, onnx.AttributeProto.FLOATS,
            onnx.AttributeProto.INTS, onnx.AttributeProto.STRINGS,
        ):
            raise CheckpointPreflightError("Unsupported complex ONNX attribute")
        val = onnx.helper.get_attribute_value(attr)
        if isinstance(val, bytes):
            val = {"utf8_or_binary_sha256": _digest(val), "size": len(val)}
        elif isinstance(val, (tuple, list)):
            val = [
                {"utf8_or_binary_sha256": _digest(x), "size": len(x)}
                if isinstance(x, bytes) else x for x in val
            ]
        if isinstance(val, float) and not math.isfinite(val):
            raise CheckpointPreflightError("Nonfinite ONNX attribute")
        if isinstance(val, list) and any(isinstance(x, float) and not math.isfinite(x) for x in val):
            raise CheckpointPreflightError("Nonfinite ONNX attribute list")
        ledger[attr.name] = val
    return dict(sorted(ledger.items()))


def audit_bytes(data: bytes, *, expected_sha256: str, source_revision: str) -> dict:
    """Produce an inspectable graph inventory bound to *literal* checkpoint bytes.

    Checks precede ONNX parsing. Only passive tensor/attribute decoding occurs.
    A successful record is deliberately marked `preflight_only`.
    """
    _require_hex_digest(expected_sha256)
    if not isinstance(source_revision, str) or len(source_revision) != 40 or any(
        ch not in "0123456789abcdef" for ch in source_revision
    ):
        raise CheckpointPreflightError("Expected immutable 40-hex source revision")
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_MODEL_BYTES:
        raise CheckpointPreflightError("Invalid or oversized ONNX checkpoint bytes")
    actual = _digest(data)
    if actual != expected_sha256:
        raise CheckpointPreflightError("Checkpoint SHA-256 mismatch before decoding")

    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
        # load_model_from_string does not chase external files or URLs.
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise CheckpointPreflightError("ONNX parser/checker rejected checkpoint") from exc
    # ONNX checker may raise ValidationError (not ValueError).
    graph = model.graph
    if not model.ir_version or not 0 < len(graph.node) <= MAX_GRAPH_NODES:
        raise CheckpointPreflightError("Unexpected ONNX IR version or graph size")
    if model.functions or model.training_info:
        raise CheckpointPreflightError("Model functions or training graphs unsupported")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise CheckpointPreflightError("Preflight expects a single input and output")
    if not model.opset_import or any(imp.domain not in ("", "ai.onnx") for imp in model.opset_import):
        raise CheckpointPreflightError("Custom ONNX operator domain unsupported")
    if any(node.domain not in ("", "ai.onnx") or len(node.output) != 1 for node in graph.node):
        raise CheckpointPreflightError("Custom-domain or multi-output node unsupported")
    if graph.sparse_initializer:
        raise CheckpointPreflightError("Sparse initializers unsupported")
    if graph.input[0].name in {x.name for x in graph.initializer}:
        raise CheckpointPreflightError("Runtime input may not alias an initializer")

    initializers = []
    total_bytes = 0
    seen_names: set[str] = set()
    for tensor in graph.initializer:
        if tensor.name in seen_names or not tensor.name:
            raise CheckpointPreflightError("Repeated or unnamed initializer")
        seen_names.add(tensor.name)
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise CheckpointPreflightError("External tensor references prohibited")
        if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
            raise CheckpointPreflightError("Unadmitted initializer dtype")
        dims = tuple(int(v) for v in tensor.dims)
        if len(dims) > 4 or any(dim <= 0 for dim in dims) or math.prod(dims) > MAX_TENSOR_ELEMENTS:
            raise CheckpointPreflightError("Initializer tensor geometry out of bounds")
        try:
            array = numpy_helper.to_array(tensor)
        except (ValueError, RuntimeError) as exc:
            raise CheckpointPreflightError("Unparseable initializer") from exc
        if (array.shape != dims or array.dtype not in (np.dtype("float32"), np.dtype("int64"))
                or (array.dtype.kind == "f" and not bool(np.isfinite(array).all()))):
            raise CheckpointPreflightError("Initializer shape/dtype/nonfinite violation")
        serialized = np.ascontiguousarray(array).tobytes(order="C")
        total_bytes += len(serialized)
        if total_bytes > MAX_INITIALIZER_BYTES:
            raise CheckpointPreflightError("Initializer storage budget exceeded")
        initializers.append({
            "name": tensor.name, "shape": list(dims), "dtype": str(array.dtype),
            "tensor_proto_sha256": _digest(tensor.SerializeToString(deterministic=True)),
            "canonical_array_sha256": _digest(serialized), "bytes": len(serialized),
        })
    if not graph.input[0].name or not graph.output[0].name:
        raise CheckpointPreflightError("Missing input/output names")
    nodes = []
    names = {graph.input[0].name, *seen_names}
    for idx, node in enumerate(graph.node):
        if not node.op_type or not node.output[0] or node.output[0] in names:
            raise CheckpointPreflightError("Invalid or repeated ONNX SSA definition")
        if not node.input or any(not name or name not in names for name in node.input):
            raise CheckpointPreflightError("Unknown or optional ONNX SSA operand")
        if node.op_type in {"If", "Loop", "Scan", "Dropout", "RandomUniform", "RandomNormal"}:
            raise CheckpointPreflightError("Stateful/control-flow ONNX operator")
        nodes.append({
            "index": idx, "op": node.op_type, "inputs": list(node.input),
            "output": node.output[0], "attrs": _attrs(node, onnx),
            "candidate_native_op": NATIVE_V32_CANDIDATE_OPS.get(node.op_type),
        })
        names.add(node.output[0])
    if graph.output[0].name not in names:
        raise CheckpointPreflightError("Graph output undefined")
    histogram = dict(sorted(Counter(n["op"] for n in nodes).items()))
    requires_extension = sorted(op for op in histogram if op not in NATIVE_V32_CANDIDATE_OPS)
    return {
        "schema": SCHEMA, "status": "preflight_only", "executed_native": False,
        "numerical_oracle_executed": False,
        "source_revision": source_revision, "checkpoint_sha256": actual,
        "checkpoint_bytes": len(data), "onnx_ir_version": model.ir_version,
        "onnx_opsets": [{"domain": x.domain, "version": x.version} for x in model.opset_import],
        "input": {"name": graph.input[0].name, "shape": _shape(graph.input[0]),
                  "dtype_id": graph.input[0].type.tensor_type.elem_type},
        "output": {"name": graph.output[0].name, "shape": _shape(graph.output[0]),
                   "dtype_id": graph.output[0].type.tensor_type.elem_type},
        "initializer_bytes": total_bytes, "initializers": initializers,
        "op_histogram": histogram, "nodes": nodes,
        "operators_requiring_extension_or_independent_admission": requires_extension,
        "limitations": (
            "Exact-byte static ONNX inventory; no compiled Semantic IR, native inference, "
            "ONNX Runtime comparison, image accuracy, or independent trained-family certification. "
            "Candidate operator names are not shape/attribute admission certificates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = audit_bytes(args.model.read_bytes(), expected_sha256=args.sha256,
                         source_revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "checkpoint_sha256": record["checkpoint_sha256"],
                      "op_histogram": record["op_histogram"],
                      "blocked": record["operators_requiring_extension_or_independent_admission"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
