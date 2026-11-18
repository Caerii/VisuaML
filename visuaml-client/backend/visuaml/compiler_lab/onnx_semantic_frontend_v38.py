"""Strict passive ONNX -> shared SemanticProgram adapter for pinned static graphs.

All input bytes are checked before ONNX parsing. No model-supplied Python runs,
no pickle, custom ops, external weights or auto-fallback. Integer shape tensors
are validated compile-time constants; float weights are per-run bound inputs.
This importer does not know the identity/name of any particular architecture.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np

from .native_semantic_codegen_v32 import NativeIRCodegenError
from .semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
    validate_semantic_program,
)

MAX_ONNX_BYTES = 16 * 1024 * 1024
_ALLOWED_ATTRS = {
    "Conv": {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"},
    "MaxPool": {"auto_pad", "kernel_shape", "pads", "strides", "dilations", "ceil_mode", "storage_order"},
    "Add": set(), "Relu": set(), "Reshape": set(), "MatMul": set(),
}
_NAME = re.compile(r"[A-Za-z_][\w.\-]{0,127}\Z")


@dataclass(frozen=True)
class ONNXNativeBinding:
    program: SemanticProgram
    input_name: str
    input_shape: tuple[int, ...]
    weight_inputs: dict[str, np.ndarray]
    tensor_sha256: dict[str, str]
    checkpoint_sha256: str
    source_revision: str

    def validated_inputs(self, sample: np.ndarray) -> dict[str, np.ndarray]:
        if (not isinstance(sample, np.ndarray)
                or sample.dtype != np.float32 or sample.shape != self.input_shape
                or not np.isfinite(sample).all()):
            raise NativeIRCodegenError("Runtime ONNX input shape/dtype/finite guard")
        result = {self.input_name: np.ascontiguousarray(sample).copy()}
        for name, weight in self.weight_inputs.items():
            if not isinstance(weight, np.ndarray) or weight.dtype != np.float32:
                raise NativeIRCodegenError("Invalid bound weight")
            detached = np.ascontiguousarray(weight).copy()
            if (not np.isfinite(detached).all()
                    or hashlib.sha256(detached.tobytes()).hexdigest() != self.tensor_sha256[name]):
                raise NativeIRCodegenError("ONNX weight mutated since admission")
            result[name] = detached
        return result


def _type(shape: tuple[int, ...]) -> ValueType:
    if not 1 <= len(shape) <= 4 or any(type(d) is not int or not 1 <= d <= 512 for d in shape):
        raise NativeIRCodegenError("Unadmitted static tensor dimensions")
    n = math.prod(shape)
    if n > 65_536:
        raise NativeIRCodegenError("ONNX intermediate exceeds native tensor budget")
    return ValueType(shape, "float32", n * 4)


def _pair(value: object, default: tuple[int, int]) -> tuple[int, int]:
    result = default if value is None else tuple(value)
    if len(result) != 2 or any(type(x) is not int or not 1 <= x <= 8 for x in result):
        raise NativeIRCodegenError("Invalid or asymmetric operator axes")
    return result


def _geometry(attrs: dict, shape: tuple[int, ...], kernel: int,
              stride: int) -> tuple[int, int, int, int]:
    mode = attrs.get("auto_pad", b"NOTSET")
    if not isinstance(mode, bytes):
        raise NativeIRCodegenError("Invalid ONNX padding mode")
    if mode == b"SAME_UPPER":
        if "pads" in attrs and any(attrs["pads"]):
            raise NativeIRCodegenError("SAME_UPPER may not combine explicit pads")
        h, w = shape[-2:]
        ph = max(0, (math.ceil(h / stride) - 1) * stride + kernel - h)
        pw = max(0, (math.ceil(w / stride) - 1) * stride + kernel - w)
        pads = (ph // 2, pw // 2, ph - ph // 2, pw - pw // 2)
    elif mode in (b"NOTSET", b"VALID"):
        pads = tuple(attrs.get("pads", (0, 0, 0, 0)))
        if mode == b"VALID" and any(pads):
            raise NativeIRCodegenError("VALID with nonzero explicit padding")
    else:
        raise NativeIRCodegenError("Unsupported ONNX auto-padding scheme")
    if len(pads) != 4 or any(type(p) is not int or not 0 <= p <= 4 for p in pads):
        raise NativeIRCodegenError("ONNX padding outside native bounds")
    return pads


def _attributes(node, onnx) -> dict:
    if node.op_type not in _ALLOWED_ATTRS:
        raise NativeIRCodegenError("Unknown ONNX graph operator " + node.op_type)
    result = {}
    for attr in node.attribute:
        if attr.name in result or attr.name not in _ALLOWED_ATTRS[node.op_type]:
            raise NativeIRCodegenError("Unknown or duplicate ONNX attribute")
        if attr.type not in (onnx.AttributeProto.INT, onnx.AttributeProto.INTS,
                             onnx.AttributeProto.STRING):
            raise NativeIRCodegenError("Unsupported ONNX attribute type")
        val = onnx.helper.get_attribute_value(attr)
        if isinstance(val, list):
            val = tuple(val)
        result[attr.name] = val
    return result


def _reshape(shape: tuple[int, ...], spec: np.ndarray) -> tuple[int, ...]:
    if spec.dtype != np.int64 or spec.ndim != 1 or not 1 <= len(spec) <= 4:
        raise NativeIRCodegenError("Reshape requires bounded INT64 constant")
    target = []
    missing = -1
    for i, raw in enumerate(spec):
        dim = int(raw)
        if dim == 0:
            if i >= len(shape):
                raise NativeIRCodegenError("Invalid zero-copy reshape axis")
            dim = shape[i]
        elif dim == -1:
            if missing != -1:
                raise NativeIRCodegenError("Multiple inferred reshape dimensions")
            missing = i
        if dim < -1 or dim > 512:
            raise NativeIRCodegenError("Reshape dimension out of range")
        target.append(dim)
    total = math.prod(shape)
    known = math.prod(x for x in target if x != -1)
    if missing != -1:
        if known == 0 or total % known:
            raise NativeIRCodegenError("Invalid inferred reshape quotient")
        target[missing] = total // known
    if any(x <= 0 for x in target) or math.prod(target) != total:
        raise NativeIRCodegenError("Reshape element count mismatch")
    return tuple(target)


def _broadcast(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    n = max(len(a), len(b))
    x, y = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    result = []
    for ai, bi in zip(x, y, strict=True):
        if ai != bi and ai != 1 and bi != 1:
            raise NativeIRCodegenError("ONNX broadcast dimension mismatch")
        result.append(max(ai, bi))
    return tuple(result)


def lower_checked_onnx(data: bytes, *, expected_sha256: str,
                       source_revision: str) -> ONNXNativeBinding:
    """Lower reviewed pure ONNX graph as normal native SemanticProgram."""
    from scripts.audit_pretrained_onnx_v37 import audit_bytes
    try:
        import onnx
        from onnx import numpy_helper
        audited = audit_bytes(data, expected_sha256=expected_sha256,
                              source_revision=source_revision)
        model = onnx.load_model_from_string(data)
    except Exception as exc:
        raise NativeIRCodegenError("ONNX source preflight failed") from exc
    if len(data) > MAX_ONNX_BYTES or audited["checkpoint_sha256"] != expected_sha256:
        raise NativeIRCodegenError("ONNX source identity failure")
    if len(model.opset_import) != 1 or not 1 <= model.opset_import[0].version <= 20:
        raise NativeIRCodegenError("Unreviewed ONNX opset")
    values = {}
    weights = {}
    constants = {}
    hashes = {}
    for tensor in model.graph.initializer:
        value = np.array(numpy_helper.to_array(tensor), copy=True)
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        hashes[tensor.name] = digest
        if value.dtype == np.float32:
            weights[tensor.name] = np.ascontiguousarray(value)
            values[tensor.name] = tuple(int(x) for x in value.shape)
            _type(values[tensor.name])
        elif value.dtype == np.int64:
            constants[tensor.name] = value
        else:
            raise NativeIRCodegenError("Unsupported initializer dtype")
    runtime = [x for x in model.graph.input if x.name not in hashes]
    if len(runtime) != 1 or audited["input"]["dtype_id"] != 1:
        raise NativeIRCodegenError("Expected single float32 ONNX runtime input")
    inp = runtime[0]
    inp_shape = tuple(audited["input"]["shape"])
    _type(inp_shape)
    if inp.name in values or inp.name in constants:
        raise NativeIRCodegenError("Runtime name aliases an initializer")
    values[inp.name] = inp_shape
    inputs = [(inp.name, _type(inp_shape))]
    inputs += [(name, _type(sh)) for name, sh in values.items() if name != inp.name]
    instructions = []
    for node in model.graph.node:
        if (len(node.output) != 1 or not node.output[0] or not _NAME.fullmatch(node.output[0])
                or node.output[0] in values or node.output[0] in constants):
            raise NativeIRCodegenError("Invalid ONNX result SSA name")
        attributes = _attributes(node, onnx)
        names = tuple(node.input)
        if not names or any(not _NAME.fullmatch(name) for name in names):
            raise NativeIRCodegenError("Malformed ONNX operands")
        if node.op_type == "Reshape":
            if len(names) != 2 or names[0] not in values or names[1] not in constants:
                raise NativeIRCodegenError("Unbound static ONNX reshape")
            shape = _reshape(values[names[0]], constants[names[1]])
            opname, operands, params = "reshape", (names[0],), ()
        else:
            if any(name not in values for name in names):
                raise NativeIRCodegenError("Unknown ONNX runtime tensor operand")
            shapes = [values[name] for name in names]
            if node.op_type == "Relu":
                if len(shapes) != 1:
                    raise NativeIRCodegenError("ReLU arity")
                shape, opname, params = shapes[0], "relu", ()
            elif node.op_type == "Add":
                if len(shapes) != 2:
                    raise NativeIRCodegenError("Add arity")
                shape, opname, params = _broadcast(*shapes), "broadcast_add", ()
            elif node.op_type == "MatMul":
                if (len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2
                        or shapes[0][1] != shapes[1][0]):
                    raise NativeIRCodegenError("MatMul requires compatible rank-two inputs")
                shape, opname, params = (shapes[0][0], shapes[1][1]), "matmul", ()
            elif node.op_type in ("Conv", "MaxPool"):
                is_conv = node.op_type == "Conv"
                if (len(shapes) != (2 if is_conv else 1) or len(shapes[0]) != 4):
                    raise NativeIRCodegenError("NCHW spatial operator arity")
                x = shapes[0]
                if is_conv:
                    if len(shapes[1]) != 4 or x[1] != shapes[1][1]:
                        raise NativeIRCodegenError("Convolution input/channel geometry")
                    kernel_shape = shapes[1][2:]
                    if attributes.get("group", 1) != 1:
                        raise NativeIRCodegenError("Grouped convolution unadmitted")
                else:
                    kernel_shape = attributes.get("kernel_shape")
                if (kernel_shape is None or len(kernel_shape) != 2
                        or kernel_shape[0] != kernel_shape[1]
                        or not 1 <= kernel_shape[0] <= 8
                        or ("kernel_shape" in attributes
                            and tuple(attributes["kernel_shape"]) != tuple(kernel_shape))):
                    raise NativeIRCodegenError("Only bounded square kernels accepted")
                stride = _pair(attributes.get("strides"), (1, 1) if is_conv else kernel_shape)
                dilation = _pair(attributes.get("dilations"), (1, 1))
                if (stride[0] != stride[1] or dilation != (1, 1)
                        or attributes.get("ceil_mode", 0) != 0
                        or attributes.get("storage_order", 0) != 0):
                    raise NativeIRCodegenError("Unsupported spatial operator variant")
                pt, pl, pb, pr = _geometry(attributes, x, kernel_shape[0], stride[0])
                h = (x[2] + pt + pb - kernel_shape[0]) // stride[0] + 1
                w = (x[3] + pl + pr - kernel_shape[1]) // stride[1] + 1
                if h <= 0 or w <= 0:
                    raise NativeIRCodegenError("Invalid spatial output dimensions")
                shape = (x[0], shapes[1][0] if is_conv else x[1], h, w)
                opname = "conv2d" if is_conv else "maxpool2d"
                params = (("stride", stride[0]), ("pad_top", pt), ("pad_left", pl),
                          ("pad_bottom", pb), ("pad_right", pr))
                if not is_conv:
                    params = (("kernel", kernel_shape[0]), *params)
            else:
                raise NativeIRCodegenError("Unsupported graph opcode")
            operands = names
        _type(shape)
        op_name = node.output[0]
        instructions.append(SemanticOperation(
            op_name, "tensor", opname, operands, op_name, _type(shape),
            parameters=tuple(params),
        ))
        values[op_name] = shape
    output = model.graph.output[0]
    if (output.name not in values or tuple(audited["output"]["shape"]) != values[output.name]
            or audited["output"]["dtype_id"] != 1):
        raise NativeIRCodegenError("ONNX graph output mismatch")
    proof = {
        "checkpoint_sha256": expected_sha256, "revision": source_revision,
        "weight_sha256": hashes, "input_name": inp.name,
        "graph": [{"op": x.op, "output": x.result, "inputs": x.operands,
                   "shape": x.result_type.shape, "parameters": x.parameters}
                  for x in instructions],
    }
    source_hash = hashlib.sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    program = SemanticProgram(
        "onnx", tuple(inputs), (), (SemanticRegion("forward", 1, tuple(instructions)),),
        (output.name,), source_hash,
    )
    validate_semantic_program(program)
    return ONNXNativeBinding(
        program, inp.name, inp_shape, weights, hashes,
        expected_sha256, source_revision,
    )
