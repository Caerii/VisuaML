"""v46 bounded internal-activation tracing and causal channel ablation.

This is a REAL pretrained ONNX graph: an exact SHA-checked original checkpoint.
Tracing changes only graph outputs; ablation inserts one explicitly reviewed Mul
after one existing ReLU node. The independently written scalar NumPy interpreter
uses v37's independently checked operator kernels, not ONNX Runtime kernels.

This does not infer what a neuron *means*. Channel causality on selected inputs
does not establish a reusable semantic circuit or multi-checkpoint transfer.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from scripts.audit_pretrained_onnx_v37 import audit_bytes
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    _attrs,
    _padding,
    _pairs,
    _reshape_onnx,
    conv2d_scalar,
    maxpool2d_scalar,
)

CHECKPOINT_SHA256 = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
CHECKPOINT_REVISION = "a19f9a8c2333de1df9b03f10f5739f468b699a1a"
TARGET = "ReLU114_Output_0"
CHANNELS = 16
ATOL = RTOL = 1e-4


class ActivationTraceError(ValueError):
    """Rejected, mismatched, unbounded, or unsupported internal intervention."""


@dataclass(frozen=True)
class ReviewedActivation:
    checkpoint_sha256: str
    target_output: str
    target_shape: tuple[int, ...]
    source_node_index: int
    source_node_op: str
    original_output: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _admitted(data: bytes):
    import onnx

    try:
        ledger = audit_bytes(
            data, expected_sha256=CHECKPOINT_SHA256,
            source_revision=CHECKPOINT_REVISION,
        )
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        nodes = [i for i, n in enumerate(model.graph.node) if TARGET in n.output]
        if len(nodes) != 1 or model.graph.node[nodes[0]].op_type != "Relu":
            raise ActivationTraceError("Unreviewed ONNX stage")
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
        values = [
            v for v in [*inferred.graph.value_info, *inferred.graph.output]
            if v.name == TARGET
        ]
        if len(values) != 1:
            raise ActivationTraceError("Missing unique inferred internal shape")
        tensor = values[0].type.tensor_type
        shape = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        if (tensor.elem_type != onnx.TensorProto.FLOAT
                or len(shape) != 4 or shape[0] != 1
                or shape[1] != CHANNELS or any(not 1 <= d <= 128 for d in shape)):
            raise ActivationTraceError("Internal activation ABI unreviewed")
        if ledger["input"]["shape"] != [1, 1, 28, 28] or ledger["output"]["shape"] != [1, 10]:
            raise ActivationTraceError("Unexpected pinned model ABI")
        info = ReviewedActivation(
            CHECKPOINT_SHA256, TARGET, shape, nodes[0],
            "Relu", ledger["output"]["name"],
        )
        return model, values[0], info
    except ActivationTraceError:
        raise
    except Exception as exc:
        raise ActivationTraceError("Original checkpoint failed preflight") from exc


def original_review(data: bytes) -> ReviewedActivation:
    return _admitted(data)[2]


def instrument_activation(data: bytes) -> tuple[bytes, ReviewedActivation]:
    """Append a typed graph output without editing executable node semantics."""
    import onnx

    model, inferred_value, info = _admitted(data)
    if TARGET in (x.name for x in model.graph.output):
        raise ActivationTraceError("Target already exported")
    model.graph.output.add().CopyFrom(inferred_value)
    onnx.checker.check_model(model, full_check=True)
    instrumented = model.SerializeToString(deterministic=True)
    return instrumented, info


def ablate_channel(data: bytes, channel: int | None) -> tuple[bytes, dict]:
    """Insert exact shape-broadcast mask after target ReLU; None is identity."""
    import onnx
    from onnx import numpy_helper

    model, _, info = _admitted(data)
    if channel is not None and (type(channel) is not int or not 0 <= channel < CHANNELS):
        raise ActivationTraceError("Channel index outside reviewed tensor shape")
    if (channel is None and not all(x not in data for x in (
        b"__visuaml_v46_", ))) or b"__visuaml_v46_" in data:
        raise ActivationTraceError("Reserved intervention symbol collision")
    intermediate_name = "__visuaml_v46_unmasked"
    mask_name = "__visuaml_v46_mask"
    mask = np.ones((1, CHANNELS, 1, 1), dtype=np.float32)
    if channel is not None:
        mask[0, channel, 0, 0] = 0.
    target = model.graph.node[info.source_node_index]
    if len(target.output) != 1 or target.output[0] != TARGET:
        raise ActivationTraceError("Unexpected target node outputs")
    target.output[0] = intermediate_name
    patch = onnx.helper.make_node(
        "Mul", [intermediate_name, mask_name], [TARGET],
        name="__visuaml_v46_channel_mask",
    )
    model.graph.node.insert(info.source_node_index + 1, patch)
    model.graph.initializer.add().CopyFrom(
        numpy_helper.from_array(mask, name=mask_name)
    )
    onnx.checker.check_model(model, full_check=True)
    patched = model.SerializeToString(deterministic=True)
    if _sha(patched) == _sha(data):
        raise ActivationTraceError("Intervention unexpectedly identical to source")
    return patched, {
        "schema": "visuaml.pretrained-channel-ablation.v46",
        "original_checkpoint_sha256": CHECKPOINT_SHA256,
        "altered_checkpoint_sha256": _sha(patched),
        "target_tensor": TARGET, "target_shape": list(info.target_shape),
        "channel": channel, "is_identity_control": channel is None,
        "mask_sha256": _sha(mask.tobytes()),
        "original_graph_nodes": len(model.graph.node) - 1,
        "modified_graph_nodes": len(model.graph.node),
        "effect": "broadcast float32 multiplication after one original ReLU",
        "original_model_bytes_mutated": False,
        "general_network_effects_proved": False,
    }


def _sample(sample: np.ndarray) -> None:
    if (not isinstance(sample, np.ndarray) or sample.dtype != np.float32
            or sample.shape != (1, 1, 28, 28) or not np.isfinite(sample).all()):
        raise ActivationTraceError("Runtime input violates original typed ABI")


def reference_trace(data: bytes, sample: np.ndarray,
                    channel: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Trace pre-mask target + post-mask logits with independent v37 scalar ops.

    The original graph is interpreted node-by-node. Channel ablation is a
    local semantic operation on one tensor in this interpreter; it does not
    execute the instrumented ONNX or import an ONNX Runtime operator kernel.
    """
    import onnx
    from onnx import numpy_helper

    _sample(sample)
    model, _, info = _admitted(data)
    if channel is not None and (type(channel) is not int or not 0 <= channel < CHANNELS):
        raise ActivationTraceError("Unadmitted intervention channel")
    tensors = {
        t.name: np.array(numpy_helper.to_array(t), copy=True)
        for t in model.graph.initializer
    }
    runtime = [item.name for item in model.graph.input if item.name not in tensors]
    if runtime != ["Input3"]:
        raise ActivationTraceError("Unexpected scalar reference input")
    tensors["Input3"] = sample.copy()
    observation = None
    for index, node in enumerate(model.graph.node):
        attrs = _attrs(node, onnx)
        args = [tensors[name] for name in node.input]
        if node.op_type == "Conv":
            kernel = _pairs(attrs.get("kernel_shape"), tuple(args[1].shape[2:]))
            stride = _pairs(attrs.get("strides"), (1, 1))
            if attrs.get("group", 1) != 1 or _pairs(
                attrs.get("dilations"), (1, 1)
            ) != (1, 1):
                raise ActivationTraceError("Unsupported scalar Conv contract")
            pads = _padding(
                tuple(args[0].shape[-2:]), kernel, stride,
                attrs.get("auto_pad", "NOTSET"),
                tuple(attrs.get("pads", (0, 0, 0, 0))),
            )
            result = conv2d_scalar(
                args[0], args[1], kernel=kernel, stride=stride, pads=pads,
            )
        elif node.op_type == "Add":
            result = args[0] + args[1]
        elif node.op_type == "Relu":
            result = np.maximum(args[0], np.float32(0))
        elif node.op_type == "MaxPool":
            kernel = _pairs(attrs.get("kernel_shape"), (0, 0))
            stride = _pairs(attrs.get("strides"), kernel)
            if attrs.get("ceil_mode", 0) or attrs.get("storage_order", 0):
                raise ActivationTraceError("Unsupported pooling variant")
            pads = _padding(
                tuple(args[0].shape[-2:]), kernel, stride,
                attrs.get("auto_pad", "NOTSET"),
                tuple(attrs.get("pads", (0, 0, 0, 0))),
            )
            result = maxpool2d_scalar(
                args[0], kernel=kernel, stride=stride, pads=pads,
            )
        elif node.op_type == "Reshape":
            result = _reshape_onnx(args[0], args[1])
        elif node.op_type == "MatMul":
            result = args[0] @ args[1]
        else:
            raise ActivationTraceError("Unsupported scalar operator in original graph")
        if (not isinstance(result, np.ndarray) or result.dtype != np.float32
                or not np.isfinite(result).all() or result.size > 65536
                or len(node.output) != 1):
            raise ActivationTraceError("Invalid intermediate scalar reference")
        if index == info.source_node_index:
            if node.output[0] != TARGET or tuple(result.shape) != info.target_shape:
                raise ActivationTraceError("Wrong target stage shape or identity")
            observation = result.copy()
            if channel is not None:
                result = result.copy()
                result[:, channel, :, :] = 0
        tensors[node.output[0]] = result
    if observation is None:
        raise ActivationTraceError("Original target was never interpreted")
    final = tensors[info.original_output]
    if final.shape != (1, 10):
        raise ActivationTraceError("Original logits shape drift")
    return observation, final


def numerical_parity(expected: np.ndarray, actual: np.ndarray) -> dict:
    if (not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray)
            or expected.dtype != np.float32 or actual.dtype != np.float32
            or expected.shape != actual.shape or expected.size == 0
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        raise ActivationTraceError("Independent numeric witness geometry mismatch")
    residual = np.abs(expected - actual)
    allowed = ATOL + RTOL * np.abs(expected)
    return {
        "max_abs_error": float(residual.max()),
        "passed": bool(np.all(residual <= allowed)),
        "scalar_reference_sha256": _sha(expected.tobytes()),
        "onnxruntime_sha256": _sha(actual.tobytes()),
    }
