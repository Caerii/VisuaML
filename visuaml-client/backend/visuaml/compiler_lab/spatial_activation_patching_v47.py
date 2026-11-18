"""v47: bounded, byte-bound spatial activation transplantation in real WASM? No: ONNX.

The exact v37/v46 pretrained ONNX checkpoint is the only source module.
An injected mask chooses between the recipient's original hidden activation
and a separately measured donor activation at the *same* typed ReLU node.
All mask entries are binary 0/1: full grafts are exact float32 selections,
not numerical interpolation, while mask=0 must be a true identity.

The independent scalar reference recomputes the downstream original ONNX
nodes after the graft, without executing ONNX Runtime's patched graph.
This tests causal activation patching, not discovery of a semantic mechanism.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    _attrs,
    _padding,
    _pairs,
    _reshape_onnx,
    maxpool2d_scalar,
)

from .onnx_activation_causality_v46 import (
    TARGET,
    ActivationTraceError,
    _admitted,
    _sample,
    reference_trace,
)

SCHEMA = "visuaml.spatial-activation-transplant.v47"
DONOR = "__visuaml_v47_donor"
MASK = "__visuaml_v47_mask"
ORIGINAL = "__visuaml_v47_original"
COMPLEMENT = "__visuaml_v47_keep"
ORIGINAL_KEPT = "__visuaml_v47_original_kept"
DONOR_SELECTED = "__visuaml_v47_donor_selected"
ONE = "__visuaml_v47_one"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_patchable_model(source: bytes) -> tuple[bytes, dict]:
    """Create a 3-input, opset-preserving ONNX graph; do not embed donor data.

    For each location the output is (1-mask)*recipient+mask*donor, where the
    exact mask is later validated as 0/1. Original graph stays immutable.
    """
    import onnx
    from onnx import numpy_helper

    model, _, info = _admitted(source)
    if any(
        name in {n for node in model.graph.node for n in (*node.input, *node.output)}
        for name in (DONOR, MASK, ORIGINAL, COMPLEMENT, ORIGINAL_KEPT, DONOR_SELECTED, ONE)
    ) or b"__visuaml_v47_" in source:
        raise ActivationTraceError("Reserved v47 patch symbol collision")
    if tuple(info.target_shape) != (1, 16, 14, 14):
        raise ActivationTraceError("v47 spatial patch only admits reviewed 16x14x14 stage")
    before = tuple((n.op_type, tuple(n.input), tuple(n.output))
                   for n in model.graph.node)
    target = model.graph.node[info.source_node_index]
    if len(target.output) != 1 or target.output[0] != TARGET:
        raise ActivationTraceError("Patch target changed")
    target.output[0] = ORIGINAL
    nodes = (
        onnx.helper.make_node("Sub", [ONE, MASK], [COMPLEMENT],
                              name="__visuaml_v47_complement"),
        onnx.helper.make_node("Mul", [ORIGINAL, COMPLEMENT], [ORIGINAL_KEPT],
                              name="__visuaml_v47_keep_original"),
        onnx.helper.make_node("Mul", [DONOR, MASK], [DONOR_SELECTED],
                              name="__visuaml_v47_choose_donor"),
        onnx.helper.make_node("Add", [ORIGINAL_KEPT, DONOR_SELECTED], [TARGET],
                              name="__visuaml_v47_merge"),
    )
    for i, node in enumerate(nodes):
        model.graph.node.insert(info.source_node_index + 1 + i, node)
    model.graph.input.add().CopyFrom(
        onnx.helper.make_tensor_value_info(
            DONOR, onnx.TensorProto.FLOAT, list(info.target_shape)
        )
    )
    model.graph.input.add().CopyFrom(
        onnx.helper.make_tensor_value_info(
            MASK, onnx.TensorProto.FLOAT, list(info.target_shape)
        )
    )
    model.graph.initializer.add().CopyFrom(
        numpy_helper.from_array(np.float32(1), name=ONE)
    )
    # The historical ONNX IR version (3) requires initializers in graph.input.
    if model.ir_version <= 3:
        model.graph.input.add().CopyFrom(
            onnx.helper.make_tensor_value_info(
                ONE, onnx.TensorProto.FLOAT, []
            )
        )
    onnx.checker.check_model(model, full_check=True)
    patched = model.SerializeToString(deterministic=True)
    if _sha(patched) == _sha(source):
        raise ActivationTraceError("Patchable ONNX source did not change")
    after = tuple((n.op_type, tuple(n.input), tuple(n.output))
                  for n in model.graph.node)
    if (len(after) != len(before) + 4
            or after[:info.source_node_index] != before[:info.source_node_index]
            or after[info.source_node_index][0] != "Relu"
            or after[info.source_node_index][1] != before[info.source_node_index][1]
            or after[info.source_node_index][2] != (ORIGINAL,)
            or after[info.source_node_index + 5:] != before[info.source_node_index + 1:]):
        raise ActivationTraceError("A non-target executable node was rewritten")
    return patched, {
        "schema": SCHEMA,
        "source_sha256": _sha(source),
        "patched_sha256": _sha(patched),
        "source_opset_versions": [
            {"domain": op.domain, "version": op.version}
            for op in model.opset_import
        ],
        "source_ir_version": model.ir_version,
        "target_name": TARGET,
        "target_shape": list(info.target_shape),
        "original_nodes": len(before),
        "modified_nodes": len(after),
        "masked_patch_exact_binary_selection": True,
        "embedded_donor_tensor": False,
        "full_model_all_input_equivalence_proved": False,
    }


def spatial_mask(channel: int, region: tuple[int, int, int, int]) -> np.ndarray:
    """One rectangle in one channel, half-open y/x indices. No soft masks."""
    if type(channel) is not int or not 0 <= channel < 16:
        raise ActivationTraceError("Unadmitted patch channel")
    if (not isinstance(region, tuple) or len(region) != 4
            or any(type(x) is not int for x in region)):
        raise ActivationTraceError("Unadmitted spatial patch geometry")
    y0, y1, x0, x1 = region
    if not (0 <= y0 < y1 <= 14 and 0 <= x0 < x1 <= 14
            and (y1-y0)*(x1-x0) <= 196):
        raise ActivationTraceError("Spatial patch is outside bounded activation plane")
    patch = np.zeros((1, 16, 14, 14), dtype=np.float32)
    patch[0, channel, y0:y1, x0:x1] = np.float32(1)
    return patch


def _admit_tensor(source, donor, mask) -> None:
    _sample(source)
    if (not isinstance(donor, np.ndarray) or not isinstance(mask, np.ndarray)
            or donor.dtype != np.float32 or mask.dtype != np.float32
            or donor.shape != (1, 16, 14, 14)
            or mask.shape != donor.shape
            or not np.isfinite(donor).all() or not np.isfinite(mask).all()
            or not np.logical_or(mask == 0, mask == 1).all()):
        raise ActivationTraceError("Required finite float32 donor and binary spatial mask")


def downstream_reference(
    source: bytes, recipient: np.ndarray, donor: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independently replay original downstream MaxPool/Reshape/MatMul/Add.

    Uses v37 pure NumPy reference operations, not the augmented ONNX graph.
    """
    import onnx
    from onnx import numpy_helper

    _admit_tensor(recipient, donor, mask)
    model, _, info = _admitted(source)
    recipient_stage, _ = reference_trace(source, recipient)
    if recipient_stage.shape != donor.shape:
        raise ActivationTraceError("Donor target ABI differs")
    # Selection uses pure float32 0/1. In exact selection mode no weights are
    # blended, and original/donor bytes must be retained at selected entries.
    graft = np.where(mask == 1, donor, recipient_stage)
    if not np.array_equal(graft[mask == 1], donor[mask == 1]):
        raise ActivationTraceError("Donor values not retained exactly")
    tensors = {
        t.name: np.array(numpy_helper.to_array(t), copy=True)
        for t in model.graph.initializer
    }
    tensors[TARGET] = graft
    # The pretrained model also contains a constant-only weight Reshape
    # upstream of the graft. Materialize that independent initializer
    # computation without replaying any unrelated upstream activations.
    constant_node = model.graph.node[0]
    if (constant_node.op_type != "Reshape"
            or tuple(constant_node.input) != (
                "Parameter193", "Parameter193_reshape1_shape"
            )
            or tuple(constant_node.output) != ("Parameter193_reshape1",)
            or any(name not in tensors for name in constant_node.input)):
        raise ActivationTraceError("Pretrained constant-folding contract changed")
    tensors[constant_node.output[0]] = _reshape_onnx(
        tensors[constant_node.input[0]], tensors[constant_node.input[1]]
    )
    for node in model.graph.node[info.source_node_index + 1:]:
        attrs = _attrs(node, onnx)
        args = [tensors[name] for name in node.input]
        if node.op_type == "MaxPool":
            if attrs.get("ceil_mode", 0) or attrs.get("storage_order", 0):
                raise ActivationTraceError("Unsupported downstream pool mode")
            kernel = _pairs(attrs.get("kernel_shape"), (0, 0))
            stride = _pairs(attrs.get("strides"), kernel)
            pads = _padding(
                tuple(args[0].shape[-2:]), kernel, stride,
                attrs.get("auto_pad", "NOTSET"),
                tuple(attrs.get("pads", (0, 0, 0, 0))),
            )
            result = maxpool2d_scalar(
                args[0], kernel=kernel, stride=stride, pads=pads
            )
        elif node.op_type == "Reshape":
            result = _reshape_onnx(args[0], args[1])
        elif node.op_type == "MatMul":
            result = args[0] @ args[1]
        elif node.op_type == "Add":
            result = args[0] + args[1]
        else:
            raise ActivationTraceError("Unreviewed downstream scalar operator")
        if (not isinstance(result, np.ndarray) or result.dtype != np.float32
                or not np.isfinite(result).all() or result.size > 65536
                or len(node.output) != 1):
            raise ActivationTraceError("Downstream typed execution failed")
        tensors[node.output[0]] = result
    final = tensors[info.original_output]
    if final.shape != (1, 10):
        raise ActivationTraceError("Unexpected downstream logits geometry")
    return graft, final


def numeric_effect(reference: np.ndarray, modified: np.ndarray) -> dict:
    if (not isinstance(reference, np.ndarray) or not isinstance(modified, np.ndarray)
            or reference.dtype != np.float32 or modified.dtype != np.float32
            or reference.shape != (1, 10) or modified.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(modified).all()):
        raise ActivationTraceError("Logit causal comparison ABI mismatch")
    delta = (modified.astype(np.float64) - reference.astype(np.float64))
    return {
        "mean_absolute_logit_delta": float(np.mean(np.abs(delta))),
        "relative_l2_logit_delta": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)
        ),
        "max_absolute_logit_delta": float(np.max(np.abs(delta))),
        "top1_changed": bool(int(np.argmax(reference)) != int(np.argmax(modified))),
        "baseline_sha256": _sha(reference.tobytes()),
        "modified_sha256": _sha(modified.tobytes()),
    }
