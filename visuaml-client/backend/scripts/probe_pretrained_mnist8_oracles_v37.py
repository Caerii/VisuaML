"""Numerically compare a pinned real CNN in ONNX Runtime and scalar NumPy.

Uses the original ONNX operator ledger, not a hand-coded MNIST topology. This
is an independent *framework oracle* check, not native Semantic IR codegen.
All model source bytes and input/output witnesses are recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.audit_pretrained_onnx_v37 import audit_bytes

SCHEMA = "visuaml.pretrained-cnn-onnx-vs-numpy.v37"
ATOL = 0.0001
RTOL = 0.0001


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _pairs(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected pair of operator dimensions")
    result = tuple(value)
    if any(type(x) is not int or x <= 0 or x > 8 for x in result):
        raise ValueError("Operator kernel/stride/dilation out of bounds")
    return result


def _padding(shape: tuple[int, int], kernel: tuple[int, int],
             stride: tuple[int, int], mode: str,
             pads: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if mode == "SAME_UPPER":
        sizes = [max(0, (math.ceil(size / step) - 1) * step + k - size)
                 for size, k, step in zip(shape, kernel, stride, strict=True)]
        return (sizes[0] // 2, sizes[1] // 2,
                sizes[0] - sizes[0] // 2, sizes[1] - sizes[1] // 2)
    if mode == "NOTSET" and len(pads) == 4 and all(type(x) is int and 0 <= x <= 3 for x in pads):
        return pads
    raise ValueError("Unadmitted padding contract")


def _attrs(node, onnx) -> dict:
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if len(attrs) != len(node.attribute):
        raise ValueError("Repeated node attribute")
    for name, value in attrs.items():
        if isinstance(value, bytes):
            attrs[name] = value.decode("ascii", errors="strict")
    return attrs


def conv2d_scalar(x: np.ndarray, w: np.ndarray, *,
                  kernel: tuple[int, int], stride: tuple[int, int],
                  pads: tuple[int, int, int, int]) -> np.ndarray:
    if x.ndim != 4 or w.ndim != 4 or x.shape[1] != w.shape[1] or tuple(w.shape[2:]) != kernel:
        raise ValueError("Invalid conv geometry")
    n, ci, h, width = x.shape
    co, _, kh, kw = w.shape
    pt, pl, pb, pr = pads
    oh = (h + pt + pb - kh) // stride[0] + 1
    ow = (width + pl + pr - kw) // stride[1] + 1
    if not 1 <= oh <= 128 or not 1 <= ow <= 128:
        raise ValueError("Convolution output shape out of bounds")
    result = np.zeros((n, co, oh, ow), dtype=np.float32)
    for batch in range(n):
        for out_channel in range(co):
            for oy in range(oh):
                for ox in range(ow):
                    total = np.float32(0.0)
                    for in_channel in range(ci):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * stride[0] + ky - pt
                                ix = ox * stride[1] + kx - pl
                                if 0 <= iy < h and 0 <= ix < width:
                                    total += x[batch, in_channel, iy, ix] * w[out_channel, in_channel, ky, kx]
                    result[batch, out_channel, oy, ox] = total
    return result


def maxpool2d_scalar(x: np.ndarray, *, kernel: tuple[int, int],
                     stride: tuple[int, int],
                     pads: tuple[int, int, int, int]) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError("Pool expects NCHW")
    n, c, h, width = x.shape
    pt, pl, pb, pr = pads
    oh = (h + pt + pb - kernel[0]) // stride[0] + 1
    ow = (width + pl + pr - kernel[1]) // stride[1] + 1
    if not 1 <= oh <= 128 or not 1 <= ow <= 128:
        raise ValueError("Pool result out of bounds")
    result = np.empty((n, c, oh, ow), dtype=np.float32)
    for batch in range(n):
        for channel in range(c):
            for oy in range(oh):
                for ox in range(ow):
                    maximum = np.float32(-np.inf)
                    for ky in range(kernel[0]):
                        for kx in range(kernel[1]):
                            iy = oy * stride[0] + ky - pt
                            ix = ox * stride[1] + kx - pl
                            if 0 <= iy < h and 0 <= ix < width:
                                maximum = max(maximum, x[batch, channel, iy, ix])
                    if not np.isfinite(maximum):
                        raise ValueError("Unadmitted empty maxpool window")
                    result[batch, channel, oy, ox] = maximum
    return result


def _reshape_onnx(x: np.ndarray, shape: np.ndarray) -> np.ndarray:
    if shape.ndim != 1 or shape.dtype != np.int64 or shape.size > 4:
        raise ValueError("Unadmitted reshape shape initializer")
    dims = [int(v) for v in shape]
    if sum(v == -1 for v in dims) > 1 or any(v < -1 or v > 1024 for v in dims):
        # The independent ONNX oracle allows the real 256-wide flatten;
        # native v32 admission still has its separate 128-axis limit.
        raise ValueError("Unadmitted reshape dimension")
    dims = [x.shape[i] if dim == 0 else dim for i, dim in enumerate(dims)]
    y = np.reshape(x, dims)
    if y.size != x.size or y.ndim > 4 or any(v > 1024 for v in y.shape):
        raise ValueError("Reshape exceeds oracle bounds")
    return y


def numpy_onnx_interpreter(model, sample: np.ndarray) -> np.ndarray:
    """Only the six explicitly reviewed operators; ONNX/PyTorch kernels unused."""
    import onnx
    from onnx import numpy_helper

    values = {tensor.name: np.array(numpy_helper.to_array(tensor), copy=True)
              for tensor in model.graph.initializer}
    runtime_inputs = [x.name for x in model.graph.input if x.name not in values]
    if len(runtime_inputs) != 1:
        raise ValueError("Ambiguous ONNX runtime input")
    values[runtime_inputs[0]] = sample.copy()
    for node in model.graph.node:
        a = _attrs(node, onnx)
        args = [values[name] for name in node.input]
        if node.op_type == "Conv":
            if len(args) != 2 or a.get("group", 1) != 1 or _pairs(a.get("dilations"), (1, 1)) != (1, 1):
                raise ValueError("Unadmitted grouped/dilated convolution")
            stride = _pairs(a.get("strides"), (1, 1))
            kernel = _pairs(a.get("kernel_shape"), tuple(args[1].shape[2:]))
            padding = _padding(tuple(args[0].shape[-2:]), kernel, stride,
                               a.get("auto_pad", "NOTSET"), tuple(a.get("pads", (0, 0, 0, 0))))
            result = conv2d_scalar(args[0], args[1], kernel=kernel, stride=stride, pads=padding)
        elif node.op_type == "MaxPool":
            if len(args) != 1 or a.get("ceil_mode", 0) != 0 or a.get("storage_order", 0) != 0:
                raise ValueError("Unadmitted pooling mode")
            kernel = _pairs(a.get("kernel_shape"), (0, 0))
            stride = _pairs(a.get("strides"), kernel)
            if _pairs(a.get("dilations"), (1, 1)) != (1, 1):
                raise ValueError("Dilated pooling unsupported")
            padding = _padding(tuple(args[0].shape[-2:]), kernel, stride,
                               a.get("auto_pad", "NOTSET"), tuple(a.get("pads", (0, 0, 0, 0))))
            result = maxpool2d_scalar(args[0], kernel=kernel, stride=stride, pads=padding)
        elif node.op_type == "Reshape":
            if len(args) != 2:
                raise ValueError("Reshape arity violation")
            result = _reshape_onnx(*args)
        elif node.op_type == "MatMul":
            if len(args) != 2 or args[0].ndim != 2 or args[1].ndim != 2:
                raise ValueError("MatMul rank violation")
            result = args[0] @ args[1]
        elif node.op_type == "Add":
            if len(args) != 2:
                raise ValueError("Add arity violation")
            result = args[0] + args[1]
        elif node.op_type == "Relu":
            if len(args) != 1:
                raise ValueError("Relu arity violation")
            result = np.maximum(args[0], np.float32(0))
        else:
            raise ValueError("Unsupported ONNX oracle node " + node.op_type)
        if not isinstance(result, np.ndarray) or result.dtype != np.float32 or not np.isfinite(result).all() or result.size > 65_536:
            raise ValueError("Intermediate dtype, finiteness or element bound violation")
        if len(node.output) != 1:
            raise ValueError("Multi-output node")
        values[node.output[0]] = result
    return values[model.graph.output[0].name]


def run(data: bytes, *, expected_sha256: str, source_revision: str) -> dict:
    import onnx
    import onnxruntime as ort

    preflight = audit_bytes(data, expected_sha256=expected_sha256,
                            source_revision=source_revision)
    if preflight["input"]["shape"] != [1, 1, 28, 28] or preflight["output"]["shape"] != [1, 10]:
        raise ValueError("Unexpected pinned CNN input/output")
    model = onnx.load_model_from_string(data)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Unexpected ONNX Runtime ABI")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if input_name != preflight["input"]["name"] or output_name != preflight["output"]["name"]:
        raise ValueError("ONNX Runtime input/output mismatch")
    rng = np.random.default_rng(37)
    random_data = rng.random((1, 1, 28, 28)).astype(np.float32)
    impulse = np.zeros_like(random_data)
    impulse[0, 0, 14, 14] = 1.0
    checkerboard = (np.indices((28, 28)).sum(axis=0) % 2).astype(np.float32)[None, None, :, :]
    probes = (
        ("zero", np.zeros_like(random_data)),
        ("ones", np.ones_like(random_data)),
        ("impulse", impulse),
        ("checkerboard", checkerboard),
        ("random_seed_37", random_data),
    )
    witnesses = []
    for name, sample in probes:
        expected = numpy_onnx_interpreter(model, sample)
        observed = session.run([output_name], {input_name: sample})[0]
        if expected.shape != observed.shape or expected.shape != (1, 10):
            raise ValueError("Pretrained output shape diverged")
        delta = np.abs(expected - observed)
        allowed = ATOL + RTOL * np.abs(expected)
        success = bool(np.isfinite(observed).all() and np.all(delta <= allowed))
        witnesses.append({
            "name": name, "input_sha256": sha(sample),
            "numpy_logits_sha256": sha(expected), "onnxruntime_logits_sha256": sha(observed),
            "max_abs_error": float(delta.max()), "max_allowed_error": float(allowed.max()),
            "pass": success,
        })
    return {
        "schema": SCHEMA, "checkpoint_sha256": expected_sha256,
        "source_revision": source_revision, "preflight_schema": preflight["schema"],
        "onnxruntime_version": ort.__version__, "onnx_version": onnx.__version__,
        "numpy_version": np.__version__, "oracle": "closed-world scalar NumPy reference",
        "framework": "ONNX Runtime CPU graph optimization disabled",
        "atol": ATOL, "rtol": RTOL,
        "witnesses": witnesses,
        "all_passed": all(w["pass"] for w in witnesses),
        "executed_native_semantic_cpp": False,
        "limitations": (
            "Independent pretrained ONNX vs NumPy oracle parity; not generated C++ "
            "execution, graph admission, native speedup, or all-input equivalence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.model.read_bytes(), expected_sha256=args.sha256,
                 source_revision=args.revision)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"],
                      "max_abs_error": max(w["max_abs_error"] for w in report["witnesses"])},
                     sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("At least one independent pretrained witness failed")


if __name__ == "__main__":
    main()
