"""Portable, PyTorch-free execution of a restricted VisuaML NumPy-AOT manifest.

Only declared operators are accepted. No Python code strings are executed. The
manifest is a data format, not an executable script. This module deliberately
imports ONLY the standard library and NumPy, so it can be copied to a fresh
Python process that does not have PyTorch installed.
"""
from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path
from zipfile import BadZipFile, ZipFile

try:
    from .ir_contract import DEFAULT_POLICY, ResourcePolicy, validate_portable_ir
except ImportError:  # standalone PyTorch-free deployment directory
    from ir_contract import DEFAULT_POLICY, ResourcePolicy, validate_portable_ir
from typing import Any

import numpy as np

SCHEMA = "visuaml.numpy-aot.v1"
OPS = frozenset({
    "linear", "matmul", "add", "sub", "mul", "relu", "sigmoid", "tanh",
    "topk", "zeros_like", "getitem", "scatter", "transpose", "reshape",
    "view", "flatten", "contiguous", "permute", "softmax", "exp", "sum",
    "amax", "rsqrt", "div", "gelu", "layer_norm", "argsort", "conv2d",
    "sin", "stack", "where", "ge", "index_add", "avg_pool2d", "interpolate_nearest", "topk_attr",
    "cat", "repeat_interleave",
})


class BundleError(ValueError):
    """Invalid or unsupported portable program; fail closed."""


def array_digest(value: np.ndarray) -> str:
    """Hash canonical C-order content plus dtype and shape (not ZIP metadata)."""
    v = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([v.dtype.str, list(v.shape)], separators=(",", ":")).encode())
    h.update(v.tobytes())
    return h.hexdigest()


def _verify_integrity(manifest: dict[str, Any], constants: dict[str, np.ndarray]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise BundleError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    expected = manifest.get("manifest_sha256")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if not isinstance(expected, str) or digest != expected:
        raise BundleError("Manifest integrity digest mismatch")
    hashes = manifest.get("constant_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(constants):
        raise BundleError("Manifest constant table mismatch")
    for name, value in constants.items():
        if value.dtype.hasobject or array_digest(value) != hashes[name]:
            raise BundleError(f"Constant failed digest check: {name}")


def load_bundle(directory: str | Path, *, policy: ResourcePolicy = DEFAULT_POLICY
                ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(directory)
    manifest_file = root / "manifest.json"
    if manifest_file.stat().st_size > policy.max_manifest_bytes:
        raise BundleError("Manifest byte budget exceeded")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    archive_path = root / "constants.npz"
    try:
        with ZipFile(archive_path) as zf:
            members = zf.infolist()
            if len(members) > policy.max_archive_members:
                raise BundleError("Archive member count limit exceeded")
            if sum(v.file_size for v in members) > policy.max_constant_bytes + 1024 * len(members):
                raise BundleError("Archive uncompressed-size budget exceeded")
            if any(v.flag_bits & 1 for v in members):
                raise BundleError("Encrypted NPZ members are not supported")
    except BadZipFile as exc:
        raise BundleError("Malformed constant archive") from exc
    with np.load(archive_path, allow_pickle=False) as archive:
        constants = {k: archive[k] for k in archive.files}
    _verify_integrity(manifest, constants)
    validate_portable_ir(manifest, constants, policy=policy)
    return manifest, constants


def _decode(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_decode(v, env) for v in value]
    if not isinstance(value, dict):
        raise BundleError(f"Unrecognized expression: {type(value).__name__}")
    if len(value) != 1:
        raise BundleError("Malformed expression tag")
    tag, v = next(iter(value.items()))
    if tag == "ref":
        if v not in env:
            raise BundleError(f"Unknown node reference {v!r}")
        return env[v]
    if tag == "map":
        if not isinstance(v, dict):
            raise BundleError("Invalid map")
        return {k: _decode(x, env) for k, x in v.items()}
    if tag == "tuple":
        return tuple(_decode(x, env) for x in v)
    if tag == "slice":
        return slice(*(_decode(x, env) for x in v))
    raise BundleError(f"Unsupported expression tag {tag}")


def _execute(op: str, args: Any, kwargs: dict[str, Any]) -> Any:
    if op == "cat":
        if len(args) != 1 or not isinstance(args[0], (tuple, list)) or not 1 <= len(args[0]) <= 32:
            raise BundleError("cat requires a bounded sequence of tensors")
        tensors = args[0]
        if any(not isinstance(x, np.ndarray) for x in tensors):
            raise BundleError("cat accepts NumPy tensors only")
        dim = kwargs.get("dim", 0)
        if type(dim) is not int or not -tensors[0].ndim <= dim < tensors[0].ndim:
            raise BundleError("cat axis out of range")
        if sum(t.nbytes for t in tensors) > 256 * 1024 * 1024:
            raise BundleError("cat exceeds output byte admission")
        try:
            return np.concatenate(tensors, axis=dim)
        except ValueError as exc:
            raise BundleError("cat tensor shapes disagree") from exc
    if op == "repeat_interleave":
        if len(args) not in (2, 3):
            raise BundleError("repeat_interleave requires tensor, integer repeats, axis")
        x, repeats = args[:2]
        dim = kwargs.get("dim", args[2] if len(args) == 3 else None)
        if (not isinstance(x, np.ndarray) or type(repeats) is not int
            or not 1 <= repeats <= 32 or type(dim) is not int
            or not -x.ndim <= dim < x.ndim
            or x.nbytes * repeats > 256 * 1024 * 1024):
            raise BundleError("repeat_interleave requires bounded scalar repeat factor and explicit axis")
        return np.repeat(x, repeats, axis=dim)
    if op == "linear":
        x, w, *other = args
        bias = other[0] if other else kwargs.get("bias")
        result = np.matmul(x, np.swapaxes(w, -1, -2))
        return result if bias is None else result + bias
    if op == "matmul":
        return np.matmul(*args)
    if op == "conv2d":
        if len(args) < 2:
            raise BundleError("conv2d expects data and weights")
        x, weight = args[:2]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
        padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
        dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
        groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
        def pair(v):
            if isinstance(v, int):
                return (v, v)
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(z, int) for z in v):
                return tuple(v)
            raise BundleError("conv2d parameters must be integer pairs")
        stride, padding, dilation = pair(stride), pair(padding), pair(dilation)
        if not isinstance(x, np.ndarray) or not isinstance(weight, np.ndarray) or x.ndim != 4 or weight.ndim != 4:
            raise BundleError("conv2d requires NCHW input and OIHW weights")
        if not isinstance(groups, int) or groups <= 0 or any(s <= 0 for s in stride + dilation) or any(p < 0 for p in padding):
            raise BundleError("Invalid conv2d stride, padding, dilation or groups")
        batch, cin, h, w = x.shape
        cout, cgroup, kh, kw = weight.shape
        if cin != cgroup * groups or cout % groups != 0:
            raise BundleError("conv2d channel/group mismatch")
        oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        if oh <= 0 or ow <= 0:
            raise BundleError("conv2d invalid output spatial dimensions")
        if bias is not None and (np.shape(bias) != (cout,)):
            raise BundleError("conv2d bias shape mismatch")
        padded = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
        # GEMM lowering: extract a read-only window view, materialize at most
        # 64 MiB of column data per group, and use optimized BLAS for the dot.
        # Larger / non-float32 tensors retain the original bounded reference
        # kernel to avoid unbounded im2col scratch allocations or dtype drift.
        per_group_elements = batch * oh * ow * cgroup * kh * kw
        if (
            x.dtype == weight.dtype == np.dtype('float32')
            and per_group_elements * x.dtype.itemsize <= 64 * 1024 * 1024
            and batch * cout * oh * ow * x.dtype.itemsize <= 256 * 1024 * 1024
        ):
            from numpy.lib.stride_tricks import sliding_window_view

            effective_h = dilation[0] * (kh - 1) + 1
            effective_w = dilation[1] * (kw - 1) + 1
            window = sliding_window_view(padded, (effective_h, effective_w), axis=(-2, -1))
            window = window[:, :, ::stride[0], ::stride[1], ::dilation[0], ::dilation[1]]
            if window.shape[2:] != (oh, ow, kh, kw):
                raise BundleError('Convolution window dimensions differ from output contract')
            ocg = cout // groups
            result = np.empty((batch, cout, oh, ow), dtype=np.float32)
            for group in range(groups):
                patches = window[:, group * cgroup:(group + 1) * cgroup]
                columns = patches.transpose(0, 2, 3, 1, 4, 5).reshape(
                    batch * oh * ow, cgroup * kh * kw
                )
                filters = weight[group * ocg:(group + 1) * ocg].reshape(ocg, -1)
                activation = columns @ filters.T
                result[:, group * ocg:(group + 1) * ocg] = activation.reshape(
                    batch, oh, ow, ocg
                ).transpose(0, 3, 1, 2)
            if bias is not None:
                result += bias[None, :, None, None]
            return result
        result = np.zeros((batch, cout, oh, ow), dtype=np.result_type(x, weight))
        ocg = cout // groups
        for group in range(groups):
            xb = padded[:, group * cgroup:(group + 1) * cgroup]
            wb = weight[group * ocg:(group + 1) * ocg]
            out = result[:, group * ocg:(group + 1) * ocg]
            for i in range(kh):
                for j in range(kw):
                    patch = xb[:, :, i*dilation[0]:i*dilation[0]+oh*stride[0]:stride[0],
                               j*dilation[1]:j*dilation[1]+ow*stride[1]:stride[1]]
                    out += np.einsum('bchw,oc->bohw', patch, wb[:, :, i, j])
        if bias is not None:
            result += bias[None, :, None, None]
        return result
    if op == "add":
        return np.add(*args, **kwargs)
    if op == "sub":
        return np.subtract(*args, **kwargs)
    if op == "mul":
        return np.multiply(*args, **kwargs)
    if op == "div":
        return np.divide(*args, **kwargs)
    if op == "exp":
        return np.exp(args[0])
    if op == "rsqrt":
        return 1 / np.sqrt(args[0])
    if op in {"sum", "amax"}:
        axis = kwargs.get("dim", args[1] if len(args) > 1 else None)
        keepdims = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
        return (np.sum if op == "sum" else np.max)(args[0], axis=axis, keepdims=keepdims)
    if op == "softmax":
        axis = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        shifted = args[0] - np.max(args[0], axis=axis, keepdims=True)
        z = np.exp(shifted)
        return z / np.sum(z, axis=axis, keepdims=True)
    if op == "gelu":
        import math
        x = args[0]
        mode = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        if mode == "tanh":
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        if mode == "none":
            return 0.5 * x * (1 + np.vectorize(math.erf, otypes=[float])(x / np.sqrt(2)).astype(x.dtype))
        raise BundleError(f"Unsupported GELU approximation {mode}")
    if op == "layer_norm":
        x = args[0]
        normalized_shape = args[1]
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        bias = args[3] if len(args) > 3 else kwargs.get("bias")
        eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
        count = len(normalized_shape)
        if count == 0 or tuple(x.shape[-count:]) != tuple(normalized_shape):
            raise BundleError("LayerNorm normalized shape does not match input")
        axes = tuple(range(-count, 0))
        mu = np.mean(x, axis=axes, keepdims=True)
        centered = x - mu
        var = np.mean(centered * centered, axis=axes, keepdims=True)
        z = centered / np.sqrt(var + eps)
        if weight is not None:
            z = z * weight
        if bias is not None:
            z = z + bias
        return z
    if op == "topk_attr":
        if (len(args) != 2 or not isinstance(args[0], (tuple, list))
                or len(args[0]) != 2 or args[1] not in ("indices", "values")):
            raise BundleError("Only .indices/.values on a topk result is portable")
        return args[0][1 if args[1] == "indices" else 0]
    if op == "sin":
        return np.sin(args[0])
    if op == "ge":
        return np.greater_equal(args[0], args[1])
    if op == "where":
        if len(args) != 3:
            raise BundleError("where requires condition, left and right operands")
        return np.where(args[0], args[1], args[2])
    if op == "stack":
        values = args[0]
        axis = kwargs.get("dim", args[1] if len(args) > 1 else 0)
        if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 2048:
            raise BundleError("stack requires a bounded list of tensors")
        if sum(v.nbytes for v in values if isinstance(v, np.ndarray)) > 256 * 1024 * 1024:
            raise BundleError("stack exceeds activation allocation budget")
        return np.stack(values, axis=int(axis))
    if op == "index_add":
        # Tensor.index_add is FUNCTIONAL; never mutate the source tensor.
        if len(args) not in (4, 5):
            raise BundleError("index_add expects tensor, dim, indices, source, optional alpha")
        x, axis, indices, source = args[:4]
        alpha = args[4] if len(args) == 5 else kwargs.get("alpha", 1)
        if int(axis) != 0 or not isinstance(indices, np.ndarray) or indices.ndim != 1:
            raise BundleError("index_add currently admits axis=0 and 1D index tensors")
        if indices.dtype.kind not in "iu" or source.ndim != x.ndim or source.shape[0] != indices.size or source.shape[1:] != x.shape[1:]:
            raise BundleError("Invalid index_add index/source shape or dtype")
        if np.any(indices < 0) or np.any(indices >= x.shape[0]):
            raise BundleError("index_add indices out of bounds")
        if type(alpha) not in (float, int) or not np.isfinite(alpha):
            raise BundleError("Invalid index_add alpha")
        result = x.copy()
        np.add.at(result, indices.astype(np.int64, copy=False), source * alpha)
        return result
    if op == "avg_pool2d":
        # Deliberately narrow until independently validated generalized pooling.
        x = args[0]
        kernel = kwargs.get("kernel_size", args[1] if len(args) > 1 else None)
        stride = kwargs.get("stride", args[2] if len(args) > 2 else None)
        padding = kwargs.get("padding", args[3] if len(args) > 3 else 0)
        ceil = kwargs.get("ceil_mode", args[4] if len(args) > 4 else False)
        include = kwargs.get("count_include_pad", args[5] if len(args) > 5 else True)
        if kernel not in (2, (2, 2)) or stride not in (None, 2, (2, 2)) or padding not in (0, (0, 0)) or ceil is not False or include is not True:
            raise BundleError("avg_pool2d supports only 2x2, stride 2, no padding")
        if not isinstance(x, np.ndarray) or x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
            raise BundleError("avg_pool2d requires even spatial dimensions in NCHW layout")
        n, c, h, w = x.shape
        return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
    if op == "interpolate_nearest":
        x = args[0]
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        scale = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)
        mode = kwargs.get("mode", args[3] if len(args) > 3 else "nearest")
        align = kwargs.get("align_corners", args[4] if len(args) > 4 else None)
        antialias = kwargs.get("antialias", args[6] if len(args) > 6 else False)
        recompute = kwargs.get("recompute_scale_factor", args[5] if len(args) > 5 else None)
        if (not isinstance(x, np.ndarray) or x.ndim != 4 or size is not None
                or scale not in (2, 2.0, (2, 2), (2.0, 2.0)) or mode != "nearest"
                or align is not None or antialias is not False or recompute is not None):
            raise BundleError("interpolate supports NCHW nearest upsampling with scale factor 2 only")
        if x.nbytes > 64 * 1024 * 1024:
            raise BundleError("interpolate exceeds bounded output allocation")
        return np.repeat(np.repeat(x, 2, axis=-2), 2, axis=-1)
    if op == "relu":
        return np.maximum(args[0], 0)
    if op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-args[0]))
    if op == "tanh":
        return np.tanh(args[0])
    if op == "zeros_like":
        return np.zeros_like(args[0])
    if op == "getitem":
        return args[0][args[1]]
    if op == "transpose":
        return np.swapaxes(args[0], int(args[1]), int(args[2]))
    if op == "permute":
        axes = args[1] if len(args) == 2 and isinstance(args[1], (tuple, list)) else args[1:]
        return np.transpose(args[0], axes)
    if op in {"view", "reshape"}:
        shape = args[1:] if len(args) != 2 or not isinstance(args[1], (tuple, list)) else args[1]
        return np.reshape(args[0], shape)
    if op == "flatten":
        x = args[0]
        start = int(args[1]) if len(args) > 1 else 0
        end = int(args[2]) if len(args) > 2 else -1
        start %= x.ndim
        end %= x.ndim
        if start > end:
            raise BundleError("Invalid flatten dimensions")
        return x.reshape(x.shape[:start] + (prod(x.shape[start:end + 1]),) + x.shape[end + 1:])
    if op == "contiguous":
        return np.ascontiguousarray(args[0])
    if op == "scatter":
        x, dim, indices, value = args
        result = x.copy()
        replacement = value if isinstance(value, np.ndarray) else np.full(indices.shape, value)
        np.put_along_axis(result, indices.astype(np.int64), replacement, axis=int(dim))
        return result
    if op == "argsort":
        x = args[0]
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        descending = kwargs.get("descending", args[2] if len(args) > 2 else False)
        stable = kwargs.get("stable", args[3] if len(args) > 3 else False)
        if stable is not True:
            raise BundleError("Portable argsort requires stable=True")
        return np.argsort(-x if descending else x, axis=dim, kind="stable")
    if op == "topk":
        x, k, *rest = args
        dim = int(kwargs.get("dim", rest[0] if rest else -1))
        largest = kwargs.get("largest", rest[1] if len(rest) > 1 else True)
        sorted_ = kwargs.get("sorted", rest[2] if len(rest) > 2 else True)
        if largest is not True or sorted_ is not True:
            raise BundleError("topk currently supports largest=True, sorted=True only")
        if k < 0 or k > x.shape[dim]:
            raise BundleError("topk k outside axis bounds")
        # Torch does not guarantee index order for equal values. Reject ties
        # rather than pretending that NumPy's stable sort is equivalent.
        sorted_x = np.sort(x, axis=dim)
        if np.any(np.diff(sorted_x, axis=dim) == 0):
            raise BundleError("topk contains ties; index order is not portable")
        ix = np.argsort(-x, axis=dim, kind="stable")
        ix = np.take(ix, range(k), axis=dim)
        return np.take_along_axis(x, ix, axis=dim), ix
    raise BundleError(f"No NumPy semantics registered for {op!r}")


def _restore_tree(flat_value: Any, shape: Any) -> Any:
    """Rebuild the original exported pytree from flattened FX outputs."""
    leaves: list[Any] = []
    def flatten(value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for x in value:
                flatten(x)
        elif isinstance(value, dict):
            for k in sorted(value):
                flatten(value[k])
        else:
            leaves.append(value)
    flatten(flat_value)
    it = iter(leaves)
    def rebuild(spec: Any) -> Any:
        if spec == "leaf":
            return next(it)
        if "tuple" in spec:
            return tuple(rebuild(x) for x in spec["tuple"])
        if "list" in spec:
            return [rebuild(x) for x in spec["list"]]
        if "dict" in spec:
            return {k: rebuild(x) for k, x in spec["dict"].items()}
        raise BundleError("Malformed pytree specification")
    result = rebuild(shape)
    try:
        next(it)
    except StopIteration:
        return result
    raise BundleError("Output pytree has more leaves than expected")



class PreparedNumpyExecutor:
    """Verified-once, reusable execution of an immutable NumPy AOT snapshot.

    The constructor copies the manifest and every constant, then checks the
    integrity digest, resource policy and SSA contract BEFORE admitting a run.
    This avoids rehashing and revalidating hundreds of constant nodes for each
    decoded token. Later mutations to the caller's objects cannot affect this
    snapshot; every invocation still validates all runtime input contracts.

    This is a trusted, in-process optimization, NOT a sandbox or a way to skip
    admission of an untrusted external artifact. Construct a new instance when
    a manifest, constant, or resource policy changes.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        constants: dict[str, np.ndarray],
        *,
        policy: ResourcePolicy = DEFAULT_POLICY,
    ) -> None:
        import copy

        self._manifest = copy.deepcopy(manifest)
        # Immutable *bytes-backed* arrays prevent a caller from re-enabling
        # write access and changing previously verified constant contents.
        self._constants = {}
        for key, value in constants.items():
            if not isinstance(value, np.ndarray):
                raise BundleError(f"Invalid constant array: {key}")
            stable = np.ascontiguousarray(value)
            self._constants[key] = np.frombuffer(
                stable.tobytes(), dtype=stable.dtype
            ).reshape(stable.shape)
        self._policy = policy
        _verify_integrity(self._manifest, self._constants)
        validate_portable_ir(self._manifest, self._constants, policy=policy)
        self._inputs = tuple(self._manifest["inputs"])
        self._instructions = tuple(self._manifest["nodes"])
        self._output = self._manifest["output"]
        self._output_tree = self._manifest.get("output_tree")
        self._constraints = self._manifest.get("input_constraints", {})

    def run(self, *inputs: Any) -> Any:
        """Execute a preverified, fixed-shape program with fresh invocation state."""
        if len(inputs) != len(self._inputs):
            raise BundleError(
                f"Expected {len(self._inputs)} inputs, received {len(inputs)}"
            )
        env: dict[str, Any] = {}
        input_bytes = 0
        for spec, value in zip(self._inputs, inputs):
            x = np.asarray(value)
            input_bytes += x.nbytes
            if input_bytes > self._policy.max_input_bytes:
                raise BundleError("Runtime input memory budget exceeded")
            if list(x.shape) != spec["shape"] or str(x.dtype) != spec["dtype"]:
                raise BundleError(
                    f"Input {spec['name']} requires {spec['shape']}/{spec['dtype']}, "
                    f"got {list(x.shape)}/{x.dtype}"
                )
            limits = self._constraints.get(spec["name"])
            if limits is not None and (
                np.any(x < limits["min_inclusive"])
                or np.any(x >= limits["max_exclusive"])
            ):
                raise BundleError(
                    f"Input {spec['name']} outside declared bounded integer domain"
                )
            env[spec["name"]] = x

        held_bytes = input_bytes
        for item in self._instructions:
            name = item["name"]
            if item["kind"] == "constant":
                result = self._constants[name]
            else:
                args = _decode(item["args"], env)
                kwargs = _decode(item["kwargs"], env)
                result = _execute(item["op"], args, kwargs)
            env[name] = result
            if isinstance(result, np.ndarray):
                held_bytes += result.nbytes
            if held_bytes > self._policy.max_retained_value_bytes:
                raise BundleError("Runtime retained-value byte budget exceeded")
        result = _decode(self._output, env)
        if self._output_tree is not None:
            return _restore_tree(result, self._output_tree)
        return result


def prepare_numpy_aot(
    manifest: dict[str, Any],
    constants: dict[str, np.ndarray],
    *,
    policy: ResourcePolicy = DEFAULT_POLICY,
) -> PreparedNumpyExecutor:
    """Validate and snapshot an AOT bundle once for safe repeated CPU calls."""
    return PreparedNumpyExecutor(manifest, constants, policy=policy)

def execute(manifest: dict[str, Any], constants: dict[str, np.ndarray], *inputs: Any,
            policy: ResourcePolicy = DEFAULT_POLICY) -> Any:
    """Execute only a closed list of operators and the captured fixed input specs.

    Each input must match its traced tensor dimensions and dtype exactly.
    Numeric equivalence to PyTorch is approximate under floating-point math.
    """
    _verify_integrity(manifest, constants)
    validate_portable_ir(manifest, constants, policy=policy)
    specs = manifest.get("inputs", [])
    if len(inputs) != len(specs):
        raise BundleError(f"Expected {len(specs)} inputs, received {len(inputs)}")
    env: dict[str, Any] = {}
    input_bytes = 0
    for spec, value in zip(specs, inputs):
        x = np.asarray(value)
        input_bytes += x.nbytes
        if input_bytes > policy.max_input_bytes:
            raise BundleError("Runtime input memory budget exceeded")
        if list(x.shape) != spec["shape"] or str(x.dtype) != spec["dtype"]:
            raise BundleError(f"Input {spec['name']} requires {spec['shape']}/{spec['dtype']}, got {list(x.shape)}/{x.dtype}")
        limits = manifest.get("input_constraints", {}).get(spec["name"])
        if limits is not None and (np.any(x < limits["min_inclusive"]) or np.any(x >= limits["max_exclusive"])):
            raise BundleError(f"Input {spec['name']} outside declared bounded integer domain")
        env[spec["name"]] = x
    seen = set(env)
    held_bytes = sum(value.nbytes for value in env.values() if isinstance(value, np.ndarray))
    for item in manifest.get("nodes", []):
        name, kind, op = item["name"], item["kind"], item["op"]
        if name in seen:
            raise BundleError(f"Duplicate name {name!r}")
        if kind == "constant":
            if name not in constants:
                raise BundleError(f"Missing constant {name}")
            result = constants[name]
        elif kind == "call" and op in OPS:
            args = _decode(item["args"], env)
            kwargs = _decode(item["kwargs"], env)
            result = _execute(op, args, kwargs)
        else:
            raise BundleError(f"Unsupported executable kind/operator: {kind}/{op}")
        env[name] = result
        seen.add(name)
        # Conservative post-operation memory accounting. This is a resource
        # admission check, not a hard sandbox limit on in-flight allocations.
        if isinstance(result, np.ndarray):
            held_bytes += result.nbytes
        if held_bytes > policy.max_retained_value_bytes:
            raise BundleError("Runtime retained-value byte budget exceeded")
    result = _decode(manifest["output"], env)
    if "output_tree" in manifest:
        return _restore_tree(result, manifest["output_tree"])
    return result


def run_bundle(directory: str | Path, *inputs: Any,
               policy: ResourcePolicy = DEFAULT_POLICY) -> Any:
    manifest, constants = load_bundle(directory, policy=policy)
    return execute(manifest, constants, *inputs, policy=policy)
