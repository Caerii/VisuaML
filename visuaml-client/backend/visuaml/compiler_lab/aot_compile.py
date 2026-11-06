"""FX -> portable NumPy-AOT manifest and parameter arrays.

This is a *genuine* independent execution artifact for a documented operator
subset, not a general compiler for arbitrary Python, ONNX, or physical devices.
Unknown operations and unencodable arguments cause compilation failure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.fx import Node

from visuaml.compiler_spike import CapturedProgram, UnsupportedOperation, _op
from .numpy_runtime import SCHEMA, OPS, array_digest


@dataclass
class AOTArtifact:
    manifest: dict[str, Any]
    constants: dict[str, np.ndarray]

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(out / "constants.npz", **self.constants)
        return out

    def run(self, *args: Any) -> Any:
        from .numpy_runtime import execute

        return execute(self.manifest, self.constants, *args)


def _encode(value: Any) -> Any:
    if isinstance(value, Node):
        return {"ref": value.name}
    if isinstance(value, tuple):
        return {"tuple": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise UnsupportedOperation("Only string-keyed dictionaries are portable")
        return {"map": {k: _encode(v) for k, v in value.items()}}
    if isinstance(value, slice):
        return {
            "slice": [_encode(value.start), _encode(value.stop), _encode(value.step)]
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise UnsupportedOperation(
        f"Unserializable FX argument {type(value).__name__}: {value!r}"
    )


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().contiguous().numpy().copy()
    elif isinstance(value, (int, float, bool, np.ndarray)):
        result = np.asarray(value).copy()
    else:
        raise UnsupportedOperation(
            f"Cannot export FX attribute of type {type(value).__name__}"
        )
    if result.dtype.hasobject or result.dtype.kind not in "biufc":
        raise UnsupportedOperation(f"Unsupported constant dtype: {result.dtype}")
    return result


def _attr(gm: torch.fx.GraphModule, target: str) -> Any:
    obj: Any = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def compile_numpy_aot(program: CapturedProgram) -> AOTArtifact:
    gm = program.graph_module
    inputs: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    constants: dict[str, np.ndarray] = {}
    output = None
    for node in gm.graph.nodes:
        op = _op(node, gm)
        if node.op == "placeholder":
            meta = node.meta.get("tensor_meta")
            if meta is None or not hasattr(meta, "shape") or not hasattr(meta, "dtype"):
                raise UnsupportedOperation(f"Input {node.name} has no tensor metadata")
            if not all(isinstance(dim, int) for dim in meta.shape):
                raise UnsupportedOperation("NumPy-AOT requires concrete input shapes")
            inputs.append(
                {
                    "name": node.name,
                    "shape": list(meta.shape),
                    "dtype": str(torch.empty((), dtype=meta.dtype).numpy().dtype),
                }
            )
        elif node.op == "output":
            output = _encode(node.args[0])
        elif node.op == "get_attr":
            constants[node.name] = _array(_attr(gm, str(node.target)))
            nodes.append({"name": node.name, "kind": "constant", "op": "constant"})
        elif node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            if op not in OPS:
                raise UnsupportedOperation(
                    f"Unsupported module {type(module).__name__} at {node.name}"
                )
            args = list(node.args)
            encoded = _encode(args)
            if isinstance(module, nn.ReLU) and module.inplace:
                raise UnsupportedOperation(
                    "In-place ReLU has alias semantics not supported by NumPy-AOT"
                )
            if isinstance(module, nn.Softmax):
                encoded.append(module.dim)
            if isinstance(module, nn.GELU):
                encoded.append(module.approximate)
            if isinstance(module, nn.LayerNorm):
                encoded.append(_encode(tuple(module.normalized_shape)))
                for postfix, param in (
                    ("weight", module.weight),
                    ("bias", module.bias),
                ):
                    if param is None:
                        encoded.append(None)
                    else:
                        pname = f"{node.name}__{postfix}"
                        constants[pname] = _array(param)
                        nodes.append(
                            {"name": pname, "kind": "constant", "op": "constant"}
                        )
                        encoded.append({"ref": pname})
                encoded.append(module.eps)
            if isinstance(module, nn.Conv2d):
                if module.padding_mode != "zeros":
                    raise UnsupportedOperation(
                        "NumPy conv2d only supports zero padding"
                    )
                if any(not isinstance(v, int) for v in module.padding):
                    raise UnsupportedOperation("NumPy conv2d requires integer padding")
                wname = f"{node.name}__weight"
                constants[wname] = _array(module.weight)
                nodes.append({"name": wname, "kind": "constant", "op": "constant"})
                encoded.append({"ref": wname})
                if module.bias is not None:
                    bname = f"{node.name}__bias"
                    constants[bname] = _array(module.bias)
                    nodes.append({"name": bname, "kind": "constant", "op": "constant"})
                    encoded.append({"ref": bname})
                else:
                    encoded.append(None)
                encoded.extend(
                    [
                        list(module.stride),
                        list(module.padding),
                        list(module.dilation),
                        module.groups,
                    ]
                )
            if isinstance(module, nn.Linear):
                wname = f"{node.name}__weight"
                if wname in constants:
                    raise UnsupportedOperation("Duplicate constant name")
                constants[wname] = _array(module.weight)
                nodes.append({"name": wname, "kind": "constant", "op": "constant"})
                encoded.append({"ref": wname})
                if module.bias is not None:
                    bname = f"{node.name}__bias"
                    constants[bname] = _array(module.bias)
                    nodes.append({"name": bname, "kind": "constant", "op": "constant"})
                    encoded.append({"ref": bname})
            nodes.append(
                {
                    "name": node.name,
                    "kind": "call",
                    "op": op,
                    "args": encoded,
                    "kwargs": _encode(dict(node.kwargs)),
                }
            )
        elif node.op in {"call_function", "call_method"}:
            if op not in OPS:
                raise UnsupportedOperation(
                    f"No portable NumPy semantics for {node.name} ({node.op} {node.target})"
                )
            nodes.append(
                {
                    "name": node.name,
                    "kind": "call",
                    "op": op,
                    "args": _encode(node.args),
                    "kwargs": _encode(dict(node.kwargs)),
                }
            )
        else:
            raise UnsupportedOperation(f"Unhandled FX node kind: {node.op}")
    if output is None:
        raise UnsupportedOperation("Program has no output")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source": "VisuaML FX restricted operator lowering",
        "numeric_contract": "Input dtype and shapes fixed; floating point output checked by witnesses; topk ties are not portable",
        "inputs": inputs,
        "nodes": nodes,
        "output": output,
        "constant_hashes": {k: array_digest(v) for k, v in sorted(constants.items())},
    }
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    from .ir_contract import validate_portable_ir

    validate_portable_ir(manifest, constants)
    return AOTArtifact(manifest, constants)
