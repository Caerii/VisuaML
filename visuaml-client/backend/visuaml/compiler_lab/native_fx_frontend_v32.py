"""Restricted trusted PyTorch FX -> existing SemanticProgram -> v32 native ABI.

Only known nn.Linear (no bias), nn.Conv2d (no bias/groups/dilation),
nn.ReLU, and residual operator.add are admitted. This API runs symbolic_trace
and ShapeProp on caller-provided Python: ONLY use with trusted local modules,
never arbitrary uploaded/model-hub Python. It is not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from math import prod

import numpy as np
import torch
from torch import nn
from torch.fx import Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from .native_semantic_codegen_v32 import NativeIRCodegenError
from .semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
    validate_semantic_program,
)


@dataclass(frozen=True)
class TrustedFXNativeBinding:
    program: SemanticProgram
    inputs: dict[str, np.ndarray]
    source_tensor_hashes: dict[str, str]


def _type(node: Node) -> ValueType:
    metadata = node.meta.get("tensor_meta")
    if (
        metadata is None
        or metadata.dtype != torch.float32
        or not hasattr(metadata, "shape")
    ):
        raise NativeIRCodegenError("FX node lacks fixed float32 tensor metadata")
    dims = tuple(metadata.shape)
    if not 1 <= len(dims) <= 4 or any(
        type(d) is not int or not 1 <= d <= 128 for d in dims
    ) or prod(dims) > 65_536:
        raise NativeIRCodegenError("FX tensor dimensions exceed native admission")
    return ValueType(dims, "float32", prod(dims) * 4)


def _weight(value: torch.Tensor) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim > 4
        or value.numel() > 65_536
    ):
        raise NativeIRCodegenError("Unsupported trusted FX weight tensor")
    result = value.detach().contiguous().numpy().copy()
    if not np.isfinite(result).all():
        raise NativeIRCodegenError("Nonfinite FX weight tensor")
    return result


def lower_trusted_fx(
    module: nn.Module, sample: torch.Tensor, *, trusted: bool = False,
) -> TrustedFXNativeBinding:
    """Snapshot a narrowly supported trusted module into pure Semantic IR.

    The trusted flag is an explicit caller acknowledgement, not an effective
    isolation/security boundary: Python tracing executes the module's code.
    """
    if trusted is not True:
        raise NativeIRCodegenError("FX tracing requires explicit trusted=True")
    if not isinstance(module, nn.Module) or module.training:
        raise NativeIRCodegenError("Expected trusted eval-mode nn.Module")
    if (
        not isinstance(sample, torch.Tensor)
        or sample.device.type != "cpu"
        or sample.dtype != torch.float32
        or not bool(torch.isfinite(sample).all())
        or sample.numel() > 65_536
    ):
        raise NativeIRCodegenError("Expected bounded finite CPU float32 sample")
    input_tensor = sample.detach().contiguous()
    try:
        traced = symbolic_trace(module)
        with torch.no_grad():
            ShapeProp(traced).propagate(input_tensor)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise NativeIRCodegenError("Trusted FX tracing/shape propagation failed") from exc

    declared: list[tuple[str, ValueType]] = []
    runtime: dict[str, np.ndarray] = {}
    operations: list[SemanticOperation] = []
    known: set[str] = set()
    params = 0
    output: str | None = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if declared:
                raise NativeIRCodegenError("Only one tensor placeholder admitted")
            name = node.name
            declared.append((name, _type(node)))
            runtime[name] = input_tensor.numpy().copy()
            known.add(name)
            continue
        if node.op == "output":
            if (
                len(node.args) != 1
                or not isinstance(node.args[0], Node)
                or node.args[0].name not in known
            ):
                raise NativeIRCodegenError("FX output must be one known tensor")
            output = node.args[0].name
            continue
        if node.name in known:
            raise NativeIRCodegenError("Repeated FX SSA definition")
        if node.op == "call_module":
            if len(node.args) != 1 or not isinstance(node.args[0], Node) or node.kwargs:
                raise NativeIRCodegenError("Unsupported module argument form")
            input_name = node.args[0].name
            if input_name not in known:
                raise NativeIRCodegenError("Unknown FX SSA input")
            layer = traced.get_submodule(str(node.target))
            if type(layer) is nn.Linear:
                if layer.bias is not None:
                    raise NativeIRCodegenError("Biased linear requires separate IR op")
                weight = _weight(layer.weight).T.copy(order="C")
                opcode = "matmul"
                extra = ()
            elif type(layer) is nn.Conv2d:
                if (
                    layer.bias is not None
                    or layer.groups != 1
                    or tuple(layer.dilation) != (1, 1)
                    or layer.padding_mode != "zeros"
                    or layer.stride[0] != layer.stride[1]
                    or layer.padding[0] != layer.padding[1]
                ):
                    raise NativeIRCodegenError("Unsupported FX convolution variant")
                weight = _weight(layer.weight)
                opcode = "conv2d"
                extra = (
                    ("stride", layer.stride[0]),
                    ("padding", layer.padding[0]),
                )
            elif type(layer) is nn.ReLU:
                if layer.inplace:
                    raise NativeIRCodegenError("In-place ReLU aliasing unsupported")
                operations.append(SemanticOperation(
                    node.name, "tensor", "relu", (input_name,),
                    node.name, _type(node),
                ))
                known.add(node.name)
                continue
            else:
                raise NativeIRCodegenError("Unsupported FX module type")
            pname = f"__weight_{params}"
            params += 1
            if pname in known or node.name == pname:
                raise NativeIRCodegenError("Generated FX weight SSA collision")
            declared.append((
                pname,
                ValueType(tuple(weight.shape), "float32", weight.nbytes),
            ))
            runtime[pname] = weight
            known.add(pname)
            operations.append(SemanticOperation(
                node.name, "tensor", opcode, (input_name, pname),
                node.name, _type(node), parameters=extra,
            ))
            known.add(node.name)
        elif node.op == "call_function" and node.target is operator.add:
            if (
                len(node.args) != 2
                or node.kwargs
                or not all(isinstance(arg, Node) and arg.name in known
                           for arg in node.args)
            ):
                raise NativeIRCodegenError("Only pure tensor residual addition")
            operations.append(SemanticOperation(
                node.name, "tensor", "add",
                tuple(arg.name for arg in node.args),
                node.name, _type(node),
            ))
            known.add(node.name)
        else:
            raise NativeIRCodegenError("Unadmitted FX graph instruction")
    if not declared or output is None or not operations:
        raise NativeIRCodegenError("Missing FX input, computations or output")
    model_hashes = {
        name: hashlib.sha256(array.tobytes(order="C")).hexdigest()
        for name, array in runtime.items() if name != declared[0][0]
    }
    provenance = {
        "operation_ledger": [
            {
                "name": op.name, "op": op.op, "operands": op.operands,
                "shape": op.result_type.shape,
                "parameters": op.parameters,
            }
            for op in operations
        ],
        "inputs": [
            (name, ty.shape, ty.dtype) for name, ty in declared
        ],
        "weight_hashes": model_hashes,
    }
    source_hash = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    program = SemanticProgram(
        "torch-fx", tuple(declared), (),
        (SemanticRegion("forward", 1, tuple(operations)),),
        (output,), source_hash,
    )
    validate_semantic_program(program)
    return TrustedFXNativeBinding(program, runtime, model_hashes)
