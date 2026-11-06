"""Experimental torch.export ingestion for a documented ATen subset.

ExportedProgram carries guards and signature semantics beyond an FX graph.
Our NumPy-AOT artifact reifies concrete input shapes/dtypes and rejects others;
this deliberately does not claim to preserve general dynamic-shape constraints.
"""
from __future__ import annotations

import copy
import json
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from visuaml.compiler_spike import CapturedProgram, UnsupportedOperation, inspect_fx
from .analysis import validate_graph
from .aot_compile import compile_numpy_aot, AOTArtifact

_ATEN = {
    torch.ops.aten.linear.default: torch.nn.functional.linear,
    torch.ops.aten.conv2d.default: torch.nn.functional.conv2d,
    torch.ops.aten.relu.default: torch.relu,
    torch.ops.aten.sigmoid.default: torch.sigmoid,
    torch.ops.aten.tanh.default: torch.tanh,
    torch.ops.aten.mm.default: torch.matmul,
    torch.ops.aten.matmul.default: torch.matmul,
    torch.ops.aten.bmm.default: torch.matmul,
    torch.ops.aten.add.Tensor: torch.add,
    torch.ops.aten.sub.Tensor: torch.sub,
    torch.ops.aten.mul.Tensor: torch.mul,
    torch.ops.aten.div.Tensor: torch.div,
    torch.ops.aten.exp.default: torch.exp,
    torch.ops.aten.rsqrt.default: torch.rsqrt,
    torch.ops.aten.sum.dim_IntList: torch.sum,
    torch.ops.aten.amax.default: torch.amax,
}


@dataclass
class ExportCapture:
    graph: CapturedProgram
    input_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]
    output_tree: Any
    source_format: str = "torch.export / restricted ATen normalization"

    def check_inputs(self, *inputs: torch.Tensor) -> None:
        if len(inputs) != len(self.input_specs):
            raise UnsupportedOperation("Input arity differs from exported contract")
        for i, (value, (shape, dtype)) in enumerate(zip(inputs, self.input_specs)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                raise UnsupportedOperation(f"Export input {i} violates captured shape/dtype contract")

    def run_reference(self, *inputs: torch.Tensor) -> Any:
        self.check_inputs(*inputs)
        return self.graph.graph_module(*inputs)

    def compile_numpy(self) -> AOTArtifact:
        artifact = compile_numpy_aot(self.graph)
        artifact.manifest["output_tree"] = self.output_tree
        unsigned = {k: v for k, v in artifact.manifest.items() if k != "manifest_sha256"}
        payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
        artifact.manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        return artifact


def _output_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return "leaf"
    if isinstance(value, tuple):
        return {"tuple": [_output_tree(x) for x in value]}
    if isinstance(value, list):
        return {"list": [_output_tree(x) for x in value]}
    if isinstance(value, dict):
        return {"dict": {k: _output_tree(v) for k, v in sorted(value.items())}}
    raise UnsupportedOperation(f"Unsupported torch.export output leaf {type(value).__name__}")


def capture_export(model: nn.Module, *example_inputs: torch.Tensor) -> ExportCapture:
    if not example_inputs or any(not isinstance(x, torch.Tensor) for x in example_inputs):
        raise ValueError("torch.export capture requires concrete tensor inputs")
    model = copy.deepcopy(model).eval()
    try:
        exported = torch.export.export(model, tuple(example_inputs), strict=True)
    except Exception as exc:
        raise UnsupportedOperation(f"torch.export could not capture model: {exc}") from exc
    gm = exported.module()
    g = gm.graph
    for node in list(g.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            if node.users:
                raise UnsupportedOperation("Cannot strip guard that contributes to computation")
            # IMPORTANT: NumPy-AOT reintroduces *fixed* dtype+shape guards.
            # Dynamic constraints and higher-order guards are not supported.
            g.erase_node(node)
        elif node.op == "call_function":
            if node.target in _ATEN:
                if "alpha" in node.kwargs and node.kwargs["alpha"] != 1:
                    raise UnsupportedOperation("Nonunit alpha in exported add/sub is unsupported")
                node.target = _ATEN[node.target]
                if "alpha" in node.kwargs:
                    node.kwargs = {k:v for k,v in node.kwargs.items() if k != "alpha"}
            elif node.target is torch.ops.aten.convolution.default:
                # ATen: (input, weight, bias, stride, padding, dilation,
                #        transposed, output_padding, groups). Only forward 2D.
                args = node.args
                if len(args) != 9 or args[6] is not False or any(int(z) != 0 for z in args[7]):
                    raise UnsupportedOperation("Transposed or output-padded convolution is unsupported")
                node.target = torch.nn.functional.conv2d
                node.args = args[:6] + (args[8],)
            elif node.target is torch.ops.aten._softmax.default:
                # ATen signature: (input, dim, half_to_float=False)
                if len(node.args) != 3 or node.args[2] is not False:
                    raise UnsupportedOperation("Unsupported ATen softmax policy")
                node.target = torch.softmax
                node.args = (node.args[0], node.args[1])
            elif node.target is torch.ops.aten.transpose.int:
                node.op = "call_method"
                node.target = "transpose"
            elif node.target is torch.ops.aten.permute.default:
                node.op = "call_method"
                node.target = "permute"
            elif node.target is torch.ops.aten.view.default:
                node.op = "call_method"
                node.target = "reshape"
            elif node.target is torch.ops.aten.t.default:
                # ATen t is only permitted on a 2D matrix.
                meta = node.args[0].meta.get("tensor_meta")
                if not meta or len(meta.shape) != 2:
                    raise UnsupportedOperation("aten.t requires matrix input")
                node.op, node.target = "call_method", "transpose"
                node.args = (node.args[0], 0, 1)
            else:
                raise UnsupportedOperation(f"Exported operator {node.target} needs a lowering rule")
    g.lint()
    gm.recompile()
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    validate_graph(gm)
    with torch.no_grad():
        output_structure = _output_tree(gm(*example_inputs))
    return ExportCapture(CapturedProgram(gm, inspect_fx(gm)),
                         tuple((tuple(x.shape), x.dtype) for x in example_inputs), output_structure)
