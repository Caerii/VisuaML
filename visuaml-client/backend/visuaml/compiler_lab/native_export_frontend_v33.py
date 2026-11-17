"""v33: guarded static torch.export -> existing SemanticProgram -> native C++17.

This consumes an already-exported program and never calls its Python forward.
Only fixed float32, functional ATen linear/mm/conv2d/relu/add are admitted.
No symbolic dimensions, lifted buffers/constants, mutations, aliases,
bias, grouped/dilated convolutions, arbitrary FX targets or user Python.
Exporting a module *upstream* can execute trusted Python: not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import prod

import numpy as np
import torch
from torch.fx import Node

from .native_semantic_codegen_v32 import NativeIRCodegenError, _admit
from .semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
    validate_semantic_program,
)


def _tensor_type(node: Node) -> ValueType:
    meta = node.meta.get("tensor_meta")
    if meta is None or meta.dtype != torch.float32:
        raise NativeIRCodegenError("Exported node requires float32 tensor metadata")
    dims = tuple(meta.shape)
    if (
        not 1 <= len(dims) <= 4
        or any(type(d) is not int or not 1 <= d <= 128 for d in dims)
        or prod(dims) > 65_536
    ):
        raise NativeIRCodegenError("Exported node requires concrete bounded shape")
    return ValueType(dims, "float32", 4 * prod(dims))


def _array(tensor: torch.Tensor) -> np.ndarray:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.ndim < 1
        or tensor.ndim > 4
        or tensor.numel() > 65_536
    ):
        raise NativeIRCodegenError("Exported parameter is not admitted float32 CPU tensor")
    result = tensor.detach().contiguous().numpy().copy()
    if not np.isfinite(result).all():
        raise NativeIRCodegenError("Nonfinite exported parameter")
    return result


def _node(value: object, known: set[str]) -> str:
    if not isinstance(value, Node) or value.name not in known:
        raise NativeIRCodegenError("Unknown exported SSA operand")
    return value.name


def _int_pair(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] != value[1]
        or not minimum <= value[0] <= maximum
    ):
        raise NativeIRCodegenError(f"Unsupported exported {name} geometry")
    return value[0]


@dataclass(frozen=True)
class ExportedNativeBinding:
    program: SemanticProgram
    inputs: dict[str, np.ndarray]
    input_name: str
    parameter_sha256: dict[str, str]
    export_contract_sha256: str

    def validated_inputs(
        self, user_input: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Reject altered parameters, invalid shapes or missing input guards."""
        checked = {}
        expected_types = dict(self.program.input_types)
        for name, original in self.inputs.items():
            current = original if name != self.input_name or user_input is None else user_input
            if isinstance(current, torch.Tensor):
                if current.device.type != "cpu" or current.requires_grad:
                    raise NativeIRCodegenError("Exported runtime input must be CPU data")
                current = current.detach().numpy()
            spec = expected_types[name]
            if (
                not isinstance(current, np.ndarray)
                or current.dtype != np.float32
                or current.shape != spec.shape
                or not np.isfinite(current).all()
            ):
                raise NativeIRCodegenError("Exported fixed input guard rejected tensor")
            copied = np.ascontiguousarray(current).copy()
            fingerprint = self.parameter_sha256.get(name)
            if fingerprint and hashlib.sha256(copied.tobytes()).hexdigest() != fingerprint:
                raise NativeIRCodegenError("Exported parameter provenance mismatch")
            checked[name] = copied
        return checked


def lower_static_export(
    exported: torch.export.ExportedProgram,
    sample: torch.Tensor,
    *,
    trusted: bool = False,
) -> ExportedNativeBinding:
    """Admit an already-exported, trusted program without executing its module.

    PyTorch export outside this function executes/traces a Python module. The
    trusted acknowledgement is not a security boundary or proof of soundness.
    """
    if trusted is not True:
        raise NativeIRCodegenError("Import requires trusted=True")
    if not isinstance(exported, torch.export.ExportedProgram):
        raise NativeIRCodegenError("Expected torch.export.ExportedProgram")
    if exported.range_constraints:
        raise NativeIRCodegenError("Symbolic/dynamic exported guards unsupported")
    signature = exported.graph_signature
    if signature.buffers_to_mutate or signature.user_inputs_to_mutate:
        raise NativeIRCodegenError("Mutation effects unsupported")
    if (
        len(signature.output_specs) != 1
        or signature.output_specs[0].kind.name != "USER_OUTPUT"
    ):
        raise NativeIRCodegenError("Expected one pure exported user output")
    if (
        not isinstance(sample, torch.Tensor)
        or sample.device.type != "cpu"
        or sample.requires_grad
        or sample.dtype != torch.float32
        or not 1 <= sample.ndim <= 4
        or sample.numel() > 65_536
        or not bool(torch.isfinite(sample).all())
    ):
        raise NativeIRCodegenError("Expected bounded finite CPU float32 example")
    placeholders = [
        node for node in exported.graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    if len(placeholders) != len(signature.input_specs):
        raise NativeIRCodegenError("Export graph signature/placeholder mismatch")
    # Never trust lifted constants or untracked buffers. This first slice is
    # deliberately narrower than PyTorch's complete ExportedProgram contract.
    kinds = tuple(spec.kind.name for spec in signature.input_specs)
    if kinds.count("USER_INPUT") != 1 or set(kinds) - {"USER_INPUT", "PARAMETER"}:
        raise NativeIRCodegenError("Only one user input and parameter tensors admitted")
    tensors: dict[str, np.ndarray] = {}
    param_nodes: dict[str, np.ndarray] = {}
    known: set[str] = set()
    user_name = ""
    user_node = None
    for node, spec in zip(placeholders, signature.input_specs, strict=True):
        if node.name != spec.arg.name:
            raise NativeIRCodegenError("Export signature does not match graph")
        typ = _tensor_type(node)
        if spec.kind.name == "USER_INPUT":
            if tuple(sample.shape) != typ.shape:
                raise NativeIRCodegenError("Static export example shape changed")
            user_name = node.name
            user_node = node
            tensors[node.name] = sample.detach().contiguous().numpy().copy()
            known.add(node.name)
        else:
            if (
                not isinstance(spec.target, str)
                or spec.target not in exported.state_dict
            ):
                raise NativeIRCodegenError("Missing lifted exported parameter")
            value = _array(exported.state_dict[spec.target])
            if value.shape != typ.shape:
                raise NativeIRCodegenError("Lifted parameter geometry mismatch")
            param_nodes[node.name] = value
    if not user_name or user_node is None:
        raise NativeIRCodegenError("Missing exported user tensor")
    declarations: list[tuple[str, ValueType]] = [
        (user_name, _tensor_type(user_node))
    ]
    known = {user_name}
    ops: list[SemanticOperation] = []
    param_counter = 0
    output = None
    allowed_targets = {
        torch.ops.aten.linear.default: "linear",
        torch.ops.aten.mm.default: "matmul",
        torch.ops.aten.matmul.default: "matmul",
        torch.ops.aten.conv2d.default: "conv2d",
        torch.ops.aten.relu.default: "relu",
        torch.ops.aten.add.Tensor: "add",
    }
    for node in exported.graph_module.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            if len(node.args) != 1 or not isinstance(node.args[0], tuple):
                raise NativeIRCodegenError("Export output must be a singleton tuple")
            values = node.args[0]
            if len(values) != 1:
                raise NativeIRCodegenError("Only single exported output supported")
            output = _node(values[0], known)
            continue
        if node.op != "call_function" or node.target not in allowed_targets:
            raise NativeIRCodegenError("Unsupported functional ATen export node")
        if node.name in known:
            raise NativeIRCodegenError("Duplicate exported SSA definition")
        kind = allowed_targets[node.target]
        args = node.args
        params: tuple[tuple[str, int], ...] = ()
        if kind in ("linear", "conv2d"):
            if len(args) < 2 or not isinstance(args[1], Node):
                raise NativeIRCodegenError("Missing explicit exported weight")
            if args[1].name not in param_nodes:
                raise NativeIRCodegenError("Exported weight not sourced from parameter")
            x = _node(args[0], known)
            weight = param_nodes[args[1].name]
            if kind == "linear":
                if len(args) not in (2, 3) or (len(args) == 3 and args[2] is not None):
                    raise NativeIRCodegenError("Biased exported linear unsupported")
                if weight.ndim != 2:
                    raise NativeIRCodegenError("Exported linear weight must be rank two")
                weight = weight.T.copy(order="C")
                opcode = "matmul"
            else:
                if not 2 <= len(args) <= 7 or weight.ndim != 4:
                    raise NativeIRCodegenError("Unsupported exported convolution signature")
                bias = args[2] if len(args) >= 3 else None
                stride = args[3] if len(args) >= 4 else [1, 1]
                padding = args[4] if len(args) >= 5 else [0, 0]
                dilation = args[5] if len(args) >= 6 else [1, 1]
                groups = args[6] if len(args) >= 7 else 1
                if bias is not None or dilation != [1, 1] and dilation != (1, 1) or type(groups) is not int or groups != 1:
                    raise NativeIRCodegenError("Unsupported exported convolution variant")
                params = (
                    ("stride", _int_pair(stride, "stride", minimum=1, maximum=4)),
                    ("padding", _int_pair(padding, "padding", minimum=0, maximum=3)),
                )
                opcode = "conv2d"
            if node.kwargs:
                raise NativeIRCodegenError("Exported convolution/linear kwargs unsupported")
            pname = f"__export_weight_{param_counter}"
            param_counter += 1
            if pname in known or pname in param_nodes:
                raise NativeIRCodegenError("Export weight SSA collision")
            declarations.append((pname, ValueType(
                tuple(weight.shape), "float32", weight.nbytes,
            )))
            tensors[pname] = weight
            known.add(pname)
            operands = (x, pname)
        elif kind == "relu":
            if len(args) != 1 or node.kwargs:
                raise NativeIRCodegenError("Unsupported exported relu form")
            operands = (_node(args[0], known),)
            opcode = "relu"
        elif kind == "matmul":
            if len(args) != 2 or node.kwargs:
                raise NativeIRCodegenError("Unsupported exported matmul form")
            lhs = _node(args[0], known)
            right = args[1]
            if isinstance(right, Node) and right.name in param_nodes:
                weight = param_nodes[right.name]
                if weight.ndim != 2:
                    raise NativeIRCodegenError("Lifted matmul parameter must be rank two")
                pname = f"__export_weight_{param_counter}"
                param_counter += 1
                if pname in known or pname in param_nodes:
                    raise NativeIRCodegenError("Export matmul weight SSA collision")
                declarations.append((
                    pname, ValueType(tuple(weight.shape), "float32", weight.nbytes),
                ))
                tensors[pname] = weight
                known.add(pname)
                operands = (lhs, pname)
            else:
                operands = (lhs, _node(right, known))
            opcode = "matmul"
        else:
            if len(args) not in (2, 3):
                raise NativeIRCodegenError("Unsupported exported add form")
            alpha = args[2] if len(args) == 3 else node.kwargs.get("alpha", 1)
            if type(alpha) not in (int, float) or alpha != 1 or set(node.kwargs) - {"alpha"}:
                raise NativeIRCodegenError("Broadcast/scaled exported add unsupported")
            operands = (_node(args[0], known), _node(args[1], known))
            opcode = "add"
        ops.append(SemanticOperation(
            node.name, "tensor", opcode, operands, node.name,
            _tensor_type(node), parameters=params,
        ))
        known.add(node.name)
    if output is None or not ops:
        raise NativeIRCodegenError("No admitted exported output")
    param_hashes = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in tensors.items() if name != user_name
    }
    contract = {
        "schema": "visuaml.static-export-bound.v33",
        "torch_version": torch.__version__,
        "input_specs": [
            (spec.kind.name, spec.arg.name, spec.target)
            for spec in signature.input_specs
        ],
        "output": output,
        "declared": [
            (name, spec.shape, spec.dtype) for name, spec in declarations
        ],
        "ops": [
            (op.name, op.op, op.operands, op.result_type.shape, op.parameters)
            for op in ops
        ],
        "parameter_hashes": param_hashes,
        "static_guard": "exact_dtype_rank_dimensions",
    }
    contract_hash = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    program = SemanticProgram(
        "torch-fx", tuple(declarations), (),
        (SemanticRegion("forward", 1, tuple(ops)),),
        (output,), contract_hash,
    )
    validate_semantic_program(program)
    _admit(program)
    return ExportedNativeBinding(
        program, tensors, user_name, param_hashes, contract_hash,
    )
