"""VisuaML: typed FX inspection, transformation and pluggable execution backends.

Deliberately small vertical slice. The IR describes *captured* tensor programs; it
is not a claim to compile arbitrary Python or arbitrary accelerator hardware.
Only the tested primitive subset has NumPy semantics. Unsupported operations
are reported before execution rather than silently miscompiled.

Optional dependency: torch for tracing and reference execution, numpy for CPU
lowering. No integration with the existing frontend is required for this API.
"""

from __future__ import annotations

import copy
import operator
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import nn
from torch.fx import GraphModule, Interpreter, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


class UnsupportedOperation(ValueError):
    """An operator cannot be represented or executed by the requested backend."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[str, ...]
    dtype: str
    bytes: int | None


@dataclass(frozen=True)
class OpRecord:
    id: str
    op: str
    target: str
    inputs: tuple[str, ...]
    tensors: tuple[TensorSpec, ...]
    estimated_flops: int | None
    decomposition: tuple[str, ...]


@dataclass(frozen=True)
class ProgramIR:
    nodes: tuple[OpRecord, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        by_op: dict[str, int] = {}
        for node in self.nodes:
            by_op[node.op] = by_op.get(node.op, 0) + 1
        return {
            "nodes": len(self.nodes),
            "operators": by_op,
            "estimated_flops": sum(n.estimated_flops or 0 for n in self.nodes),
            "unknown_flop_nodes": [n.id for n in self.nodes if n.estimated_flops is None],
            "tensor_output_bytes": sum(t.bytes or 0 for n in self.nodes for t in n.tensors),
            "unknown_tensor_bytes": [n.id for n in self.nodes if any(t.bytes is None for t in n.tensors)],
        }


@dataclass
class CapturedProgram:
    graph_module: GraphModule
    ir: ProgramIR


def _op(node: Node, gm: GraphModule) -> str:
    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    if node.op == "get_attr":
        return "constant"
    if node.op == "call_module":
        mod = gm.get_submodule(str(node.target))
        for cls, name in ((nn.Linear, "linear"), (nn.Conv2d, "conv2d"), (nn.ReLU, "relu"),
                          (nn.Sigmoid, "sigmoid"), (nn.Tanh, "tanh"),
                          (nn.Softmax, "softmax"), (nn.GELU, "gelu"), (nn.LayerNorm, "layer_norm")):
            if isinstance(mod, cls):
                return name
        return f"module:{type(mod).__name__}"
    if node.op == "call_method":
        name = str(node.target)
        return name if name in {"relu", "reshape", "view", "transpose", "flatten", "scatter", "sigmoid", "tanh", "contiguous", "permute", "softmax", "exp", "sum", "amax", "rsqrt", "gelu"} else f"method:{name}"
    if node.op == "call_function":
        t = node.target
        mapping = {
            torch.matmul: "matmul", operator.matmul: "matmul",
            torch.mm: "matmul", torch.bmm: "matmul",
            torch.add: "add", operator.add: "add",
            torch.sub: "sub", operator.sub: "sub",
            torch.mul: "mul", operator.mul: "mul",
            torch.relu: "relu", torch.sigmoid: "sigmoid", torch.tanh: "tanh",
            torch.topk: "topk", torch.argsort: "argsort", torch.zeros_like: "zeros_like",
            operator.getitem: "getitem",
            torch.nn.functional.linear: "linear", torch.nn.functional.conv2d: "conv2d",
            torch.nn.functional.relu: "relu",
            torch.softmax: "softmax", torch.nn.functional.softmax: "softmax",
            torch.nn.functional.gelu: "gelu", torch.nn.functional.layer_norm: "layer_norm",
            torch.exp: "exp", torch.sum: "sum", torch.amax: "amax", torch.rsqrt: "rsqrt",
            torch.div: "div", operator.truediv: "div", torch.transpose: "transpose",
            torch.permute: "permute", torch.nn.functional.scaled_dot_product_attention: "sdpa",
        }
        return mapping.get(t, f"function:{getattr(t, '__name__', str(t))}")
    return f"unknown:{node.op}"


def _tensor_spec(meta: Any) -> TensorSpec | None:
    if not hasattr(meta, "shape") or not hasattr(meta, "dtype"):
        return None
    shape = tuple(str(d) for d in meta.shape)
    dtype = meta.dtype
    concrete = all(isinstance(x, int) for x in meta.shape)
    nbytes = int(prod(meta.shape) * torch.empty((), dtype=dtype).element_size()) if concrete else None
    return TensorSpec(shape=shape, dtype=str(dtype), bytes=nbytes)


def _tensors(node: Node) -> tuple[TensorSpec, ...]:
    meta = node.meta.get("tensor_meta")
    # FX TensorMetadata is a namedtuple; handle it as one tensor, not a list.
    spec = _tensor_spec(meta)
    if spec is not None:
        return (spec,)
    if isinstance(meta, (tuple, list)):
        return tuple(spec for x in meta if (spec := _tensor_spec(x)) is not None)
    return ()


def _flops(node: Node, kind: str, gm: GraphModule, tensors: tuple[TensorSpec, ...]) -> int | None:
    if kind in {"input", "output", "constant", "getitem", "transpose", "view", "reshape", "flatten", "contiguous"}:
        return 0
    if not tensors or not all(s.isdecimal() for s in tensors[0].shape):
        return None
    shape = tuple(int(s) for s in tensors[0].shape)
    if kind == "conv2d":
        if node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            if isinstance(module, nn.Conv2d):
                return 2 * prod(shape) * (module.in_channels // module.groups) * prod(module.kernel_size)
        if len(node.args) >= 2 and isinstance(node.args[1], Node):
            wm = node.args[1].meta.get("tensor_meta")
            if wm is not None and len(wm.shape) == 4 and all(isinstance(d, int) for d in wm.shape):
                return 2 * prod(shape) * int(wm.shape[1]) * int(wm.shape[2]) * int(wm.shape[3])
        return None
    if kind == "linear":
        if node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            return 2 * prod(shape) * module.in_features
        if len(node.args) >= 2 and isinstance(node.args[1], Node):
            weight_meta = node.args[1].meta.get("tensor_meta")
            if weight_meta is not None and len(weight_meta.shape) >= 2 and isinstance(weight_meta.shape[-1], int):
                return 2 * prod(shape) * int(weight_meta.shape[-1])
        return None
    if kind == "matmul":
        # Only compute a count when the RHS inner dimension is statically known.
        rhs = node.args[1] if len(node.args) > 1 else None
        meta = rhs.meta.get("tensor_meta") if isinstance(rhs, Node) else None
        if meta is not None and hasattr(meta, "shape") and len(meta.shape) >= 2 and isinstance(meta.shape[-2], int):
            return 2 * prod(shape) * int(meta.shape[-2])
        return None
    if kind in {"add", "sub", "mul", "div", "relu", "sigmoid", "tanh", "exp", "rsqrt", "gelu"}:
        return prod(shape)  # Approximate scalar-operation count, not device instructions.
    return None


def inspect_fx(gm: GraphModule) -> ProgramIR:
    records: list[OpRecord] = []
    placeholders: list[str] = []
    outputs: list[str] = []
    for node in gm.graph.nodes:
        kind = _op(node, gm)
        specs = _tensors(node)
        if kind == "input":
            placeholders.append(node.name)
        if kind == "output":
            outputs.extend(n.name for n in node.all_input_nodes)
        decomposition = ("transpose(weight)", "matmul(input, weight.T)", "add(bias)") if kind == "linear" else ()
        records.append(OpRecord(
            id=node.name, op=kind, target=str(node.target),
            inputs=tuple(n.name for n in node.all_input_nodes),
            tensors=specs,
            estimated_flops=_flops(node, kind, gm, specs),
            decomposition=decomposition,
        ))
    return ProgramIR(tuple(records), tuple(placeholders), tuple(outputs))


def capture(model: nn.Module, *example_inputs: torch.Tensor) -> CapturedProgram:
    """Trace a deterministic tensor computation. FX limitations apply."""
    if not example_inputs or not all(isinstance(x, torch.Tensor) for x in example_inputs):
        raise ValueError("capture requires at least one example torch.Tensor")
    model = copy.deepcopy(model).eval()
    gm = symbolic_trace(model)
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    return CapturedProgram(gm, inspect_fx(gm))


def decompose_linear(captured: CapturedProgram, *example_inputs: torch.Tensor) -> CapturedProgram:
    """Lower nn.Linear layers into transpose+matmul+add FX nodes.

    The rewrite is algebraic, not guaranteed bitwise identical to fused GEMM.
    Users should compare outputs at realistic dtype tolerances.
    """
    gm = copy.deepcopy(captured.graph_module)
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_module" or not isinstance(gm.get_submodule(str(node.target)), nn.Linear):
            continue
        mod = gm.get_submodule(str(node.target))
        if len(node.args) != 1 or node.kwargs:
            raise UnsupportedOperation(f"Cannot decompose unsupported Linear invocation: {node.name}")
        with graph.inserting_before(node):
            weight = graph.get_attr(f"{node.target}.weight")
            wt = graph.call_method("transpose", (weight, 0, 1))
            result = graph.call_function(torch.matmul, (node.args[0], wt))
            if mod.bias is not None:
                bias = graph.get_attr(f"{node.target}.bias")
                result = graph.call_function(torch.add, (result, bias))
        node.replace_all_uses_with(result)
        graph.erase_node(node)
    graph.lint()
    gm.recompile()
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    return CapturedProgram(gm, inspect_fx(gm))


def fuse_linear_pattern(captured: CapturedProgram, *example_inputs: torch.Tensor) -> CapturedProgram:
    """Fuse a safe structural pattern: add(matmul(x, weight.T), bias) -> linear.

    Pattern fusion only. It is *not* a general e-graph optimizer or a promise
    of lower latency on any particular backend. Floating point rounding may
    change because torch can choose a different fused GEMM implementation.
    """
    gm = copy.deepcopy(captured.graph_module)
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (torch.add, operator.add) or len(node.args) != 2:
            continue
        matmul, bias = node.args
        if not isinstance(matmul, Node) or matmul.op != "call_function" or matmul.target not in (torch.matmul, operator.matmul):
            continue
        if len(matmul.users) != 1 or len(matmul.args) != 2:
            continue
        x, transposed = matmul.args
        if not isinstance(transposed, Node) or transposed.op != "call_method" or transposed.target != "transpose":
            continue
        if len(transposed.args) != 3 or transposed.args[1:] != (0, 1):
            continue
        weight = transposed.args[0]
        if not isinstance(weight, Node) or weight.op != "get_attr":
            continue
        if not isinstance(bias, Node) or bias.op != "get_attr":
            continue
        # Require the weight and bias to come from the same nn.Linear module.
        weight_path, _, weight_field = str(weight.target).rpartition(".")
        bias_path, _, bias_field = str(bias.target).rpartition(".")
        if weight_path != bias_path or weight_field != "weight" or bias_field != "bias":
            continue
        if not isinstance(gm.get_submodule(weight_path), nn.Linear):
            continue
        with graph.inserting_before(node):
            fused = graph.call_function(torch.nn.functional.linear, (x, weight, bias))
        node.replace_all_uses_with(fused)
        graph.erase_node(node)
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    return CapturedProgram(gm, inspect_fx(gm))


@dataclass(frozen=True)
class Placement:
    node: str
    operator: str
    device: str
    reason: str


@dataclass(frozen=True)
class LoweringPlan:
    backend: str
    placements: tuple[Placement, ...]
    unsupported: tuple[str, ...]
    boundary_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Backend(Protocol):
    name: str
    native_ops: frozenset[str]
    fallback_ops: frozenset[str]

    def run(self, gm: GraphModule, *args: Any) -> Any: ...


BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    if not backend.name or not backend.native_ops:
        raise ValueError("Backend needs a name and native operations")
    BACKENDS[backend.name] = backend


def plan(ir: ProgramIR, backend: Backend) -> LoweringPlan:
    positions: list[Placement] = []
    unsupported: list[str] = []
    devices: dict[str, str] = {}
    cuts: set[tuple[str, str]] = set()
    for node in ir.nodes:
        if node.op in {"input", "output", "constant"}:
            device = "host"
            why = "interface/state"
        elif "*" in backend.native_ops or node.op in backend.native_ops:
            device = backend.name
            why = "native"
        elif node.op in backend.fallback_ops:
            device = "host"
            why = "explicit digital fallback"
        else:
            unsupported.append(node.id)
            device = "unsupported"
            why = "operator not implemented"
        positions.append(Placement(node.id, node.op, device, why))
        devices[node.id] = device
        for upstream in node.inputs:
            left = devices.get(upstream, "host")
            # Only count host / accelerator cuts, not free input/parameter binding.
            if {left, device} == {"host", backend.name}:
                cuts.add((upstream, node.id))
    return LoweringPlan(backend.name, tuple(positions), tuple(unsupported), len(cuts))


def to_visuaml_graph(ir: ProgramIR, placement: LoweringPlan | None = None) -> dict[str, Any]:
    """Project semantic IR onto VisuaML's existing React Flow graph schema.

    The displayed coordinates are initial layout hints, not a Dagre/ELK pass.
    Layout and display filtering do not affect the source computational IR.
    """
    placement_by_id = {p.node: p for p in placement.placements} if placement else {}
    nodes = []
    edges = []
    for index, node in enumerate(ir.nodes):
        spec = node.tensors[0] if node.tensors else None
        assigned = placement_by_id.get(node.id)
        nodes.append({
            "id": node.id,
            "type": "transformer",
            "data": {
                "label": node.op,
                "name": node.id,
                "target": node.target,
                "op": node.op,
                "args": [],
                "kwargs": {},
                "layerType": node.op,
                "color": "#0d9488" if assigned and assigned.reason == "native" else "#64748b",
                "outputShape": str(list(spec.shape)) if spec else None,
                "estimatedFlops": node.estimated_flops,
                "decomposition": list(node.decomposition),
                "tensors": [asdict(t) for t in node.tensors],
                "backend": assigned.device if assigned else None,
                "loweringReason": assigned.reason if assigned else None,
            },
            "position": {"x": 240 * (index % 4), "y": 140 * (index // 4)},
        })
        for source in node.inputs:
            edges.append({
                "id": f"{source}->{node.id}",
                "source": source,
                "target": node.id,
                "data": {},
            })
    return {"nodes": nodes, "edges": edges}


class _NumpyInterpreter(Interpreter):
    def __init__(self, gm: GraphModule, *, quant_bits: int | None = None, noise_std: float = 0.0, seed: int = 0):
        super().__init__(gm)
        self.bits = quant_bits
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def get_attr(self, target: str, args: tuple, kwargs: dict) -> Any:
        return _to_numpy(super().get_attr(target, args, kwargs))

    def _analog(self, a: np.ndarray) -> np.ndarray:
        if self.bits is not None:
            if self.bits < 2:
                raise ValueError("quant_bits must be >= 2")
            levels = (2 ** (self.bits - 1)) - 1
            bound = float(np.max(np.abs(a))) if a.size else 0.0
            if bound != 0:
                a = np.round(a * levels / bound) * (bound / levels)
        if self.noise_std:
            a = a + self.rng.normal(0, self.noise_std, size=a.shape)
        return a

    def call_module(self, target: str, args: tuple, kwargs: dict) -> Any:
        module = self.fetch_attr(target)
        if isinstance(module, nn.Linear):
            x = args[0]
            weight = module.weight.detach().cpu().numpy()
            bias = None if module.bias is None else module.bias.detach().cpu().numpy()
            out = x @ weight.T
            out = out if bias is None else out + bias
            return self._analog(out) if self.bits is not None or self.noise_std else out
        if isinstance(module, nn.ReLU):
            return np.maximum(args[0], 0)
        if isinstance(module, nn.Sigmoid):
            return 1.0 / (1.0 + np.exp(-args[0]))
        if isinstance(module, nn.Tanh):
            return np.tanh(args[0])
        if isinstance(module, nn.Softmax):
            return self.call_function(torch.softmax, (args[0],), {"dim": module.dim})
        if isinstance(module, nn.GELU):
            return self.call_function(torch.nn.functional.gelu, (args[0],), {"approximate": module.approximate})
        raise UnsupportedOperation(f"NumPy backend cannot execute module {target}: {type(module).__name__}")

    def call_method(self, target: str, args: tuple, kwargs: dict) -> Any:
        x, *rest = args
        if target in {"reshape", "view"}:
            shape = rest if len(rest) != 1 or not isinstance(rest[0], (list, tuple)) else rest[0]
            return np.reshape(x, shape)
        if target == "transpose":
            return np.swapaxes(x, int(rest[0]), int(rest[1]))
        if target == "flatten":
            start = int(rest[0]) if rest else 0
            end = int(rest[1]) if len(rest) > 1 else -1
            start %= x.ndim
            end %= x.ndim
            shape = x.shape[:start] + (int(prod(x.shape[start:end + 1])),) + x.shape[end + 1:]
            return x.reshape(shape)
        if target == "relu":
            return np.maximum(x, 0)
        if target == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if target == "tanh":
            return np.tanh(x)
        if target == "contiguous":
            return np.ascontiguousarray(x)
        if target == "permute":
            axes = rest[0] if len(rest) == 1 and isinstance(rest[0], (tuple, list)) else rest
            return np.transpose(x, axes)
        if target == "softmax":
            return self.call_function(torch.softmax, (x, *rest), kwargs)
        if target == "scatter":
            dim, indices, value = rest
            result = x.copy()
            np.put_along_axis(result, indices.astype(np.int64), value if isinstance(value, np.ndarray) else np.full(indices.shape, value), axis=dim)
            return result
        raise UnsupportedOperation(f"NumPy backend cannot execute method {target}")

    def call_function(self, target: Callable, args: tuple, kwargs: dict) -> Any:
        kind = _function_kind(target)
        if kind == "getitem":
            return operator.getitem(*args)
        if kind == "matmul":
            out = np.matmul(*args)
            return self._analog(out) if self.bits is not None or self.noise_std else out
        if kind == "linear":
            x, weight, *rest = args
            bias = rest[0] if rest else kwargs.get("bias")
            out = np.matmul(x, weight.T)
            out = out + bias if bias is not None else out
            return self._analog(out) if self.bits is not None or self.noise_std else out
        if kind == "add":
            return np.add(*args, **kwargs)
        if kind == "sub":
            return np.subtract(*args, **kwargs)
        if kind == "mul":
            return np.multiply(*args, **kwargs)
        if kind == "div":
            return np.divide(*args, **kwargs)
        if kind == "exp":
            return np.exp(args[0])
        if kind == "rsqrt":
            return 1 / np.sqrt(args[0])
        if kind in {"sum", "amax"}:
            reducer = np.sum if kind == "sum" else np.max
            return reducer(args[0], axis=kwargs.get("dim", args[1] if len(args) > 1 else None),
                           keepdims=kwargs.get("keepdim", args[2] if len(args) > 2 else False))
        if kind == "permute":
            return np.transpose(args[0], args[1])
        if kind == "transpose":
            return np.swapaxes(args[0], int(args[1]), int(args[2]))
        if kind == "softmax":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            shifted = args[0] - np.max(args[0], axis=dim, keepdims=True)
            ex = np.exp(shifted)
            return ex / np.sum(ex, axis=dim, keepdims=True)
        if kind == "gelu":
            import math
            x = args[0]
            mode = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
            if mode == "tanh":
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            if mode == "none":
                return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)).astype(x.dtype))
            raise UnsupportedOperation(f"Unsupported GELU approximation: {mode}")
        if kind == "relu":
            return np.maximum(args[0], 0)
        if kind == "sigmoid":
            return 1 / (1 + np.exp(-args[0]))
        if kind == "tanh":
            return np.tanh(args[0])
        if kind == "argsort":
            x = args[0]
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            descending = kwargs.get("descending", args[2] if len(args) > 2 else False)
            stable = kwargs.get("stable", args[3] if len(args) > 3 else False)
            if not stable:
                raise UnsupportedOperation("Portable argsort requires stable=True")
            return np.argsort(-x if descending else x, axis=dim, kind="stable")
        if kind == "topk":
            x, k = args[:2]
            dim = kwargs.get("dim", args[2] if len(args) > 2 else -1)
            largest = kwargs.get("largest", True)
            sorted_ = kwargs.get("sorted", True)
            if not largest or not sorted_:
                raise UnsupportedOperation("NumPy topk subset requires largest=True, sorted=True")
            indices = np.argsort(-x, axis=dim)
            indices = np.take(indices, range(k), axis=dim)
            return np.take_along_axis(x, indices, axis=dim), indices
        if kind == "zeros_like":
            return np.zeros_like(args[0])
        raise UnsupportedOperation(f"NumPy backend cannot execute {getattr(target, '__name__', target)}")


def _function_kind(target: Callable) -> str:
    node = type("FakeNode", (), {"op": "call_function", "target": target})()
    return _op(node, None)  # type: ignore[arg-type]


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    return value


@dataclass
class ReferenceBackend:
    name: str = "torch"
    native_ops: frozenset[str] = frozenset({"*"})
    fallback_ops: frozenset[str] = frozenset()

    def run(self, gm: GraphModule, *args: Any) -> Any:
        return gm(*args)


@dataclass
class NumpyBackend:
    name: str = "numpy"
    native_ops: frozenset[str] = frozenset({
        "linear", "matmul", "add", "sub", "mul", "relu", "sigmoid", "tanh",
        "topk", "zeros_like", "getitem", "scatter", "transpose", "reshape", "view", "flatten", "contiguous",
        "permute", "softmax", "exp", "sum", "amax", "rsqrt", "div", "gelu", "argsort",
    })
    fallback_ops: frozenset[str] = frozenset()

    def run(self, gm: GraphModule, *args: Any) -> Any:
        return _NumpyInterpreter(gm).run(*(_to_numpy(x) for x in args))


@dataclass
class OpticalSurrogateBackend(NumpyBackend):
    """Nonphysical quantized/noisy linear surrogate; NOT an Ayo device model."""
    name: str = "optical-surrogate"
    native_ops: frozenset[str] = frozenset({"matmul", "linear"})
    fallback_ops: frozenset[str] = frozenset({
        "add", "sub", "mul", "relu", "sigmoid", "tanh", "topk", "zeros_like", "getitem",
        "scatter", "transpose", "reshape", "view", "flatten", "contiguous",
        "permute", "softmax", "exp", "sum", "amax", "rsqrt", "div", "gelu", "argsort",
    })
    quant_bits: int = 8
    noise_std: float = 0.0
    seed: int = 42

    def run(self, gm: GraphModule, *args: Any) -> Any:
        return _NumpyInterpreter(gm, quant_bits=self.quant_bits,
                                 noise_std=self.noise_std, seed=self.seed).run(*(_to_numpy(x) for x in args))


def lower(captured: CapturedProgram, backend_name: str):
    backend = BACKENDS[backend_name]
    planned = plan(captured.ir, backend)
    if planned.unsupported:
        raise UnsupportedOperation(f"Backend {backend_name} unsupported nodes: {', '.join(planned.unsupported)}")
    return LoweredProgram(captured, backend, planned)


@dataclass
class LoweredProgram:
    captured: CapturedProgram
    backend: Backend
    plan: LoweringPlan

    def run(self, *args: Any) -> Any:
        return self.backend.run(self.captured.graph_module, *args)


def compare_outputs(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compare actual sample outputs; not a proof of semantic equivalence."""
    ref = _to_numpy(reference)
    cand = _to_numpy(candidate)
    if isinstance(ref, (tuple, list)) and isinstance(cand, (tuple, list)):
        if len(ref) != len(cand):
            raise ValueError("Output structure differs")
        return {"children": [compare_outputs(a, b) for a, b in zip(ref, cand)]}
    a, b = np.asarray(ref), np.asarray(cand)
    if a.shape != b.shape:
        raise ValueError(f"Output shapes differ: {a.shape} vs {b.shape}")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"shape": list(a.shape), "max_abs_error": float(np.max(delta)) if delta.size else 0.0,
            "mean_abs_error": float(np.mean(delta)) if delta.size else 0.0,
            "all_finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}


register_backend(ReferenceBackend())
register_backend(NumpyBackend())
register_backend(OpticalSurrogateBackend())
