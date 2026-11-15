"""Executable CPU/NumPy hybrid interpreter with *real* data-boundary copies.

A NumPy partition is a separately dispatched software adapter, not physical
accelerator execution. Timings include Python overhead; no overlap, DMA, PCIe,
optical conversion, or GPU speedup is implied. Only explicit, pure operators
are admitted. Capturing arbitrary PyTorch model source is not a sandbox.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.fx import Interpreter, Node

from visuaml.compiler_spike import CapturedProgram, _op

from .verification import verify_outputs


class HybridExecutionError(ValueError):
    """Unsupported executable placement or violated runtime contract."""


DEVICE_OPERATIONS = frozenset({"matmul", "add", "sub", "mul", "relu", "sigmoid", "tanh", "linear"})


@dataclass(frozen=True)
class RuntimeTransfer:
    producer: str
    consumer: str
    direction: str
    bytes: int
    elapsed_ns: int


@dataclass(frozen=True)
class RuntimeEvent:
    node: str
    op: str
    adapter: str
    elapsed_ns: int
    output_bytes: int


@dataclass(frozen=True)
class HybridRun:
    events: tuple[RuntimeEvent, ...]
    transfers: tuple[RuntimeTransfer, ...]
    wall_ns: int
    peak_live_bytes: int
    output: Any

    def as_dict(self) -> dict:
        return {
            "schema": "visuaml.hybrid-execution.v1",
            "runtime": "CPU torch host + NumPy software adapter",
            "events": [asdict(e) for e in self.events],
            "transfers": [asdict(e) for e in self.transfers],
            "wall_ns": self.wall_ns,
            "critical_path_ns": self.wall_ns,  # strictly sequential execution
            "peak_retained_value_bytes": self.peak_live_bytes,
            "total_transfer_bytes": sum(t.bytes for t in self.transfers),
            "warnings": [
                "No parallel overlap: measured sequential interpreter wall time is the critical path.",
                "NumPy is a distinct CPU software adapter, not an accelerator benchmark.",
                "Repeated edges are counted as distinct logical transfers; caching is not modeled.",
                "Linear-module weight/bias copies are counted in bytes; their duration is included in operator dispatch.",
            ],
        }


def _nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_nbytes(x) for x in value.values())
    return 0


def _convert(value: Any, destination: str) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(_convert(v, destination) for v in value)
    if isinstance(value, dict):
        return {k: _convert(v, destination) for k, v in value.items()}
    if destination == "device":
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
                raise HybridExecutionError("NumPy adapter supports only CPU float32/float64 tensors")
            return value.detach().contiguous().numpy().copy()
        if isinstance(value, np.ndarray):
            return value.copy()
    if destination == "host" and isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value)).clone()
    return value


def _device_compute(node: Node, module: torch.fx.GraphModule, args: tuple, kwargs: dict):
    kind = _op(node, module)
    if kind not in DEVICE_OPERATIONS:
        raise HybridExecutionError(f"Unsupported NumPy adapter operation: {kind}")
    if kwargs:
        raise HybridExecutionError("NumPy adapter accepts only positional arguments")
    if kind == "matmul":
        return np.matmul(args[0], args[1])
    if kind == "add":
        return np.add(args[0], args[1])
    if kind == "sub":
        return np.subtract(args[0], args[1])
    if kind == "mul":
        return np.multiply(args[0], args[1])
    if kind == "relu":
        return np.maximum(args[0], 0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-args[0]))
    if kind == "tanh":
        return np.tanh(args[0])
    if kind == "linear":
        if node.op == "call_module":
            layer = module.get_submodule(str(node.target))
            if not isinstance(layer, nn.Linear):
                raise HybridExecutionError("Linear module type mismatch")
            weights = _convert(layer.weight, "device")
            bias = _convert(layer.bias, "device") if layer.bias is not None else None
            result = np.matmul(args[0], weights.T)
        else:
            weights = args[1]
            bias = args[2] if len(args) > 2 else None
            result = np.matmul(args[0], weights.T)
        return result + bias if bias is not None else result
    raise HybridExecutionError("Operator admission bug")


class _Interpreter(Interpreter):
    def __init__(self, captured: CapturedProgram, placements: Mapping[str, str], max_bytes: int):
        super().__init__(captured.graph_module, garbage_collect_values=False)
        self.placements = placements
        self.max_bytes = max_bytes
        self.locations: dict[str, str] = {}
        self.events: list[RuntimeEvent] = []
        self.transfers: list[RuntimeTransfer] = []
        self.peak_bytes = 0

    def _copy_args(self, node: Node, destination: str, args: Any, kwargs: Any) -> tuple[Any, Any]:
        crossings = [p for p in node.all_input_nodes if self.locations.get(p.name, "host") != destination]
        started = time.perf_counter_ns()
        converted_args = _convert(args, destination)
        converted_kwargs = _convert(kwargs, destination)
        elapsed = time.perf_counter_ns() - started
        total_bytes = sum(_nbytes(self.env[p]) for p in crossings)
        for predecessor in crossings:
            nbytes = _nbytes(self.env[predecessor])
            # Apportion one measured aggregate conversion, not a separate dummy copy.
            share = round(elapsed * nbytes / total_bytes) if total_bytes else 0
            origin = self.locations.get(predecessor.name, "host")
            self.transfers.append(RuntimeTransfer(predecessor.name, node.name,
                                                 f"{origin}->{destination}", nbytes, share))
        return converted_args, converted_kwargs

    def run_node(self, node: Node) -> Any:
        kind = _op(node, self.module)
        if node.op in ("placeholder", "get_attr"):
            result = super().run_node(node)
            self.locations[node.name] = "host"
        elif node.op == "output":
            output = super().run_node(node)
            args, _ = self._copy_args(node, "host", (output,), {})
            result = args[0]
            self.locations[node.name] = "host"
        else:
            destination = self.placements[node.name]
            if destination == "device" and kind not in DEVICE_OPERATIONS:
                raise HybridExecutionError(f"Device cannot execute {kind}: {node.name}")
            if kind.endswith("_") or kind.startswith(("function:", "module:", "method:")):
                raise HybridExecutionError(f"Unsupported or mutating operator: {kind}")
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            args, kwargs = self._copy_args(node, destination, args, kwargs)
            started = time.perf_counter_ns()
            if destination == "device":
                result = _device_compute(node, self.module, args, kwargs)
            elif node.op == "call_module":
                result = self.module.get_submodule(str(node.target))(*args, **kwargs)
            elif node.op == "call_function":
                result = node.target(*args, **kwargs)
            elif node.op == "call_method":
                result = getattr(args[0], str(node.target))(*args[1:], **kwargs)
            else:
                raise HybridExecutionError(f"Unsupported node category {node.op}")
            elapsed = time.perf_counter_ns() - started
            if destination == "device" and node.op == "call_module" and kind == "linear":
                layer = self.module.get_submodule(str(node.target))
                for part, tensor in (("weight", layer.weight), ("bias", layer.bias)):
                    if tensor is not None:
                        self.transfers.append(RuntimeTransfer(f"{node.target}.{part}", node.name,
                                                             "host->device", _nbytes(tensor), 0))
                # The weight-copy timing is included in the kernel event.
            if _nbytes(result) > self.max_bytes:
                raise HybridExecutionError("Single output exceeds memory admission")
            self.locations[node.name] = destination
            self.events.append(RuntimeEvent(node.name, kind, destination, elapsed, _nbytes(result)))
        # Interpreter is intentionally sequential and holds values for diagnosis.
        live = sum(_nbytes(value) for value in self.env.values()) + _nbytes(result)
        self.peak_bytes = max(self.peak_bytes, live)
        if live > self.max_bytes:
            raise HybridExecutionError("Retained activation memory exceeds budget")
        return result


def execute_hybrid(
    captured: CapturedProgram,
    *inputs: torch.Tensor,
    placements: Mapping[str, str] | None = None,
    max_retained_bytes: int = 256 * 1024 * 1024,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[HybridRun, dict]:
    """Execute a captured pure FX subset through two software runtime adapters.

    The original model's Python code was executed at *capture time* and is not
    sandboxed by this function. Only trusted, allowlisted local models belong here.
    """
    if not isinstance(captured, CapturedProgram):
        raise HybridExecutionError("Expected captured model")
    if type(max_retained_bytes) is not int or not 1024 <= max_retained_bytes <= 2 ** 30:
        raise HybridExecutionError("Invalid explicit memory budget")
    gm = captured.graph_module
    nodes = tuple(gm.graph.nodes)
    if len(nodes) > 1024:
        raise HybridExecutionError("Graph exceeds node budget")
    calls = {n.name for n in nodes if n.op in {"call_module", "call_function", "call_method"}}
    assignments = dict(placements or {})
    if set(assignments) - calls:
        raise HybridExecutionError("Placement specifies a nonexistent or non-compute node")
    plan = {name: assignments.get(name, "host") for name in calls}
    if any(x not in ("host", "device") for x in plan.values()):
        raise HybridExecutionError("Unknown placement adapter")
    input_nodes = [n for n in nodes if n.op == "placeholder"]
    if len(inputs) != len(input_nodes) or not all(isinstance(x, torch.Tensor) and x.device.type == "cpu" for x in inputs):
        raise HybridExecutionError("Input arity/type/device mismatch")
    for n, x in zip(input_nodes, inputs):
        spec = n.meta.get("tensor_meta")
        if spec is None or tuple(x.shape) != tuple(spec.shape) or x.dtype != spec.dtype:
            raise HybridExecutionError("Input shape or dtype outside captured contract")
    with torch.no_grad():
        reference = gm(*inputs)
        worker = _Interpreter(captured, plan, max_retained_bytes)
        started = time.perf_counter_ns()
        observed = worker.run(*inputs)
        elapsed = time.perf_counter_ns() - started
    receipt = verify_outputs(reference, observed, atol=atol, rtol=rtol)
    if not receipt.passed:
        raise HybridExecutionError("Hybrid execution violated differential witness contract")
    return HybridRun(tuple(worker.events), tuple(worker.transfers), elapsed,
                     worker.peak_bytes, observed), receipt.as_dict()
