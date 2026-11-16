"""Reproducible graph inspection and explicitly labeled static cost estimates.

These estimates do NOT constitute real device throughput/energy measurements.
Activation liveness ignores aliasing, kernel scratch space and allocator reuse;
parameters are accounted separately from temporary activations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.fx import GraphModule, Node

from visuaml.compiler_spike import CapturedProgram, _op


@dataclass(frozen=True)
class GraphDiagnostic:
    code: str
    level: str
    message: str
    node: str | None = None


@dataclass(frozen=True)
class CostReport:
    schema: str
    graph_digest: str
    node_count: int
    known_flops: int
    uncounted_flop_nodes: tuple[str, ...]
    activation_bytes_written: int
    peak_live_activation_bytes: int
    parameter_bytes: int
    tensor_edge_bytes: int
    unmeasured_tensor_nodes: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canon(value: Any) -> Any:
    if isinstance(value, Node):
        return {"node": value.name}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        # Byte reinterpretation also supports bfloat16 where NumPy has no dtype.
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        return {
            "tensor": {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest,
            }
        }
    if isinstance(value, tuple):
        return {"tuple": [_canon(v) for v in value]}
    if isinstance(value, list):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _canon(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, slice):
        return {"slice": [_canon(value.start), _canon(value.stop), _canon(value.step)]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return {"opaque_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def graph_digest(gm: GraphModule) -> str:
    """Content-based fingerprint, not a formal equivalence proof or signed receipt."""
    items = []
    for n in gm.graph.nodes:
        if n.op == "call_module":
            mod = gm.get_submodule(str(n.target))
            target = f"{type(mod).__module__}.{type(mod).__qualname__}:{mod!r}"
        elif n.op == "call_function":
            target = f"{getattr(n.target, '__module__', '')}.{getattr(n.target, '__qualname__', str(n.target))}"
        else:
            target = str(n.target)
        items.append([n.name, n.op, target, _canon(n.args), _canon(n.kwargs)])
    state = [(k, _canon(v)) for k, v in sorted(gm.state_dict().items())]
    payload = json.dumps(
        {"schema": "visuaml.graph-sha256.v1", "nodes": items, "state": state},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_graph(gm: GraphModule) -> tuple[GraphDiagnostic, ...]:
    """Reject structural invalidity; flag capture/shape limitations distinctly."""
    try:
        gm.graph.lint()
    except Exception as exc:
        raise ValueError(f"FX graph failed structural verification: {exc}") from exc
    nodes = list(gm.graph.nodes)
    if not nodes or nodes[-1].op != "output":
        raise ValueError("FX graph has no terminal output")
    defined: set[str] = set()
    warnings: list[GraphDiagnostic] = []
    for node in nodes:
        if node.name in defined:
            raise ValueError(f"Duplicate node name: {node.name}")
        for arg in node.all_input_nodes:
            if arg.name not in defined:
                raise ValueError(
                    f"Non-topological dependency: {node.name} <- {arg.name}"
                )
        defined.add(node.name)
        if node.op in {"call_module", "call_function", "call_method"}:
            if node.meta.get("tensor_meta") is None:
                warnings.append(
                    GraphDiagnostic(
                        "UNKNOWN_OUTPUT_SPEC",
                        "warning",
                        "Output shape/dtype unavailable",
                        node.name,
                    )
                )
            kind = _op(node, gm)
            if kind.endswith("_"):
                warnings.append(
                    GraphDiagnostic(
                        "POTENTIAL_MUTATION",
                        "warning",
                        "In-place operator may invalidate SSA/alias assumptions",
                        node.name,
                    )
                )
    return tuple(warnings)


def cost_report(program: CapturedProgram) -> CostReport:
    gm, ir = program.graph_module, program.ir
    warnings = validate_graph(gm)
    nodes = list(gm.graph.nodes)
    index = {n.name: i for i, n in enumerate(nodes)}
    last_use = {n.name: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        for dep in n.all_input_nodes:
            last_use[dep.name] = max(last_use[dep.name], i)
    sizes = {n.id: sum(spec.bytes or 0 for spec in n.tensors) for n in ir.nodes}
    unknown = tuple(
        n.id
        for n in ir.nodes
        if n.op not in {"output", "constant"}
        and (not n.tensors or any(spec.bytes is None for spec in n.tensors))
    )
    written = 0
    peak = 0
    live = 0
    edges = 0
    releases: dict[int, list[str]] = {}
    for name, last in last_use.items():
        releases.setdefault(last, []).append(name)
    for i, node in enumerate(nodes):
        # model parameters and buffers are persistent storage, not activations.
        if node.op not in {"get_attr", "output"}:
            written += sizes[node.name]
            live += sizes[node.name]
            peak = max(peak, live)
        for dep in node.all_input_nodes:
            if dep.op not in {"get_attr", "placeholder"}:
                edges += sizes[dep.name]
        # Free after the final consumer, not when the producer executes.
        for prior_name in releases.get(i, ()):
            prior = nodes[index[prior_name]]
            if prior.op not in {"get_attr", "output"}:
                live -= sizes[prior_name]
    state_bytes = sum(v.nelement() * v.element_size() for v in gm.state_dict().values())
    counted = sum(n.estimated_flops or 0 for n in ir.nodes)
    uncounted = tuple(
        n.id
        for n in ir.nodes
        if n.estimated_flops is None and n.op not in {"input", "output", "constant"}
    )
    return CostReport(
        schema="visuaml.cost-report.v1",
        graph_digest=graph_digest(gm),
        node_count=len(nodes),
        known_flops=counted,
        uncounted_flop_nodes=uncounted,
        activation_bytes_written=written,
        peak_live_activation_bytes=peak,
        parameter_bytes=state_bytes,
        tensor_edge_bytes=edges,
        unmeasured_tensor_nodes=unknown,
        warnings=(
            "Illustrative operation-count and activation accounting; excludes kernel scratch space, allocation overhead, transfer costs, aliasing and hardware measurements.",
        )
        + tuple(f"{w.code}: {w.node}" for w in warnings),
    )
