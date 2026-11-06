"""Exact binary graph-cut placement for a *hypothetical* host/accelerator cost model.

This partitions a static graph into HOST and DEVICE assignments, minimizing
  sum(node execution costs) + sum(cross-boundary tensor transfer costs).
It uses a directed max-flow/min-cut reduction. This is exact for the stated
binary additive objective, NOT measured latency, physically valid kernel
selection, or an end-to-end multi-device performance prediction.

One producer-to-consumer edge counts one transfer; repeated values and caching
are not modeled. We reject unknown tensor sizes rather than invent bytes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from visuaml.compiler_spike import ProgramIR

from .backends import BackendCapabilities


class PlacementError(ValueError):
    pass


@dataclass(frozen=True)
class HypotheticalCostModel:
    """All costs are abstract units; none are claimed hardware measurements."""

    name: str = "illustrative-profile-v1"
    host_overhead: float = 1.0
    device_overhead: float = 4.0
    host_flops_per_unit: float = 500.0
    device_flops_per_unit: float = 5000.0
    transfer_bytes_per_unit: float = 4096.0
    transfer_overhead: float = 2.0
    host_op_costs: Mapping[str, float] | None = None
    device_op_costs: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        for k in ("host_overhead", "device_overhead", "transfer_overhead"):
            v = getattr(self, k)
            if not isfinite(v) or v < 0:
                raise PlacementError(f"{k} must be finite and nonnegative")
        for k in (
            "host_flops_per_unit",
            "device_flops_per_unit",
            "transfer_bytes_per_unit",
        ):
            v = getattr(self, k)
            if not isfinite(v) or v <= 0:
                raise PlacementError(f"{k} must be finite and positive")
        for costs in (self.host_op_costs, self.device_op_costs):
            if costs is not None and any(
                not isfinite(v) or v < 0 for v in costs.values()
            ):
                raise PlacementError(
                    "Explicit per-op costs must be finite and nonnegative"
                )


@dataclass(frozen=True)
class CostedAssignment:
    node: str
    op: str
    assigned: str
    host_cost: float
    device_cost: float | None


@dataclass(frozen=True)
class CrossBoundary:
    producer: str
    consumer: str
    bytes: int
    estimated_cost: float


@dataclass(frozen=True)
class OptimizedPlacement:
    schema: str
    profile: str
    backend: str
    assignments: tuple[CostedAssignment, ...]
    boundaries: tuple[CrossBoundary, ...]
    estimated_total_cost: float
    all_host_baseline: float
    unconstrained_greedy_cost: float
    savings_vs_all_host: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Flow:
    def __init__(self, n: int):
        self.g: list[list[list[float | int]]] = [[] for _ in range(n)]

    def add(self, u: int, v: int, capacity: float) -> None:
        if capacity < 0:
            raise PlacementError("Negative cut capacity")
        f: list[float | int] = [v, capacity, len(self.g[v])]
        b: list[float | int] = [u, 0.0, len(self.g[u])]
        self.g[u].append(f)
        self.g[v].append(b)

    def minimum_cut(self, s: int, t: int) -> tuple[float, set[int]]:
        flow = 0.0
        eps = 1e-10
        n = len(self.g)
        while True:
            levels = [-1] * n
            levels[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, _ in self.g[u]:
                    if cap > eps and levels[v] < 0:
                        levels[v] = levels[u] + 1
                        q.append(v)
            if levels[t] < 0:
                break
            at = [0] * n

            def dfs(u: int, pushed: float, at: list[int], levels: list[int]) -> float:
                if u == t:
                    return pushed
                while at[u] < len(self.g[u]):
                    e = self.g[u][at[u]]
                    v, cap, rev = e
                    if cap > eps and levels[v] == levels[u] + 1:
                        sent = dfs(int(v), min(pushed, float(cap)), at, levels)
                        if sent > eps:
                            e[1] = float(cap) - sent
                            r = self.g[int(v)][int(rev)]
                            r[1] = float(r[1]) + sent
                            return sent
                    at[u] += 1
                return 0.0

            while True:
                sent = dfs(s, float("inf"), at, levels)
                if sent <= eps:
                    break
                flow += sent
        reachable = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.g[u]:
                v = int(v)
                if cap > eps and v not in reachable:
                    reachable.add(v)
                    q.append(v)
        return flow, reachable


def optimize_binary_placement(
    ir: ProgramIR,
    capabilities: BackendCapabilities,
    costs: HypotheticalCostModel | None = None,
    *,
    max_nodes: int = 500,
) -> OptimizedPlacement:
    """Exact binary optimum for the DECLARED estimated objective.

    Requires concrete tensor sizes to avoid making up transfer economics.
    The generated plan is informational: an independent lowering backend must
    still compile and validate execution before claiming device support.
    """
    costs = costs or HypotheticalCostModel()
    nodes = list(ir.nodes)
    if len(nodes) > max_nodes:
        raise PlacementError("Graph exceeds exact graph-cut size admission limit")
    names = [n.id for n in nodes]
    if len(set(names)) != len(names):
        raise PlacementError("Duplicate IR value IDs")
    index = {n: i for i, n in enumerate(names)}
    before: set[str] = set()
    specs: dict[str, int] = {}
    for n in nodes:
        for dep in n.inputs:
            if dep not in before:
                raise PlacementError(
                    f"Forward reference or missing producer: {dep}->{n.id}"
                )
        before.add(n.id)
        if n.op not in {"input", "output", "constant"} and (
            not n.tensors or any(t.bytes is None for t in n.tensors)
        ):
            raise PlacementError(
                f"Unknown output bytes for {n.id}; refusing to guess transfer cost"
            )
        specs[n.id] = sum(t.bytes or 0 for t in n.tensors)

    source, sink = len(nodes), len(nodes) + 1
    graph = _Flow(len(nodes) + 2)
    node_costs: dict[str, tuple[float, float | None]] = {}
    costs_all_host = 0.0
    # A conservative finite infinity, larger than every possible valid cost.
    edge_pairs = [(dep, n.id) for n in nodes for dep in n.inputs]
    # An invalid (unsupported) device choice must cost more than every feasible
    # all-host assignment, including arbitrary user-provided explicit op costs.
    # Ignoring overrides here makes the purported exact solver select a backend
    # that cannot execute an operator whenever host costs exceed the sentinel.
    explicit = [
        *(costs.host_op_costs or {}).values(),
        *(costs.device_op_costs or {}).values(),
    ]
    max_override = max([0.0, *explicit])
    max_unary = len(nodes) * (
        costs.host_overhead
        + costs.device_overhead
        + 1
        + max_override
        + sum((n.estimated_flops or 0) for n in nodes)
        / min(costs.host_flops_per_unit, costs.device_flops_per_unit)
    )
    max_transfer = sum(
        costs.transfer_overhead + specs[dep] / costs.transfer_bytes_per_unit
        for dep, _ in edge_pairs
    )
    infinite = max(1e12, (max_unary + max_transfer + 1) * 8)
    if not isfinite(infinite):
        raise PlacementError("Non-finite optimization objective")

    for n in nodes:
        special = n.op in {"input", "constant", "output"}
        operations = capabilities.native_ops
        supported = (
            not special
            and (n.op in operations or "*" in operations)
            and all(v.dtype in capabilities.supported_dtypes for v in n.tensors)
            and (
                capabilities.max_tensor_rank is None
                or all(len(v.shape) <= capabilities.max_tensor_rank for v in n.tensors)
            )
        )
        flop = n.estimated_flops
        host_explicit = (costs.host_op_costs or {}).get(n.op)
        device_explicit = (costs.device_op_costs or {}).get(n.op)
        if special:
            host, device = 0.0, None
        else:
            if flop is None and host_explicit is None:
                raise PlacementError(
                    f"Missing static cost for {n.id}; provide explicit host op cost"
                )
            host = (
                host_explicit
                if host_explicit is not None
                else costs.host_overhead + flop / costs.host_flops_per_unit
            )
            device = (
                device_explicit
                if device_explicit is not None
                else (costs.device_overhead + flop / costs.device_flops_per_unit)
                if flop is not None
                else None
            )
            if not supported:
                device = None
            if supported and device is None:
                raise PlacementError(
                    f"Missing device cost for supported operation {n.op}"
                )
        assert host is not None
        costs_all_host += host
        node_costs[n.id] = (host, device)
        # S -> node costs host assignment; node -> T costs device assignment.
        graph.add(source, index[n.id], host)
        graph.add(index[n.id], sink, device if device is not None else infinite)

    for a, b in edge_pairs:
        weight = costs.transfer_overhead + specs[a] / costs.transfer_bytes_per_unit
        graph.add(index[a], index[b], weight)
        graph.add(index[b], index[a], weight)
    _, src_side = graph.minimum_cut(source, sink)
    assignments: list[CostedAssignment] = []
    selected: dict[str, str] = {}
    unary = 0.0
    greedy_labels: dict[str, str] = {}
    greedy_unary = 0.0
    for n in nodes:
        host, device = node_costs[n.id]
        assignment = "device" if index[n.id] in src_side else "host"
        if assignment == "device" and device is None:
            raise RuntimeError(f"Cut assigned unsupported operation {n.id} to device")
        selected[n.id] = assignment
        unary += device if assignment == "device" else host
        greedy = "device" if device is not None and device < host else "host"
        greedy_labels[n.id] = greedy
        greedy_unary += device if greedy == "device" else host
        assignments.append(CostedAssignment(n.id, n.op, assignment, host, device))
    boundaries: list[CrossBoundary] = []
    greedy_edge = 0.0
    for a, b in edge_pairs:
        c = costs.transfer_overhead + specs[a] / costs.transfer_bytes_per_unit
        if selected[a] != selected[b]:
            boundaries.append(CrossBoundary(a, b, specs[a], c))
        if greedy_labels[a] != greedy_labels[b]:
            greedy_edge += c
    total = unary + sum(e.estimated_cost for e in boundaries)
    if total > costs_all_host + 1e-6:
        raise RuntimeError("Graph cut violated all-host baseline; implementation bug")
    return OptimizedPlacement(
        "visuaml.binary-cut-placement.v1",
        costs.name,
        capabilities.name,
        tuple(assignments),
        tuple(boundaries),
        total,
        costs_all_host,
        greedy_unary + greedy_edge,
        costs_all_host - total,
        (
            "Optimization is exact only for the supplied additive binary objective.",
            "Profile costs are illustrative; no real hardware benchmarking or execution occurs.",
            "Repeated transfers, caching, IO conversion, kernel fusion and memory limits are not modeled.",
        ),
    )
