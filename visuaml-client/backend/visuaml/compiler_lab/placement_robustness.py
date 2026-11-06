"""Auditable sensitivity analysis for *hypothetical* binary backend placement.

A graph cut optimizes one cost estimate, but accelerator designs often have
uncertain transfer rates and overheads. This module asks whether the preferred
placement survives a specified set of cost scenarios. For sufficiently small
programs it also computes the exact minimum-maximum-regret placement over all
legal binary assignments. For larger programs it clearly labels a restricted
candidate search; no statistical confidence or measured hardware performance
is inferred from the scenario set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from math import isfinite
from typing import Any

from visuaml.compiler_spike import ProgramIR

from .backends import BackendCapabilities
from .placement_optimizer import (
    HypotheticalCostModel,
    PlacementError,
    optimize_binary_placement,
)


@dataclass(frozen=True)
class ScenarioPlacement:
    profile: str
    cost_model: dict[str, Any]
    optimum_cost: float
    all_host_cost: float
    device_nodes: tuple[str, ...]


@dataclass(frozen=True)
class RobustCandidate:
    device_nodes: tuple[str, ...]
    scenario_costs: tuple[float, ...]
    scenario_regrets: tuple[float, ...]
    worst_regret: float
    worst_cost: float
    mean_regret: float


@dataclass(frozen=True)
class PlacementSensitivity:
    schema: str
    backend: str
    input_sha256: str
    backend_contract: dict[str, Any]
    search: str
    scenarios: tuple[ScenarioPlacement, ...]
    preferred: RobustCandidate
    node_device_frequency: tuple[tuple[str, int], ...]
    candidates_checked: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cost_of_assignment(
    ir: ProgramIR,
    plan: dict[str, str],
    host: dict[str, float],
    device: dict[str, float | None],
    costs: HypotheticalCostModel,
) -> float:
    """Replay precisely the binary additive cost assumed by graph-cut placement."""
    total = 0.0
    widths: dict[str, int] = {}
    for node in ir.nodes:
        label = plan[node.id]
        value = host[node.id] if label == "host" else device[node.id]
        if value is None:
            raise PlacementError(f"Illegal device assignment for {node.id}")
        total += value
        widths[node.id] = sum(t.bytes or 0 for t in node.tensors)
    for node in ir.nodes:
        for dependency in node.inputs:
            if plan[dependency] != plan[node.id]:
                total += (
                    costs.transfer_overhead
                    + widths[dependency] / costs.transfer_bytes_per_unit
                )
    if not isfinite(total):
        raise PlacementError("Non-finite scenario cost")
    return total


def analyze_placement_sensitivity(
    ir: ProgramIR,
    capabilities: BackendCapabilities,
    profiles: Sequence[HypotheticalCostModel],
    *,
    max_exact_device_nodes: int = 12,
    max_scenarios: int = 32,
) -> PlacementSensitivity:
    """Compare graph-cut plans across cost profiles, optionally exactly minimax.

    Regret is additive: candidate cost minus the best plan *for that profile*.
    With <= max_exact_device_nodes eligible nodes, exhaustively check every
    possible legal labeling. Otherwise use the per-scenario optima and all-host
    as candidates, reporting that restricted search explicitly.
    """
    if not 2 <= len(profiles) <= max_scenarios:
        raise PlacementError("Expected between 2 and max_scenarios cost profiles")
    if max_exact_device_nodes < 0 or max_exact_device_nodes > 18:
        raise PlacementError("Exact-search admission must be in [0, 18]")
    if len({p.name for p in profiles}) != len(profiles):
        raise PlacementError("Cost profiles must have distinct names")

    optima = [optimize_binary_placement(ir, capabilities, p) for p in profiles]
    nodes = tuple(n.id for n in ir.nodes)
    allowed = tuple(
        n.id
        for n in ir.nodes
        if all(
            next(a for a in result.assignments if a.node == n.id).device_cost
            is not None
            for result in optima
        )
    )
    allowed_set = set(allowed)
    backend_contract = {
        "name": capabilities.name,
        "revision": capabilities.revision,
        "native_ops": sorted(capabilities.native_ops),
        "fallback_ops": sorted(capabilities.fallback_ops),
        "supported_dtypes": sorted(capabilities.supported_dtypes),
        "max_tensor_rank": capabilities.max_tensor_rank,
        "is_physical_hardware": capabilities.is_physical_hardware,
    }
    input_payload = {
        "ir": ir.as_dict(),
        "backend_contract": backend_contract,
        "profiles": [asdict(p) for p in profiles],
    }
    input_digest = sha256(
        json.dumps(
            input_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    scenarios = tuple(
        ScenarioPlacement(
            p.name,
            asdict(p),
            result.estimated_total_cost,
            result.all_host_baseline,
            tuple(a.node for a in result.assignments if a.assigned == "device"),
        )
        for p, result in zip(profiles, optima)
    )

    if len(allowed) <= max_exact_device_nodes:
        candidate_sets = [
            frozenset(allowed[i] for i, bit in enumerate(bits) if bit)
            for bits in product((False, True), repeat=len(allowed))
        ]
        method = "exhaustive-minimax-regret"
    else:
        candidate_sets = [frozenset(), *(frozenset(s.device_nodes) for s in scenarios)]
        method = "restricted-scenario-candidates"
    # Deduplicate while preserving candidate order for reproducible results.
    candidate_sets = list(dict.fromkeys(candidate_sets))
    assignments = [
        (
            {a.node: a.host_cost for a in result.assignments},
            {a.node: a.device_cost for a in result.assignments},
        )
        for result in optima
    ]
    scored: list[RobustCandidate] = []
    for selected in candidate_sets:
        if not selected.issubset(allowed_set):
            continue
        plan = {node: ("device" if node in selected else "host") for node in nodes}
        scenario_costs = tuple(
            _cost_of_assignment(ir, plan, h, d, p)
            for (h, d), p in zip(assignments, profiles)
        )
        regrets = tuple(
            max(0.0, c - s.optimum_cost) for c, s in zip(scenario_costs, scenarios)
        )
        scored.append(
            RobustCandidate(
                tuple(n for n in nodes if n in selected),
                scenario_costs,
                regrets,
                max(regrets),
                max(scenario_costs),
                sum(regrets) / len(regrets),
            )
        )
    if not scored:
        raise PlacementError("No valid assignment candidates")
    best = min(
        scored,
        key=lambda c: (
            c.worst_regret,
            c.worst_cost,
            len(c.device_nodes),
            c.device_nodes,
        ),
    )
    frequency = tuple((n, sum(n in s.device_nodes for s in scenarios)) for n in nodes)
    return PlacementSensitivity(
        schema="visuaml.placement-sensitivity.v2",
        backend=capabilities.name,
        input_sha256=input_digest,
        backend_contract=backend_contract,
        search=method,
        scenarios=scenarios,
        preferred=best,
        node_device_frequency=frequency,
        candidates_checked=len(scored),
        warnings=(
            "Costs and scenarios are illustrative, not calibrated hardware measurements.",
            "Device frequency counts the supplied scenarios; it is not a probability estimate.",
            "The additive objective ignores caching, memory constraints, concurrency and kernel fusion.",
        )
        + (
            ("Restricted candidate search is not a globally optimal minimax solution.",)
            if method != "exhaustive-minimax-regret"
            else ()
        ),
    )
