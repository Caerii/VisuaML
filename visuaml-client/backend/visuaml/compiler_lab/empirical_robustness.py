"""Sensitivity of evidence-guided placement to observed calibration errors.

This is an EMPIRICAL residual envelope, NOT a confidence/credible interval,
future-error guarantee, device measurement, or physical uncertainty model.
We form five explicit additive-cost scenarios from the largest training and
held-out absolute residuals and evaluate placements across them. A scenario
ensemble must not be interpreted as an observed probability distribution.

No runtime lowering occurs here; the graph-cut result remains a cost-model
recommendation and requires separate executable backend validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from visuaml.compiler_spike import ProgramIR

from .backends import BackendCapabilities
from .calibration_evidence import HoldoutEvaluation
from .evidence_placement import EvidencePlacementReport, optimize_with_evidence
from .placement_optimizer import HypotheticalCostModel
from .placement_robustness import PlacementSensitivity, analyze_placement_sensitivity


@dataclass(frozen=True)
class EmpiricalRobustPlacement:
    schema: str
    unit: str
    source_kind: str
    nominal: EvidencePlacementReport
    empirical_error_us: tuple[tuple[str, float], ...]
    sensitivity: PlacementSensitivity
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _envelope(evidence: HoldoutEvaluation) -> float:
    # Include training residuals: a fitted line can match one split unusually well.
    error = max(evidence.max_abs_error_us, evidence.calibration.residual_max_abs_us)
    if not isfinite(error) or error < 0:
        raise ValueError("Calibration residual envelope must be finite and nonnegative")
    return float(error)


def optimize_with_empirical_envelopes(
    ir: ProgramIR,
    capabilities: BackendCapabilities,
    *,
    host_hardware_id: str,
    device_hardware_id: str,
    transfer_hardware_id: str,
    host: dict[str, HoldoutEvaluation],
    device: dict[str, HoldoutEvaluation],
    transfer: HoldoutEvaluation,
    allow_synthetic: bool = False,
    max_exact_device_nodes: int = 12,
) -> EmpiricalRobustPlacement:
    """Minimax-regret placement over specified empirical-residual scenarios.

    This is *not* robust to unseen distribution shifts; the max observed error
    is used only to construct an auditable deterministic sensitivity exercise.
    Admission, provenance, supported-op and range checks are inherited from
    ``optimize_with_evidence`` and run before building any scenarios.
    """
    nominal = optimize_with_evidence(
        ir,
        capabilities,
        host_hardware_id=host_hardware_id,
        device_hardware_id=device_hardware_id,
        transfer_hardware_id=transfer_hardware_id,
        host=host,
        device=device,
        transfer=transfer,
        allow_synthetic=allow_synthetic,
    )
    host_nominal = dict(nominal.predicted_host_us)
    device_nominal = dict(nominal.predicted_device_us)
    host_delta_by_op = {op: _envelope(item) for op, item in host.items()}
    device_delta_by_op = {op: _envelope(item) for op, item in device.items()}
    transfer_delta = _envelope(transfer)
    op_by_id = {node.id: node.op for node in ir.nodes}
    host_delta = {node: host_delta_by_op[op_by_id[node]] for node in host_nominal}
    device_delta = {node: device_delta_by_op[op_by_id[node]] for node in device_nominal}

    def profile(name: str, *, host_sign: int, device_sign: int, transfer_sign: int):
        return HypotheticalCostModel(
            name=name,
            host_node_costs={
                key: max(0.0, price + host_sign * host_delta[key])
                for key, price in host_nominal.items()
            },
            device_node_costs={
                key: max(0.0, price + device_sign * device_delta[key])
                for key, price in device_nominal.items()
            },
            transfer_overhead=max(
                0.0, nominal.transfer_overhead_us + transfer_sign * transfer_delta
            ),
            transfer_bytes_per_unit=1 / nominal.transfer_us_per_byte,
        )

    profiles = (
        profile("nominal", host_sign=0, device_sign=0, transfer_sign=0),
        profile("device-favorable", host_sign=1, device_sign=-1, transfer_sign=-1),
        profile("host-favorable", host_sign=-1, device_sign=1, transfer_sign=1),
        profile("joint-upper", host_sign=1, device_sign=1, transfer_sign=1),
        profile("joint-lower", host_sign=-1, device_sign=-1, transfer_sign=-1),
    )
    sensitivity = analyze_placement_sensitivity(
        ir,
        capabilities,
        profiles,
        max_exact_device_nodes=max_exact_device_nodes,
    )
    errors = tuple(
        sorted(
            [(f"host:{k}", v) for k, v in host_delta_by_op.items()]
            + [(f"device:{k}", v) for k, v in device_delta_by_op.items()]
            + [("transfer:per-edge", transfer_delta)]
        )
    )
    return EmpiricalRobustPlacement(
        "visuaml.empirical-robust-placement.v1",
        "microseconds",
        nominal.source_kind,
        nominal,
        errors,
        sensitivity,
        (
            "Residual envelopes describe errors observed on this training/holdout split, NOT statistical confidence or future-error bounds.",
            "The five scenarios are illustrative perturbations of holdout-admitted additive latency estimates, not probabilities.",
            "Transfer error is applied per modeled edge, and the cost model omits communication overlap, memory residency and fusion.",
            "A placement decision is not executable backend lowering or evidence of physical speedup.",
        ),
    )
