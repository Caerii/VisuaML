"""Holdout-evidence-gated, *per-node* binary hardware placement.

The cost model is explicit and narrow: each op's predictor is an admitted
nonnegative-affine latency fit, evaluated at declared work and output bytes.
Transfers use a separately admitted affine fit with workload_units=1 for
one producer-consumer boundary. Units are MICROSECONDS throughout.

Admission is not cryptographic provenance and not device execution. The
minimum cut is optimal only under this additive, non-overlapped model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from visuaml.compiler_spike import ProgramIR

from .backends import BackendCapabilities
from .calibration_evidence import HoldoutEvaluation, admit_latency_for_planning
from .hardware_calibration import CalibrationError
from .placement_optimizer import (
    HypotheticalCostModel,
    OptimizedPlacement,
    PlacementError,
    optimize_binary_placement,
)


@dataclass(frozen=True)
class EvidencePlacementReport:
    schema: str
    input_sha256: str
    source_kind: str
    unit: str
    host_hardware_id: str
    device_hardware_id: str
    transfer_hardware_id: str
    evidence_digests: tuple[tuple[str, str, str], ...]
    predicted_host_us: tuple[tuple[str, float], ...]
    predicted_device_us: tuple[tuple[str, float], ...]
    transfer_overhead_us: float
    transfer_us_per_byte: float
    placement: OptimizedPlacement
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def optimize_with_evidence(
    ir: ProgramIR,
    capabilities: BackendCapabilities,
    *,
    host_hardware_id: str,
    device_hardware_id: str,
    transfer_hardware_id: str,
    host: Mapping[str, HoldoutEvaluation],
    device: Mapping[str, HoldoutEvaluation],
    transfer: HoldoutEvaluation,
    allow_synthetic: bool = False,
) -> EvidencePlacementReport:
    """Place only observed/in-domain ops. Reject unknown costs, don't invent FLOPS.

    Host evidence must cover every non-I/O operation. Device evidence must
    cover every operator the backend declares native in this particular IR.
    Every graph edge must be covered by the transfer evidence, even if a
    particular placement ends up not crossing that edge.
    """
    if any(
        not isinstance(x, str) or not x.strip()
        for x in (host_hardware_id, device_hardware_id, transfer_hardware_id)
    ):
        raise CalibrationError("Hardware identities must be nonempty")
    if type(allow_synthetic) is not bool:
        raise CalibrationError("Synthetic admission must be a boolean")
    special = {"input", "output", "constant"}
    op_names = {n.op for n in ir.nodes if n.op not in special}
    native_names = {
        n.op
        for n in ir.nodes
        if n.op not in special
        and (n.op in capabilities.native_ops or "*" in capabilities.native_ops)
        and all(t.dtype in capabilities.supported_dtypes for t in n.tensors)
        and (
            capabilities.max_tensor_rank is None
            or all(len(t.shape) <= capabilities.max_tensor_rank for t in n.tensors)
        )
    }
    if set(host) != op_names or set(device) != native_names:
        raise CalibrationError(
            "Operator evidence coverage must exactly match host and native workloads"
        )
    admitted_host = {
        op: admit_latency_for_planning(ev, allow_synthetic=allow_synthetic)
        for op, ev in host.items()
    }
    admitted_device = {
        op: admit_latency_for_planning(ev, allow_synthetic=allow_synthetic)
        for op, ev in device.items()
    }
    admitted_transfer = admit_latency_for_planning(
        transfer, allow_synthetic=allow_synthetic
    )
    for op, a in admitted_host.items():
        if (a.hardware_id, a.operation) != (host_hardware_id, op):
            raise CalibrationError(
                f"Host evidence does not identify requested host/op {op}"
            )
    for op, a in admitted_device.items():
        if (a.hardware_id, a.operation) != (device_hardware_id, op):
            raise CalibrationError(
                f"Device evidence does not identify requested device/op {op}"
            )
    if (admitted_transfer.hardware_id, admitted_transfer.operation) != (
        transfer_hardware_id,
        "transfer",
    ):
        raise CalibrationError(
            "Transfer evidence must identify the link and 'transfer' operation"
        )
    kinds = {
        x.source_kind
        for x in (*admitted_host.values(), *admitted_device.values(), admitted_transfer)
    }
    if len(kinds) != 1:
        raise CalibrationError(
            "Mixed measured/synthetic evidence cannot enter one placement objective"
        )
    if admitted_transfer.us_per_byte <= 0:
        raise CalibrationError(
            "Transfer coefficient must be strictly positive to define transfer cost"
        )
    per_node_host: dict[str, float] = {}
    per_node_device: dict[str, float] = {}
    tensor_bytes: dict[str, int] = {}
    digests: list[tuple[str, str, str]] = []
    for area, group in (
        ("host", admitted_host),
        ("device", admitted_device),
        ("transfer", {"transfer": admitted_transfer}),
    ):
        for op, evidence in group.items():
            digests.append(
                (f"{area}:{op}", evidence.calibration_sha256, evidence.holdout_sha256)
            )
    for n in ir.nodes:
        if any(t.bytes is None or t.bytes < 0 for t in n.tensors):
            raise PlacementError(f"Unknown or negative output bytes: {n.id}")
        tensor_bytes[n.id] = sum(t.bytes for t in n.tensors if t.bytes is not None)
        if n.op in special:
            continue
        if (
            n.estimated_flops is None
            or type(n.estimated_flops) is not int
            or n.estimated_flops < 0
        ):
            raise PlacementError(f"Unknown or invalid declared work for {n.id}")
        work, count = n.estimated_flops, tensor_bytes[n.id]
        per_node_host[n.id] = admitted_host[n.op].estimate_us(work, count)
        if n.op in native_names:
            per_node_device[n.id] = admitted_device[n.op].estimate_us(work, count)
    # Inputs/constants can also cross boundaries, so their bytes must be in-domain.
    for n in ir.nodes:
        for dep in n.inputs:
            if dep not in tensor_bytes:
                raise PlacementError(f"Missing/forward dependency: {dep}")
            admitted_transfer.estimate_us(1, tensor_bytes[dep])
    overhead = admitted_transfer.baseline_us + admitted_transfer.us_per_work_unit
    perbyte = admitted_transfer.us_per_byte
    if not math.isfinite(overhead) or overhead < 0:
        raise CalibrationError("Invalid transfer overhead")
    profile = HypotheticalCostModel(
        name="evidence-estimated-microseconds",
        host_node_costs=per_node_host,
        device_node_costs=per_node_device,
        transfer_overhead=overhead,
        transfer_bytes_per_unit=1.0 / perbyte,
    )
    # Use the true supported dialect, not a wildcard that could admit a
    # device op without evidence. The underlying optimizer checks dtype/rank.
    placement = optimize_binary_placement(ir, capabilities, profile)
    digests.sort()
    payload = {
        "ir": ir.as_dict(),
        "backend": {},
        "host_hardware_id": host_hardware_id,
        "device_hardware_id": device_hardware_id,
        "transfer_hardware_id": transfer_hardware_id,
        "evidence": digests,
        "host_predictions": per_node_host,
        "device_predictions": per_node_device,
        "transfer_overhead": overhead,
        "transfer_per_byte": perbyte,
    }
    # dataclasses containing frozensets are not JSON serializable; resolve
    # backend capabilities into a stable plain-data contract.
    payload["backend"] = {
        "name": capabilities.name,
        "revision": capabilities.revision,
        "native_ops": sorted(capabilities.native_ops),
        "fallback_ops": sorted(capabilities.fallback_ops),
        "dtypes": sorted(capabilities.supported_dtypes),
        "max_tensor_rank": capabilities.max_tensor_rank,
    }
    digest = sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return EvidencePlacementReport(
        "visuaml.evidence-placement.v1",
        digest,
        next(iter(kinds)),
        "microseconds",
        host_hardware_id,
        device_hardware_id,
        transfer_hardware_id,
        tuple(digests),
        tuple(sorted(per_node_host.items())),
        tuple(sorted(per_node_device.items())),
        overhead,
        perbyte,
        placement,
        (
            "Provenance and measurement source-kind are provider-declared, not independently authenticated.",
            "The objective sums independently fitted operator and per-edge transfer latencies; it does not model concurrency, shared transfers, or memory limits.",
            "Output tensor bytes are an explicit proxy for data movement, not a measured transfer trace.",
            "An optimized placement is not an executable accelerator binary or verified device speedup.",
        ),
    )
