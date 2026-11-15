"""Out-of-sample admission for evidence-supported latency estimates.

All measurement source labels are PROVIDED, not authenticated. No device is
queried. Passing this gate only supports the stated affine fit on held-out
observations inside its measured domain; it does not prove future performance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

from .hardware_calibration import (
    CalibrationError, LatencyCalibration, LatencyObservation, calibrate_latency,
)


@dataclass(frozen=True)
class HoldoutEvaluation:
    schema: str
    calibration: LatencyCalibration
    holdout_sha256: str
    holdout_count: int
    max_abs_error_us: float
    mean_abs_error_us: float
    max_relative_error: float
    mean_relative_error: float
    allowed_max_abs_us: float
    allowed_max_relative: float
    passed: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _digest_records(records: Sequence[LatencyObservation]) -> str:
    return sha256(json.dumps([asdict(r) for r in sorted(records, key=lambda r: r.measurement_id)],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_latency_holdout(
    train: Sequence[LatencyObservation], holdout: Sequence[LatencyObservation], *,
    allowed_max_abs_us: float, allowed_max_relative: float,
) -> HoldoutEvaluation:
    """Evaluate disjoint, in-domain evidence without allowing test-data leakage."""
    if not isinstance(train, (list, tuple)) or not isinstance(holdout, (list, tuple)):
        raise CalibrationError("Calibration train and holdout must be ordered record sequences")
    if not 4 <= len(holdout) <= 100_000:
        raise CalibrationError("Require 4..100000 independent holdout observations")
    for v in (allowed_max_abs_us, allowed_max_relative):
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise CalibrationError("Holdout error budgets must be finite nonnegative numbers")
    calibration = calibrate_latency(list(train))
    ids_train = {r.measurement_id for r in train}
    ids_holdout = [r.measurement_id for r in holdout]
    if len(set(ids_holdout)) != len(ids_holdout) or ids_train.intersection(ids_holdout):
        raise CalibrationError("Holdout evidence IDs must be unique and disjoint from calibration")
    if any(not isinstance(r, LatencyObservation) or (r.hardware_id, r.operation, r.source_kind) != (
        calibration.hardware_id, calibration.operation, calibration.source_kind) for r in holdout):
        raise CalibrationError("Holdout device, operator, or source provenance is incompatible")
    errs: list[float] = []
    relatives: list[float] = []
    for record in holdout:
        estimate = calibration.predict_us(work_units=record.workload_units, bytes_moved=record.bytes_moved)
        error = abs(estimate - record.latency_us)
        errs.append(error)
        relatives.append(error / max(record.latency_us, 1e-6))
    maximum = max(errs)
    relative = max(relatives)
    success = maximum <= allowed_max_abs_us and relative <= allowed_max_relative
    return HoldoutEvaluation(
        "visuaml.holdout-latency-evidence.v1",calibration,_digest_records(holdout),len(holdout),
        maximum,sum(errs)/len(errs),relative,sum(relatives)/len(relatives),
        float(allowed_max_abs_us),float(allowed_max_relative),success,
        ("Source-kind labels and provenance are provider-declared, not cryptographically authenticated.",
         "The fit is evaluated only inside its observed calibration domain.",
         "A passing synthetic holdout must never be represented as a physical-device measurement.",),
    )


@dataclass(frozen=True)
class AdmittedLatency:
    hardware_id: str
    operation: str
    source_kind: str
    calibration_sha256: str
    holdout_sha256: str
    baseline_us: float
    us_per_work_unit: float
    us_per_byte: float
    workload_range: tuple[float,float]
    transfer_range: tuple[float,float]

    def estimate_us(self, work_units: float, bytes_moved: float) -> float:
        # No extrapolation: backend optimizer must not guess unseen shapes.
        if (type(work_units) not in (int,float) or type(bytes_moved) not in (int,float)
            or not math.isfinite(work_units) or not math.isfinite(bytes_moved)
            or not self.workload_range[0] <= work_units <= self.workload_range[1]
            or not self.transfer_range[0] <= bytes_moved <= self.transfer_range[1]):
            raise CalibrationError("Estimate outside admitted holdout-validated domain")
        return self.baseline_us+self.us_per_work_unit*work_units+self.us_per_byte*bytes_moved


def admit_latency_for_planning(evidence: HoldoutEvaluation, *, allow_synthetic: bool = False) -> AdmittedLatency:
    """Admit only passing out-of-sample evidence; synthetic requires explicit opt-in."""
    if not evidence.passed:
        raise CalibrationError("Failed holdout evidence cannot feed a placement cost model")
    c = evidence.calibration
    if c.source_kind != "measured" and not allow_synthetic:
        raise CalibrationError("Synthetic cost model needs explicit experimental opt-in")
    return AdmittedLatency(c.hardware_id,c.operation,c.source_kind,c.dataset_sha256,
        evidence.holdout_sha256,c.baseline_us,c.us_per_work_unit,c.us_per_byte,
        c.work_unit_range,c.byte_range)
