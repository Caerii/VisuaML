"""Unitful, provenance-bound latency calibration from supplied measurements.

No photonic hardware is queried or assumed. A measurement's source kind is
DECLARED by its provider, not cryptographically attested by this library.
Synthetic inputs must stay explicitly synthetic in every derived report.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


class CalibrationError(ValueError):
    """Insufficient, invalid, incompatible, or misleading calibration data."""


@dataclass(frozen=True)
class LatencyObservation:
    measurement_id: str
    hardware_id: str
    operation: str
    source_kind: Literal["measured", "synthetic"]
    workload_units: float
    bytes_moved: float
    latency_us: float
    provenance: str

    def __post_init__(self) -> None:
        for field in ("measurement_id", "hardware_id", "operation", "provenance"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise CalibrationError(f"Invalid observation field: {field}")
        if self.source_kind not in ("measured", "synthetic"):
            raise CalibrationError("Source kind must be measured or synthetic")
        for field in ("workload_units", "bytes_moved", "latency_us"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CalibrationError(f"{field} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise CalibrationError(f"{field} must be finite and nonnegative")


@dataclass(frozen=True)
class LatencyCalibration:
    schema: str
    hardware_id: str
    operation: str
    source_kind: str
    dataset_sha256: str
    sample_count: int
    baseline_us: float
    us_per_work_unit: float
    us_per_byte: float
    work_unit_range: tuple[float, float]
    byte_range: tuple[float, float]
    residual_max_abs_us: float
    residual_mean_abs_us: float
    fit_kind: str = "nonnegative-affine-ols"

    def predict_us(self, *, work_units: float, bytes_moved: float, allow_extrapolation: bool = False) -> float:
        for value in (work_units, bytes_moved):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise CalibrationError("Prediction inputs must be finite, nonnegative numbers")
        if not allow_extrapolation and not (
            self.work_unit_range[0] <= work_units <= self.work_unit_range[1]
            and self.byte_range[0] <= bytes_moved <= self.byte_range[1]
        ):
            raise CalibrationError("Prediction extrapolates beyond the observed calibration domain")
        return self.baseline_us + self.us_per_work_unit * work_units + self.us_per_byte * bytes_moved

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_latency(records: list[LatencyObservation]) -> LatencyCalibration:
    """Fit an identifiable nonnegative affine model to ONE operation/device/source.

    Requires at least 6 distinct, full-rank observations. The linear fit is
    descriptive, not a verified physical model or a benchmark of another device.
    No synthetic result may be re-labeled measured by aggregation.
    """
    if not 6 <= len(records) <= 100_000:
        raise CalibrationError("Require 6..100000 observations")
    if any(not isinstance(r, LatencyObservation) for r in records):
        raise CalibrationError("Only typed LatencyObservation records are accepted")
    labels = {(r.hardware_id, r.operation, r.source_kind) for r in records}
    if len(labels) != 1:
        raise CalibrationError("Mixing hardware, operations, or measured/synthetic sources is forbidden")
    ids = [r.measurement_id for r in records]
    if len(set(ids)) != len(ids):
        raise CalibrationError("Duplicate measurement IDs")
    sorted_records = sorted(records, key=lambda r: r.measurement_id)
    X = np.array([[1.0, r.workload_units, r.bytes_moved] for r in sorted_records], dtype=np.float64)
    Y = np.array([r.latency_us for r in sorted_records], dtype=np.float64)
    if np.linalg.matrix_rank(X) != 3:
        raise CalibrationError("Cannot separately identify baseline, work and transfer coefficients")
    coefficients, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if rank != 3 or not np.all(np.isfinite(coefficients)):
        raise CalibrationError("Unstable or rank-deficient calibration")
    if np.any(coefficients < -1e-9):
        raise CalibrationError("Negative physical-cost coefficient; declared model is invalid for this dataset")
    coefficients = np.maximum(coefficients, 0)
    residuals = X @ coefficients - Y
    if not np.all(np.isfinite(residuals)):
        raise CalibrationError("Non-finite residual")
    document = [asdict(r) for r in sorted_records]
    content = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    hardware, op, source = next(iter(labels))
    return LatencyCalibration(
        schema="visuaml.latency-calibration.v1",
        hardware_id=hardware,
        operation=op,
        source_kind=source,
        dataset_sha256=digest,
        sample_count=len(records),
        baseline_us=float(coefficients[0]),
        us_per_work_unit=float(coefficients[1]),
        us_per_byte=float(coefficients[2]),
        work_unit_range=(float(X[:, 1].min()), float(X[:, 1].max())),
        byte_range=(float(X[:, 2].min()), float(X[:, 2].max())),
        residual_max_abs_us=float(np.max(np.abs(residuals))),
        residual_mean_abs_us=float(np.mean(np.abs(residuals))),
    )
