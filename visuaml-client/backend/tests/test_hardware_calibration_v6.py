"""Synthetic reference math for calibration schema; NOT hardware benchmarks."""

import json
from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.hardware_calibration import (
    CalibrationError,
    LatencyObservation,
    calibrate_latency,
)


def fixtures():
    out = []
    for i in range(18):
        units = 10.0 + (i * 23) % 97
        moved = 5.0 + (i * 37) % 101
        out.append(
            LatencyObservation(
                measurement_id=f"synthetic-{i:02d}",
                hardware_id="synthetic-device-not-ayo",
                operation="matmul",
                source_kind="synthetic",
                workload_units=units,
                bytes_moved=moved,
                latency_us=3.0 + 0.25 * units + 0.4 * moved,
                provenance="unit-test-affine-fixture",
            )
        )
    return out


def test_affine_fit_recovers_declared_synthetic_coefficients():
    result = calibrate_latency(fixtures())
    assert result.source_kind == "synthetic"
    assert result.hardware_id == "synthetic-device-not-ayo"
    assert result.baseline_us == pytest.approx(3.0)
    assert result.us_per_work_unit == pytest.approx(0.25)
    assert result.us_per_byte == pytest.approx(0.4)
    assert result.residual_max_abs_us < 1e-10
    assert len(result.dataset_sha256) == 64
    assert result.predict_us(work_units=37, bytes_moved=21) == pytest.approx(20.65)
    json.dumps(result.as_dict(), allow_nan=False)


def test_dataset_fingerprint_independent_of_row_order():
    a = fixtures()
    b = list(reversed(a))
    assert calibrate_latency(a) == calibrate_latency(b)


def test_provenance_modification_changes_fingerprint():
    a = fixtures()
    b = a.copy()
    b[0] = replace(b[0], provenance="different-capture")
    assert calibrate_latency(a).dataset_sha256 != calibrate_latency(b).dataset_sha256


def test_refuse_mixing_claimed_measurements_with_synthetic_data():
    a = fixtures()
    a[0] = replace(a[0], source_kind="measured")
    with pytest.raises(CalibrationError, match="Mixing"):
        calibrate_latency(a)


def test_refuse_devices_and_operations_mixed():
    for field in ("hardware_id", "operation"):
        a = fixtures()
        a[0] = replace(a[0], **{field: "different"})
        with pytest.raises(CalibrationError, match="Mixing"):
            calibrate_latency(a)


def test_refuse_underdetermined_or_insufficient_datasets():
    with pytest.raises(CalibrationError, match="6"):
        calibrate_latency(fixtures()[:3])
    a = fixtures()
    a = [replace(x, bytes_moved=2 * x.workload_units) for x in a]
    with pytest.raises(CalibrationError, match="identify"):
        calibrate_latency(a)


def test_refuse_negative_latency_or_non_finite_observations():
    for field, value in (
        ("latency_us", -1.0),
        ("bytes_moved", np.inf),
        ("workload_units", np.nan),
    ):
        with pytest.raises(CalibrationError):
            replace(fixtures()[0], **{field: value})


def test_refuse_negative_fitted_slope():
    a = fixtures()
    a = [
        replace(r, latency_us=100 - 0.25 * r.workload_units + 0.4 * r.bytes_moved)
        for r in a
    ]
    with pytest.raises(CalibrationError, match="Negative"):
        calibrate_latency(a)


def test_refuse_duplicate_evidence_id():
    a = fixtures()
    a[1] = replace(a[1], measurement_id=a[0].measurement_id)
    with pytest.raises(CalibrationError, match="Duplicate"):
        calibrate_latency(a)


def test_extrapolation_requires_explicit_opt_in():
    report = calibrate_latency(fixtures())
    with pytest.raises(CalibrationError, match="extrapolates"):
        report.predict_us(work_units=1000, bytes_moved=5)
    assert (
        report.predict_us(work_units=1000, bytes_moved=5, allow_extrapolation=True) > 0
    )


def test_residual_diagnostics_reveal_model_mismatch():
    a = fixtures()
    b = [
        replace(r, latency_us=r.latency_us + (2 if i % 2 else 0))
        for i, r in enumerate(a)
    ]
    report = calibrate_latency(b)
    assert report.residual_max_abs_us > 0.2
    assert report.residual_mean_abs_us > 0.1


def test_refuse_invalid_admission_and_overflow():
    a = fixtures()
    with pytest.raises(CalibrationError):
        replace(a[0], measurement_id="")
    with pytest.raises(CalibrationError):
        replace(a[0], latency_us=True)
    with pytest.raises(CalibrationError):
        calibrate_latency(a * 0)
