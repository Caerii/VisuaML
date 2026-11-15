"""V7 evidence gate: disjoint holdout, no invented device economics."""

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.calibration_evidence import (
    admit_latency_for_planning,
    validate_latency_holdout,
)
from visuaml.compiler_lab.hardware_calibration import (
    CalibrationError,
    LatencyObservation,
)


def _data(count=30, source="synthetic"):
    out = []
    for i in range(count):
        work = 10 + (i * 23) % 101
        moved = 5 + (i * 37) % 103
        out.append(
            LatencyObservation(
                f"{source}-{i:03d}",
                "test-device-not-ayo",
                "matmul",
                source,
                float(work),
                float(moved),
                5.0 + 0.12 * work + 0.21 * moved,
                "synthetic-affine-control",
            )
        )
    # Training deliberately covers the evaluation domain; avoid extrapolation.
    anchors = [
        LatencyObservation(
            f"{source}-anchor-{i}",
            "test-device-not-ayo",
            "matmul",
            source,
            float(w),
            float(b),
            5.0 + 0.12 * w + 0.21 * b,
            "synthetic-affine-control",
        )
        for i, (w, b) in enumerate(((0, 0), (200, 0), (0, 200), (200, 200)))
    ]
    return [*out[:20], *anchors], out[20:]


def test_holdout_fits_affine_and_reports_provenance():
    train, hold = _data()
    result = validate_latency_holdout(
        train, hold, allowed_max_abs_us=1e-8, allowed_max_relative=1e-8
    )
    assert result.passed
    assert result.max_abs_error_us < 1e-8
    assert result.calibration.baseline_us == pytest.approx(5.0)
    assert result.calibration.source_kind == "synthetic"
    assert len(result.holdout_sha256) == 64


def test_measured_label_only_is_not_independent_attestation():
    train, hold = _data(source="measured")
    result = validate_latency_holdout(
        train, hold, allowed_max_abs_us=1e-8, allowed_max_relative=1e-8
    )
    admitted = admit_latency_for_planning(result)
    assert admitted.source_kind == "measured"
    assert admitted.estimate_us(
        train[-1].workload_units, train[-1].bytes_moved
    ) == pytest.approx(train[-1].latency_us)
    # Passing does not constitute external attestation of the label.
    assert "provider-declared" in result.warnings[0]


def test_synthetic_needs_explicit_approval_for_planning():
    train, hold = _data()
    v = validate_latency_holdout(
        train, hold, allowed_max_abs_us=0.01, allowed_max_relative=0.01
    )
    with pytest.raises(CalibrationError, match="Synthetic"):
        admit_latency_for_planning(v)
    admitted = admit_latency_for_planning(v, allow_synthetic=True)
    assert admitted.estimate_us(70, 50) == pytest.approx(5 + 0.12 * 70 + 0.21 * 50)


def test_out_of_domain_holdout_rejected():
    train, hold = _data()
    hold = list(hold)
    hold[0] = replace(hold[0], workload_units=9999.0)
    with pytest.raises(CalibrationError, match="extrapolates"):
        validate_latency_holdout(
            train, hold, allowed_max_abs_us=2, allowed_max_relative=0.2
        )


def test_holdout_leakage_and_duplicate_ids_rejected():
    train, hold = _data()
    h = list(hold)
    h[0] = replace(h[0], measurement_id=train[0].measurement_id)
    with pytest.raises(CalibrationError, match="disjoint"):
        validate_latency_holdout(
            train, h, allowed_max_abs_us=1, allowed_max_relative=0.1
        )
    h = list(hold)
    h[0] = replace(h[0], measurement_id=h[1].measurement_id)
    with pytest.raises(CalibrationError, match="unique"):
        validate_latency_holdout(
            train, h, allowed_max_abs_us=1, allowed_max_relative=0.1
        )


def test_source_hardware_and_op_mixing_rejected():
    train, hold = _data()
    for field, value in (
        ("hardware_id", "other"),
        ("operation", "relu"),
        ("source_kind", "measured"),
    ):
        h = list(hold)
        h[0] = replace(h[0], **{field: value})
        with pytest.raises(CalibrationError, match="incompatible"):
            validate_latency_holdout(
                train, h, allowed_max_abs_us=1, allowed_max_relative=0.1
            )


def test_holdout_detects_misspecified_cost_model_and_cannot_admit():
    train, hold = _data()
    h = [replace(r, latency_us=r.latency_us + 8.0) for r in hold]
    result = validate_latency_holdout(
        train, h, allowed_max_abs_us=1.0, allowed_max_relative=0.1
    )
    assert not result.passed
    assert result.max_abs_error_us > 7.0
    with pytest.raises(CalibrationError, match="Failed holdout"):
        admit_latency_for_planning(result, allow_synthetic=True)


def test_holdout_budgets_reject_nan_bool_and_negative():
    train, hold = _data()
    for value in (True, np.nan, -1):
        with pytest.raises(CalibrationError, match="budgets"):
            validate_latency_holdout(
                train, hold, allowed_max_abs_us=value, allowed_max_relative=0.1
            )


def test_holdout_hash_changes_with_evidence_content():
    train, hold = _data()
    one = validate_latency_holdout(
        train, hold, allowed_max_abs_us=1, allowed_max_relative=1
    )
    h = list(hold)
    h[0] = replace(h[0], provenance="different-run")
    two = validate_latency_holdout(
        train, h, allowed_max_abs_us=1, allowed_max_relative=1
    )
    assert one.holdout_sha256 != two.holdout_sha256


def test_empty_and_short_holdout_rejected():
    train, hold = _data()
    with pytest.raises(CalibrationError, match="4"):
        validate_latency_holdout(
            train, hold[:3], allowed_max_abs_us=1, allowed_max_relative=0.5
        )


def test_admitted_model_fails_closed_outside_domain():
    train, hold = _data()
    v = validate_latency_holdout(
        train, hold, allowed_max_abs_us=1, allowed_max_relative=1
    )
    model = admit_latency_for_planning(v, allow_synthetic=True)
    with pytest.raises(CalibrationError, match="outside"):
        model.estimate_us(999999, 0)
