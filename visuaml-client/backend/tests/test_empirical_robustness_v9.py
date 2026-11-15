"""V9: explicitly nonstatistical envelopes around holdout-admitted costs."""
from __future__ import annotations

from dataclasses import replace

import pytest

from visuaml.compiler_lab.backends import BackendCapabilities
from visuaml.compiler_lab.calibration_evidence import validate_latency_holdout
from visuaml.compiler_lab.empirical_robustness import optimize_with_empirical_envelopes
from visuaml.compiler_lab.hardware_calibration import CalibrationError, LatencyObservation
from visuaml.compiler_spike import OpRecord, ProgramIR, TensorSpec


def evidence(hardware, operation, coefficients, *, jitter=0.15):
    a, b, c = coefficients
    observations = []
    for i in range(48):
        work = 1.0 + ((i % 36) * 47) % 193
        nbytes = 16.0 + ((i % 36) * 37) % 199
        observations.append(LatencyObservation(
            f"{hardware}-{operation}-{i:02d}", hardware, operation,
            "synthetic", work, nbytes,
            a + b*work + c*nbytes + jitter*((i%5)-2)/2,
            "v9-synthetic-fixture-not-hardware",
        ))
    return validate_latency_holdout(
        observations[:36], observations[36:],
        allowed_max_abs_us=1.0, allowed_max_relative=0.5,
    )


def sample():
    def t(n): return (TensorSpec(("1", "4"), "torch.float32", n),)
    ir = ProgramIR((
        OpRecord("x", "input", "", (), t(48), 0, ()),
        OpRecord("m1", "matmul", "", ("x",), t(56), 85, ()),
        OpRecord("r", "relu", "", ("m1",), t(56), 12, ()),
        OpRecord("m2", "matmul", "", ("r",), t(64), 133, ()),
        OpRecord("o", "output", "", ("m2",), (), 0, ()),
    ), ("x",), ("m2",))
    caps = BackendCapabilities("toy-device", "r1", frozenset({"matmul"}), frozenset({"relu"}), frozenset({"torch.float32"}))
    h = {"matmul": evidence("cpu", "matmul", (7.0, 0.3, 0.01)), "relu": evidence("cpu", "relu", (2.0, 0.04, 0.005))}
    d = {"matmul": evidence("not-real-hardware", "matmul", (0.3, 0.02, 0.001))}
    tr = evidence("link", "transfer", (0.5, 0.1, 0.007))
    return ir, caps, h, d, tr


def run(*, allow_synthetic=True, max_exact=12):
    ir, caps, h, d, tr = sample()
    return optimize_with_empirical_envelopes(
        ir, caps, host_hardware_id="cpu", device_hardware_id="not-real-hardware",
        transfer_hardware_id="link", host=h, device=d, transfer=tr,
        allow_synthetic=allow_synthetic, max_exact_device_nodes=max_exact,
    )


def test_ensemble_contains_observed_residuals_and_five_profiles():
    r = run()
    assert r.unit == "microseconds" and r.source_kind == "synthetic"
    assert len(r.sensitivity.scenarios) == 5
    assert r.sensitivity.scenarios[0].profile == "nominal"
    assert all(uncertainty >= 0 for _, uncertainty in r.empirical_error_us)
    assert any(uncertainty > 0.02 for _, uncertainty in r.empirical_error_us)
    assert r.sensitivity.search == "exhaustive-minimax-regret"
    assert r.sensitivity.preferred.worst_regret >= 0
    assert len(r.nominal.input_sha256) == len(r.sensitivity.input_sha256) == 64
    assert "NOT statistical" in " ".join(r.warnings)


def test_nominal_scenario_matches_direct_evidence_fit():
    r = run()
    nominal = r.sensitivity.scenarios[0]
    assert nominal.optimum_cost == pytest.approx(r.nominal.placement.estimated_total_cost)
    assert nominal.device_nodes == tuple(x.node for x in r.nominal.placement.assignments if x.assigned == "device")


def test_synthetic_is_not_accepted_as_measured():
    with pytest.raises(CalibrationError, match="Synthetic"):
        run(allow_synthetic=False)


def test_failed_holdout_is_not_accepted():
    ir, caps, h, d, tr = sample()
    h = dict(h)
    h["relu"] = replace(h["relu"], passed=False)
    with pytest.raises(CalibrationError, match="Failed holdout"):
        optimize_with_empirical_envelopes(ir, caps, host_hardware_id="cpu", device_hardware_id="not-real-hardware", transfer_hardware_id="link", host=h, device=d, transfer=tr, allow_synthetic=True)


def test_out_of_domain_work_rejected():
    ir, caps, h, d, tr = sample()
    nodes = list(ir.nodes)
    nodes[1] = replace(nodes[1], estimated_flops=90000)
    with pytest.raises(CalibrationError, match="outside"):
        optimize_with_empirical_envelopes(replace(ir, nodes=tuple(nodes)), caps, host_hardware_id="cpu", device_hardware_id="not-real-hardware", transfer_hardware_id="link", host=h, device=d, transfer=tr, allow_synthetic=True)


def test_changed_residuals_change_sensitivity_fingerprint():
    ir, caps, h, d, tr = sample()
    first = run()
    altered = dict(h)
    altered["matmul"] = evidence("cpu", "matmul", (7.0, 0.3, 0.01), jitter=0.33)
    second = optimize_with_empirical_envelopes(ir, caps, host_hardware_id="cpu", device_hardware_id="not-real-hardware", transfer_hardware_id="link", host=altered, device=d, transfer=tr, allow_synthetic=True)
    assert first.sensitivity.input_sha256 != second.sensitivity.input_sha256


def test_restricted_search_discloses_nonoptimality():
    r = run(max_exact=1)
    assert r.sensitivity.search == "restricted-scenario-candidates"
    assert "not a globally optimal" in " ".join(r.sensitivity.warnings)
