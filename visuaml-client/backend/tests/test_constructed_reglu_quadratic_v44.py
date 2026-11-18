"""v44: from neural weights to smaller but different analytically built gated network.

A 2-unit ReGLU representation is not equal-cost to a 255-ReLU representation.
Numerical witnesses show exactness under the stated binary64 integer budget.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    AnalyticRelu,
    compile_bounded_program,
    evaluate_bounded,
)
from visuaml.compiler_lab.constructed_reglu_quadratic_v44 import (
    ConstructedGatedQuadratic,
    GatedConstructionError,
    bundle_fingerprint,
    check_real_algebraic_identity,
    construct_from_symbolic,
    evaluate_constructed,
    permute_gated_units,
    recover_from_gated_weights,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    RecoveredPolynomial,
    SymbolicRecoveryError,
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    execute_program,
)


def quadratic() -> I32Program:
    def op(name: str, value: int | None = None) -> VMOperation:
        return VMOperation(name, value)
    return I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))


def _source_recovered() -> tuple[AnalyticRelu, RecoveredPolynomial]:
    source_net, _ = compile_bounded_program(quadratic())
    detached = AnalyticRelu(
        source_net.w_in.copy(), source_net.b_in.copy(),
        source_net.w_out.copy(), source_net.b_out.copy()
    )
    return detached, recover_quadratic_from_weights(detached)


def test_construct_two_gated_units_from_symbolically_recovered_neural_function():
    original, poly = _source_recovered()
    generated, receipt = construct_from_symbolic(poly)
    assert generated.gate_weights.shape == (2,)
    assert receipt["hidden_gated_units"] == 2
    assert receipt["v41_reference_relu_units"] == 255
    assert receipt["serialized_scalar_parameter_count"] == 10
    assert receipt["v41_reference_scalar_parameters"] == 766
    assert receipt["different_activation_family"] is True
    assert receipt["speedup_measured"] is False
    assert receipt["full_domain_signed_i8_evaluated"] is True
    assert receipt["all_i32_gated_network_proved"] is False
    assert receipt["source_program_consulted_for_construction"] is False
    assert receipt["polynomial_digest"] == poly.digest()
    for x in range(-128, 128):
        assert evaluate_constructed(generated, x) == evaluate_bounded(original, x)
        assert evaluate_constructed(generated, x) == execute_program(quadratic(), x).value


def test_gated_network_restores_symbolic_rule_without_source_manifest():
    _, poly = _source_recovered()
    net, _ = construct_from_symbolic(poly)
    inversion = recover_from_gated_weights(net)
    assert inversion == poly
    scrambled = permute_gated_units(net, (1, 0))
    assert bundle_fingerprint(scrambled) != bundle_fingerprint(net)
    assert recover_from_gated_weights(scrambled) == poly
    with pytest.raises(GatedConstructionError, match="bijective"):
        permute_gated_units(net, (0, 0))


def test_new_gating_architecture_extends_old_relu_domain_for_safe_exact_values():
    old, poly = _source_recovered()
    network, _ = construct_from_symbolic(poly)
    # The previously archived x=128 neural counterexample no longer applies
    # to THIS new gated network, but all-i32 wrap equivalence remains unproved.
    x = 128
    old_out = int(float(
        (old.w_out @ np.maximum(
            0, old.w_in @ np.array([float(x)]) + old.b_in
        ))[0] + old.b_out[0]
    ))
    new_out = evaluate_constructed(network, x)
    assert old_out == 16005
    assert new_out == 16007
    assert new_out == execute_program(quadratic(), x).value
    for sample in (-10000, -1000, -129, -128, -1, 0, 1, 127, 128, 129, 1000, 10000):
        assert evaluate_constructed(network, sample) == (
            sample * sample - 3 * sample + 7
        )


def test_gated_weight_validation_and_numeric_budget_fail_closed():
    _, poly = _source_recovered()
    net, _ = construct_from_symbolic(poly)
    with pytest.raises(GatedConstructionError, match="budget"):
        evaluate_constructed(net, 1 << 30)
    with pytest.raises(GatedConstructionError, match="numeric range"):
        evaluate_constructed(net, (1 << 31))
    with pytest.raises(GatedConstructionError, match="Input"):
        evaluate_constructed(net, True)
    malformed = (
        replace(net, value_weights=np.array([1., 1.])),
        replace(net, output_weights=np.array([1., 2.])),
        replace(net, gate_bias=np.array([0., 1.])),
        replace(net, gate_weights=np.array([1., -1.], np.float32)),
        replace(net, gate_weights=np.array([np.nan, -1.])),
        replace(net, skip=True),
        replace(net, schema="untrusted-network"),
    )
    for invalid in malformed:
        with pytest.raises(GatedConstructionError):
            recover_from_gated_weights(invalid)


def test_pretrained_polynomial_digest_mismatch_rejected_at_build_time():
    _, poly = _source_recovered()
    with pytest.raises(GatedConstructionError, match="bound neural function"):
        construct_from_symbolic(replace(poly, c=8))
    with pytest.raises(SymbolicRecoveryError, match="degree label"):
        construct_from_symbolic(replace(poly, a=0))


def test_z3_exact_real_case_split_and_wrong_gate_negative_control():
    pytest.importorskip("z3")
    proof = check_real_algebraic_identity()
    assert proof["nonnegative_domain_solver"] == "unsat"
    assert proof["negative_domain_solver"] == "unsat"
    assert proof["negative_control"] == "sat"
    assert proof["all_real_algebraic_identity_smt_checked"]
    assert not proof["float64_all_real_bitwise_proved"]
    assert not proof["wasm_i32_wrap_equivalence_proved"]
    assert not proof["machine_checked_proof_object"]
