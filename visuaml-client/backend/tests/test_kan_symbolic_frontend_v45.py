"""v45: construct verifiable modular KAN from previously extracted neural symbol."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    AnalyticRelu,
    compile_bounded_program,
)
from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    KANAdmissionError,
    execute_kan_scalar,
    execute_kan_vector,
    polynomial_identity,
)
from visuaml.compiler_lab.kan_symbolic_frontend_v45 import (
    ablate_edge,
    causal_edge_witnesses,
    quadratic_to_kan,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import I32Program, VMOperation


def _symbolic():
    def op(name, imm=None):
        return VMOperation(name, imm)
    program = I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))
    old, _ = compile_bounded_program(program)
    detached = AnalyticRelu(
        old.w_in.copy(), old.b_in.copy(),
        old.w_out.copy(), old.b_out.copy()
    )
    return recover_quadratic_from_weights(detached)


def test_construct_symbolic_multkan_not_just_a_densely_fitted_network():
    poly = _symbolic()
    constructed = quadratic_to_kan(poly)
    assert constructed.exact_symbolic_construction
    assert constructed.scalar_witnesses == 256
    assert constructed.source_symbol_sha256 == poly.digest()
    assert constructed.graph_sha256 == constructed.program.digest()
    assert constructed.symbolic_coefficients == (
        Fraction(7), Fraction(-3), Fraction(1)
    )
    assert constructed.program.nodes[0].combine == "product"
    assert constructed.program.nodes[1].combine == "sum"
    for x in range(-128, 128):
        assert execute_kan_scalar(
            constructed.program, {"x": x}
        )["result"] == x*x - 3*x + 7


def test_construction_numerically_agrees_with_separate_vector_kernel():
    model = quadratic_to_kan(_symbolic()).program
    rng = np.random.default_rng(45)
    points = rng.uniform(-128, 127, size=400).astype(np.float64)
    batch = execute_kan_vector(model, {"x": points})["result"]
    scalar = np.array([
        execute_kan_scalar(model, {"x": float(x)})["result"]
        for x in points
    ])
    np.testing.assert_allclose(batch, scalar, atol=1e-10, rtol=1e-12)


def test_edge_interventions_explain_explicit_constructed_terms():
    model = quadratic_to_kan(_symbolic()).program
    samples = (-5, -1, 0, 1, 5)
    record = causal_edge_witnesses(model, samples)
    assert record["sample_count"] == 5
    assert not record["pretrained_mechanism_causality_proved"]
    assert set(record["edge_interventions"]) == {
        "identity_left", "identity_right", "quadratic",
        "linear", "constant",
    }
    assert record["edge_interventions"]["quadratic"]["max_abs_output_change"] == 25
    assert record["edge_interventions"]["linear"]["max_abs_output_change"] == 15
    assert record["edge_interventions"]["constant"]["max_abs_output_change"] == 7
    assert record["edge_interventions"]["quadratic"]["changed_sample_count"] == 4
    assert record["edge_interventions"]["constant"]["changed_sample_count"] == 5
    assert polynomial_identity(ablate_edge(
        model, "quadratic"
    ), output="result") == (Fraction(7), Fraction(-3))
    assert polynomial_identity(ablate_edge(
        model, "linear"
    ), output="result") == (Fraction(7), Fraction(0), Fraction(1))
    assert polynomial_identity(ablate_edge(
        model, "constant"
    ), output="result") == (Fraction(0), Fraction(-3), Fraction(1))


def test_reject_unknown_edge_and_inconsistent_symbolic_certificate():
    model = quadratic_to_kan(_symbolic()).program
    with pytest.raises(KANAdmissionError, match="existing"):
        ablate_edge(model, "nonexistent")
    poly = _symbolic()
    with pytest.raises(KANAdmissionError, match="identity"):
        # A symbolic digest must match the recovered finite-domain function;
        # the source polynomial compiler currently checks declared provenance
        # separately from the KAN compiler's structural identity.
        quadratic_to_kan(replace(poly, c=8))
    with pytest.raises(KANAdmissionError, match="intervention"):
        causal_edge_witnesses(model, (1, 1, 2))
