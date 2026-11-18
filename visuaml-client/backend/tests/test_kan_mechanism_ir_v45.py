"""v45: tests for compositional KAN/MultKAN mechanism contracts.

Exact rational symbolic identities are limited to polynomial edges. Numerical
agreement does not establish causal relevance or symbolic equivalence for
spline, Fourier and Gaussian RBF bases.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANAdmissionError,
    KANNode,
    KANProgram,
    edge_scalar,
    execute_kan_scalar,
    execute_kan_vector,
    polynomial_identity,
    validate_kan,
)


def quadratic_kan() -> KANProgram:
    return KANProgram(
        ("x",),
        (
            KANNode("squared", "product", (
                EdgeFunction("u", "x", "polynomial", (0., 1.)),
                EdgeFunction("v", "x", "polynomial", (0., 1.)),
            )),
            KANNode("y", "sum", (
                EdgeFunction("quadratic_term", "squared", "polynomial", (0., 1.)),
                EdgeFunction("linear_term", "x", "polynomial", (0., -3.)),
                EdgeFunction("constant_term", "x", "polynomial", (7.,)),
            )),
        ),
        ("y",),
    )


def test_constructed_multkan_polynomial_is_exact_rational_and_scalar():
    p = quadratic_kan()
    validate_kan(p)
    assert len(p.digest()) == 64
    assert polynomial_identity(p, output="y") == (
        Fraction(7), Fraction(-3), Fraction(1)
    )
    for x in range(-128, 128):
        assert execute_kan_scalar(p, {"x": x})["y"] == x*x - 3*x + 7


def test_scalar_numpy_batch_agrees_for_kan_and_multkan():
    p = quadratic_kan()
    values = np.arange(-128, 128, dtype=np.float64)
    out = execute_kan_vector(p, {"x": values})["y"]
    expected = values**2 - 3*values + 7
    np.testing.assert_array_equal(out, expected)
    for i in (-128, -127, -1, 0, 1, 43, 127):
        assert out[i + 128] == execute_kan_scalar(p, {"x": float(i)})["y"]


def test_basis_families_have_explicit_math_and_numeric_parity():
    input_values = np.linspace(-2, 2, 57, dtype=np.float64)
    basis_edges = (
        EdgeFunction("poly", "x", "polynomial", (1., 2., 0.5)),
        EdgeFunction("spline", "x", "linear_spline", (-1., 2., 0.),
                     (-1.5, 0., 1.5)),
        EdgeFunction("rbf", "x", "gaussian_rbf", (1.5, -0.5),
                     (-1., 1., 0.7, 1.1)),
        EdgeFunction("fourier", "x", "fourier", (0.25, 1., -0.5, 0.25, 0.3),
                     (1.2,)),
    )
    for edge in basis_edges:
        p = KANProgram(("x",), (KANNode("out", "sum", (edge,)),),
                       ("out",), (-2., 2.))
        batched = execute_kan_vector(p, {"x": input_values})["out"]
        expected = np.array([execute_kan_scalar(p, {"x": float(v)})["out"]
                             for v in input_values])
        np.testing.assert_allclose(batched, expected, atol=1e-12, rtol=1e-12)
        if edge.basis != "polynomial":
            with pytest.raises(KANAdmissionError, match="Nonpolynomial"):
                polynomial_identity(p, output="out")


def test_exact_polynomial_certificate_rejects_wrong_output():
    p = quadratic_kan()
    broken = replace(p, nodes=(
        p.nodes[0], replace(p.nodes[1], edges=(
            p.nodes[1].edges[0],
            replace(p.nodes[1].edges[1], coefficients=(0., -2.)),
            p.nodes[1].edges[2],
        )),
    ))
    assert polynomial_identity(broken, output="y") == (
        Fraction(7), Fraction(-2), Fraction(1)
    )
    assert polynomial_identity(broken, output="y") != polynomial_identity(p, output="y")


def test_kan_shape_and_effect_safety_guards():
    p = quadratic_kan()
    for bad in (
        replace(p, inputs=("x", "x")),
        replace(p, outputs=("missing",)),
        replace(p, nodes=(replace(p.nodes[0], name="x"), p.nodes[1])),
        replace(p, nodes=(replace(p.nodes[0], edges=(
            replace(p.nodes[0].edges[0], source="y"),
        ) + p.nodes[0].edges[1:]), p.nodes[1])),
        replace(p, domain=(0., 0.)),
        replace(p, schema="untrusted"),
    ):
        with pytest.raises(KANAdmissionError):
            validate_kan(bad)
    with pytest.raises(KANAdmissionError, match="domain"):
        execute_kan_scalar(p, {"x": 200})
    with pytest.raises(KANAdmissionError, match="ABI"):
        execute_kan_scalar(p, {"unknown": 1.})
    with pytest.raises(KANAdmissionError, match="dtype"):
        execute_kan_vector(p, {"x": np.array([1, 2], dtype=np.int64)})


def test_invalid_basis_parameters_fail_closed():
    variants = (
        EdgeFunction("x", "x", "polynomial", (float("nan"),)),
        EdgeFunction("x", "x", "polynomial", tuple(range(8))),
        EdgeFunction("x", "x", "linear_spline", (0., 1., 2.),
                     (0., 2., 1.)),
        EdgeFunction("x", "x", "gaussian_rbf", (1.,), (0., 0.)),
        EdgeFunction("x", "x", "fourier", (0., 1., 1.), (0.,)),
        EdgeFunction("x", "x", "madeup", (0.,)),
    )
    for edge in variants:
        program = KANProgram(("x",), (KANNode("out", "sum", (edge,)),), ("out",))
        with pytest.raises(KANAdmissionError):
            validate_kan(program)


def test_scalar_edges_honor_piecewise_constant_extrapolation():
    e = EdgeFunction("spl", "x", "linear_spline", (-3., 4., -2.),
                     (-1., 0., 2.))
    assert edge_scalar(e, -100.) == -3.
    assert edge_scalar(e, 100.) == -2.
    assert edge_scalar(e, 0.) == 4.
    assert edge_scalar(e, 1.) == 1.
