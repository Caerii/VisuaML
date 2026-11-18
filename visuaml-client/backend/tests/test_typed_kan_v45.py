"""Adversarial and independent oracles for typed KAN / MultKAN v45.

Explicitly tests five different edge bases, product composition, polynomial
extraction, gradients and native C++17 emission. These are not accuracy,
causal-interpretability, pretrained-transfer, or throughput claims.
"""
from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pytest
from visuaml.compiler_lab.native_kan_codegen_v45 import (
    emit_kan_cpp17,
    run_kan_cpp17,
)
from visuaml.compiler_lab.torch_kan_v45 import TorchKAN
from visuaml.compiler_lab.typed_kan_v45 import (
    KANAdmissionError,
    KANEdge,
    KANNode,
    KANProgram,
    ablate_kan_edge,
    ablation_witness,
    evaluate_numpy,
    evaluate_scalar,
    from_recovered_v43_polynomial,
    recover_polynomial,
    validate_kan,
)


def edge(source, family, coefficients, *, domain=(-1.0, 1.0), knots=(), scale=1.0):
    return KANEdge(source, family, tuple(coefficients), domain, tuple(knots), scale)


def single(family, coefficients, **kwargs):
    return KANProgram(1, ((KANNode((edge(0, family, coefficients, **kwargs),)),),))


def examples():
    return {
        "power": single("power", (0.5, -1.0, 2.0)),
        "chebyshev": single("chebyshev", (0.5, 0.0, 0.5)),
        "fourier": single("fourier", (0.1, 0.8, -0.4, 0.2, -0.3), scale=2.0),
        "gaussian": single(
            "gaussian", (0.7, -0.4, 0.2),
            knots=(-0.6, 0.1, 0.7), scale=0.35,
        ),
        "linear_spline": single(
            "linear_spline", (0.5, -0.75, 1.0),
            knots=(-1.0, 0.0, 1.0),
        ),
    }


@pytest.mark.parametrize("kind", [
    "power", "chebyshev", "fourier", "gaussian", "linear_spline"
])
def test_typed_kan_five_distinct_basis_oracles_agree(kind):
    import torch

    graph = examples()[kind]
    assert validate_kan(graph)["basis_families"] == [kind]
    values = np.linspace(-1, 1, 129, dtype=np.float64)[:, None]
    scalar = np.array([
        evaluate_scalar(graph, (float(row[0]),)) for row in values
    ], dtype=np.float64)
    vectorized = evaluate_numpy(graph, values)
    neural = TorchKAN(graph)(torch.tensor(values, dtype=torch.float64))
    torch_values = neural.detach().numpy()
    assert vectorized.shape == (129, 1)
    np.testing.assert_allclose(vectorized, scalar, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(torch_values, scalar, atol=1e-11, rtol=1e-11)
    assert len(graph.fingerprint()) == 64
    assert graph.fingerprint() == validate_kan(graph)["program_sha256"]


@pytest.mark.parametrize("kind", [
    "power", "chebyshev", "fourier", "gaussian", "linear_spline"
])
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")),
                    reason="C++17 unavailable")
def test_native_cpp17_from_same_typed_kan_graph_matches_independent_oracles(kind):
    import torch

    graph = examples()[kind]
    input_data = np.linspace(-1, 1, 85).astype(np.float64)[:, None]
    observed, receipt = run_kan_cpp17(graph, input_data)
    numpy_result = evaluate_numpy(graph, input_data)
    scalar = np.array([
        evaluate_scalar(graph, (float(x),)) for x in input_data[:, 0]
    ])
    other = TorchKAN(graph)(torch.tensor(input_data)).detach().numpy()
    for expected in (numpy_result, scalar, other):
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    assert receipt["native_execution"]
    assert receipt["output_shape"] == [85, 1]
    assert not receipt["numerical_parity_verified"]
    assert not receipt["performance_speedup_proved"]
    assert receipt["cpp_source_sha256"] == hashlib.sha256(
        emit_kan_cpp17(graph).encode()
    ).hexdigest()
    assert len(receipt["native_binary_sha256"]) == 64


def test_chebyshev_recovers_exact_symbolic_square_and_spline_rejects_symbolic():
    import sympy as sp

    cheby = examples()["chebyshev"]
    variables, recovered = recover_polynomial(cheby)
    assert len(variables) == 1 and len(recovered) == 1
    assert sp.expand(recovered[0] - variables[0] ** 2) == 0
    power = examples()["power"]
    _, expression = recover_polynomial(power)
    assert expression[0] == sp.Rational(1, 2) - variables[0] + 2 * variables[0] ** 2
    for other in ("fourier", "gaussian", "linear_spline"):
        with pytest.raises(KANAdmissionError, match="only admits"):
            recover_polynomial(examples()[other])


def xy_multkan():
    """Two layers, explicitly multiplied edge-transformed subexpressions."""
    return KANProgram(
        2,
        (
            (
                KANNode((edge(0, "power", (0.0, 1.0)),)),
                KANNode((edge(1, "power", (0.0, 1.0)),)),
            ),
            (
                KANNode(
                    (
                        edge(0, "power", (0.0, 1.0)),
                        edge(1, "power", (0.0, 1.0)),
                    ),
                    aggregation="product",
                    bias=0.0,
                ),
            ),
        ),
    )


def test_multkan_product_is_a_genuinely_compositional_mechanism():
    import sympy as sp
    import torch

    graph = xy_multkan()
    report = validate_kan(graph)
    assert report["has_multiplication_nodes"]
    assert report["stages"] == 2 and report["edges"] == 4
    rng = np.random.default_rng(45)
    data = rng.uniform(-1, 1, (111, 2)).astype(np.float64)
    expected = (data[:, 0] * data[:, 1])[:, None]
    np.testing.assert_allclose(evaluate_numpy(graph, data), expected, atol=1e-12)
    for x, y in data[:30]:
        assert abs(evaluate_scalar(graph, (float(x), float(y)))[0] - x * y) < 1e-12
    observed_torch = TorchKAN(graph)(torch.tensor(data)).detach().numpy()
    np.testing.assert_allclose(observed_torch, expected, atol=1e-12)
    variables, expression = recover_polynomial(graph)
    assert sp.expand(expression[0] - variables[0] * variables[1]) == 0
    if shutil.which("g++") or shutil.which("clang++"):
        observed_cpp, _ = run_kan_cpp17(graph, data)
        np.testing.assert_allclose(observed_cpp, expected, atol=1e-12)


def test_import_v43_symbolic_polynomial_as_constructed_kan_edge():
    import sympy as sp

    from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
        AnalyticRelu,
        compile_bounded_program,
    )
    from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
        recover_quadratic_from_weights,
    )
    from visuaml.compiler_lab.wasm_state_vm_v40 import I32Program, VMOperation

    def op(name, value=None):
        return VMOperation(name, value)
    old_wasm = I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))
    original_net, _ = compile_bounded_program(old_wasm)
    detached = AnalyticRelu(
        original_net.w_in, original_net.b_in,
        original_net.w_out, original_net.b_out,
    )
    poly = recover_quadratic_from_weights(detached)
    graph = from_recovered_v43_polynomial(poly)
    assert graph.stages[0][0].edges[0].coefficients == (7.0, -3.0, 1.0)
    variables, expressions = recover_polynomial(graph)
    assert sp.expand(expressions[0] -
                     (variables[0] ** 2 - 3 * variables[0] + 7)) == 0
    values = np.arange(-128, 128, dtype=np.float64)[:, None]
    expected = (values[:, 0] ** 2 - 3 * values[:, 0] + 7)[:, None]
    np.testing.assert_array_equal(evaluate_numpy(graph, values), expected)
    if shutil.which("g++") or shutil.which("clang++"):
        result, _ = run_kan_cpp17(graph, values)
        np.testing.assert_array_equal(result, expected)
    with pytest.raises(KANAdmissionError, match="outside declared domain"):
        evaluate_scalar(graph, (128.0,))


def test_kan_torch_parameters_support_gradient_updates_without_hiding_learning():
    import torch

    graph = single("power", (0., 0., 0.))
    model = TorchKAN(graph, trainable=True)
    x = torch.linspace(-1, 1, 60, dtype=torch.float64)[:, None]
    y = 0.5 - x + 2 * x * x
    before = model(x).detach().numpy()
    optim = torch.optim.Adam(model.parameters(), lr=0.12)
    for _ in range(130):
        optim.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optim.step()
    learned = model(x).detach().numpy()
    assert np.mean((learned - y.numpy()) ** 2) < 2e-5
    assert np.mean((before - y.numpy()) ** 2) > 0.1
    # The typed graph is immutable until explicitly re-exported. A trained
    # parameter tensor cannot silently reuse the original source hash.
    assert graph.stages[0][0].edges[0].coefficients == (0.0, 0.0, 0.0)


def test_reject_out_of_domain_bases_topology_and_nonfinite_parameters():
    base = examples()["power"]
    bad_graphs = (
        KANProgram(0, base.stages),
        KANProgram(9, base.stages),
        KANProgram(1, ((),)),
        single("fourier", (0., 1.)),
        single("power", (float("nan"),)),
        single("chebyshev", (1.,), knots=(0.,)),
        single("linear_spline", (0., 1.), knots=(0., 0.)),
        single("gaussian", (0., 1.), knots=(0., -1.)),
        single("linear_spline", (0., 1.), knots=(-0.5, 0.5)),
        KANProgram(1, ((KANNode((edge(2, "power", (1.,)),)),),)),
    )
    for graph in bad_graphs:
        with pytest.raises(KANAdmissionError):
            validate_kan(graph)
    data = np.array([[1.1]], dtype=np.float64)
    for call in (
        lambda: evaluate_numpy(base, data),
        lambda: evaluate_scalar(base, (1.1,)),
        lambda: TorchKAN(base)(__import__("torch").tensor(data)),
    ):
        with pytest.raises(KANAdmissionError, match="outside declared domain"):
            call()
    with pytest.raises(KANAdmissionError):
        evaluate_numpy(base, np.ones((4097, 1), dtype=np.float64))
    with pytest.raises(KANAdmissionError):
        evaluate_numpy(base, np.ones((1, 1), dtype=np.float32))
    if shutil.which("g++") or shutil.which("clang++"):
        with pytest.raises(KANAdmissionError):
            run_kan_cpp17(base, data)


def test_edge_intervention_preserves_typed_graph_and_measures_counterfactual():
    graph = xy_multkan()
    input_data = np.array([[0.25, 0.5], [-0.5, 0.75], [1.0, -1.0]], np.float64)
    before = evaluate_numpy(graph, input_data)
    assert np.any(before != 0)
    changed, certificate = ablate_kan_edge(
        graph, stage_index=1, node_index=0, edge_index=0
    )
    assert certificate["source_sha256"] == graph.fingerprint()
    assert certificate["target_sha256"] == changed.fingerprint()
    assert certificate["source_sha256"] != certificate["target_sha256"]
    assert certificate["causal_in_this_graph_only"] is True
    assert certificate["external_pretrained_mechanism_certified"] is False
    np.testing.assert_array_equal(evaluate_numpy(changed, input_data), np.zeros((3, 1)))
    result = ablation_witness(
        graph, input_data, stage_index=1, node_index=0, edge_index=0
    )
    assert result["witness_inputs"] == 3
    assert result["maximum_absolute_output_intervention"] == 1
    assert result["full_pretrained_model_intervention_executed"] is False
    assert result["counterfactual_executed"] is True
    with pytest.raises(KANAdmissionError):
        ablate_kan_edge(graph, 1, 0, 2)
    with pytest.raises(KANAdmissionError):
        ablate_kan_edge(graph, True, 0, 0)
    with pytest.raises(KANAdmissionError):
        ablate_kan_edge(changed, 1, 0, 0)
