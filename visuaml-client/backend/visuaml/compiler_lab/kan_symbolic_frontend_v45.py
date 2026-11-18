"""v45: compile verified symbolic quadratic to modular KAN/MultKAN graph.

A named product node performs the square, three explicit univariate edges
compose the recovered polynomial, and all coefficient identities are checked
with exact rational arithmetic. No fitted weights or learned feature labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .analytic_relu_specialization_v41 import function_hash
from .kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANAdmissionError,
    KANNode,
    KANProgram,
    execute_kan_scalar,
    polynomial_identity,
    validate_kan,
)
from .neural_symbolic_recovery_v43 import (
    RecoveredPolynomial,
    SymbolicRecoveryError,
    compile_polynomial_wasm,
)


@dataclass(frozen=True)
class KANLowering:
    program: KANProgram
    symbolic_coefficients: tuple[Fraction, ...]
    source_symbol_sha256: str
    graph_sha256: str
    scalar_witnesses: int
    exact_symbolic_construction: bool


def quadratic_to_kan(poly: RecoveredPolynomial) -> KANLowering:
    """No source WASM or v41 weight archive is consulted in this compiler."""
    try:
        compile_polynomial_wasm(poly)
    except SymbolicRecoveryError as exc:
        raise KANAdmissionError("Unadmitted recovered polynomial") from exc
    domain_outputs = tuple(
        poly.a * x * x + poly.b * x + poly.c
        for x in range(int(poly.domain[0]), int(poly.domain[1]) + 1)
    )
    if function_hash(domain_outputs) != poly.recovered_function_sha256:
        raise KANAdmissionError("Recovered polynomial/function identity mismatch")
    program = KANProgram(
        inputs=("x",),
        nodes=(
            KANNode("square", "product", (
                EdgeFunction("identity_left", "x", "polynomial", (0., 1.)),
                EdgeFunction("identity_right", "x", "polynomial", (0., 1.)),
            )),
            KANNode("result", "sum", (
                EdgeFunction("quadratic", "square", "polynomial",
                             (0., float(poly.a))),
                EdgeFunction("linear", "x", "polynomial",
                             (0., float(poly.b))),
                EdgeFunction("constant", "x", "polynomial",
                             (float(poly.c),)),
            )),
        ),
        outputs=("result",),
        domain=(float(poly.domain[0]), float(poly.domain[1])),
    )
    validate_kan(program)
    symbolic = polynomial_identity(program, output="result")
    expected = (Fraction(poly.c), Fraction(poly.b), Fraction(poly.a))
    while len(expected) > 1 and expected[-1] == 0:
        expected = expected[:-1]
    if symbolic != expected:
        raise KANAdmissionError("Constructed MultKAN symbolic identity mismatch")
    for x in range(int(poly.domain[0]), int(poly.domain[1]) + 1):
        observed = execute_kan_scalar(program, {"x": float(x)})["result"]
        expected_value = poly.a * x * x + poly.b * x + poly.c
        if not np.isfinite(observed) or observed != expected_value:
            raise KANAdmissionError("Constructed KAN finite-domain witness mismatch")
    return KANLowering(
        program, symbolic, poly.digest(), program.digest(),
        int(poly.domain[1] - poly.domain[0]) + 1,
        True,
    )


def ablate_edge(program: KANProgram, edge_name: str) -> KANProgram:
    """Counterfactual: remove exactly one edge; zero product edges if needed."""
    from dataclasses import replace

    validate_kan(program)
    found = 0
    nodes = []
    for node in program.nodes:
        edges = []
        for edge in node.edges:
            if edge.name == edge_name:
                found += 1
                # Replacing by a constant-zero *univariate function* keeps
                # all sources and node fanin intact (and product semantics).
                edges.append(replace(
                    edge, basis="polynomial", coefficients=(0.,), parameters=()
                ))
            else:
                edges.append(edge)
        nodes.append(replace(node, edges=tuple(edges)))
    if found != 1:
        raise KANAdmissionError("Ablation requires one exact existing KAN edge")
    mutated = replace(program, nodes=tuple(nodes))
    validate_kan(mutated)
    return mutated


def causal_edge_witnesses(program: KANProgram, samples: tuple[int, ...]) -> dict:
    """Interventions on the KAN itself, not evidence of a pretrained model circuit."""
    validate_kan(program)
    if (not isinstance(samples, tuple) or not 3 <= len(samples) <= 2048
            or len(set(samples)) != len(samples)
            or any(type(x) is not int or not program.domain[0] <= x <= program.domain[1]
                   for x in samples)):
        raise KANAdmissionError("Invalid bounded intervention inputs")
    baseline = np.array(
        [execute_kan_scalar(program, {program.inputs[0]: v})[program.outputs[0]]
         for v in samples], dtype=np.float64
    )
    per_edge = {}
    for node in program.nodes:
        for edge in node.edges:
            intervened = ablate_edge(program, edge.name)
            values = np.array(
                [execute_kan_scalar(
                    intervened, {program.inputs[0]: v}
                )[program.outputs[0]] for v in samples], dtype=np.float64
            )
            diff = baseline - values
            per_edge[edge.name] = {
                "intervention_program_sha256": intervened.digest(),
                "max_abs_output_change": float(np.max(np.abs(diff))),
                "mean_abs_output_change": float(np.mean(np.abs(diff))),
                "changed_sample_count": int(np.count_nonzero(diff)),
                "source_edge": edge.source,
                "edge_basis": edge.basis,
            }
    data = np.asarray(samples, dtype="<i8").tobytes()
    return {
        "schema": "visuaml.kan-edge-intervention.v45",
        "program_sha256": program.digest(),
        "inputs_sha256": hashlib.sha256(data).hexdigest(),
        "sample_count": len(samples),
        "edge_interventions": per_edge,
        "causality_scope": "controlled synthetic constructed KAN only",
        "pretrained_mechanism_causality_proved": False,
    }
