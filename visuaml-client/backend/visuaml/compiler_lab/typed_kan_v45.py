"""v45 bounded typed Kolmogorov-Arnold / MultKAN composition IR.

Research semantics: a stage is a set of nodes, each aggregating independently
parameterized univariate functions of preceding-stage outputs. Nodes may sum
(edge-KAN) or multiply (restricted MultKAN interaction). The five admitted edge
bases are power, Chebyshev, Fourier, Gaussian RBF and linear B-spline hats.

This is architecture and execution infrastructure, NOT a claim that arbitrary
pretrained model circuits are identifiable, that KANs always beat MLPs, or
that symbolic labels imply causal evidence. The numeric contract is float64,
bounded-input, immutable and free of external callbacks.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

SCHEMA = "visuaml.typed-kan-program.v45"
BasisFamily = Literal["power", "chebyshev", "fourier", "gaussian", "linear_spline"]
Aggregation = Literal["sum", "product"]
MAX_WIDTH = 8
MAX_STAGES = 4
MAX_EDGES = 64
MAX_BASIS = 12
MAX_ROWS = 4096


class KANAdmissionError(ValueError):
    """Malformed basis, tensor, topology, input range, or output magnitude."""


@dataclass(frozen=True)
class KANEdge:
    source: int
    family: BasisFamily
    coefficients: tuple[float, ...]
    domain: tuple[float, float]
    knots: tuple[float, ...] = ()
    scale: float = 1.0


@dataclass(frozen=True)
class KANNode:
    edges: tuple[KANEdge, ...]
    aggregation: Aggregation = "sum"
    bias: float = 0.0


@dataclass(frozen=True)
class KANProgram:
    input_dim: int
    stages: tuple[tuple[KANNode, ...], ...]
    schema: str = SCHEMA

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()


def _finite_float(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e9


def validate_kan(program: KANProgram) -> dict:
    if (not isinstance(program, KANProgram) or program.schema != SCHEMA
            or type(program.input_dim) is not int
            or not 1 <= program.input_dim <= MAX_WIDTH
            or not isinstance(program.stages, tuple)
            or not 1 <= len(program.stages) <= MAX_STAGES):
        raise KANAdmissionError("Invalid bounded typed KAN schema")
    previous = program.input_dim
    edges = 0
    basis_families = set()
    for stage_index, stage in enumerate(program.stages):
        if (not isinstance(stage, tuple) or not 1 <= len(stage) <= MAX_WIDTH):
            raise KANAdmissionError("Invalid stage width")
        for node in stage:
            if (not isinstance(node, KANNode) or node.aggregation not in ("sum", "product")
                    or not isinstance(node.edges, tuple) or not 1 <= len(node.edges) <= MAX_WIDTH
                    or not _finite_float(node.bias)):
                raise KANAdmissionError("Invalid typed KAN node")
            edges += len(node.edges)
            if edges > MAX_EDGES:
                raise KANAdmissionError("KAN edge budget exceeded")
            for edge in node.edges:
                if (not isinstance(edge, KANEdge) or type(edge.source) is not int
                        or not 0 <= edge.source < previous):
                    raise KANAdmissionError("Invalid KAN edge source index")
                if edge.family not in (
                    "power", "chebyshev", "fourier", "gaussian", "linear_spline"
                ):
                    raise KANAdmissionError("Unrecognized univariate edge basis")
                if (not isinstance(edge.coefficients, tuple)
                        or not 1 <= len(edge.coefficients) <= MAX_BASIS
                        or not all(_finite_float(x) for x in edge.coefficients)):
                    raise KANAdmissionError("Invalid bounded basis coefficients")
                if (not isinstance(edge.domain, tuple) or len(edge.domain) != 2
                        or not all(_finite_float(x) for x in edge.domain)
                        or not edge.domain[0] < edge.domain[1]):
                    raise KANAdmissionError("Invalid declared basis input domain")
                if not _finite_float(edge.scale) or not 0 < edge.scale <= 1e6:
                    raise KANAdmissionError("Invalid basis scale / period")
                if not isinstance(edge.knots, tuple):
                    raise KANAdmissionError("Invalid knots")
                if edge.family == "fourier":
                    if edge.knots or len(edge.coefficients) % 2 != 1:
                        raise KANAdmissionError("Fourier needs a0, cosine/sine pairs, no knots")
                elif edge.family in ("gaussian", "linear_spline"):
                    if (len(edge.knots) != len(edge.coefficients)
                            or not 2 <= len(edge.knots) <= MAX_BASIS
                            or not all(_finite_float(k) for k in edge.knots)
                            or not all(a < b for a, b in zip(edge.knots, edge.knots[1:], strict=False))):
                        raise KANAdmissionError("Ordered basis knots/centers required")
                    if (edge.family == "linear_spline"
                            and (edge.domain[0] < edge.knots[0]
                                 or edge.domain[1] > edge.knots[-1])):
                        raise KANAdmissionError("Spline edge domain outside knots")
                elif edge.knots:
                    raise KANAdmissionError("Polynomial or Chebyshev may not use knots")
                basis_families.add(edge.family)
        previous = len(stage)
        if stage_index >= MAX_STAGES:
            raise KANAdmissionError("Excessive KAN depth")
    return {
        "schema": SCHEMA, "stages": len(program.stages), "edges": edges,
        "basis_families": sorted(basis_families),
        "output_dim": previous, "input_dim": program.input_dim,
        "has_multiplication_nodes": any(
            node.aggregation == "product" for stage in program.stages for node in stage
        ),
        "program_sha256": program.fingerprint(),
    }


def _basis_scalar(edge: KANEdge, value: float) -> float:
    if not math.isfinite(value) or not edge.domain[0] <= value <= edge.domain[1]:
        raise KANAdmissionError("KAN basis input outside declared domain")
    c = edge.coefficients
    if edge.family == "power":
        result = 0.0
        for coefficient in reversed(c):
            result = result * value + coefficient
    elif edge.family == "chebyshev":
        u = (2.0 * value - (edge.domain[0] + edge.domain[1])) / (
            edge.domain[1] - edge.domain[0]
        )
        t0, t1 = 1.0, u
        result = c[0]
        if len(c) >= 2:
            result += c[1] * t1
        for coefficient in c[2:]:
            t0, t1 = t1, 2.0 * u * t1 - t0
            result += coefficient * t1
    elif edge.family == "fourier":
        theta = 2.0 * math.pi * (value - edge.domain[0]) / edge.scale
        result = c[0]
        for index in range(1, (len(c) + 1) // 2):
            result += (c[2 * index - 1] * math.cos(index * theta)
                       + c[2 * index] * math.sin(index * theta))
    elif edge.family == "gaussian":
        result = sum(
            coefficient * math.exp(-((value - center) / edge.scale) ** 2)
            for coefficient, center in zip(c, edge.knots, strict=True)
        )
    else:
        # Degree-one B-spline interpolation; no arbitrary extrapolation.
        index = min(max(0, int(np.searchsorted(edge.knots, value) - 1)),
                    len(edge.knots) - 2)
        start, end = edge.knots[index], edge.knots[index + 1]
        t = (value - start) / (end - start)
        result = (1 - t) * c[index] + t * c[index + 1]
    if not math.isfinite(result) or abs(result) > 1e12:
        raise KANAdmissionError("KAN basis overflow")
    return float(result)


def evaluate_scalar(program: KANProgram, inputs: tuple[float, ...]) -> tuple[float, ...]:
    validate_kan(program)
    if (not isinstance(inputs, tuple) or len(inputs) != program.input_dim
            or not all(_finite_float(x) for x in inputs)):
        raise KANAdmissionError("Invalid scalar inputs")
    previous = tuple(float(value) for value in inputs)
    for stage in program.stages:
        current = []
        for node in stage:
            values = [_basis_scalar(edge, previous[edge.source]) for edge in node.edges]
            if node.aggregation == "sum":
                result = float(node.bias + math.fsum(values))
            else:
                result = float(node.bias + math.prod(values))
            if not math.isfinite(result) or abs(result) > 1e12:
                raise KANAdmissionError("KAN composed output overflow")
            current.append(result)
        previous = tuple(current)
    return previous


def _basis_numpy(edge: KANEdge, x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all() or not ((edge.domain[0] <= x) & (x <= edge.domain[1])).all():
        raise KANAdmissionError("KAN basis input outside declared domain")
    c = edge.coefficients
    if edge.family == "power":
        result = np.zeros_like(x, dtype=np.float64)
        for coefficient in reversed(c):
            result = result * x + coefficient
    elif edge.family == "chebyshev":
        u = (2 * x - (edge.domain[0] + edge.domain[1])) / (
            edge.domain[1] - edge.domain[0]
        )
        t0, t1 = np.ones_like(u), u
        result = np.full_like(x, c[0], dtype=np.float64)
        if len(c) >= 2:
            result = result + c[1] * t1
        for coefficient in c[2:]:
            t0, t1 = t1, 2 * u * t1 - t0
            result = result + coefficient * t1
    elif edge.family == "fourier":
        theta = 2 * np.pi * (x - edge.domain[0]) / edge.scale
        result = np.full_like(x, c[0], dtype=np.float64)
        for i in range(1, (len(c) + 1) // 2):
            result = result + c[2 * i - 1] * np.cos(i * theta) + c[2 * i] * np.sin(i * theta)
    elif edge.family == "gaussian":
        result = np.zeros_like(x)
        for coefficient, center in zip(c, edge.knots, strict=True):
            result = result + coefficient * np.exp(-((x - center) / edge.scale) ** 2)
    else:
        result = np.interp(x, np.array(edge.knots), np.array(c)).astype(np.float64)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e12:
        raise KANAdmissionError("Nonfinite or out-of-budget basis result")
    return result


def evaluate_numpy(program: KANProgram, data: np.ndarray) -> np.ndarray:
    """Vectorized explicit edge evaluation (no implicit framework training)."""
    validate_kan(program)
    if (not isinstance(data, np.ndarray) or data.dtype != np.float64
            or data.ndim != 2 or data.shape[1] != program.input_dim
            or not 1 <= data.shape[0] <= MAX_ROWS
            or not np.isfinite(data).all()):
        raise KANAdmissionError("Input must be bounded finite float64 [B,features]")
    prev = data.copy()
    for stage in program.stages:
        out = np.empty((len(data), len(stage)), dtype=np.float64)
        for index, node in enumerate(stage):
            edges = [_basis_numpy(e, prev[:, e.source]) for e in node.edges]
            if node.aggregation == "sum":
                # Explicit order makes subsequent independent parity meaningful.
                value = np.full(len(data), node.bias)
                for e in edges:
                    value = value + e
            else:
                value = np.full(len(data), node.bias) + np.prod(np.stack(edges), axis=0)
            if not np.isfinite(value).all() or np.max(np.abs(value)) > 1e12:
                raise KANAdmissionError("Composed KAN output outside numeric budget")
            out[:, index] = value
        prev = out
    return prev


def recover_polynomial(program: KANProgram):
    """Exact symbolic polynomial extraction for power/Chebyshev-only KANs.

    Rational.from_float captures the *actual float64 binary coefficients*.
    The full graph is structurally traversed; product nodes are expanded.
    RBF/Fourier/spline structures are correctly rejected as NOT certified
    polynomials, even if a few samples happen to look polynomial.
    """
    validate_kan(program)
    if any(edge.family not in ("power", "chebyshev")
           for stage in program.stages for node in stage for edge in node.edges):
        raise KANAdmissionError("Symbolic polynomial extraction only admits power/Chebyshev")
    import sympy as sp

    inputs = tuple(sp.Symbol(f"x{i}", real=True) for i in range(program.input_dim))
    previous = list(inputs)
    for stage in program.stages:
        current = []
        for node in stage:
            terms = []
            for edge in node.edges:
                variable = previous[edge.source]
                if edge.family == "power":
                    expr = sum(sp.Rational(float(c)) * variable**i
                               for i, c in enumerate(edge.coefficients))
                else:
                    lo, hi = (sp.Rational(float(v)) for v in edge.domain)
                    u = (2 * variable - lo - hi) / (hi - lo)
                    expr = sum(sp.Rational(float(c)) * sp.chebyshevt(i, u)
                               for i, c in enumerate(edge.coefficients))
                terms.append(expr)
            bias = sp.Rational(float(node.bias))
            expr = bias + (sp.Add(*terms) if node.aggregation == "sum" else sp.Mul(*terms))
            expr = sp.expand(expr)
            if sp.count_ops(expr) > 2000 or any(sp.degree(expr, x) > 32 for x in inputs):
                raise KANAdmissionError("Symbolic KAN complexity budget exceeded")
            current.append(expr)
        previous = current
    return inputs, tuple(previous)


def from_recovered_v43_polynomial(poly) -> KANProgram:
    """Construct a power-edge KAN from v43 typed symbolic coefficients.

    Produces a new edge-function representation; does not silently reinterpret
    real-valued evaluation as i32 modular execution on all inputs.
    """
    from .neural_symbolic_recovery_v43 import (
        RecoveredPolynomial,
        compile_polynomial_wasm,
    )
    if not isinstance(poly, RecoveredPolynomial):
        raise KANAdmissionError("Requires typed recovered polynomial")
    compile_polynomial_wasm(poly)
    edge = KANEdge(
        0, "power", (float(poly.c), float(poly.b), float(poly.a)),
        (float(LOW), float(HIGH)),
    )
    graph = KANProgram(1, ((KANNode((edge,)),),))
    validate_kan(graph)
    return graph


LOW, HIGH = -128, 127
