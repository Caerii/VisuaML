"""v45 typed, fail-closed KAN edge-function / MultKAN compositional research IR.

This is a small transparent dialect, not a general pykan checkpoint parser.
An edge is a bounded univariate function of an existing graph value.
Nodes combine incoming edge signals by SUM or PRODUCT (MultKAN-style).
Supported edge bases: monomial polynomial, piecewise-linear spline,
Gaussian RBF, Fourier. Only polynomial edges admit exact rational symbolic
normalization; transcendental edges get numerical witnesses, not proof.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction

SCHEMA = "visuaml.kan-mechanism-ir.v45"
MAX_INPUTS = 8
MAX_NODES = 64
MAX_EDGES = 128
MAX_DEGREE = 4
MAX_COEFF = 1_000_000
MAX_MAGNITUDE = 1e12


class KANAdmissionError(ValueError):
    """Out-of-domain, malformed or ambiguous analytic edge composition."""


@dataclass(frozen=True)
class EdgeFunction:
    name: str
    source: str
    basis: str
    coefficients: tuple[float, ...]
    parameters: tuple[float, ...] = ()


@dataclass(frozen=True)
class KANNode:
    name: str
    combine: str
    edges: tuple[EdgeFunction, ...]


@dataclass(frozen=True)
class KANProgram:
    inputs: tuple[str, ...]
    nodes: tuple[KANNode, ...]
    outputs: tuple[str, ...]
    domain: tuple[float, float] = (-128.0, 127.0)
    schema: str = SCHEMA

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()


def _numbers(values: tuple[float, ...], *, maximum: int) -> None:
    if (not isinstance(values, tuple) or len(values) > maximum
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or abs(v) > MAX_COEFF for v in values)):
        raise KANAdmissionError("Invalid finite coefficient/parameter tuple")


def validate_kan(program: KANProgram) -> None:
    if (not isinstance(program, KANProgram) or program.schema != SCHEMA
            or not isinstance(program.inputs, tuple)
            or not 1 <= len(program.inputs) <= MAX_INPUTS
            or not isinstance(program.nodes, tuple)
            or not 1 <= len(program.nodes) <= MAX_NODES
            or not isinstance(program.outputs, tuple)
            or not 1 <= len(program.outputs) <= 8):
        raise KANAdmissionError("Invalid bounded KAN program")
    lo, hi = program.domain
    if (type(lo) not in (float, int) or type(hi) not in (float, int)
            or not math.isfinite(lo) or not math.isfinite(hi)
            or not -MAX_MAGNITUDE <= lo < hi <= MAX_MAGNITUDE):
        raise KANAdmissionError("Invalid declared input domain")
    values = set()
    for x in program.inputs:
        if not isinstance(x, str) or not x.isidentifier() or x in values:
            raise KANAdmissionError("Repeated/malformed KAN input")
        values.add(x)
    edge_names: set[str] = set()
    count = 0
    for node in program.nodes:
        if (not isinstance(node, KANNode) or not node.name.isidentifier()
                or node.name in values or node.combine not in ("sum", "product")
                or not isinstance(node.edges, tuple) or not 1 <= len(node.edges) <= 8):
            raise KANAdmissionError("Malformed, cyclic or repeated KAN node")
        for edge in node.edges:
            if (not isinstance(edge, EdgeFunction) or not edge.name.isidentifier()
                    or edge.name in edge_names or edge.source not in values):
                raise KANAdmissionError("Unknown KAN source or repeated edge")
            edge_names.add(edge.name)
            _numbers(edge.coefficients, maximum=32)
            _numbers(edge.parameters, maximum=32)
            if edge.basis == "polynomial":
                if (not 1 <= len(edge.coefficients) <= MAX_DEGREE + 1
                        or edge.parameters):
                    raise KANAdmissionError("Polynomial must have degree <=4")
            elif edge.basis == "linear_spline":
                # B-spline degree 1 represented by ordered knots and values;
                # extrapolation is constant at either endpoint.
                if (not 2 <= len(edge.parameters) <= 16
                        or len(edge.coefficients) != len(edge.parameters)
                        or any(a >= b for a, b in zip(
                            edge.parameters, edge.parameters[1:], strict=False
                        ))):
                    raise KANAdmissionError("Spline knots must be strictly increasing")
            elif edge.basis == "gaussian_rbf":
                # Coefficients per center. Parameters: centers followed by widths.
                k = len(edge.coefficients)
                if (not 1 <= k <= 8 or len(edge.parameters) != 2 * k
                        or any(width <= 0 for width in edge.parameters[k:])):
                    raise KANAdmissionError("RBF requires paired centers/positive scales")
            elif edge.basis == "fourier":
                # a0 + sum_{k=1}^K [a_k cos(k*w*x) + b_k sin(k*w*x)]
                if (not len(edge.parameters) == 1
                        or not 0 < edge.parameters[0] <= 100
                        or len(edge.coefficients) < 3
                        or (len(edge.coefficients) - 1) % 2
                        or len(edge.coefficients) > 17):
                    raise KANAdmissionError("Bounded Fourier basis and frequency required")
            else:
                raise KANAdmissionError("Unsupported KAN basis")
            count += 1
            if count > MAX_EDGES:
                raise KANAdmissionError("KAN edge budget exceeded")
        values.add(node.name)
    if any(name not in values for name in program.outputs):
        raise KANAdmissionError("Unknown KAN result")
    # Requiring the final node to be observable rules out unused graph tails.
    if program.nodes[-1].name not in program.outputs:
        raise KANAdmissionError("KAN output must observe final node")


def edge_scalar(edge: EdgeFunction, value: float) -> float:
    """Reference scalar function, with an independent vector evaluator below."""
    if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise KANAdmissionError("Unbounded/nonfinite KAN activation")
    a, p = edge.coefficients, edge.parameters
    if edge.basis == "polynomial":
        total = 0.0
        for coefficient in reversed(a):
            total = total * value + coefficient
    elif edge.basis == "linear_spline":
        if value <= p[0]:
            total = a[0]
        elif value >= p[-1]:
            total = a[-1]
        else:
            index = next(i for i in range(len(p) - 1) if p[i] <= value < p[i + 1])
            t = (value - p[index]) / (p[index + 1] - p[index])
            total = a[index] * (1 - t) + a[index + 1] * t
    elif edge.basis == "gaussian_rbf":
        k = len(a)
        total = math.fsum(
            a[i] * math.exp(-0.5 * ((value - p[i]) / p[k + i]) ** 2)
            for i in range(k)
        )
    elif edge.basis == "fourier":
        total = a[0]
        for harmonic in range(1, (len(a) - 1) // 2 + 1):
            theta = harmonic * p[0] * value
            total += a[2 * harmonic - 1] * math.cos(theta)
            total += a[2 * harmonic] * math.sin(theta)
    else:
        raise KANAdmissionError("Unknown basis")
    if not math.isfinite(total) or abs(total) > MAX_MAGNITUDE:
        raise KANAdmissionError("KAN edge activation exceeds finite bound")
    return float(total)


def execute_kan_scalar(program: KANProgram, inputs: dict[str, float]) -> dict[str, float]:
    validate_kan(program)
    if set(inputs) != set(program.inputs):
        raise KANAdmissionError("Input ABI mismatch")
    values: dict[str, float] = {}
    for name in program.inputs:
        value = inputs[name]
        if (type(value) not in (float, int) or not math.isfinite(value)
                or not program.domain[0] <= value <= program.domain[1]):
            raise KANAdmissionError("KAN input outside declared domain")
        values[name] = float(value)
    for node in program.nodes:
        intermediate = [edge_scalar(edge, values[edge.source]) for edge in node.edges]
        result = math.fsum(intermediate) if node.combine == "sum" else math.prod(intermediate)
        if not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
            raise KANAdmissionError("KAN node activation violates bound")
        values[node.name] = result
    return {name: values[name] for name in program.outputs}


def execute_kan_vector(program: KANProgram, inputs: dict[str, object]):
    """Separate NumPy-vectorized implementation; scalar code not reused."""
    import numpy as np

    validate_kan(program)
    if set(inputs) != set(program.inputs):
        raise KANAdmissionError("Vector input ABI mismatch")
    arrays: dict[str, np.ndarray] = {}
    size = None
    for name in program.inputs:
        a = np.asarray(inputs[name])
        if (a.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or a.ndim != 1 or not 1 <= a.size <= 16384
                or not np.isfinite(a).all()
                or not np.logical_and(a >= program.domain[0],
                                      a <= program.domain[1]).all()):
            raise KANAdmissionError("Vector input shape/dtype/domain violation")
        if size is not None and a.size != size:
            raise KANAdmissionError("Batch axis mismatch")
        size = a.size
        arrays[name] = np.asarray(a, dtype=np.float64).copy()
    for node in program.nodes:
        edges = []
        for edge in node.edges:
            x = arrays[edge.source]
            a, p = edge.coefficients, edge.parameters
            if edge.basis == "polynomial":
                z = np.polynomial.polynomial.polyval(x, a)
            elif edge.basis == "linear_spline":
                z = np.interp(x, p, a)
            elif edge.basis == "gaussian_rbf":
                k = len(a)
                z = np.zeros_like(x)
                for i in range(k):
                    z += a[i] * np.exp(-0.5 * ((x - p[i]) / p[k + i]) ** 2)
            elif edge.basis == "fourier":
                z = np.full_like(x, a[0])
                for i in range(1, (len(a) - 1) // 2 + 1):
                    z += a[2 * i - 1] * np.cos(i * p[0] * x)
                    z += a[2 * i] * np.sin(i * p[0] * x)
            else:
                raise KANAdmissionError("Unknown vector basis")
            if not np.isfinite(z).all():
                raise KANAdmissionError("Nonfinite vector edge")
            edges.append(z)
        result = np.sum(np.stack(edges), axis=0) if node.combine == "sum" else np.prod(
            np.stack(edges), axis=0
        )
        if not np.isfinite(result).all() or np.any(np.abs(result) > MAX_MAGNITUDE):
            raise KANAdmissionError("Vector node magnitude limit")
        arrays[node.name] = result
    return {key: arrays[key] for key in program.outputs}


def polynomial_identity(program: KANProgram, *, output: str,
                        max_degree: int = 8) -> tuple[Fraction, ...]:
    """Exact rational symbolic normalization of polynomial-only sum/product DAGs.

    This is a structural identity; it is not a causal explanation for
    arbitrary pretrained checkpoints or a float64 bitwise equivalence proof.
    """
    validate_kan(program)
    if len(program.inputs) != 1 or output not in program.outputs:
        raise KANAdmissionError("Polynomial certificate requires one input/output")
    x = program.inputs[0]
    polys: dict[str, tuple[Fraction, ...]] = {x: (Fraction(0), Fraction(1))}

    def mul(a, b):
        if len(a) + len(b) - 2 > max_degree:
            raise KANAdmissionError("Symbolic polynomial degree budget exceeded")
        result = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                result[i + j] += ai * bj
        return tuple(result)

    for node in program.nodes:
        pieces = []
        for edge in node.edges:
            if edge.basis != "polynomial":
                raise KANAdmissionError("Nonpolynomial KAN edge is not an exact polynomial")
            coefficients = tuple(Fraction(str(v)) for v in edge.coefficients)
            power = (Fraction(1),)
            result = (Fraction(0),)
            for c in coefficients:
                width = max(len(result), len(power))
                result = tuple(
                    (result[i] if i < len(result) else 0)
                    + c * (power[i] if i < len(power) else 0)
                    for i in range(width)
                )
                power = mul(power, polys[edge.source])
            pieces.append(result)
        if node.combine == "sum":
            width = max(map(len, pieces))
            result = tuple(sum(
                (p[i] if i < len(p) else Fraction(0) for p in pieces), Fraction(0)
            ) for i in range(width))
        else:
            result = (Fraction(1),)
            for p in pieces:
                result = mul(result, p)
        if len(result) > max_degree + 1:
            raise KANAdmissionError("KAN symbolic polynomial degree exceeded")
        polys[node.name] = result
    value = list(polys[output])
    while len(value) > 1 and value[-1] == 0:
        value.pop()
    return tuple(value)
