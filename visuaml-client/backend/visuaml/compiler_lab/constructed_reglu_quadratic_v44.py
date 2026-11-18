"""v44 exact constructive ReGLU representation of recovered quadratic rules.

For EVERY real x (under exact real arithmetic):
  ReLU(x)*x + ReLU(-x)*(-x) = x^2.

Thus two gated units + linear skip implement a*x^2 + b*x + c with
analytically derived weights. The numeric float64 implementation is certified
only when all integer intermediates and partial sums stay exactly representable.

This changes the primitive family (ReGLU instead of ordinary ReLU) and
does not establish runtime speedup, universal neural computation, or i32-wrap
equivalence for the gating network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .analytic_relu_specialization_v41 import HIGH, LOW, function_hash
from .neural_symbolic_recovery_v43 import RecoveredPolynomial, compile_polynomial_wasm

SCHEMA = "visuaml.constructed-reglu-quadratic.v44"
EXTRACTED = "visuaml.reglu-manifest-free-quadratic.v44"
EXACT_INTEGER_LIMIT = 1 << 52
MAX_NUMERIC_ABS_INPUT = 1 << 30


class GatedConstructionError(ValueError):
    """Unsupported bilinear-gate template or unsafe numeric arithmetic."""


@dataclass(frozen=True)
class ConstructedGatedQuadratic:
    gate_weights: np.ndarray
    value_weights: np.ndarray
    gate_bias: np.ndarray
    output_weights: np.ndarray
    skip: int
    bias: int
    schema: str = SCHEMA


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _admit(net: ConstructedGatedQuadratic) -> tuple[int, int, int]:
    if not isinstance(net, ConstructedGatedQuadratic) or net.schema != SCHEMA:
        raise GatedConstructionError("Unknown gated neural circuit schema")
    for value in (net.gate_weights, net.value_weights, net.gate_bias, net.output_weights):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or not np.isfinite(value).all()):
            raise GatedConstructionError("Expected finite float64 analytical weights")
    if (net.gate_weights.shape != (2,)
            or net.value_weights.shape != (2,)
            or net.gate_bias.shape != (2,)
            or net.output_weights.shape != (2,)):
        raise GatedConstructionError("Only exactly two scalar gating channels")
    if not np.array_equal(net.gate_bias, np.zeros((2,), dtype=np.float64)):
        raise GatedConstructionError("Unsupported gated bias")
    pairs = sorted((
        float(net.gate_weights[i]),
        float(net.value_weights[i]),
        float(net.output_weights[i]),
    ) for i in range(2))
    if pairs[0][:2] != (-1.0, -1.0) or pairs[1][:2] != (1.0, 1.0):
        raise GatedConstructionError("Unrecognized sign-paired multiplication circuit")
    if pairs[0][2] != pairs[1][2]:
        raise GatedConstructionError("Asymmetric quadratic readout")
    raw_a = pairs[0][2]
    if not raw_a.is_integer():
        raise GatedConstructionError("Noninteger quadratic coefficient")
    a = int(raw_a)
    b, c = net.skip, net.bias
    if any(type(x) is not int or not -(1 << 31) <= x < (1 << 31)
           for x in (a, b, c)):
        raise GatedConstructionError("Expected signed i32 polynomial coefficients")
    return a, b, c


def _budget(a: int, b: int, c: int, x: int) -> None:
    if type(x) is not int or abs(x) > MAX_NUMERIC_ABS_INPUT:
        raise GatedConstructionError("Input outside bounded numeric range")
    # Require both gated terms, the linear skip and final addition to be
    # exact binary64 integers. Do not assert correctness after overflow.
    bound = abs(a) * x * x + abs(b) * abs(x) + abs(c)
    if bound >= EXACT_INTEGER_LIMIT:
        raise GatedConstructionError("Binary64 exact-integer arithmetic budget exceeded")


def construct_from_symbolic(poly: RecoveredPolynomial) -> tuple[ConstructedGatedQuadratic, dict]:
    """Compile a certified v43 polynomial, not arbitrary source code, to 2 ReGLU gates."""
    if not isinstance(poly, RecoveredPolynomial):
        raise GatedConstructionError("Requires typed recovered symbolic polynomial")
    # Reuses the established typed i32/digest admission contract.
    compile_polynomial_wasm(poly)
    outputs = tuple(
        poly.a * x * x + poly.b * x + poly.c for x in range(LOW, HIGH + 1)
    )
    if function_hash(outputs) != poly.recovered_function_sha256:
        raise GatedConstructionError("Polynomial does not reproduce its bound neural function")
    net = ConstructedGatedQuadratic(
        gate_weights=np.array([1., -1.], dtype=np.float64),
        value_weights=np.array([1., -1.], dtype=np.float64),
        gate_bias=np.array([0., 0.], dtype=np.float64),
        output_weights=np.array([poly.a, poly.a], dtype=np.float64),
        skip=poly.b,
        bias=poly.c,
    )
    if _admit(net) != (poly.a, poly.b, poly.c):
        raise GatedConstructionError("Constructed gating template was invalid")
    for x in range(LOW, HIGH + 1):
        if evaluate_constructed(net, x) != outputs[x - LOW]:
            raise GatedConstructionError("Gated construction numerical mismatch")
    certificate = {
        "schema": SCHEMA,
        "numeric_contract": "exact integer float64 on explicitly budgeted inputs",
        "real_arithmetic_identity": "relu(x)*x + relu(-x)*(-x) = x*x",
        "construction_type": "two independent ReGLU multiplication gates plus linear skip",
        "hidden_gated_units": 2,
        "serialized_scalar_parameter_count": 10,
        "v41_reference_relu_units": 255,
        "v41_reference_scalar_parameters": 766,
        "different_activation_family": True,
        "speedup_measured": False,
        "all_i32_gated_network_proved": False,
        "full_domain_signed_i8_evaluated": True,
        "polynomial_digest": poly.digest(),
        "weight_sha256": {
            "gate": _sha(net.gate_weights),
            "value": _sha(net.value_weights),
            "gate_bias": _sha(net.gate_bias),
            "output": _sha(net.output_weights),
        },
        "source_program_consulted_for_construction": False,
        "limitations": (
            "Two bilinear ReGLU gates, not equivalent cost to 255 ordinary ReLU gates; "
            "not a general neural net optimizer, universal transformer VM, WASM "
            "i32 wraparound gate or proof of numerical exactness at unsafe magnitudes."
        ),
    }
    return net, certificate


def evaluate_constructed(net: ConstructedGatedQuadratic, x: int) -> int:
    """Actual vectorized binary64 ReLU*linear gate execution, not a polynomial shortcut."""
    a, b, c = _admit(net)
    _budget(a, b, c, x)
    value = np.float64(x)
    gates = np.maximum(np.float64(0), net.gate_weights * value + net.gate_bias)
    linear = net.value_weights * value
    out = float(np.sum(net.output_weights * gates * linear) + b * value + c)
    if not np.isfinite(out) or not out.is_integer():
        raise GatedConstructionError("Gated network failed exact integer contract")
    return int(out)


def recover_from_gated_weights(net: ConstructedGatedQuadratic) -> RecoveredPolynomial:
    """Recover the polynomial without the source program or its construction receipt."""
    a, b, c = _admit(net)
    outputs = []
    for x in range(LOW, HIGH + 1):
        outputs.append(evaluate_constructed(net, x))
    return RecoveredPolynomial(
        a, b, c, 2 if a else (1 if b else 0),
        (LOW, HIGH), function_hash(tuple(outputs)),
        (LOW, LOW + 1, LOW + 2),
        False,
    )


def permute_gated_units(
    net: ConstructedGatedQuadratic, order: tuple[int, int]
) -> ConstructedGatedQuadratic:
    _admit(net)
    if (not isinstance(order, tuple) or len(order) != 2
            or any(type(x) is not int for x in order) or set(order) != {0, 1}):
        raise GatedConstructionError("Expected two-unit bijective permutation")
    indices = np.array(order, dtype=np.int64)
    return ConstructedGatedQuadratic(
        net.gate_weights[indices].copy(),
        net.value_weights[indices].copy(),
        net.gate_bias[indices].copy(),
        net.output_weights[indices].copy(),
        net.skip, net.bias,
    )


def bundle_fingerprint(net: ConstructedGatedQuadratic) -> str:
    _admit(net)
    record = {
        "schema": net.schema,
        "gate": _sha(net.gate_weights),
        "value": _sha(net.value_weights),
        "gate_bias": _sha(net.gate_bias),
        "output": _sha(net.output_weights),
        "skip": net.skip,
        "bias": net.bias,
    }
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
