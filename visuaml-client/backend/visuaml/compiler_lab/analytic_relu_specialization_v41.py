"""v41 deterministic ReLU specialization of one bounded WASM i32 function.

A *known* pure typed i32 program is specialized into an analytically built
two-affine-layer scalar ReLU network with no gradient descent or training data.
It is exact only for the admitted finite signed-i8 input domain, and it is NOT
a universal neural VM or reconstruction of original code from arbitrary weights.

The inverse extracts the complete finite-domain input/output function from
neural weights WITHOUT a source program, runtime traces, or a manifest. It
recovers *behavior* under an explicitly recognized architecture template.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .wasm_state_vm_v40 import I32Program, execute_program, validate_program

SCHEMA = "visuaml.analytic-relu-specialization.v41"
EXTRACTION = "visuaml.manifest-free-relu-function-extraction.v41"
LOW = -128
HIGH = 127
COUNT = HIGH - LOW + 1
HIDDEN = COUNT - 1
EXACT_SAFE_SUM = 1 << 52


class NeuralSpecializationError(ValueError):
    """Unsupported network layout, invalid tensor, or nonexact arithmetic."""


@dataclass(frozen=True)
class AnalyticRelu:
    """y = W_out @ ReLU(W_in @ [x] + b_in) + b_out."""

    w_in: np.ndarray
    b_in: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    source_program_sha256: str | None = None


@dataclass(frozen=True)
class RecoveredFunction:
    domain: tuple[int, int]
    outputs: tuple[int, ...]
    piecewise_slopes: tuple[int, ...]
    function_sha256: str
    manifest_used: bool = False


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(value: np.ndarray) -> str:
    return _digest(np.ascontiguousarray(value, dtype="<f8").tobytes(order="C"))


def _expected_biases() -> tuple[int, ...]:
    # Channel 0 spans the full domain, others are hinges at every interior knot.
    return tuple([-LOW, *(-x for x in range(LOW + 1, HIGH))])


def _guard_weights(net: AnalyticRelu) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(net, AnalyticRelu):
        raise NeuralSpecializationError("Expected analytic ReLU tensor bundle")
    arrays = (net.w_in, net.b_in, net.w_out, net.b_out)
    shapes = ((HIDDEN, 1), (HIDDEN,), (1, HIDDEN), (1,))
    copies = []
    for value, shape in zip(arrays, shapes, strict=True):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()
                or not np.equal(value, np.rint(value)).all()):
            raise NeuralSpecializationError("Expected finite exact-integer float64 weight tensors")
        copies.append(np.ascontiguousarray(value).copy())
    w1, b1, w2, b2 = copies
    if not np.array_equal(w1, np.ones_like(w1)):
        raise NeuralSpecializationError("Unrecognized ReLU input weight pattern")
    if sorted(int(x) for x in b1) != sorted(_expected_biases()):
        raise NeuralSpecializationError("Unrecognized domain hinge layout")
    # An absolute upper bound on all intermediate products and partial sums
    # ensures binary64 integer operations are exact throughout the domain.
    peak_activations = np.maximum(0, HIGH + b1)
    absolute_bound = int(abs(b2[0])) + sum(
        int(abs(w2[0, i])) * int(peak_activations[i])
        for i in range(HIDDEN)
    )
    if absolute_bound >= EXACT_SAFE_SUM:
        raise NeuralSpecializationError("Binary64 exact-integer sum bound exceeded")
    return w1, b1, w2, b2


def evaluate_bounded(net: AnalyticRelu, value: int) -> int:
    if type(value) is not int or not LOW <= value <= HIGH:
        raise NeuralSpecializationError("Input outside specialized finite i8 domain")
    w1, b1, w2, b2 = _guard_weights(net)
    activation = np.maximum(np.float64(0), w1 @ np.array([float(value)]) + b1)
    result = float((w2 @ activation)[0] + b2[0])
    if not np.isfinite(result) or not result.is_integer():
        raise NeuralSpecializationError("Noninteger output violates exact circuit contract")
    return int(result)


def function_hash(outputs: tuple[int, ...]) -> str:
    return _digest(json.dumps({
        "domain": [LOW, HIGH], "outputs": list(outputs)
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def decompile_function_from_weights(net: AnalyticRelu) -> RecoveredFunction:
    """Identify hinge basis and recover the *function*, not the source program."""
    _, biases, out, b_out = _guard_weights(net)
    ordered = sorted(
        ((int(biases[i]), int(out[0, i])) for i in range(HIDDEN)),
        reverse=True,
    )
    expected = _expected_biases()
    if tuple(b for b, _ in ordered) != expected:
        raise NeuralSpecializationError("Inconsistent sorted ReLU hinge basis")
    weights = [w for _, w in ordered]
    y = [int(b_out[0])]
    slope = weights[0]
    y.append(y[0] + slope)
    for second_difference in weights[1:]:
        slope += second_difference
        y.append(y[-1] + slope)
    if len(y) != COUNT or any(not -(1 << 31) <= value < (1 << 31) for value in y):
        raise NeuralSpecializationError("Recovered function outside bounded i32 result domain")
    return RecoveredFunction(
        (LOW, HIGH), tuple(y),
        tuple(y[i + 1] - y[i] for i in range(COUNT - 1)),
        function_hash(tuple(y)),
        manifest_used=False,
    )


def compile_bounded_program(program: I32Program) -> tuple[AnalyticRelu, dict]:
    """Compile a known typed program by exhausting the declared 256 inputs.

    This is a bounded *Futamura-like program specialization* via tabulation,
    NOT a fixed-weight universal instruction interpreter.
    """
    validate_program(program)
    observed = tuple(execute_program(program, value).value for value in range(LOW, HIGH + 1))
    slope = observed[1] - observed[0]
    differences = tuple(
        observed[i + 1] - 2 * observed[i] + observed[i - 1]
        for i in range(1, COUNT - 1)
    )
    coefficients = (slope, *differences)
    if len(coefficients) != HIDDEN:
        raise NeuralSpecializationError("Hinge coefficient construction failed")
    net = AnalyticRelu(
        np.ones((HIDDEN, 1), dtype=np.float64),
        np.array(_expected_biases(), dtype=np.float64),
        np.array([coefficients], dtype=np.float64),
        np.array([observed[0]], dtype=np.float64),
        program.digest(),
    )
    recovered = decompile_function_from_weights(net)
    if recovered.outputs != observed:
        raise NeuralSpecializationError("Analytic inverse does not match source interpreter")
    for value in range(LOW, HIGH + 1):
        if evaluate_bounded(net, value) != observed[value - LOW]:
            raise NeuralSpecializationError("Constructed neural circuit diverged")
    receipt = {
        "schema": SCHEMA,
        "program_sha256": program.digest(),
        "input_domain": [LOW, HIGH],
        "hidden_relu_units": HIDDEN,
        "tensor_sha256": {
            "w_in": _tensor_sha(net.w_in),
            "b_in": _tensor_sha(net.b_in),
            "w_out": _tensor_sha(net.w_out),
            "b_out": _tensor_sha(net.b_out),
        },
        "recovered_function_sha256": recovered.function_sha256,
        "integer_domain_exactly_verified": True,
        "manifest_free_function_extraction": True,
        "source_program_decompiled": False,
        "wasmtime_executed_by_this_function": False,
        "universal_neural_interpreter": False,
        "all_i32_inputs_proved": False,
        "limitations": (
            "One finite-domain static i32 program specialized by exhaustive "
            "tabulation to analytic ReLU hinges; float64 exact-integer sum "
            "budget; not arbitrary i32, long-running VM, transformer "
            "attention, arbitrary learned-network program recovery or "
            "all-input proof."
        ),
    }
    return net, receipt


def permute_hidden_neurons(net: AnalyticRelu,
                           permutation: tuple[int, ...]) -> AnalyticRelu:
    """Reorder hidden neural 'binary layout' without changing its function."""
    _guard_weights(net)
    if (not isinstance(permutation, tuple) or len(permutation) != HIDDEN
            or any(type(x) is not int for x in permutation)
            or set(permutation) != set(range(HIDDEN))):
        raise NeuralSpecializationError("Expected bijective hidden-neuron permutation")
    index = np.array(permutation, dtype=np.int64)
    return AnalyticRelu(
        net.w_in[index].copy(), net.b_in[index].copy(),
        net.w_out[:, index].copy(), net.b_out.copy(),
        net.source_program_sha256,
    )
