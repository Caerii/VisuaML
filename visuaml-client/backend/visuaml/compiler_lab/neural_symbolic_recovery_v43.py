"""v43: recover a symbolic quadratic from analytic neural weights, then recompile.

The extraction input is ONLY a v41 bounded ReLU tensor bundle, without the
program, construction receipt, runtime outputs, or source identifiers.
Recovered polynomial behavior is certified on ALL 256 admitted i8 inputs.

A separate independent Z3 check compares a reconstructed i32 bitvector
program against an independently supplied source I32Program for ALL 2^32
i32 inputs. The neural circuit itself remains certified ONLY on [-128,127].
Never collapse these two different equivalence statements into one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .analytic_relu_specialization_v41 import (
    HIGH,
    LOW,
    AnalyticRelu,
    decompile_function_from_weights,
)
from .wasm_state_vm_v40 import (
    I32_MAX,
    I32_MIN,
    I32Program,
    VMAdmissionError,
    VMOperation,
    encode_program,
    validate_program,
)

SCHEMA = "visuaml.neural-to-symbolic-polynomial.v43"
MAX_DEGREE = 2


class SymbolicRecoveryError(ValueError):
    """The neural function cannot be represented by the admitted polynomial."""


@dataclass(frozen=True)
class RecoveredPolynomial:
    a: int
    b: int
    c: int
    degree: int
    domain: tuple[int, int]
    recovered_function_sha256: str
    inference_points: tuple[int, int, int]
    source_program_read: bool = False
    schema: str = SCHEMA

    def expression(self) -> str:
        """Canonical textual display, not executable code generation."""
        return f"({self.a})*x*x + ({self.b})*x + ({self.c})"

    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _admit_coefficients(a: int, b: int, c: int) -> None:
    if any(type(x) is not int or not I32_MIN <= x <= I32_MAX for x in (a, b, c)):
        raise SymbolicRecoveryError("Polynomial coefficients require signed i32")


def _evaluate_integer(poly: RecoveredPolynomial, value: int) -> int:
    return poly.a * value * value + poly.b * value + poly.c


def recover_quadratic_from_weights(net: AnalyticRelu) -> RecoveredPolynomial:
    """Discover the finite-domain symbolic rule without the original WASM program.

    Hypothesis class: univariate integer polynomial of degree <= 2.
    Infer from 3 samples, independently check all remaining 253 samples.
    Reject piecewise/nonpolynomial neural behaviors even when first 3 fit.
    """
    recovered = decompile_function_from_weights(net)
    if recovered.domain != (LOW, HIGH) or len(recovered.outputs) != HIGH - LOW + 1:
        raise SymbolicRecoveryError("Unexpected neural reconstruction domain")
    y0, y1, y2 = recovered.outputs[:3]
    second = y2 - 2 * y1 + y0
    if second % 2:
        raise SymbolicRecoveryError("Nonintegral quadratic coefficient")
    a = second // 2
    b = y1 - y0 - a * (2 * LOW + 1)
    c = y0 - a * LOW * LOW - b * LOW
    _admit_coefficients(a, b, c)
    degree = 2 if a else (1 if b else 0)
    poly = RecoveredPolynomial(
        a, b, c, degree, (LOW, HIGH),
        recovered.function_sha256,
        (LOW, LOW + 1, LOW + 2),
    )
    for x in range(LOW, HIGH + 1):
        predicted = _evaluate_integer(poly, x)
        if predicted != recovered.outputs[x - LOW]:
            raise SymbolicRecoveryError(
                f"Polynomial hypothesis falsified by recovered neural function at x={x}"
            )
    return poly


def compile_polynomial_wasm(poly: RecoveredPolynomial) -> tuple[I32Program, bytes]:
    """Build the existing strict typed i32 WASM dialect from symbolic coefficients."""
    if not isinstance(poly, RecoveredPolynomial) or poly.schema != SCHEMA:
        raise SymbolicRecoveryError("Unexpected symbolic program schema")
    _admit_coefficients(poly.a, poly.b, poly.c)
    if poly.degree != (2 if poly.a else (1 if poly.b else 0)):
        raise SymbolicRecoveryError("Incorrect symbolic degree label")
    if poly.domain != (LOW, HIGH) or poly.source_program_read:
        raise SymbolicRecoveryError("Unadmitted symbolic provenance contract")
    # Typed i32 stack machine, not a bespoke WASM binary splicer:
    # x*x*a + x*b + c, all arithmetic explicitly wraps modulo 2^32.
    ops = (
        VMOperation("local.get", 0),
        VMOperation("local.get", 0),
        VMOperation("i32.mul"),
        VMOperation("i32.const", poly.a),
        VMOperation("i32.mul"),
        VMOperation("local.get", 0),
        VMOperation("i32.const", poly.b),
        VMOperation("i32.mul"),
        VMOperation("i32.add"),
        VMOperation("i32.const", poly.c),
        VMOperation("i32.add"),
        VMOperation("end"),
    )
    program = I32Program(0, ops)
    validate_program(program)
    raw = encode_program(program)
    return program, raw


def compile_polynomial_cpp17(poly: RecoveredPolynomial) -> str:
    """Generate a safe integer-modular C++17 ABI without undefined signed overflow.

    Constants are typed checked signed i32 values, not user-supplied source
    fragments. std::uint32_t arithmetic is defined modulo 2^32; memcpy
    preserves the signed result bit-pattern on the tested 32-bit-int target.
    """
    _admit_coefficients(poly.a, poly.b, poly.c)
    A, B, C = (f"{x & 0xFFFFFFFF}u" for x in (poly.a, poly.b, poly.c))
    return (
        "// v43 exact i32 modular polynomial; compiler-generated after typed admission.\n"
        "#include <cstdint>\n#include <cstring>\n"
        "#include <climits>\n"
        "static_assert(CHAR_BIT == 8 && sizeof(std::int32_t) == 4"
        " && sizeof(std::uint32_t) == 4, \"32-bit i32 required\");\n"
        "extern \"C\" std::int32_t visuaml_recovered_i32(std::int32_t x) {\n"
        "    const std::uint32_t u = static_cast<std::uint32_t>(x);\n"
        f"    const std::uint32_t z = ({A} * u * u) + ({B} * u) + {C};\n"
        "    std::int32_t result;\n"
        "    std::memcpy(&result, &z, sizeof(result));\n"
        "    return result;\n"
        "}\n"
    )


def _encode_wasm_in_z3(source: I32Program, x, z3):
    """Independent symbolic execution of the admitted stack-machine ISA."""
    validate_program(source)
    values = [x, *[z3.BitVecVal(0, 32) for _ in range(source.local_count)]]
    stack = []
    for operation in source.operations:
        name = operation.opcode
        imm = operation.immediate
        if name == "i32.const":
            stack.append(z3.BitVecVal(imm, 32))
        elif name == "local.get":
            stack.append(values[imm])
        elif name == "local.set":
            values[imm] = stack.pop()
        elif name == "local.tee":
            values[imm] = stack[-1]
        elif name in {"i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s"}:
            right, left = stack.pop(), stack.pop()
            if name == "i32.add":
                stack.append(left + right)
            elif name == "i32.sub":
                stack.append(left - right)
            elif name == "i32.mul":
                stack.append(left * right)
            elif name == "i32.lt_s":
                stack.append(z3.If(left < right, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)))
            else:
                stack.append(z3.If(left > right, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)))
        elif name == "i32.eqz":
            stack.append(z3.If(
                stack.pop() == z3.BitVecVal(0, 32),
                z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)
            ))
        elif name == "drop":
            stack.pop()
        elif name == "end":
            break
        else:
            raise VMAdmissionError("Unadmitted source operation in Z3 interpreter")
    if len(stack) != 1:
        raise VMAdmissionError("Symbolic WASM did not return exactly one i32")
    return stack[0]


def check_all_i32_equivalence(
    source: I32Program, candidate: RecoveredPolynomial,
    *, timeout_ms: int = 15000,
) -> dict:
    """Use 32-bit bit-vector SMT to disprove or establish stated source equivalence.

    UNSAT means *no i32 input* produces different bitvector results, under the
    encoded pure-source semantics. SAT yields a concrete falsifying i32.
    UNKNOWN (timeout or incompleteness) is an error, not success.
    This is an SMT solver result, NOT a machine-checked proof object.
    """
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise SymbolicRecoveryError("Unbounded SMT solver budget")
    import z3

    _admit_coefficients(candidate.a, candidate.b, candidate.c)
    x = z3.BitVec("input_i32", 32)
    source_out = _encode_wasm_in_z3(source, x, z3)
    expected = (
        z3.BitVecVal(candidate.a, 32) * x * x
        + z3.BitVecVal(candidate.b, 32) * x
        + z3.BitVecVal(candidate.c, 32)
    )
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(source_out != expected)
    formula = solver.to_smt2().encode("utf-8")
    outcome = solver.check()
    result = {
        "schema": "visuaml.smt-i32-bitvector-equivalence.v43",
        "symbolic_source_sha256": source.digest(),
        "symbolic_candidate_sha256": candidate.digest(),
        "smtlib_sha256": hashlib.sha256(formula).hexdigest(),
        "solver": f"Z3 {z3.get_version_string()}",
        "query": "exists input_i32: source(input) != polynomial(input), bitvector32",
        "solver_outcome": str(outcome),
        "all_i32_source_vs_symbolic_equivalent": outcome == z3.unsat,
        "neural_all_i32_equivalent": False,
        "machine_checked_proof_object": False,
    }
    if outcome == z3.unknown:
        raise SymbolicRecoveryError("SMT solver returned unknown: " + solver.reason_unknown())
    if outcome == z3.sat:
        model = solver.model()
        unsigned = int(model.eval(x, model_completion=True).as_long())
        signed = unsigned - (1 << 32) if unsigned >= 1 << 31 else unsigned
        result["counterexample_input_i32"] = signed
        result["source_output_u32"] = int(
            model.eval(source_out, model_completion=True).as_long()
        )
        result["candidate_output_u32"] = int(
            model.eval(expected, model_completion=True).as_long()
        )
    return result
