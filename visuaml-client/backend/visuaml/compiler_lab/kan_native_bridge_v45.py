"""v45 exact KAN polynomial DAG -> existing typed WASM and safe C++17 backends.

This is a restricted *native lowering* of a mathematically normalized KAN:
one input, one output, exact rational polynomial degree<=2 with integer i32
coefficients, and declared [-128,127] domain. The compiled i32 programs have
full bitvector modular semantics, while the original KAN graph is certified
only on its finite input domain. No general arbitrary KAN/native lowering.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from .analytic_relu_specialization_v41 import function_hash
from .kan_mechanism_ir_v45 import (
    KANAdmissionError,
    KANProgram,
    execute_kan_scalar,
    polynomial_identity,
    validate_kan,
)
from .neural_symbolic_recovery_v43 import (
    RecoveredPolynomial,
    compile_polynomial_cpp17,
    compile_polynomial_wasm,
)
from .wasm_state_vm_v40 import I32Program


@dataclass(frozen=True)
class KANBackendLowering:
    symbolic: RecoveredPolynomial
    typed_wasm: I32Program
    wasm_bytes: bytes
    cpp17_source: str
    receipt: dict


def lower_exact_polynomial_kan(program: KANProgram) -> KANBackendLowering:
    """No original source-program or pretrained checkpoint required."""
    validate_kan(program)
    if (len(program.inputs) != 1 or len(program.outputs) != 1
            or program.domain != (-128., 127.)):
        raise KANAdmissionError("Native polynomial lowering supports one signed-i8-domain scalar")
    normalized = polynomial_identity(
        program, output=program.outputs[0], max_degree=2
    )
    if len(normalized) > 3:
        raise KANAdmissionError("Native KAN integer degree exceeds two")
    coefficients = [Fraction(0)] * 3
    for index, coeff in enumerate(normalized):
        if coeff.denominator != 1:
            raise KANAdmissionError("Native KAN requires integral symbolic coefficients")
        coefficients[index] = coeff
    c, b, a = (int(v) for v in coefficients)
    for parameter in (a, b, c):
        if not -(1 << 31) <= parameter < 1 << 31:
            raise KANAdmissionError("Native KAN coefficient outside 32-bit i32")
    outputs = tuple(
        a * x * x + b * x + c for x in range(-128, 128)
    )
    if any(not -(1 << 31) <= value < 1 << 31 for value in outputs):
        raise KANAdmissionError("Native KAN polynomial output exceeds signed-i32 domain")
    for x, expected in zip(range(-128, 128), outputs, strict=True):
        observed = execute_kan_scalar(program, {program.inputs[0]: float(x)})[
            program.outputs[0]
        ]
        if observed != expected:
            raise KANAdmissionError("KAN float64 execution does not implement symbolic polynomial")
    symbolic = RecoveredPolynomial(
        a, b, c, 2 if a else (1 if b else 0), (-128, 127),
        function_hash(outputs), (-128, -127, -126), False,
    )
    machine, binary = compile_polynomial_wasm(symbolic)
    cpp = compile_polynomial_cpp17(symbolic)
    receipt = {
        "schema": "visuaml.kan-polynomial-to-native.v45",
        "source_kan_graph_sha256": program.digest(),
        "exact_symbolic_coefficients_low_to_high": [
            str(value) for value in normalized
        ],
        "reconstructed_symbol_sha256": symbolic.digest(),
        "source_function_sha256": symbolic.recovered_function_sha256,
        "output_kan_domain": [-128, 127],
        "checked_in_domain_integer_witnesses": len(outputs),
        "native_wasm_sha256": hashlib.sha256(binary).hexdigest(),
        "native_cpp17_source_sha256": hashlib.sha256(cpp.encode()).hexdigest(),
        "typed_binary_roundtrip": True,
        "general_kan_compiler": False,
        "all_i32_original_kan_equivalent": False,
        "native_code_performance_measured": False,
        "limitations": (
            "Only exact polynomial KAN DAGs of integral degree<=2, bounded "
            "scalar signed-i8 input interval, and i32-range outputs. The "
            "generated conventional WASM/C++ wrap i32 over the full bitvector "
            "domain, while KAN graph floating-point evaluations are admitted "
            "only on their original declared finite interval."
        ),
    }
    return KANBackendLowering(symbolic, machine, binary, cpp, receipt)
