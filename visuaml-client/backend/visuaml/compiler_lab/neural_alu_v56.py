"""v56 shared fixed-weight neural ALU selector for bounded WASM i32.

Six candidate circuits run on the *same* two input registers; the selected
32-bit output is a bilinear contraction against one-hot opcode PROGRAM DATA.
The host does not choose between add/sub/mul/lt/gt/eqz implementations.

This does NOT remove host stack-arity handling, opcode fetch, or locals.
The v49 bit gates remain the arithmetic substrate, not learned parameters.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    WIDTH,
    BitCircuitWeights,
    _add,
    _eqz,
    _mul,
    _signed_lt,
    _sub,
    construct_bit_weights,
)

SCHEMA = "visuaml.constructed-fixed-alu.v56"
ALU_OPS = (
    "i32.add", "i32.sub", "i32.mul", "i32.lt_s",
    "i32.gt_s", "i32.eqz",
)
OP_COUNT = len(ALU_OPS)


class NeuralALUError(ValueError):
    """Invalid opcode or corrupted exact-binary datapath."""


@dataclass(frozen=True)
class ALUWeights:
    """A single fixed set of coefficients shared across programs."""

    output_gains: np.ndarray

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        h.update(np.ascontiguousarray(
            self.output_gains, dtype="<f8"
        ).tobytes())
        return h.hexdigest()


def construct_alu_weights() -> ALUWeights:
    return ALUWeights(np.ones(OP_COUNT, dtype=np.float64))


def _admit_binary(a: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(a, np.ndarray) or a.dtype != np.float64
            or a.shape != shape or not np.isfinite(a).all()
            or not np.isin(a, (0., 1.)).all()):
        raise NeuralALUError("ALU input/output must be exact binary tensor")


def onehot_alu(name: str) -> np.ndarray:
    if name not in ALU_OPS:
        raise NeuralALUError("Opcode outside admitted v56 ALU subset")
    result = np.zeros(OP_COUNT, dtype=np.float64)
    result[ALU_OPS.index(name)] = 1.
    return result


def execute_alu(
    opcode: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    fixed: ALUWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
) -> np.ndarray:
    """Compute all six exact bit-gate candidates then fixed-weight select.

    The unary eqz uses left; right is ignored for that candidate. Runtime
    host code must still supply the correct stack effect and operand order.
    """
    _admit_binary(opcode, (OP_COUNT,))
    _admit_binary(left, (WIDTH,))
    _admit_binary(right, (WIDTH,))
    if opcode.sum() != 1.:
        raise NeuralALUError("Opcode selector is not uniquely one-hot")
    fixed = construct_alu_weights() if fixed is None else fixed
    if not isinstance(fixed, ALUWeights):
        raise NeuralALUError("Unsupported ALU coefficient object")
    _admit_binary(fixed.output_gains, (OP_COUNT,))
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    lt = np.zeros(WIDTH, dtype=np.float64)
    gt = np.zeros(WIDTH, dtype=np.float64)
    eq = np.zeros(WIDTH, dtype=np.float64)
    lt[0] = _signed_lt(left, right, bit_fixed)
    gt[0] = _signed_lt(right, left, bit_fixed)
    eq[0] = _eqz(left, bit_fixed)
    candidates = np.stack((
        _add(left, right, fixed=bit_fixed),
        _sub(left, right, fixed=bit_fixed),
        _mul(left, right, bit_fixed.carry_thresholds),
        lt, gt, eq,
    ))
    _admit_binary(candidates, (OP_COUNT, WIDTH))
    gate = opcode * fixed.output_gains
    result = gate @ candidates
    _admit_binary(result, (WIDTH,))
    if gate.sum() != 1.:
        raise NeuralALUError("Fixed ALU gate fails unique selection")
    return result
