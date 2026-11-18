"""v57 source-bound, fixed-weight typed microoperation token IR.

Compile canonical original WASM into PROGRAM DATA (opcode/constant/local/
branch tokens). A single fixed coefficient pair derives stack-effect and
local-mode classes from runtime PC-selected opcode one-hot inputs. Opcode
semantics are fixed across all programs; their tokens are never per-program
neural weights.

Component-stage milestone: this does not yet replace the v56 host stack
effect dispatcher. Exact source validation and independent Wasmtime are
required before runtime execution claims.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import _bits
from .neural_alu_v56 import ALU_OPS
from .neural_control_routing_v52 import SLOTS
from .wasm_control_flow_v51 import (
    MAX_LOCALS,
    MAX_NESTING,
    MAX_OPERATIONS,
    ControlProgram,
    _OPCODES,
    _syntax,
    encode_control,
)

SCHEMA = "visuaml.typed-neural-microoperations.v57"
OPCODES = tuple(_OPCODES)
EFFECTS = ("retain", "push", "pop", "unary", "binary")
LOCAL_MODES = ("none", "get", "set", "tee")
STACK_EFFECT = {
    "i32.const": "push",
    "local.get": "push",
    "local.set": "pop",
    "drop": "pop",
    "if": "pop",
    "br_if": "pop",
    "i32.eqz": "unary",
    **{op: "binary" for op in ALU_OPS if op != "i32.eqz"},
}
LOCAL_MODE = {
    "local.get": "get",
    "local.set": "set",
    "local.tee": "tee",
}
LOCALS_WIDTH = MAX_LOCALS + 1


class MicroopAdmissionError(ValueError):
    """Malformed tokens, untrusted source identity, or fixed decoder outputs."""


@dataclass(frozen=True)
class MicroopWeights:
    effects: np.ndarray
    local_modes: np.ndarray
    alu_selectors: np.ndarray

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        for value in (self.effects, self.local_modes, self.alu_selectors):
            h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class ProgramMicroopTokens:
    opcode_rows: np.ndarray
    i32_immediates: np.ndarray
    local_selectors: np.ndarray
    branch_depths: np.ndarray
    length: int
    source_wasm_sha256: str
    source_program_sha256: str

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        for value in (
            self.opcode_rows, self.i32_immediates,
            self.local_selectors, self.branch_depths,
        ):
            h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        h.update(str(self.length).encode())
        h.update(self.source_wasm_sha256.encode())
        h.update(self.source_program_sha256.encode())
        return h.hexdigest()


@dataclass(frozen=True)
class SelectedMicroop:
    opcode: np.ndarray
    effect: np.ndarray
    local_mode: np.ndarray
    alu_selector: np.ndarray
    immediate_bits: np.ndarray
    local_selector: np.ndarray
    branch_depth: np.ndarray


def _binary(value: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.shape != shape or not np.isfinite(value).all()
            or not np.isin(value, (0., 1.)).all()):
        raise MicroopAdmissionError("Noncanonical binary microop tensor")


def _onehot(value: np.ndarray, shape: tuple[int, ...], *, allow_zero=False):
    _binary(value, shape)
    if value.sum() not in ((0., 1.) if allow_zero else (1.,)):
        raise MicroopAdmissionError("Invalid one-hot microop selection")


def construct_microop_weights() -> MicroopWeights:
    effects = np.zeros((len(OPCODES), len(EFFECTS)), dtype=np.float64)
    modes = np.zeros((len(OPCODES), len(LOCAL_MODES)), dtype=np.float64)
    alu = np.zeros((len(OPCODES), len(ALU_OPS)), dtype=np.float64)
    for i, name in enumerate(OPCODES):
        effects[i, EFFECTS.index(STACK_EFFECT.get(name, "retain"))] = 1.
        modes[i, LOCAL_MODES.index(LOCAL_MODE.get(name, "none"))] = 1.
        if name in ALU_OPS:
            alu[i, ALU_OPS.index(name)] = 1.
    return MicroopWeights(effects, modes, alu)


def compile_microop_tokens(program: ControlProgram) -> ProgramMicroopTokens:
    _syntax(program)
    opcodes = np.zeros((SLOTS, len(OPCODES)), dtype=np.float64)
    imm = np.zeros((SLOTS, 32), dtype=np.float64)
    locals_ = np.zeros((SLOTS, LOCALS_WIDTH), dtype=np.float64)
    branches = np.zeros((SLOTS, MAX_NESTING), dtype=np.float64)
    for pc, operation in enumerate(program.operations):
        opcode = operation.opcode
        opcodes[pc, OPCODES.index(opcode)] = 1.
        if opcode == "i32.const":
            imm[pc] = _bits(operation.immediate)
        elif opcode in LOCAL_MODE:
            locals_[pc, operation.immediate] = 1.
        elif opcode in ("br", "br_if"):
            branches[pc, operation.immediate] = 1.
    return ProgramMicroopTokens(
        opcodes, imm, locals_, branches, len(program.operations),
        hashlib.sha256(encode_control(program)).hexdigest(), program.digest(),
    )


def validate_microop_tokens(
    tokens: ProgramMicroopTokens, program: ControlProgram,
) -> None:
    if not isinstance(tokens, ProgramMicroopTokens):
        raise MicroopAdmissionError("Unexpected token representation")
    count = len(program.operations)
    if (tokens.length != count or count > MAX_OPERATIONS
            or tokens.source_wasm_sha256 != hashlib.sha256(
                encode_control(program)
            ).hexdigest()
            or tokens.source_program_sha256 != program.digest()):
        raise MicroopAdmissionError("Source/structural token identity mismatch")
    for value, shape in (
        (tokens.opcode_rows, (SLOTS, len(OPCODES))),
        (tokens.i32_immediates, (SLOTS, 32)),
        (tokens.local_selectors, (SLOTS, LOCALS_WIDTH)),
        (tokens.branch_depths, (SLOTS, MAX_NESTING)),
    ):
        _binary(value, shape)
        if np.any(value[count:]):
            raise MicroopAdmissionError("Noncanonical padded opcode data")
    if np.any(tokens.opcode_rows[:count].sum(axis=1) != 1.):
        raise MicroopAdmissionError("Non-onehot source opcode row")
    expected = compile_microop_tokens(program)
    if any(not np.array_equal(x, y) for x, y in (
        (tokens.opcode_rows, expected.opcode_rows),
        (tokens.i32_immediates, expected.i32_immediates),
        (tokens.local_selectors, expected.local_selectors),
        (tokens.branch_depths, expected.branch_depths),
    )):
        raise MicroopAdmissionError("Program tokens differ from canonical WASM")


def read_microop(
    pc: np.ndarray,
    tokens: ProgramMicroopTokens,
    fixed: MicroopWeights | None = None,
) -> SelectedMicroop:
    """Shared fixed-weight decoder, source-token addressing by one-hot PC."""
    _onehot(pc, (SLOTS,))
    fixed = construct_microop_weights() if fixed is None else fixed
    if not isinstance(fixed, MicroopWeights):
        raise MicroopAdmissionError("Unexpected microop weights")
    for value, shape in (
        (fixed.effects, (len(OPCODES), len(EFFECTS))),
        (fixed.local_modes, (len(OPCODES), len(LOCAL_MODES))),
        (fixed.alu_selectors, (len(OPCODES), len(ALU_OPS))),
    ):
        _binary(value, shape)
    opcode = pc @ tokens.opcode_rows
    _onehot(opcode, (len(OPCODES),))
    effect = opcode @ fixed.effects
    mode = opcode @ fixed.local_modes
    alu = opcode @ fixed.alu_selectors
    _onehot(effect, (len(EFFECTS),))
    _onehot(mode, (len(LOCAL_MODES),))
    _onehot(alu, (len(ALU_OPS),), allow_zero=True)
    if alu.sum() != sum(opcode[OPCODES.index(name)] for name in ALU_OPS):
        raise MicroopAdmissionError("ALU selector presence disagrees with opcode")
    return SelectedMicroop(
        opcode, effect, mode, alu,
        pc @ tokens.i32_immediates,
        pc @ tokens.local_selectors,
        pc @ tokens.branch_depths,
    )
