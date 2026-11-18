"""v58 fixed-weight typed microstep for straight-line WASM data effects.

A single source-bound microop drives operand consumption, i32 ALU, immediate
materialization, local get/set/tee, and result push through constructed
binary gates. No runtime Python opcode branch selects the data operation.

This circuit admits *data instructions only*; structural control is a
separate v52-v54 subsystem. The v58 straight-line executor is a restricted
integration gate, NOT a general WASM interpreter or native accelerator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_binary import is_exact_binary
from .fused_state_v58 import local_access, pop_pair, push_one

from .bit_neural_vm_v49 import WIDTH, BitCircuitWeights, construct_bit_weights
from .neural_alu_v56 import (
    ALU_OPS,
    ALUWeights,
    construct_alu_weights,
    execute_alu,
)
from .neural_local_bank_v57 import (
    LocalBankWeights,
    construct_local_bank_weights,
    read_local,
    write_local,
)
from .neural_operand_storage_v55 import (
    ACTIONS,
    OperandState,
    OperandWeights,
    fixed_operand_weights,
    onehot,
    step_operands,
    top_word,
    validate_operands,
)
from .typed_microop_ir_v57 import (
    EFFECTS, LOCALS_WIDTH, LOCAL_MODES, OPCODES, MicroopAdmissionError,
    SelectedMicroop,
)

DATA_OPS = frozenset((
    "nop", "i32.const", "drop", "local.get", "local.set", "local.tee",
    *ALU_OPS,
))


@dataclass(frozen=True)
class MicrostepWeights:
    """Fixed coefficients shared across every program and microstep."""

    pop_gain: np.ndarray  # [2] for first and second operand pops
    produce_gain: np.ndarray  # [3] immediate, local-read, ALU
    write_gain: np.ndarray  # [1] local-bank write mask


@dataclass(frozen=True)
class MicrostepResult:
    operands: OperandState
    locals: np.ndarray
    consumed_count: int
    produced_count: int


def construct_microstep_weights() -> MicrostepWeights:
    return MicrostepWeights(
        np.ones(2, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
    )


def _binary(a: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(a, np.ndarray) or a.dtype != np.float64
            or a.shape != shape or not is_exact_binary(a)):
        raise MicroopAdmissionError("Noncanonical v58 microstep tensor")


def _hot(a: np.ndarray, shape: tuple[int, ...]) -> None:
    _binary(a, shape)
    if a.sum() != 1.:
        raise MicroopAdmissionError("Expected one-hot fixed microstep selector")


def step_data_microop(
    operands: OperandState,
    locals_: np.ndarray,
    token: SelectedMicroop,
    *,
    micro_fixed: MicrostepWeights | None = None,
    operand_fixed: OperandWeights | None = None,
    local_fixed: LocalBankWeights | None = None,
    alu_fixed: ALUWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    use_compiled_alu: bool = False,
    use_fused_state: bool = False,
) -> MicrostepResult:
    """Execute a fixed-weight data microstep without Python opcode dispatch.

    Every candidate datapath is computed; masks select final state.
    Validation may reject malformed/unsupported data but cannot substitute
    host-computed values for a corrupted neural output.
    """
    if not isinstance(token, SelectedMicroop):
        raise MicroopAdmissionError("Unknown typed microstep")
    validate_operands(operands)
    _binary(locals_, (LOCALS_WIDTH, WIDTH))
    _hot(token.opcode, (len(OPCODES),))
    _hot(token.effect, (len(EFFECTS),))
    _hot(token.local_mode, (len(LOCAL_MODES),))
    _binary(token.alu_selector, (len(ALU_OPS),))
    _binary(token.immediate_bits, (WIDTH,))
    _binary(token.local_selector, (LOCALS_WIDTH,))
    if token.alu_selector.sum() not in (0., 1.):
        raise MicroopAdmissionError("Invalid v58 ALU selector")
    static_data = np.array(
        [float(name in DATA_OPS) for name in OPCODES],
        dtype=np.float64,
    )
    if token.opcode @ static_data != 1.:
        raise MicroopAdmissionError("Control opcode outside v58 data microstep")

    fixed = construct_microstep_weights() if micro_fixed is None else micro_fixed
    if not isinstance(fixed, MicrostepWeights):
        raise MicroopAdmissionError("Unknown v58 fixed coefficients")
    for array, shape in (
        (fixed.pop_gain, (2,)),
        (fixed.produce_gain, (3,)),
        (fixed.write_gain, (1,)),
    ):
        _binary(array, shape)

    operand_fixed = (
        fixed_operand_weights() if operand_fixed is None else operand_fixed
    )
    local_fixed = construct_local_bank_weights() if local_fixed is None else local_fixed
    alu_fixed = construct_alu_weights() if alu_fixed is None else alu_fixed
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    # Compile the two pop gates from the source-bound effect class.
    effect = token.effect
    first = (effect[2] + effect[3] + effect[4]) * fixed.pop_gain[0]
    second = effect[4] * fixed.pop_gain[1]
    # Tee reads the original top, before the two pop gates take effect.
    if use_fused_state:
        after_second, top, right, left = pop_pair(operands, first, second, operand_fixed)
    else:
        pop_first = np.array((1. - first, 0., first, 0.), dtype=np.float64)
        pop_second = np.array((1. - second, 0., second, 0.), dtype=np.float64)
        _hot(pop_first, (len(ACTIONS),))
        _hot(pop_second, (len(ACTIONS),))
        top = top_word(operands)
        after_first, right = step_operands(operands, pop_first, fixed=operand_fixed)
        after_second, left = step_operands(after_first, pop_second, fixed=operand_fixed)
    # Single-pop unary operators consume the first popped word; binary
    # operations use the second popped word as the left operand.
    alu_left = effect[3] * right + effect[4] * left
    alu_right = effect[4] * right
    _binary(alu_left, (WIDTH,))
    _binary(alu_right, (WIDTH,))

    is_alu = token.alu_selector.sum()
    default = onehot(ALU_OPS.index("i32.add"), len(ALU_OPS))
    admitted_alu = token.alu_selector + (1. - is_alu) * default
    _hot(admitted_alu, (len(ALU_OPS),))
    candidate_alu = execute_alu(
        admitted_alu, alu_left, alu_right,
        fixed=alu_fixed, bit_fixed=bit_fixed, use_compiled_candidates=use_compiled_alu,
    )

    # Read and candidate-write are always evaluated using valid one-hot
    # addressing. Nonlocal opcodes select slot 0 but discard their result.
    local_mode = token.local_mode
    active_local = 1. - local_mode[0]
    default_local = onehot(0, LOCALS_WIDTH)
    local_slot = token.local_selector + (1. - active_local) * default_local
    _hot(local_slot, (LOCALS_WIDTH,))
    if not use_fused_state:
        local_read = read_local(locals_, local_slot, fixed=local_fixed)
    local_write = local_mode[2] * right + local_mode[3] * top
    _binary(local_write, (WIDTH,))
    write_mask = (local_mode[2] + local_mode[3]) * fixed.write_gain[0]
    if use_fused_state:
        local_read, result_bank = local_access(
            locals_, local_slot, local_write, write_mask, local_fixed,
        )
    else:
        modified_bank = write_local(locals_, local_slot, local_write, fixed=local_fixed)
        result_bank = (1. - write_mask) * locals_ + write_mask * modified_bank
    _binary(result_bank, (LOCALS_WIDTH, WIDTH))

    const_mask = token.opcode[OPCODES.index("i32.const")]
    local_read_mask = local_mode[1]
    sources = np.array((
        const_mask, local_read_mask, is_alu,
    )) * fixed.produce_gain
    candidate = (
        sources[0] * token.immediate_bits +
        sources[1] * local_read +
        sources[2] * candidate_alu
    )
    _binary(candidate, (WIDTH,))
    push_expected = effect[1] + effect[3] + effect[4]
    if (sources.sum() != push_expected
            or first + second != effect[2] + effect[3] + 2. * effect[4]):
        raise MicroopAdmissionError("Fixed microstep lost stack effect authority")
    if use_fused_state:
        final = push_one(after_second, push_expected, candidate, operand_fixed)
    else:
        push = np.array((1. - push_expected, push_expected, 0., 0.), dtype=np.float64)
        _hot(push, (len(ACTIONS),))
        final, _ = step_operands(after_second, push, value=candidate, fixed=operand_fixed)
    return MicrostepResult(
        final, result_bank, int(first + second), int(push_expected),
    )
