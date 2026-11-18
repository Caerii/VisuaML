"""Factor the straight-line v58 operand and register gate equations."""
from __future__ import annotations

import numpy as np

from .neural_local_bank_v57 import LocalBankWeights, _binary as _local_binary
from .neural_operand_storage_v55 import (
    ACTIONS, MAX_STACK, _DOWN, _PREFIX, _PREVIOUS, _UP,
    OperandCircuitError, OperandState, OperandWeights, _binary, validate_operands,
)
from .typed_microop_ir_v57 import LOCALS_WIDTH, MicroopAdmissionError


def pop_pair(state: OperandState, first: float, second: float, fixed: OperandWeights):
    """Contract the two retain/pop gates without constructing action tensors.

    The owning microstep has admitted the input state and the binary
    coefficients that form the two pop/retain gates. Validate the selected gains and bounds in pop order.
    """
    if not isinstance(fixed, OperandWeights):
        raise OperandCircuitError("Unknown circuit weights")
    _binary(fixed.action_gain, (len(ACTIONS),))
    depth = int(np.argmax(state.height))
    if first and depth == 0:
        raise OperandCircuitError("Bounded operand stack underflow")
    if (1. - first) * fixed.action_gain[0] + first * fixed.action_gain[2] != 1.:
        raise OperandCircuitError("Noncanonical one-hot operand token")
    if second and depth - first == 0:
        raise OperandCircuitError("Bounded operand stack underflow")
    if (1. - second) * fixed.action_gain[0] + second * fixed.action_gain[2] != 1.:
        raise OperandCircuitError("Noncanonical one-hot operand token")

    height = state.height
    first_slot = height @ _PREVIOUS
    top = first_slot @ state.cells
    after_first = (1. - first) * height + first * (height @ _DOWN)
    second_slot = after_first @ _PREVIOUS
    # When the first pop is active, the second slot lies strictly below it.
    # A retain leaves the original cells available to the second read.
    left = second * (second_slot @ state.cells)
    after_second = (1. - second) * after_first + second * (after_first @ _DOWN)
    # A pop only clears a suffix of an already validated prefix.
    cells = state.cells * (after_second @ _PREFIX)[:, None]
    return OperandState(after_second, cells), top, first * top, left


def push_one(state: OperandState, push: float, value: np.ndarray, fixed: OperandWeights):
    """Mix the final push and retain gates using the zero inactive slots."""
    if push and state.height[MAX_STACK]:
        raise OperandCircuitError("Bounded operand stack overflow")
    if (1. - push) * fixed.action_gain[0] + push * fixed.action_gain[1] != 1.:
        raise OperandCircuitError("Noncanonical one-hot operand token")
    height = (1. - push) * state.height + push * (state.height @ _UP)
    cells = state.cells + (push * state.height[:MAX_STACK, None]) * value[None, :]
    result = OperandState(height, cells)
    validate_operands(result)
    return result


def local_access(registers, selector, value, write, fixed: LocalBankWeights):
    """Mix a live local read and candidate write without copying an intermediate bank."""
    if selector.sum() != 1.:
        raise MicroopAdmissionError("Local read requires one-hot source")
    if not isinstance(fixed, LocalBankWeights):
        raise MicroopAdmissionError("Invalid local-bank weights")
    _local_binary(fixed.read_gain, (LOCALS_WIDTH,))
    read = selector * fixed.read_gain
    if read.sum() != 1.:
        raise MicroopAdmissionError("Local read gate ablated")
    _local_binary(fixed.write_gain, (LOCALS_WIDTH,))
    selected = selector * fixed.write_gain
    if selected.sum() != 1.:
        raise MicroopAdmissionError("Local write gate ablated")
    mask = (write * selected)[:, None]
    return read @ registers, (1. - mask) * registers + mask * value[None, :]
