"""v55 fixed-capacity, bit-sliced neural operand-stack storage prototype.

Finite one-hot addressing and fixed bilinear action mixing operate on exact
i32 bit-vectors. No host Python list determines the next stored operand
state. Host validation still admits action and truncation-height tokens.
This component is NOT integrated with the v54 WASM executor yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_binary import is_exact_binary

from .wasm_control_flow_v51 import MAX_STACK

WIDTH = 32
HEIGHTS = MAX_STACK + 1
ACTIONS = ("retain", "push", "pop", "truncate")


class OperandCircuitError(ValueError):
    """Invalid one-hot stack height, bit-data, action, or circuit coefficients."""


@dataclass(frozen=True)
class OperandWeights:
    action_gain: np.ndarray


@dataclass(frozen=True)
class OperandState:
    height: np.ndarray  # [0..MAX_STACK] one-hot
    cells: np.ndarray  # [MAX_STACK,32] bit-sliced i32 values


def onehot(position: int, length: int) -> np.ndarray:
    if type(position) is not int or not 0 <= position < length:
        raise OperandCircuitError("Invalid bounded token position")
    result = np.zeros(length, dtype=np.float64)
    result[position] = 1.
    return result


def fixed_operand_weights() -> OperandWeights:
    return OperandWeights(np.ones(len(ACTIONS), dtype=np.float64))


def empty_operands() -> OperandState:
    return OperandState(
        onehot(0, HEIGHTS),
        np.zeros((MAX_STACK, WIDTH), dtype=np.float64),
    )


_UP = np.zeros((HEIGHTS, HEIGHTS), dtype=np.float64)
_DOWN = np.zeros((HEIGHTS, HEIGHTS), dtype=np.float64)
_PREFIX = np.zeros((HEIGHTS, MAX_STACK), dtype=np.float64)
_PREVIOUS = np.zeros((HEIGHTS, MAX_STACK), dtype=np.float64)
for depth in range(HEIGHTS):
    _UP[depth, min(depth + 1, MAX_STACK)] = 1.
    _DOWN[depth, max(depth - 1, 0)] = 1.
    _PREFIX[depth, :depth] = 1.
    if depth:
        _PREVIOUS[depth, depth - 1] = 1.


def _binary(value: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.shape != shape or not is_exact_binary(value)):
        raise OperandCircuitError("Noncanonical binary operand tensor")


def _hot(value: np.ndarray, shape: tuple[int, ...]) -> None:
    _binary(value, shape)
    if value.sum() != 1.:
        raise OperandCircuitError("Noncanonical one-hot operand token")


def validate_operands(state: OperandState) -> None:
    if not isinstance(state, OperandState):
        raise OperandCircuitError("Unknown operand state")
    _hot(state.height, (HEIGHTS,))
    _binary(state.cells, (MAX_STACK, WIDTH))
    live = state.height @ _PREFIX
    if np.any(state.cells * (1. - live[:, None])):
        raise OperandCircuitError("Nonzero inactive operand slots")


def top_word(state: OperandState) -> np.ndarray:
    """Fixed one-hot dot-product read of the previous live slot."""
    validate_operands(state)
    return state.height @ _PREVIOUS @ state.cells


def step_operands(
    state: OperandState,
    action: np.ndarray,
    *,
    value: np.ndarray | None = None,
    truncate_height: np.ndarray | None = None,
    fixed: OperandWeights | None = None,
) -> tuple[OperandState, np.ndarray]:
    """Apply the admitted one-hot event and its fixed-weight gain.

    The output second value is the popped 32-bit word for a pop event,
    otherwise zero. Unit-weight push/pop use the sparse form of the
    contraction; other actions and non-unit active gains use all
    candidates and the general fixed action-gain mixing.
    """
    validate_operands(state)
    _hot(action, (len(ACTIONS),))
    value = (
        np.zeros(WIDTH, dtype=np.float64) if value is None else value
    )
    truncate_height = (
        state.height.copy() if truncate_height is None else truncate_height
    )
    _binary(value, (WIDTH,))
    _hot(truncate_height, (HEIGHTS,))
    fixed = fixed_operand_weights() if fixed is None else fixed
    if not isinstance(fixed, OperandWeights):
        raise OperandCircuitError("Unknown circuit weights")
    _binary(fixed.action_gain, (len(ACTIONS),))
    depth = int(np.argmax(state.height))
    target = int(np.argmax(truncate_height))
    if action[1] and depth == MAX_STACK:
        raise OperandCircuitError("Bounded operand stack overflow")
    if action[2] and depth == 0:
        raise OperandCircuitError("Bounded operand stack underflow")
    if action[3] and target > depth:
        raise OperandCircuitError("Truncation must not grow the live stack")

    # A unique one-hot height selects exactly one row, and a unit
    # action gain discards every other candidate. Materialize only the
    # selected transition. Keep independent output buffers, including
    # the popped word, because callers may retain or mutate snapshots.
    push, pop = action[1:3]
    if (push * fixed.action_gain[1] + pop * fixed.action_gain[2] == 1.
            and all(type(array) is np.ndarray for array in
                    (state.height, state.cells, action, value, truncate_height, fixed.action_gain))):
        cells = state.cells.copy()
        if push:
            cells[depth] = value
            result = OperandState(state.height @ _UP, cells)
            returned = np.zeros(WIDTH, dtype=np.float64)
        else:
            returned = cells[depth - 1].copy()
            cells[depth - 1] = 0.
            result = OperandState(state.height @ _DOWN, cells)
        validate_operands(result)
        return result, returned

    push_slot = state.height[:MAX_STACK]
    pop_slot = state.height @ _PREVIOUS
    pop_value = pop_slot @ state.cells

    push_cells = (
        state.cells * (1. - push_slot[:, None])
        + push_slot[:, None] * value[None, :]
    )
    pop_cells = state.cells * (1. - pop_slot[:, None])
    retain_on_truncate = truncate_height @ _PREFIX
    trim_cells = state.cells * retain_on_truncate[:, None]

    choices_height = np.stack((
        state.height,
        state.height @ _UP,
        state.height @ _DOWN,
        truncate_height,
    ))
    choices_cells = np.stack((
        state.cells, push_cells, pop_cells, trim_cells,
    ))
    gates = action * fixed.action_gain
    result = OperandState(
        gates @ choices_height,
        np.einsum("a,ajk->jk", gates, choices_cells),
    )
    validate_operands(result)
    returned = action[2] * pop_value
    _binary(returned, (WIDTH,))
    return result, returned
