"""v54 constructed fixed-weight frame-event decoder for bounded WASM.

No Python opcode conditional chooses a frame action. The reviewed opcode is
one-hot PROGRAM DATA; branch decisions are live bits from v52 routing and
the current v53 frame depth. This component is not yet wired to execution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPCODES = (
    "block", "loop", "if", "else", "end", "br", "br_if", "return", "other",
)
ACTIONS = ("retain", "push", "pop", "branch", "return")
DEFAULT = (1, 1, 0, 2, 0, 3, 0, 4, 0)


class EventCircuitError(ValueError):
    """Malformed program token or noncanonical frame event output."""


@dataclass(frozen=True)
class EventWeights:
    default: np.ndarray
    conditional_gains: np.ndarray


def onehot(index: int, size: int) -> np.ndarray:
    if type(index) is not int or not 0 <= index < size:
        raise EventCircuitError("Invalid static token index")
    result = np.zeros(size, dtype=np.float64)
    result[index] = 1.
    return result


def fixed_event_weights() -> EventWeights:
    default = np.zeros((len(OPCODES), len(ACTIONS)), dtype=np.float64)
    for opcode, action in enumerate(DEFAULT):
        default[opcode, action] = 1.
    return EventWeights(default, np.ones(3, dtype=np.float64))


def _admit(array: np.ndarray, shape: tuple[int, ...], *, one: bool) -> None:
    if (not isinstance(array, np.ndarray) or array.dtype != np.float64
            or array.shape != shape or not np.isfinite(array).all()
            or not np.isin(array, (0., 1.)).all()
            or (one and array.sum() != 1.)):
        raise EventCircuitError("Noncanonical binary/one-hot event tensor")


def select_frame_action(
    opcode: np.ndarray,
    *,
    conditional_sequential: np.ndarray,
    has_else: np.ndarray,
    frame_active: np.ndarray,
    branch_taken: np.ndarray,
    weights: EventWeights | None = None,
) -> np.ndarray:
    """Choose one-hot frame event using fixed bilinear/ReLU selectors.

    All four guard vectors are binary shape-(1,) gate data. For 'if',
    sequential indicates a true arm; 'has_else' preserves false-arm label.
    The caller must source branch_taken from the v52 neural PC circuit.
    """
    weights = fixed_event_weights() if weights is None else weights
    if not isinstance(weights, EventWeights):
        raise EventCircuitError("Unrecognized fixed circuit")
    _admit(opcode, (len(OPCODES),), one=True)
    for value in (
        conditional_sequential, has_else, frame_active, branch_taken
    ):
        _admit(value, (1,), one=False)
    _admit(weights.default, (len(OPCODES), len(ACTIONS)), one=False)
    _admit(weights.conditional_gains, (3,), one=False)
    if not np.all(weights.default.sum(axis=1) == 1.):
        raise EventCircuitError("Corrupt opcode action matrix")

    # Fixed ReLU OR of true-arm selection and lexical else presence:
    # OR(a,b) = a + b - ReLU(a+b-1), exact for binary a,b.
    true_arm = conditional_sequential[0]
    else_present = has_else[0]
    enter_if = true_arm + else_present - np.maximum(
        true_arm + else_present - 1., 0.
    )
    shifts = np.array((
        opcode[2] * enter_if * weights.conditional_gains[0],
        opcode[4] * frame_active[0] * weights.conditional_gains[1],
        opcode[6] * branch_taken[0] * weights.conditional_gains[2],
    ))
    chosen = opcode @ weights.default
    # Change exactly one default RETAIN path to its conditional action.
    chosen = chosen + np.array((
        -shifts.sum(), shifts[0], shifts[1], shifts[2], 0.,
    ))
    _admit(chosen, (len(ACTIONS),), one=True)
    return chosen


def decode_opcode_token(opcode: str) -> np.ndarray:
    """Host lexical admission only; runtime event selection is circuit-based."""
    return onehot(
        OPCODES.index(opcode) if opcode in OPCODES[:-1] else len(OPCODES)-1,
        len(OPCODES),
    )
