"""Host-facing protocol for v55 tensor-backed constructed operand storage.

This adapter never keeps a parallel Python list of live i32 values. It
delegates every push, pop and truncate to the fixed-weight v55 circuit.
Host indexing is limited to the already-decoded top-of-stack operation,
trace enumeration and structural stack-height checks.
"""
from __future__ import annotations

import numpy as np

from .neural_operand_storage_v55 import (
    ACTIONS,
    HEIGHTS,
    OperandCircuitError,
    OperandState,
    OperandWeights,
    _unit_pop,
    empty_operands,
    fixed_operand_weights,
    onehot,
    step_operands,
    top_word,
    validate_operands,
)


class TensorOperandStack:
    """A bounded storage protocol whose only live backing is OperandState."""

    def __init__(self, fixed: OperandWeights | None = None):
        self.fixed = fixed_operand_weights() if fixed is None else fixed
        self.state: OperandState = empty_operands()

    def __len__(self) -> int:
        return int(np.argmax(self.state.height))

    def __bool__(self) -> bool:
        return len(self) != 0

    def append(self, bits: np.ndarray) -> None:
        self.state, _ = step_operands(
            self.state, onehot(1, len(ACTIONS)),
            value=bits, fixed=self.fixed,
        )

    def pop(self) -> np.ndarray:
        self.state, popped = step_operands(
            self.state, onehot(2, len(ACTIONS)), fixed=self.fixed,
        )
        return popped

    def __getitem__(self, index: int) -> np.ndarray:
        if type(index) is not int or index != -1 or not self:
            raise OperandCircuitError("Only live top-of-stack read is admitted")
        return top_word(self.state)

    def __delitem__(self, index: slice) -> None:
        if (not isinstance(index, slice) or index.stop is not None
                or index.step is not None or type(index.start) is not int):
            raise OperandCircuitError("Only exact tail truncation admitted")
        self.state, _ = step_operands(
            self.state, onehot(3, len(ACTIONS)),
            truncate_height=onehot(index.start, HEIGHTS),
            fixed=self.fixed,
        )

    def __iter__(self):
        # Host trace observation only. Numerical execution never reads a
        # Python-owned parallel list of operand values.
        for index in range(len(self)):
            yield self.state.cells[index].copy()


class _PreparedPopStack(TensorOperandStack):
    """Invocation-owned operand state with the canonical v55 pop prepared."""

    def __init__(self):
        super().__init__()
        validate_operands(self.state)

    def pop(self) -> np.ndarray:
        depth = len(self)
        if depth == 0:
            raise OperandCircuitError("Bounded operand stack underflow")
        # Entry is checked. Inherited push and truncation validate their
        # outputs on every edge, including branch restoration. This pop
        # clears exactly the retired row and applies the fixed down matrix,
        # so binary live words and zero inactive rows hold inductively.
        self.state, popped = _unit_pop(self.state, depth)
        return popped
