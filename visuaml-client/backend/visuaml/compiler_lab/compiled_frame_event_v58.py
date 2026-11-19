"""An invocation-owned binding of the v54 frame-event gate graph."""
from __future__ import annotations

import numpy as np

from .constructed_alu_v58 import compiled_alu
from .neural_frame_events_v54 import ACTIONS, OPCODES, EventCircuitError, EventWeights


class CompiledFrameEvents:
    """Bind admitted source opcode rows and consume four binary gate inputs."""

    def __init__(self, opcode_tokens: tuple[np.ndarray, ...], weights: EventWeights):
        opcodes = np.stack(opcode_tokens)
        if (opcodes.shape != (len(opcode_tokens), len(OPCODES))
                or opcodes.dtype != np.float64
                or weights.default.shape != (len(OPCODES), len(ACTIONS))
                or weights.conditional_gains.shape != (3,)):
            raise EventCircuitError("Invalid compiled event gate dimensions")
        # Contract the two source-only operands of the reference graph.
        default = opcodes @ weights.default
        conditional = opcodes[:, [2, 4, 6]] * weights.conditional_gains
        self._rows = np.ascontiguousarray(np.concatenate((default, conditional), axis=1), dtype=np.float64)
        self._rows.flags.writeable = False
        self._row_pointers = tuple(self._rows.ctypes.data + i * self._rows.strides[0]
                                   for i in range(len(self._rows)))
        kernels = compiled_alu()
        self._kernel = kernels.frame_event
        self.binary_sha256 = kernels.binary_sha256
        # Each invocation consumes its output before the next call. It is
        # never published as an execution result or retained by step_frames.
        self.vector = np.empty(len(ACTIONS), dtype=np.float64)
        self._output_pointer = self.vector.ctypes.data

    def select(self, pc: int, sequential: bool, has_else: bool, frame_active: bool, branch_taken: bool) -> int:
        action = self._kernel(self._row_pointers[pc], sequential, has_else, frame_active,
                              branch_taken, self._output_pointer)
        if action < 0:
            raise EventCircuitError("Noncanonical binary/one-hot event tensor")
        return action
