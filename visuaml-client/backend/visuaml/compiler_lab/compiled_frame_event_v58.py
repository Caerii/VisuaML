"""An invocation-owned binding of the v54 frame-event gate graph."""
from __future__ import annotations

import numpy as np

from .constructed_alu_v58 import compiled_alu
from .neural_frame_circuit_v53 import FrameState, validate_state
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


class _CompiledFrameRetain(CompiledFrameEvents):
    """Keep canonical frames in private slabs across individual instructions."""

    def __init__(self, opcode_tokens, weights, initial: FrameState, *, resident: bool):
        super().__init__(opcode_tokens, weights)
        validate_state(initial)
        shapes = (initial.depth.shape, initial.kinds.shape, initial.bases.shape)
        cuts = np.cumsum((initial.depth.size, initial.kinds.size, initial.bases.size))
        self._elements = int(cuts[-1])
        self._buffers = tuple(np.empty(self._elements, dtype=np.float64)
                              for _ in range(1 if resident else 2))
        self._views = tuple(FrameState(*(part.reshape(shape) for part, shape in
                                        zip(np.split(slab, cuts[:-1]), shapes, strict=True)))
                            for slab in self._buffers)
        self._pointers = tuple(slab.ctypes.data for slab in self._buffers)
        self._active = 0
        self._resident = resident
        self._retain_kernel = compiled_alu().frame_event_retain
        self.publish_checked(initial)

    @property
    def state(self) -> FrameState:
        return self._views[self._active]

    def publish_checked(self, state: FrameState) -> None:
        """Import a state already validated by the original v53 transition."""
        current = self.state
        current.depth[:] = state.depth
        current.kinds[:] = state.kinds
        current.bases[:] = state.bases

    def select(self, pc: int, sequential: bool, has_else: bool, frame_active: bool, branch_taken: bool) -> int:
        destination = None if self._resident else self._pointers[1 - self._active]
        action = self._retain_kernel(
            self._row_pointers[pc], sequential, has_else, frame_active, branch_taken,
            self._output_pointer, self._pointers[self._active], destination, self._elements,
        )
        if action < 0:
            raise EventCircuitError("Noncanonical binary/one-hot event tensor")
        if action == 0 and not self._resident:
            self._active = 1 - self._active
        return action
