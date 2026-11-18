"""Sparse evaluation of source-bound one-hot PC gates for the v58 executor."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bit_neural_vm_v49 import BitCircuitWeights, WIDTH
from .constructed_alu_v58 import compiled_alu
from .neural_control_routing_v52 import (
    SLOTS, TERMINAL, ControlRouteTokens, ControlRouterWeights, NeuralControlError,
    _onehot, validate_route_tokens,
)
from .wasm_control_flow_v51 import ControlProgram


def _snapshot(value, shape):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float64:
        raise NeuralControlError("Invalid compact routing tensor")
    return np.array(value, copy=True, order="C", subok=False)


class CompactRouter:
    """Program-scoped, admitted route tables and coefficient snapshots."""

    def __init__(self, program: ControlProgram, tokens: ControlRouteTokens,
                 fixed: ControlRouterWeights, bit_fixed: BitCircuitWeights):
        # Preserve canonical admission of both dense token matrices. A compact
        # row is valid only because each target was proven one-hot at admission.
        if not isinstance(tokens, ControlRouteTokens):
            raise NeuralControlError("Unknown program route-token representation")
        tokens = replace(
            tokens,
            sequential=_snapshot(tokens.sequential, (SLOTS, SLOTS)),
            alternate=_snapshot(tokens.alternate, (SLOTS, SLOTS)),
            guard_kind=_snapshot(tokens.guard_kind, (SLOTS, 3)),
        )
        validate_route_tokens(tokens, program)
        if not isinstance(fixed, ControlRouterWeights) or not isinstance(bit_fixed, BitCircuitWeights):
            raise NeuralControlError("Unknown fixed neural control weights")
        self._targets = np.column_stack((
            np.argmax(tokens.sequential, axis=1), np.argmax(tokens.alternate, axis=1),
        )).astype(np.int32)
        self._targets[tokens.instruction_count:TERMINAL] = -1
        self._tensors = (
            self._targets, tokens.guard_kind,
            _snapshot(fixed.guard_weights, (3,)), _snapshot(fixed.path_weights, (2,)),
            _snapshot(bit_fixed.and_kernel, (3,)),
        )
        if not all(np.isfinite(value).all() for value in self._tensors[2:]):
            raise NeuralControlError("Invalid neural gate coefficients")
        self._pointers = tuple(value.ctypes.data for value in self._tensors)
        self._route = compiled_alu().route

    def step(self, pc: np.ndarray, condition: np.ndarray, *, capture_trace: bool = True):
        # Copies isolate the read-only native inputs from external array writes.
        pc = _snapshot(pc, (SLOTS,))
        condition = _snapshot(condition, (WIDTH,))
        details = np.empty(5)
        status = self._route(
            pc.ctypes.data, SLOTS, condition.ctypes.data, *self._pointers, details.ctypes.data,
        )
        if status == 1:
            raise NeuralControlError("Invalid binary condition or noncanonical active PC")
        if status:
            raise NeuralControlError("Constructed route lost unique next PC")
        next_pc = int(details[0])
        return _onehot(next_pc), ({
            "nonzero_condition_gate": int(details[1]),
            "branch_guard_gate": int(details[2]),
            "sequential_target": int(details[3]),
            "alternative_target": int(details[4]),
            "next_pc": next_pc,
        } if capture_trace else None)
