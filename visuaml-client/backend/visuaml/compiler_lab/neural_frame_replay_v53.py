"""v53 executable-trace differential audit for the constructed frame circuit.

Consumes *actual* v52 WASM routed execution, replays each frame transition
with fixed coefficients, and checks every frame depth/kind and branch stack
restoration. This is an independent shadow circuit, NOT YET the authority
driving v52's runtime frame state or physical operand stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neural_control_routing_v52 import RoutedResult, execute_routed_control
from .neural_frame_circuit_v53 import (
    ACTIONS,
    HEIGHTS,
    KINDS,
    MAX_NESTING,
    FrameCircuitError,
    FrameCircuitWeights,
    FrameTokens,
    empty_frames,
    fixed_frame_weights,
    onehot,
    step_frames,
)
from .wasm_control_flow_v51 import ControlProgram, _syntax

FRAME_KINDS = ("block", "loop", "if")


@dataclass(frozen=True)
class FrameReplayResult:
    routed: RoutedResult
    frame_transitions: int
    branch_restorations_checked: int
    deepest_frame: int
    fixed_frame_weights: tuple[float, ...]


def _optional(index: int | None, width: int) -> np.ndarray:
    return (np.zeros(width, dtype=np.float64)
            if index is None else onehot(index, width))


def replay_routed_frames(
    program: ControlProgram, argument: int, *, fuel: int = 1500,
    weights: FrameCircuitWeights | None = None,
) -> FrameReplayResult:
    """Audit the live v52 trace against independently evolved v53 frame state."""
    links = _syntax(program)
    weights = fixed_frame_weights() if weights is None else weights
    routed = execute_routed_control(program, argument, fuel=fuel)
    state = empty_frames()
    restorations = 0
    deepest = 0
    for record in routed.trace:
        pc = record["pc"]
        operation = program.operations[pc]
        opcode = operation.opcode
        next_pc = record["next_pc"]
        depth = int(np.argmax(state.depth))
        action = 0
        kind = None
        base = None
        branch_depth = None
        if opcode in FRAME_KINDS:
            # A false if WITHOUT else skips the entire frame. A false if
            # WITH else must preserve the frame until the matching end.
            if (opcode != "if" or links[pc][1] is not None
                    or next_pc == pc + 1):
                action = 1
                kind = FRAME_KINDS.index(opcode)
                base = len(record["stack_i32"])
        elif opcode == "else" or (opcode == "end" and depth):
            action = 2
        elif opcode in ("br", "br_if"):
            if record["neural_route"]["branch_guard_gate"]:
                action = 3
                branch_depth = operation.immediate
        elif opcode == "return":
            action = 4
        height = len(record["stack_i32"])
        if not 0 <= height < HEIGHTS:
            raise FrameCircuitError("Executed operand height exceeds circuit capacity")
        tokens = FrameTokens(
            onehot(action, ACTIONS),
            _optional(kind, KINDS),
            _optional(base, HEIGHTS),
            _optional(branch_depth, MAX_NESTING),
            onehot(height, HEIGHTS),
        )
        state, restored = step_frames(state, tokens, weights)
        labels = [
            FRAME_KINDS[int(np.argmax(row))]
            for row in state.kinds
            if row.sum()
        ]
        if labels != record["control_labels"]:
            raise FrameCircuitError(
                f"Constructed frame mismatch at PC {pc}: "
                f"{labels} != {record['control_labels']}"
            )
        if action == 3:
            restorations += 1
            if int(np.argmax(restored)) != height:
                raise FrameCircuitError(
                    f"Constructed operand restoration mismatch at PC {pc}"
                )
        deepest = max(deepest, int(np.argmax(state.depth)))
    if int(np.argmax(state.depth)) != 0:
        raise FrameCircuitError("Executed program halted with live frames")
    return FrameReplayResult(
        routed, len(routed.trace), restorations, deepest,
        tuple(float(x) for x in weights.action_weights),
    )
