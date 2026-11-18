"""v53 prototype: fixed-capacity, shared-weight control-frame transition circuit.

This is an ISOLATED component. The v52 production executor still handles
frames on the host; do not describe this as full neural WASM execution.
Action/branch tokens are inputs; the circuit never dispatches on Python
opcode strings and never selects an event path using Python conditionals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wasm_control_flow_v51 import MAX_NESTING, MAX_STACK

KINDS = 3  # block, loop, if
ACTIONS = 5  # retain, push, pop, branch, return
HEIGHTS = MAX_STACK + 1
LEVELS = MAX_NESTING + 1


class FrameCircuitError(ValueError):
    """Noncanonical constructed-frame state, tokens, or output."""


@dataclass(frozen=True)
class FrameCircuitWeights:
    """One fixed set of coefficients for every WASM source program."""

    action_weights: np.ndarray


@dataclass(frozen=True)
class FrameState:
    depth: np.ndarray  # [LEVELS] one-hot
    kinds: np.ndarray  # [MAX_NESTING, 3] inactive row = all zero
    bases: np.ndarray  # [MAX_NESTING, HEIGHTS] inactive row = all zero


@dataclass(frozen=True)
class FrameTokens:
    action: np.ndarray  # [retain, push, pop, branch, return]
    kind: np.ndarray  # new pushed kind [block, loop, if]
    base: np.ndarray  # one-hot stack height on entering
    branch_depth: np.ndarray  # one-hot lexical depth for branch
    operand_height: np.ndarray  # externally updated live stack height


def onehot(index: int, width: int) -> np.ndarray:
    if type(index) is not int or not 0 <= index < width:
        raise FrameCircuitError("Invalid one-hot index")
    result = np.zeros(width, dtype=np.float64)
    result[index] = 1.
    return result


def fixed_frame_weights() -> FrameCircuitWeights:
    return FrameCircuitWeights(np.ones(ACTIONS, dtype=np.float64))


def empty_frames() -> FrameState:
    return FrameState(
        onehot(0, LEVELS),
        np.zeros((MAX_NESTING, KINDS), dtype=np.float64),
        np.zeros((MAX_NESTING, HEIGHTS), dtype=np.float64),
    )


def _binary(array: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(array, np.ndarray) or array.dtype != np.float64
            or array.shape != shape or not np.isfinite(array).all()
            or not np.isin(array, (0., 1.)).all()):
        raise FrameCircuitError("Noncanonical binary tensor")


def _hot(array: np.ndarray, shape: tuple[int, ...]) -> None:
    _binary(array, shape)
    if float(array.sum()) != 1.:
        raise FrameCircuitError("Noncanonical one-hot tensor")


def validate_state(state: FrameState) -> None:
    if not isinstance(state, FrameState):
        raise FrameCircuitError("Unknown frame state")
    _hot(state.depth, (LEVELS,))
    _binary(state.kinds, (MAX_NESTING, KINDS))
    _binary(state.bases, (MAX_NESTING, HEIGHTS))
    # A fixed prefix matrix computes active frame slots from one-hot depth.
    active = state.depth @ _PREFIX
    if (not np.array_equal(state.kinds.sum(axis=1), active)
            or not np.array_equal(state.bases.sum(axis=1), active)):
        raise FrameCircuitError("Frame metadata mismatches active depth")


_SHIFT_UP = np.zeros((LEVELS, LEVELS), dtype=np.float64)
_SHIFT_DOWN = np.zeros((LEVELS, LEVELS), dtype=np.float64)
_PREFIX = np.zeros((LEVELS, MAX_NESTING), dtype=np.float64)
_TARGET = np.zeros((LEVELS, MAX_NESTING, MAX_NESTING), dtype=np.float64)
for d in range(LEVELS):
    _SHIFT_UP[d, min(d + 1, MAX_NESTING)] = 1.
    _SHIFT_DOWN[d, max(d - 1, 0)] = 1.
    _PREFIX[d, :d] = 1.
    for k in range(d):
        _TARGET[d, k, d - 1 - k] = 1.


def step_frames(
    state: FrameState,
    tokens: FrameTokens,
    weights: FrameCircuitWeights | None = None,
) -> tuple[FrameState, np.ndarray]:
    """Compute ALL event candidates, then select using fixed bilinear masks.

    The returned height is an operand-stack truncation *request*. The caller
    still applies the actual stack data movement; integration is outstanding.
    """
    validate_state(state)
    if not isinstance(tokens, FrameTokens):
        raise FrameCircuitError("Unknown frame instruction data")
    for value, shape in (
        (tokens.action, (ACTIONS,)),
        (tokens.kind, (KINDS,)),
        (tokens.base, (HEIGHTS,)),
        (tokens.branch_depth, (MAX_NESTING,)),
        (tokens.operand_height, (HEIGHTS,)),
    ):
        _binary(value, shape)
    _hot(tokens.action, (ACTIONS,))
    _hot(tokens.operand_height, (HEIGHTS,))
    weights = fixed_frame_weights() if weights is None else weights
    if not isinstance(weights, FrameCircuitWeights):
        raise FrameCircuitError("Unknown weights")
    _binary(weights.action_weights, (ACTIONS,))

    d = int(np.argmax(state.depth))  # admission only; NOT event dispatch
    push, pop, branch = tokens.action[1:4]
    if push and (d >= MAX_NESTING
                 or tokens.kind.sum() != 1.
                 or tokens.base.sum() != 1.):
        raise FrameCircuitError("Cannot enter a noncanonical/full frame")
    if pop and d == 0:
        raise FrameCircuitError("Frame stack underflow")
    if branch and (d == 0 or tokens.branch_depth.sum() != 1.):
        raise FrameCircuitError("Missing target branch depth")

    # Fixed tensor contraction selects target frame from two input one-hots.
    target = np.einsum(
        "d,k,dki->i", state.depth, tokens.branch_depth, _TARGET,
        optimize=True,
    )
    if branch and target.sum() != 1.:
        raise FrameCircuitError("Branch target is not a live frame")
    target_kind = target @ state.kinds
    loop = target_kind[1]
    # For loops keep target frame; for blocks/ifs discard target frame.
    target_level = target @ np.eye(MAX_NESTING, LEVELS, k=0)
    target_level_plus = target @ np.eye(MAX_NESTING, LEVELS, k=1)
    branch_level = loop * target_level_plus + (1. - loop) * target_level

    retain_depth = state.depth
    pushed_depth = state.depth @ _SHIFT_UP
    popped_depth = state.depth @ _SHIFT_DOWN
    returned_depth = onehot(0, LEVELS)
    choices_depth = np.stack((
        retain_depth, pushed_depth, popped_depth,
        branch_level, returned_depth,
    ))

    # Construct all frame metadata candidates without an opcode dispatch.
    slot = state.depth[:MAX_NESTING]
    pushed_kinds = state.kinds + slot[:, None] * tokens.kind[None, :]
    pushed_bases = state.bases + slot[:, None] * tokens.base[None, :]
    popped_mask = popped_depth @ _PREFIX
    branched_mask = branch_level @ _PREFIX
    choices_kind = np.stack((
        state.kinds,
        pushed_kinds,
        state.kinds * popped_mask[:, None],
        state.kinds * branched_mask[:, None],
        np.zeros_like(state.kinds),
    ))
    choices_base = np.stack((
        state.bases,
        pushed_bases,
        state.bases * popped_mask[:, None],
        state.bases * branched_mask[:, None],
        np.zeros_like(state.bases),
    ))
    action = tokens.action * weights.action_weights
    out = FrameState(
        action @ choices_depth,
        np.einsum("a,ajk->jk", action, choices_kind),
        np.einsum("a,ajk->jk", action, choices_base),
    )
    restore = target @ state.bases
    # For branch, restore saved frame base; otherwise keep externally
    # supplied operand height. The actual data stack is not modified here.
    height = branch * restore + (1. - branch) * tokens.operand_height
    validate_state(out)
    _hot(height, (HEIGHTS,))
    return out, height
