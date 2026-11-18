"""v53 experimental frame-authoritative routed WASM executor.

Constructed v52 one-hot PC routing remains authoritative. The v53 finite
frame-circuit output now determines live label depth and actual branch
operand-stack restoration. Host still decodes/dispatches instructions,
derives frame event tokens, maintains lexical source metadata and physical
operand contents. Not universal WASM or a fully neural instruction decoder.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    BitCircuitWeights,
    _add,
    _bits,
    _eqz,
    _int,
    _mul,
    _signed_lt,
    _sub,
    construct_bit_weights,
)
from .neural_control_routing_v52 import (
    TERMINAL,
    ControlRouteTokens,
    ControlRouterWeights,
    _onehot,
    compile_route_tokens,
    construct_router_weights,
    neural_pc_step,
    validate_route_tokens,
)
from .neural_frame_circuit_v53 import (
    ACTIONS,
    HEIGHTS,
    KINDS,
    MAX_NESTING,
    FrameCircuitWeights,
    FrameTokens,
    empty_frames,
    fixed_frame_weights,
    onehot as frame_onehot,
    step_frames,
)
from .wasm_control_flow_v51 import (
    _BINARY,
    MAX_FUEL,
    MAX_STACK,
    ControlAdmissionError,
    ControlExecutionTrap,
    ControlFuelExhausted,
    ControlProgram,
    _syntax,
    encode_control,
)


@dataclass(frozen=True)
class FrameRoutedResult:
    result_i32: int
    trace: tuple[dict, ...]
    steps: int
    source_wasm_sha256: str
    fixed_router_weights_sha256: str
    fixed_arithmetic_weights_sha256: str
    program_route_tokens_sha256: str
    fixed_frame_weights: tuple[float, ...]
    pc_routed_by_constructed_gates: bool = True
    label_depth_driven_by_constructed_gates: bool = True
    stack_restoration_driven_by_constructed_gates: bool = True
    instruction_dispatch_host_interpreted: bool = True


def _frame_optional(index: int | None, width: int) -> np.ndarray:
    return (np.zeros(width, dtype=np.float64)
            if index is None else frame_onehot(index, width))


def _pop(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ControlExecutionTrap("Typed operand stack underflow")
    return stack.pop()


def execute_frame_routed_control(
    program: ControlProgram, argument: int, *, fuel: int = 500,
    tokens: ControlRouteTokens | None = None,
    fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    frame_fixed: FrameCircuitWeights | None = None,
) -> FrameRoutedResult:
    """EXECUTE using the one-hot route output as the next PC.

    v53 gate output owns frame depths and branch operand-stack restoration.
    Host still dispatches opcode arithmetic and carries lexical PC metadata.
    """
    _syntax(program)
    if type(fuel) is not int or not 1 <= fuel <= MAX_FUEL:
        raise ControlAdmissionError("Invalid execution fuel")
    tokens = compile_route_tokens(program) if tokens is None else tokens
    validate_route_tokens(tokens, program)
    fixed = construct_router_weights() if fixed is None else fixed
    bit_fixed = construct_bit_weights() if bit_fixed is None else bit_fixed
    links = _syntax(program)
    locals_ = [_bits(argument), *[
        _bits(0) for _ in range(program.local_count)
    ]]
    stack: list[np.ndarray] = []
    frame_state = empty_frames()
    frame_meta: list[tuple[str, int, int, int | None]] = []
    pc_vector = _onehot(0)
    frame_fixed = fixed_frame_weights() if frame_fixed is None else frame_fixed
    records = []
    count = len(program.operations)

    for tick in range(fuel):
        pc = int(np.argmax(pc_vector))
        if pc == TERMINAL or pc >= count:
            raise ControlExecutionTrap("Reached terminal without WASM return")
        op = program.operations[pc]
        kind, imm = op.opcode, op.immediate
        # Condition is read from the ACTUAL binary neural stack before
        # consuming the operand. The gate itself is independent of host
        # opcode branch selection and uses v49 fixed ReLU zero testing.
        condition = (
            stack[-1] if stack
            else np.zeros(32, dtype=np.float64)
        )
        selected, route = neural_pc_step(
            pc_vector, condition, tokens, fixed, bit_fixed,
        )
        next_pc = int(np.argmax(selected))
        sequential_pc = route["sequential_target"]
        taken = bool(route["branch_guard_gate"])
        halted = False

        action = 0  # retain, enter, close, branch, return
        entry_kind = None
        entry_base = None
        branch_depth = None
        entering_metadata = None
        depth_before = int(np.argmax(frame_state.depth))
        if kind in ("block", "loop", "if"):
            end, else_pc = links[pc]
            if kind == "if":
                _pop(stack)
            if kind != "if" or else_pc is not None or next_pc == sequential_pc:
                action = 1
                entry_kind = ("block", "loop", "if").index(kind)
                entry_base = len(stack)
                entering_metadata = (kind, pc, end, else_pc)
        elif kind == "else":
            if (not frame_meta or frame_meta[-1][0] != "if"
                    or frame_meta[-1][3] != pc):
                raise ControlExecutionTrap("Unexpected dynamic else")
            base = int(np.argmax(frame_state.bases[depth_before - 1]))
            if len(stack) != base:
                raise ControlExecutionTrap("Incorrect empty if-arm result")
            action = 2
        elif kind == "end":
            if depth_before:
                if frame_meta[-1][2] != pc:
                    raise ControlExecutionTrap("Mismatched dynamic frame end")
                base = int(np.argmax(frame_state.bases[depth_before - 1]))
                if len(stack) != base:
                    raise ControlExecutionTrap("Unbalanced empty control frame")
                action = 2
            elif pc == count - 1 and len(stack) == 1:
                halted = True
            else:
                raise ControlExecutionTrap("Function must end with one i32")
        elif kind in ("br", "br_if"):
            if kind == "br_if":
                _pop(stack)
            if taken:
                action = 3
                branch_depth = imm
        elif kind == "return":
            if len(stack) != 1 or next_pc != TERMINAL:
                raise ControlExecutionTrap("Unexpected return stack or target")
            action = 4
            halted = True
        elif kind == "nop":
            pass
        elif kind == "i32.const":
            stack.append(_bits(imm))
        elif kind == "local.get":
            stack.append(locals_[imm].copy())
        elif kind == "local.set":
            locals_[imm] = _pop(stack)
        elif kind == "local.tee":
            if not stack:
                raise ControlExecutionTrap("local.tee stack underflow")
            locals_[imm] = stack[-1].copy()
        elif kind == "drop":
            _pop(stack)
        elif kind == "i32.eqz":
            value = _pop(stack)
            bits = np.zeros(32, dtype=np.float64)
            bits[0] = _eqz(value, bit_fixed)
            stack.append(bits)
        elif kind in _BINARY:
            right, left = _pop(stack), _pop(stack)
            if kind == "i32.add":
                value = _add(left, right, fixed=bit_fixed)
            elif kind == "i32.sub":
                value = _sub(left, right, fixed=bit_fixed)
            elif kind == "i32.mul":
                value = _mul(left, right, bit_fixed.carry_thresholds)
            else:
                value = np.zeros(32, dtype=np.float64)
                value[0] = (_signed_lt(left, right, bit_fixed)
                            if kind == "i32.lt_s" else
                            _signed_lt(right, left, bit_fixed))
            stack.append(value)
        else:
            raise ControlExecutionTrap("Unadmitted dynamic operation")
        event_tokens = FrameTokens(
            frame_onehot(action, ACTIONS),
            _frame_optional(entry_kind, KINDS),
            _frame_optional(entry_base, HEIGHTS),
            _frame_optional(branch_depth, MAX_NESTING),
            frame_onehot(len(stack), HEIGHTS),
        )
        frame_state, restored = step_frames(frame_state, event_tokens, frame_fixed)
        depth_after = int(np.argmax(frame_state.depth))
        if action == 3:
            restored_height = int(np.argmax(restored))
            if restored_height > len(stack):
                raise ControlExecutionTrap("Neural restoration beyond live operands")
            # The actual data stack is trimmed using the constructed output,
            # not a host-selected dynamic branch-frame base.
            del stack[restored_height:]
        frame_meta = frame_meta[:depth_after]
        if action == 1:
            frame_meta.append(entering_metadata)
        if len(frame_meta) != depth_after:
            raise ControlExecutionTrap("Frame metadata not aligned to circuit depth")
        labels = [
            ("block", "loop", "if")[int(np.argmax(frame_state.kinds[i]))]
            for i in range(depth_after)
        ]
        if len(stack) > MAX_STACK:
            raise ControlExecutionTrap("Operand stack resource budget exceeded")
        records.append({
            "step": tick, "pc": pc, "opcode": kind,
            # TERMINAL is a routing sink, represented as WASM len(program)
            # in public trace for parity with the v51 host reference.
            "next_pc": count if next_pc == TERMINAL else next_pc,
            "stack_i32": [_int(v) for v in stack],
            "locals_i32": [_int(v) for v in locals_],
            "control_labels": labels,
            "halted": halted,
            "neural_route": route,
        })
        pc_vector = selected
        if halted:
            if next_pc != TERMINAL:
                raise ControlExecutionTrap("WASM halted at nonterminal routing PC")
            return FrameRoutedResult(
                _int(stack[-1]), tuple(records), len(records),
                hashlib.sha256(encode_control(program)).hexdigest(),
                fixed.digest(), bit_fixed.digest(), tokens.digest(),
                tuple(float(x) for x in frame_fixed.action_weights),
            )
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} neural-routed transitions"
    )
