"""v57 source-bound microoperation-driven WASM execution path.

Constructed v52 one-hot PC routing remains authoritative. The v53 finite
frame-circuit output now determines live label depth and actual branch
operand-stack restoration. Host still decodes/dispatches opcode effects and maintains lexical source
metadata, but all live operands reside in v55 tensor storage. Not universal WASM or a fully neural instruction decoder.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bit_neural_vm_v49 import (
    BitCircuitWeights,
    _bits,
    _int,
    construct_bit_weights,
)
from .neural_alu_v56 import (
    ALUWeights,
    construct_alu_weights,
    execute_alu,
)
from .neural_control_routing_v52 import (
    TERMINAL,
    ControlRouterWeights,
    ControlRouteTokens,
    _onehot,
    compile_route_tokens,
    construct_router_weights,
    neural_pc_step,
    validate_route_tokens,
)
from .neural_frame_circuit_v53 import (
    HEIGHTS,
    KINDS,
    MAX_NESTING,
    FrameCircuitWeights,
    FrameTokens,
    empty_frames,
    fixed_frame_weights,
    step_frames,
)
from .neural_frame_circuit_v53 import onehot as frame_onehot
from .neural_frame_events_v54 import (
    EventWeights,
    decode_opcode_token,
    fixed_event_weights,
    select_frame_action,
)
from .neural_operand_adapter_v55 import TensorOperandStack
from .neural_operand_storage_v55 import OperandWeights, fixed_operand_weights
from .neural_local_bank_v57 import (
    LocalBankWeights,
    construct_local_bank_weights,
    read_local,
    write_local,
)
from .typed_microop_ir_v57 import (
    MicroopWeights,
    ProgramMicroopTokens,
    compile_microop_tokens,
    construct_microop_weights,
    read_microop,
    validate_microop_tokens,
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
    event_selection_gate_driven: bool = True
    tensor_operand_storage_drives_execution: bool = True
    neural_alu_selection_drives_execution: bool = True
    microop_token_selection_drives_execution: bool = True
    fixed_microop_weights_sha256: str = ""
    fixed_alu_weights_sha256: str = ""
    fixed_event_weight_digest: str = ""


def _event_digest(weights: EventWeights) -> str:
    h = hashlib.sha256(b"visuaml.v54.fixed-event-gates")
    for value in (weights.default, weights.conditional_gains):
        h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return h.hexdigest()


def _frame_optional(index: int | None, width: int) -> np.ndarray:
    return (np.zeros(width, dtype=np.float64)
            if index is None else frame_onehot(index, width))


def _pop(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ControlExecutionTrap("Typed operand stack underflow")
    return stack.pop()


def execute_microop_routed_control(
    program: ControlProgram, argument: int, *, fuel: int = 500,
    tokens: ControlRouteTokens | None = None,
    fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
    frame_fixed: FrameCircuitWeights | None = None,
    event_fixed: EventWeights | None = None,
    operand_fixed: OperandWeights | None = None,
    alu_fixed: ALUWeights | None = None,
    microop_fixed: MicroopWeights | None = None,
    microop_tokens: ProgramMicroopTokens | None = None,
    local_fixed: LocalBankWeights | None = None,
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
    microop_tokens = (
        compile_microop_tokens(program) if microop_tokens is None
        else microop_tokens
    )
    validate_microop_tokens(microop_tokens, program)
    microop_fixed = (
        construct_microop_weights() if microop_fixed is None
        else microop_fixed
    )
    local_fixed = (
        construct_local_bank_weights() if local_fixed is None
        else local_fixed
    )
    local_bank = np.zeros((5, 32), dtype=np.float64)
    local_bank[0] = _bits(argument)
    operand_fixed = (
        fixed_operand_weights() if operand_fixed is None else operand_fixed
    )
    stack = TensorOperandStack(operand_fixed)
    alu_fixed = construct_alu_weights() if alu_fixed is None else alu_fixed

    frame_state = empty_frames()
    frame_meta: list[tuple[str, int, int, int | None]] = []
    pc_vector = _onehot(0)
    frame_fixed = fixed_frame_weights() if frame_fixed is None else frame_fixed
    event_fixed = fixed_event_weights() if event_fixed is None else event_fixed
    opcode_tokens = tuple(decode_opcode_token(op.opcode) for op in program.operations)
    records = []
    count = len(program.operations)

    for tick in range(fuel):
        pc = int(np.argmax(pc_vector))
        if pc == TERMINAL or pc >= count:
            raise ControlExecutionTrap("Reached terminal without WASM return")
        op = program.operations[pc]
        kind, imm = op.opcode, op.immediate
        microop = read_microop(pc_vector, microop_tokens, microop_fixed)
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

        entry_kind = None
        entry_base = None
        branch_depth = None
        entering_metadata = None
        depth_before = int(np.argmax(frame_state.depth))
        if kind in ("block", "loop", "if"):
            end, else_pc = links[pc]
            if kind == "if":
                _pop(stack)
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
        elif kind == "end":
            if depth_before:
                if frame_meta[-1][2] != pc:
                    raise ControlExecutionTrap("Mismatched dynamic frame end")
                base = int(np.argmax(frame_state.bases[depth_before - 1]))
                if len(stack) != base:
                    raise ControlExecutionTrap("Unbalanced empty control frame")
            elif pc == count - 1 and len(stack) == 1:
                halted = True
            else:
                raise ControlExecutionTrap("Function must end with one i32")
        elif kind in ("br", "br_if"):
            if kind == "br_if":
                _pop(stack)
            branch_depth = int(np.argmax(microop.branch_depth))
        elif kind == "return":
            if len(stack) != 1 or next_pc != TERMINAL:
                raise ControlExecutionTrap("Unexpected return stack or target")
            halted = True
        elif kind == "nop":
            pass
        elif kind == "i32.const":
            stack.append(microop.immediate_bits.copy())
        elif kind == "local.get":
            stack.append(read_local(
                local_bank, microop.local_selector, fixed=local_fixed,
            ))
        elif kind == "local.set":
            local_bank = write_local(
                local_bank, microop.local_selector, _pop(stack),
                fixed=local_fixed,
            )
        elif kind == "local.tee":
            if not stack:
                raise ControlExecutionTrap("local.tee stack underflow")
            local_bank = write_local(
                local_bank, microop.local_selector, stack[-1],
                fixed=local_fixed,
            )
        elif kind == "drop":
            _pop(stack)
        elif kind == "i32.eqz":
            left = _pop(stack)
            result = execute_alu(
                microop.alu_selector, left, np.zeros(32, dtype=np.float64),
                fixed=alu_fixed, bit_fixed=bit_fixed,
            )
            stack.append(result)
        elif kind in _BINARY:
            right, left = _pop(stack), _pop(stack)
            # All candidate operations use the same v49 analytic circuit.
            # The fixed ALU output gate, not Python opcode dispatch,
            # chooses the i32 result.
            result = execute_alu(
                alu_tokens[pc], left, right, fixed=alu_fixed,
                bit_fixed=bit_fixed,
            )
            stack.append(result)
        else:
            raise ControlExecutionTrap("Unadmitted dynamic operation")
        # Frame-event action is emitted exclusively by the constructed
        # opcode/guard circuit; host opcode branches above handle only
        # operand effects, lexical admission and return validation.
        event_vector = select_frame_action(
            opcode_tokens[pc],
            conditional_sequential=np.array([
                float(next_pc == sequential_pc)
            ], dtype=np.float64),
            has_else=np.array([
                float(kind == "if" and links[pc][1] is not None)
            ], dtype=np.float64),
            frame_active=np.array([
                float(depth_before > 0)
            ], dtype=np.float64),
            branch_taken=np.array([
                float(taken)
            ], dtype=np.float64),
            weights=event_fixed,
        )
        action = int(np.argmax(event_vector))
        if action == 1 and entering_metadata is None:
            raise ControlExecutionTrap("Gate emitted unsupported frame enter")
        if action == 3 and branch_depth is None:
            raise ControlExecutionTrap("Gate emitted branch without depth token")
        event_tokens = FrameTokens(
            event_vector,
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
            "locals_i32": [
                _int(v) for v in local_bank[:program.local_count + 1]
            ],
            "control_labels": labels,
            "halted": halted,
            "neural_route": route,
            "neural_frame_event": action,
            "microop_effect": int(np.argmax(microop.effect)),
            "microop_local_mode": int(np.argmax(microop.local_mode)),
            "neural_alu_opcode": (
                int(np.argmax(microop.alu_selector))
                if microop.alu_selector.sum() else None
            ),
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
                fixed_event_weight_digest=_event_digest(event_fixed),
                fixed_alu_weights_sha256=alu_fixed.digest(),
                fixed_microop_weights_sha256=microop_fixed.digest(),
            )
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} neural-routed transitions"
    )
