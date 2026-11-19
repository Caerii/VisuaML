"""v52 constructed, shared-weight PC routing for bounded real WASM control flow.

This layer advances v51 by replacing HOST selection of the NEXT PC with
an exact one-hot/bilinear ReLU-gated routing circuit. The same fixed routing
weights run different program-specific instruction/target TOKEN TABLES.
Lexical target compilation and the runtime label stack, stack operations,
opcode dispatch and fuel are still HOST operations. No claim that this is
a stock softmax transformer or a fully neural WebAssembly interpreter.

The selected PC is ACTUALLY fed back into the executor's next instruction;
a separate v51 host-PC interpreter is used only for differential validation.
Program tables are data, not per-program trained interpreter weights.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .exact_binary import is_exact_binary

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
from .wasm_control_flow_v51 import (
    _BINARY,
    _BLOCKS,
    MAX_FUEL,
    MAX_OPERATIONS, SCALED_MAX_OPERATIONS,
    MAX_STACK,
    ControlAdmissionError,
    ControlExecutionTrap,
    ControlFrame,
    ControlFuelExhausted,
    ControlProgram,
    _syntax,
    encode_control,
)

SCHEMA = "visuaml.fixed-control-routing.v52"
SLOTS = MAX_OPERATIONS + 1
TERMINAL = MAX_OPERATIONS


class NeuralControlError(ControlAdmissionError):
    """Malformed static program targets or neural control selector."""


@dataclass(frozen=True)
class ControlRouterWeights:
    """Same gate coefficients are used for every program, including GCD."""

    guard_weights: np.ndarray
    path_weights: np.ndarray

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        for array in (self.guard_weights, self.path_weights):
            h.update(str(array.shape).encode())
            h.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class ControlRouteTokens:
    sequential: np.ndarray
    alternate: np.ndarray
    guard_kind: np.ndarray
    instruction_count: int
    source_wasm_sha256: str
    source_program_sha256: str

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        for array in (self.sequential, self.alternate, self.guard_kind):
            h.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        h.update(str(self.instruction_count).encode())
        h.update(self.source_wasm_sha256.encode())
        return h.hexdigest()


@dataclass(frozen=True)
class RoutedResult:
    result_i32: int
    trace: tuple[dict, ...]
    steps: int
    source_wasm_sha256: str
    fixed_router_weights_sha256: str
    fixed_arithmetic_weights_sha256: str
    program_route_tokens_sha256: str
    pc_routed_by_constructed_gates: bool = True
    label_stack_host_interpreted: bool = True
    instruction_dispatch_host_interpreted: bool = True


def construct_router_weights() -> ControlRouterWeights:
    return ControlRouterWeights(
        np.ones(3, dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )


def _onehot(position: int) -> np.ndarray:
    result = np.zeros(SLOTS, dtype=np.float64)
    result[position] = 1.
    return result


def _admit_onehot(array: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(array, np.ndarray) or array.dtype != np.float64
            or array.shape != shape or not is_exact_binary(array)):
        raise NeuralControlError("Noncanonical one-hot routing tensor")


def _target(position: int, count: int, terminal: int) -> int:
    if not 0 <= position <= count:
        raise NeuralControlError("Lexical branch target outside admitted program")
    return terminal if position == count else position


def compile_route_coordinates(program: ControlProgram, terminal: int):
    """Resolve canonical lexical edges without materializing dense PC tensors."""
    links = _syntax(program)
    count = len(program.operations)
    if type(terminal) is not int or not count <= terminal <= SCALED_MAX_OPERATIONS:
        raise NeuralControlError("Terminal is outside the admitted route coordinate range")
    targets = np.full((terminal + 1, 2), -1, dtype=np.int32)
    guards = np.zeros((terminal + 1, 3), dtype=np.float64)
    lexical: list[int] = []
    for pc, op in enumerate(program.operations):
        successor = _target(pc + 1, count, terminal)
        targets[pc, 0] = successor
        other = successor
        kind = op.opcode
        if kind in _BLOCKS:
            lexical.append(pc)
            if kind == "if":
                end, otherwise = links[pc]
                other = _target(
                    otherwise + 1 if otherwise is not None else end + 1, count, terminal
                )
                guards[pc, 2] = 1.
        elif kind == "else":
            if not lexical or program.operations[lexical[-1]].opcode != "if":
                raise NeuralControlError("Unmatched conditional alternate")
            enclosing = lexical[-1]
            if links[enclosing][1] != pc:
                raise NeuralControlError("Else does not match lexical if")
            other = _target(links[enclosing][0] + 1, count, terminal)
            guards[pc, 0] = 1.
        elif kind == "end":
            if lexical:
                enclosing = lexical.pop()
                if links[enclosing][0] != pc:
                    raise NeuralControlError("Mismatched lexical end")
            elif pc != count - 1:
                raise NeuralControlError("Unmatched nonfinal function end")
        elif kind in ("br", "br_if"):
            index = len(lexical) - 1 - op.immediate
            enclosing = lexical[index]
            open_kind = program.operations[enclosing].opcode
            branch_to = (
                enclosing + 1 if open_kind == "loop"
                else links[enclosing][0] + 1
            )
            other = _target(branch_to, count, terminal)
            guards[pc, 0 if kind == "br" else 1] = 1.
        elif kind == "return":
            other = terminal
            guards[pc, 0] = 1.
        targets[pc, 1] = other
    if lexical:
        raise NeuralControlError("Nonempty structural label stack at function end")
    targets[terminal] = terminal
    return targets, guards


def compile_route_tokens(program: ControlProgram) -> ControlRouteTokens:
    """Trusted structural admission for the legacy dense route tensors."""
    count = len(program.operations)
    if count > MAX_OPERATIONS:
        raise NeuralControlError("Large programs require compact routing")
    targets, guards = compile_route_coordinates(program, TERMINAL)
    sequential = np.zeros((SLOTS, SLOTS), dtype=np.float64)
    alternate = np.zeros((SLOTS, SLOTS), dtype=np.float64)
    active = np.r_[np.arange(count), TERMINAL]
    sequential[active, targets[active, 0]] = 1.
    alternate[active, targets[active, 1]] = 1.
    return ControlRouteTokens(
        sequential, alternate, guards, count,
        hashlib.sha256(encode_control(program)).hexdigest(),
        program.digest(),
    )


def validate_route_tokens(
    tokens: ControlRouteTokens, program: ControlProgram
) -> None:
    if not isinstance(tokens, ControlRouteTokens):
        raise NeuralControlError("Unknown program route-token representation")
    count = len(program.operations)
    if (tokens.instruction_count != count
            or tokens.source_wasm_sha256 != hashlib.sha256(
                encode_control(program)
            ).hexdigest()
            or tokens.source_program_sha256 != program.digest()):
        raise NeuralControlError("Routing table identity mismatches original WASM")
    _admit_onehot(tokens.sequential, (SLOTS, SLOTS))
    _admit_onehot(tokens.alternate, (SLOTS, SLOTS))
    _admit_onehot(tokens.guard_kind, (SLOTS, 3))
    for matrix in (tokens.sequential, tokens.alternate):
        if not np.all(np.sum(matrix[:count], axis=1) == 1.):
            raise NeuralControlError("Incomplete or ambiguous active routing row")
        if np.any(matrix[count:TERMINAL]):
            raise NeuralControlError("Noncanonical padded routing table")
        if matrix[TERMINAL, TERMINAL] != 1.:
            raise NeuralControlError("Missing terminal routing sink")
    if (np.any(tokens.guard_kind[count:])
            or np.any(np.sum(tokens.guard_kind, axis=1) > 1)):
        raise NeuralControlError("Ambiguous condition selectors")
    expected = compile_route_tokens(program)
    if any(not np.array_equal(x, y) for x, y in (
        (tokens.sequential, expected.sequential),
        (tokens.alternate, expected.alternate),
        (tokens.guard_kind, expected.guard_kind),
    )):
        raise NeuralControlError("Altered routing targets do not match reviewed source")


def neural_pc_step(
    pc: np.ndarray,
    top_condition: np.ndarray,
    tokens: ControlRouteTokens,
    fixed: ControlRouterWeights,
    bit_fixed: BitCircuitWeights,
) -> tuple[np.ndarray, dict]:
    """Real fixed-weight bilinear program-counter selector, no opcode branch.

    A binary condition comes from v49's ReLU bit-vector zero test.
    Target matrices are compiled PROGRAM DATA, not interpreter weights.
    """
    _admit_onehot(pc, (SLOTS,))
    _admit_onehot(top_condition, (32,))
    if float(np.sum(pc)) != 1.:
        raise NeuralControlError("Current PC is not uniquely active")
    if not isinstance(fixed, ControlRouterWeights):
        raise NeuralControlError("Unknown fixed neural control weights")
    for value, shape in (
        (fixed.guard_weights, (3,)), (fixed.path_weights, (2,))
    ):
        if (not isinstance(value, np.ndarray)
                or value.shape != shape or value.dtype != np.float64
                or not np.isfinite(value).all()):
            raise NeuralControlError("Invalid neural gate coefficients")
    flags = pc @ tokens.guard_kind
    zero = np.float64(_eqz(top_condition, bit_fixed))
    nonzero = np.float64(1.) - zero
    # Every candidate is calculated, with one fixed guarded bilinear blend.
    take = (fixed.guard_weights[0]*flags[0]
            + fixed.guard_weights[1]*flags[1]*nonzero
            + fixed.guard_weights[2]*flags[2]*zero)
    normal = pc @ tokens.sequential
    branch = pc @ tokens.alternate
    result = (fixed.path_weights[0]*(1. - take)*normal
              + fixed.path_weights[1]*take*branch)
    _admit_onehot(result, (SLOTS,))
    if float(np.sum(result)) != 1.:
        raise NeuralControlError("Constructed route lost unique next PC")
    return result, {
        "nonzero_condition_gate": int(nonzero),
        "branch_guard_gate": int(take),
        "sequential_target": int(np.argmax(normal)),
        "alternative_target": int(np.argmax(branch)),
        "next_pc": int(np.argmax(result)),
    }


def _pop(stack: list[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ControlExecutionTrap("Typed operand stack underflow")
    return stack.pop()


def execute_routed_control(
    program: ControlProgram, argument: int, *, fuel: int = 500,
    tokens: ControlRouteTokens | None = None,
    fixed: ControlRouterWeights | None = None,
    bit_fixed: BitCircuitWeights | None = None,
) -> RoutedResult:
    """EXECUTE using the one-hot route output as the next PC.

    Host still handles the instruction's stack/local effects and dynamic
    control-frame maintenance, not the conditional target decision.
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
    frames: list[ControlFrame] = []
    pc_vector = _onehot(0)
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

        if kind in ("block", "loop", "if"):
            end, else_pc = links[pc]
            if kind == "if":
                _pop(stack)
                if next_pc == sequential_pc or else_pc is not None:
                    frames.append(ControlFrame(
                        kind, pc, end, else_pc, len(stack)
                    ))
            else:
                frames.append(ControlFrame(
                    kind, pc, end, else_pc, len(stack)
                ))
        elif kind == "else":
            if (not frames or frames[-1].kind != "if"
                    or frames[-1].else_pc != pc):
                raise ControlExecutionTrap("Unexpected dynamic else")
            frame = frames.pop()
            if len(stack) != frame.stack_base:
                raise ControlExecutionTrap("Incorrect empty if-arm result")
        elif kind == "end":
            if frames:
                frame = frames.pop()
                if frame.end_pc != pc or len(stack) != frame.stack_base:
                    raise ControlExecutionTrap("Unbalanced empty control frame")
            else:
                if pc != count-1 or len(stack) != 1:
                    raise ControlExecutionTrap("Function must end with one i32")
                halted = True
        elif kind in ("br", "br_if"):
            if kind == "br_if":
                _pop(stack)
            if taken:
                target_index = len(frames)-1-imm
                if target_index < 0:
                    raise ControlExecutionTrap("Invalid dynamic branch depth")
                frame = frames[target_index]
                del stack[frame.stack_base:]
                if frame.kind == "loop":
                    del frames[target_index+1:]
                else:
                    del frames[target_index:]
        elif kind == "return":
            if len(stack) != 1 or next_pc != TERMINAL:
                raise ControlExecutionTrap("Unexpected return stack or target")
            frames.clear()
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
        if len(stack) > MAX_STACK:
            raise ControlExecutionTrap("Operand stack resource budget exceeded")
        records.append({
            "step": tick, "pc": pc, "opcode": kind,
            # TERMINAL is a routing sink, represented as WASM len(program)
            # in public trace for parity with the v51 host reference.
            "next_pc": count if next_pc == TERMINAL else next_pc,
            "stack_i32": [_int(v) for v in stack],
            "locals_i32": [_int(v) for v in locals_],
            "control_labels": [f.kind for f in frames],
            "halted": halted,
            "neural_route": route,
        })
        pc_vector = selected
        if halted:
            if next_pc != TERMINAL:
                raise ControlExecutionTrap("WASM halted at nonterminal routing PC")
            return RoutedResult(
                _int(stack[-1]), tuple(records), len(records),
                hashlib.sha256(encode_control(program)).hexdigest(),
                fixed.digest(), bit_fixed.digest(), tokens.digest(),
            )
    raise ControlFuelExhausted(
        f"Explicit bounded fuel exhausted after {fuel} neural-routed transitions"
    )
