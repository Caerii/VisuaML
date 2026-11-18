"""v48: a fixed, analytically routed neural token machine for multiple programs.

NOT yet a standard transformer: it uses exact one-hot attention-like matrix
selection and ReLU/ReGLU-style gated arithmetic, rather than softmax,
residual transformer layers, or an unbounded instruction stream.

The same immutable routing matrices execute *different* admitted typed WASM
programs. Program bytecode lives in the token inputs, not in model weights.
Every instruction transition can be compared against the independent v40
typed interpreter. Supported language: pure straight-line i32 WASM subset,
no imports, memory, tables, calls, branches, traps, or modulo overflow.

All program inputs and intermediate values must be exact signed integers in
[-2**24,2**24]; this avoids claiming float64 operations implement general
WASM i32 modular arithmetic. In particular, no all-i32 proof is claimed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from .wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    VMState,
    encode_program,
    validate_program,
)

SCHEMA = "visuaml.fixed-routed-neural-token-vm.v48"
MAX_OPS = 32
MAX_STACK = 16
MAX_LOCALS = 4
LOCAL_WIDTH = MAX_LOCALS + 1
PC_WIDTH = MAX_OPS + 1
DEPTH_WIDTH = MAX_STACK + 1
SAFE_EXACT_INTEGER = 1 << 24

OPCODES = (
    "i32.const", "local.get", "local.set", "local.tee", "i32.add",
    "i32.sub", "i32.mul", "i32.eqz", "i32.lt_s", "i32.gt_s",
    "drop", "end",
)
INDEX = {name: index for index, name in enumerate(OPCODES)}
BINARY = ("i32.add", "i32.sub", "i32.mul", "i32.lt_s", "i32.gt_s")


class NeuralTokenVMError(ValueError):
    """Invalid tokens, neural state, or numerical proof boundary."""


@dataclass(frozen=True)
class AnalyticWeights:
    """Fixed weights *shared across all programs*: never trained on any program."""

    program_next: np.ndarray
    depth_push: np.ndarray
    depth_pop: np.ndarray
    stack_push: np.ndarray
    stack_top: np.ndarray
    stack_second: np.ndarray
    operation_control: np.ndarray

    def digest(self) -> str:
        values = (
            self.program_next, self.depth_push, self.depth_pop,
            self.stack_push, self.stack_top, self.stack_second,
            self.operation_control,
        )
        h = hashlib.sha256(SCHEMA.encode())
        h.update(",".join(OPCODES).encode())
        for value in values:
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class TokenProgram:
    opcode_tokens: np.ndarray
    immediate_tokens: np.ndarray
    local_tokens: np.ndarray
    instruction_count: int
    local_count: int
    source_wasm_sha256: str
    schema: str = SCHEMA

    def digest(self) -> str:
        h = hashlib.sha256(self.schema.encode())
        for value in (self.opcode_tokens, self.immediate_tokens, self.local_tokens):
            h.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        h.update(f"{self.instruction_count}:{self.local_count}".encode())
        return h.hexdigest()


@dataclass(frozen=True)
class NeuralMachineState:
    pc: np.ndarray
    depth: np.ndarray
    stack: np.ndarray
    locals: np.ndarray
    halted: bool = False


def construct_fixed_weights() -> AnalyticWeights:
    """Deterministically synthesize routing matrices with no optimization/training."""
    program_next = np.zeros((PC_WIDTH, PC_WIDTH), dtype=np.float64)
    for i in range(PC_WIDTH - 1):
        program_next[i, i + 1] = 1.
    depth_push = np.zeros((DEPTH_WIDTH, DEPTH_WIDTH), dtype=np.float64)
    depth_pop = np.zeros((DEPTH_WIDTH, DEPTH_WIDTH), dtype=np.float64)
    stack_push = np.zeros((DEPTH_WIDTH, MAX_STACK), dtype=np.float64)
    stack_top = np.zeros((DEPTH_WIDTH, MAX_STACK), dtype=np.float64)
    stack_second = np.zeros((DEPTH_WIDTH, MAX_STACK), dtype=np.float64)
    for depth in range(DEPTH_WIDTH):
        if depth < MAX_STACK:
            depth_push[depth, depth + 1] = 1.
            stack_push[depth, depth] = 1.
        if depth > 0:
            depth_pop[depth, depth - 1] = 1.
            stack_top[depth, depth - 1] = 1.
        if depth > 1:
            stack_second[depth, depth - 2] = 1.
    # Push, pop, binary, unary, local-write, halt flags from one-hot opcode.
    controls = np.zeros((len(OPCODES), 6), dtype=np.float64)
    for name, idx in INDEX.items():
        controls[idx] = (
            float(name in ("i32.const", "local.get")),
            float(name in (*BINARY, "local.set", "drop")),
            float(name in BINARY),
            float(name == "i32.eqz"),
            float(name in ("local.set", "local.tee")),
            float(name == "end"),
        )
    return AnalyticWeights(
        program_next, depth_push, depth_pop, stack_push,
        stack_top, stack_second, controls,
    )


def _admit_program(program: I32Program) -> None:
    validate_program(program)
    if (program.local_count > MAX_LOCALS
            or len(program.operations) > MAX_OPS):
        raise NeuralTokenVMError("Bounded program exceeds constructed machine")
    depth = 0
    for op in program.operations:
        if op.opcode not in INDEX:
            raise NeuralTokenVMError("Unsupported instruction")
        if (op.opcode == "i32.const"
                and not -256 <= op.immediate <= 256):
            raise NeuralTokenVMError("Unadmitted immediate magnitude")
        if op.opcode in ("i32.const", "local.get"):
            depth += 1
        elif op.opcode in (*BINARY, "local.set", "drop"):
            depth -= 1
        if not 0 <= depth <= MAX_STACK:
            raise NeuralTokenVMError("Static stack depth exceeds neural capacity")


def tokenize_program(program: I32Program) -> TokenProgram:
    """Program-data tokens, never specialized interpreter weights."""
    _admit_program(program)
    opcode = np.zeros((PC_WIDTH, len(OPCODES)), dtype=np.float64)
    immediates = np.zeros(PC_WIDTH, dtype=np.float64)
    locals_ = np.zeros((PC_WIDTH, LOCAL_WIDTH), dtype=np.float64)
    for i, operation in enumerate(program.operations):
        opcode[i, INDEX[operation.opcode]] = 1.
        if operation.opcode == "i32.const":
            immediates[i] = operation.immediate
        if operation.opcode.startswith("local."):
            locals_[i, operation.immediate] = 1.
    return TokenProgram(
        opcode, immediates, locals_, len(program.operations),
        program.local_count,
        hashlib.sha256(encode_program(program)).hexdigest(),
    )


def detokenize_program(tokens: TokenProgram) -> I32Program:
    """Recover typed code from a REVIEWED, CANONICAL token encoding only.

    This is token disassembly. It is NOT arbitrary neural-weight decompilation
    and cannot identify unique source programs from functional behavior.
    """
    if not isinstance(tokens, TokenProgram) or tokens.schema != SCHEMA:
        raise NeuralTokenVMError("Unknown token schema")
    if (not 2 <= tokens.instruction_count <= MAX_OPS
            or not 0 <= tokens.local_count <= MAX_LOCALS):
        raise NeuralTokenVMError("Invalid token program bounds")
    for value, shape in (
        (tokens.opcode_tokens, (PC_WIDTH, len(OPCODES))),
        (tokens.immediate_tokens, (PC_WIDTH,)),
        (tokens.local_tokens, (PC_WIDTH, LOCAL_WIDTH)),
    ):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()
                or np.max(np.abs(value)) > 256):
            raise NeuralTokenVMError("Invalid finite token tensor shape/dtype")
    ops = []
    for index in range(tokens.instruction_count):
        row = tokens.opcode_tokens[index]
        if not np.isin(row, [0., 1.]).all() or np.sum(row) != 1.:
            raise NeuralTokenVMError("Missing or noncanonical one-hot opcode")
        name = OPCODES[int(np.argmax(row))]
        immediate = float(tokens.immediate_tokens[index])
        local_row = tokens.local_tokens[index]
        if name == "i32.const":
            if immediate != int(immediate) or not -256 <= immediate <= 256:
                raise NeuralTokenVMError("Invalid integer immediate token")
            if np.any(local_row):
                raise NeuralTokenVMError("Unexpected local selector")
            ops.append(VMOperation(name, int(immediate)))
        elif name.startswith("local."):
            if (immediate != 0. or not np.isin(local_row, [0., 1.]).all()
                    or np.sum(local_row) != 1.):
                raise NeuralTokenVMError("Invalid one-hot local token")
            idx = int(np.argmax(local_row))
            if idx > tokens.local_count:
                raise NeuralTokenVMError("Local index beyond actual declaration")
            ops.append(VMOperation(name, idx))
        else:
            if immediate != 0. or np.any(local_row):
                raise NeuralTokenVMError("Unexpected opcode payload")
            ops.append(VMOperation(name))
    if (np.any(tokens.opcode_tokens[tokens.instruction_count:])
            or np.any(tokens.immediate_tokens[tokens.instruction_count:])
            or np.any(tokens.local_tokens[tokens.instruction_count:])):
        raise NeuralTokenVMError("Unadmitted nonempty token padding")
    program = I32Program(tokens.local_count, tuple(ops))
    _admit_program(program)
    if hashlib.sha256(encode_program(program)).hexdigest() != tokens.source_wasm_sha256:
        raise NeuralTokenVMError("Token bytes do not match source wasm digest")
    return program


def initial_neural_state(program: I32Program, argument: int) -> NeuralMachineState:
    _admit_program(program)
    if type(argument) is not int or not -128 <= argument <= 127:
        raise NeuralTokenVMError("Declared exact neural input domain is [-128,127]")
    pc = np.zeros(PC_WIDTH, dtype=np.float64)
    pc[0] = 1.
    depth = np.zeros(DEPTH_WIDTH, dtype=np.float64)
    depth[0] = 1.
    stack = np.zeros(MAX_STACK, dtype=np.float64)
    local = np.zeros(LOCAL_WIDTH, dtype=np.float64)
    local[0] = argument
    return NeuralMachineState(pc, depth, stack, local)


def _admit_state(state: NeuralMachineState) -> None:
    if not isinstance(state, NeuralMachineState) or state.halted:
        raise NeuralTokenVMError("Invalid or halted neural state")
    for value, shape in (
        (state.pc, (PC_WIDTH,)),
        (state.depth, (DEPTH_WIDTH,)),
        (state.stack, (MAX_STACK,)),
        (state.locals, (LOCAL_WIDTH,)),
    ):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()
                or not np.equal(value, np.rint(value)).all()
                or np.max(np.abs(value)) > SAFE_EXACT_INTEGER):
            raise NeuralTokenVMError("Nonexact or out-of-bounds neural state")
    for field in (state.pc, state.depth):
        if not np.isin(field, [0., 1.]).all() or np.sum(field) != 1.:
            raise NeuralTokenVMError("Invalid one-hot machine position")


def _positive_step(value: np.float64) -> np.float64:
    """For integer value, exact ReLU gate 1 when value >= 1, else 0."""
    return np.maximum(0., value) - np.maximum(0., value - 1.)


def _reglu_mul(a: np.float64, b: np.float64) -> np.float64:
    """ReGLU-style exact-real bilinear primitive; no Python integer multiply."""
    return np.maximum(0., a) * b - np.maximum(0., -a) * b


def neural_step(state: NeuralMachineState, tokens: TokenProgram,
                fixed: AnalyticWeights) -> NeuralMachineState:
    """One complete instruction through fixed matrix selection and ReGLU gates.

    No Python opcode branch occurs in execution: every candidate operation
    is computed, and fixed one-hot flags gate each write/control transition.
    """
    _admit_state(state)
    # Matrix attention-like fetch from a separate program token table.
    flags = state.pc @ tokens.opcode_tokens
    immediate = np.float64(state.pc @ tokens.immediate_tokens)
    local_selector = state.pc @ tokens.local_tokens

    push, pop, binary, unary, localwrite, halt = flags @ fixed.operation_control
    top_selector = state.depth @ fixed.stack_top
    second_selector = state.depth @ fixed.stack_second
    push_selector = state.depth @ fixed.stack_push
    top = np.float64(state.stack @ top_selector)
    second = np.float64(state.stack @ second_selector)
    local_value = np.float64(state.locals @ local_selector)

    # Compute all candidate values with real ReLU/ReGLU circuits.
    options = np.zeros(len(OPCODES), dtype=np.float64)
    options[INDEX["i32.const"]] = immediate
    options[INDEX["local.get"]] = local_value
    options[INDEX["i32.add"]] = second + top
    options[INDEX["i32.sub"]] = second - top
    options[INDEX["i32.mul"]] = _reglu_mul(top, second)
    options[INDEX["i32.eqz"]] = np.maximum(
        0., 1. - np.maximum(0., top) - np.maximum(0., -top)
    )
    options[INDEX["i32.lt_s"]] = _positive_step(top - second)
    options[INDEX["i32.gt_s"]] = _positive_step(second - top)
    write_value = np.float64(flags @ options)

    # Same network for every instruction: masked stack and local writes
    # are bilinear gates on activation values and one-hot location tags.
    write_slots = push * push_selector + binary * second_selector + unary * top_selector
    clear_slots = pop * top_selector
    new_stack = (
        state.stack * (1. - write_slots)
        + write_value * write_slots
    ) * (1. - clear_slots)
    new_locals = (
        state.locals * (1. - localwrite * local_selector)
        + top * (localwrite * local_selector)
    )
    new_depth = (
        (1. - push - pop) * state.depth
        + push * (state.depth @ fixed.depth_push)
        + pop * (state.depth @ fixed.depth_pop)
    )
    new_pc = state.pc @ fixed.program_next
    next_state = NeuralMachineState(
        new_pc, new_depth, new_stack, new_locals, bool(halt == 1.)
    )
    # No modulo wrap: reject rather than silently claiming general i32.
    for value in (new_stack, new_locals):
        if (not np.isfinite(value).all() or np.max(np.abs(value)) > SAFE_EXACT_INTEGER
                or not np.equal(value, np.rint(value)).all()):
            raise NeuralTokenVMError("Neural arithmetic exceeded exact bounded integer contract")
    if not np.isin(new_depth, [0., 1.]).all() or np.sum(new_depth) != 1.:
        raise NeuralTokenVMError("Neural depth transition invalid")
    if not np.isin(new_pc, [0., 1.]).all() or np.sum(new_pc) != 1.:
        raise NeuralTokenVMError("Neural program counter transition invalid")
    return next_state


def decode_state(state: NeuralMachineState, program: I32Program) -> VMState:
    """External inspection only: extract numerical state without opcode dispatch."""
    if not isinstance(state, NeuralMachineState):
        raise NeuralTokenVMError("Invalid decoded state")
    depth = int(np.argmax(state.depth))
    pc = int(np.argmax(state.pc))
    if (depth > MAX_STACK or any(not float(x).is_integer()
                                 for x in state.stack[:depth])):
        raise NeuralTokenVMError("Unadmitted decoded neural stack")
    return VMState(
        pc, tuple(int(x) for x in state.stack[:depth]),
        tuple(int(x) for x in state.locals[:program.local_count+1]),
        state.halted,
    )


def run_neural(program: I32Program, argument: int,
               *, fixed: AnalyticWeights | None = None) -> dict:
    """Run different *program tokens* with identical analytically fixed weights."""
    _admit_program(program)
    fixed = fixed if fixed is not None else construct_fixed_weights()
    tokens = tokenize_program(program)
    if detokenize_program(tokens) != program:
        raise NeuralTokenVMError("Structured token disassembly failed")
    current = initial_neural_state(program, argument)
    trace = []
    for operation in program.operations:
        current = neural_step(current, tokens, fixed)
        trace.append({
            "step": len(trace),
            "opcode": operation.opcode,
            "state": asdict(decode_state(current, program)),
        })
    if not current.halted:
        raise NeuralTokenVMError("Neural machine did not halt")
    final = decode_state(current, program)
    if len(final.stack) != 1:
        raise NeuralTokenVMError("Neural machine did not return one i32")
    return {
        "schema": SCHEMA,
        "program_sha256": program.digest(),
        "wasm_sha256": tokens.source_wasm_sha256,
        "program_token_sha256": tokens.digest(),
        "fixed_weights_sha256": fixed.digest(),
        "input_i32": argument,
        "result_i32": final.stack[0],
        "step_trace": trace,
        "no_weights_trained": True,
        "program_is_separate_data": True,
        "architecture": "fixed one-hot matrix fetch plus ReLU/ReGLU gated state writes",
        "standard_softmax_transformer_executed": False,
        "general_wasm_support": False,
        "all_i32_modular_arithmetic_verified": False,
        "arbitrary_pretrained_weight_decompilation": False,
    }



def compare_typed_programs_smt(
    left: I32Program, right: I32Program, *, timeout_ms: int = 15000
) -> dict:
    """Independent all-i32 bitvector equivalence of typed SOURCE programs.

    This does not prove float64 neural execution outside its finite domain.
    Importantly, UNSAT can demonstrate nonunique program representations of
    the same behavior, *refuting* unique arbitrary decompilation from outputs.
    """
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise NeuralTokenVMError("Invalid bounded solver budget")
    _admit_program(left)
    _admit_program(right)
    import z3

    from .neural_symbolic_recovery_v43 import _encode_wasm_in_z3

    x = z3.BitVec("input_i32", 32)
    a = _encode_wasm_in_z3(left, x, z3)
    b = _encode_wasm_in_z3(right, x, z3)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(a != b)
    result = solver.check()
    if result == z3.unknown:
        raise NeuralTokenVMError(
            "SMT equivalence inconclusive: " + solver.reason_unknown()
        )
    evidence = {
        "schema": "visuaml.typed-program-equivalence-class.v48",
        "left_wasm_sha256": hashlib.sha256(encode_program(left)).hexdigest(),
        "right_wasm_sha256": hashlib.sha256(encode_program(right)).hexdigest(),
        "solver": f"Z3 {z3.get_version_string()}",
        "solver_result": str(result),
        "all_i32_typed_programs_equivalent": result == z3.unsat,
        "neural_all_i32_equivalence_proved": False,
        "unique_source_recovered_from_behavior": False,
        "machine_checked_proof_object": False,
    }
    if result == z3.sat:
        model = solver.model()
        u = int(model.eval(x, model_completion=True).as_long())
        evidence["counterexample_i32"] = u if u < (1 << 31) else u - (1 << 32)
        evidence["left_result_u32"] = int(
            model.eval(a, model_completion=True).as_long()
        )
        evidence["right_result_u32"] = int(
            model.eval(b, model_completion=True).as_long()
        )
    return evidence
