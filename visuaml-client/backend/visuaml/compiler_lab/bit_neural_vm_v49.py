"""v49 exact 32-bit modular arithmetic in an analytic bit-gated neural token VM.

Unlike the v48 float64-value interpreter, every stack and local value here is
a 32-element binary float64 vector. Fixed ReLU threshold gates implement
carry/borrow, a 32x32 bilinear ReGLU partial-product grid implements MUL,
and ReLU staircases extract carries. No Python integer arithmetic is used
to execute an instruction. Converting integers to/from bits is restricted
to the external ABI and oracle inspection.

This is NOT yet a standard transformer or unrestricted WASM: one pure
straight-line exported i32 function, no memory, imports, branches, loops or
host effects. Exact *width* does not constitute an all-input SMT proof of
the entire constructed circuit; independent original-Wasmtime witnesses and
gate-level truth-table tests are required.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .exact_binary import is_exact_binary

from .fixed_neural_token_vm_v48 import (
    DEPTH_WIDTH,
    INDEX,
    LOCAL_WIDTH,
    MAX_LOCALS,
    MAX_OPS,
    MAX_STACK,
    OPCODES,
    PC_WIDTH,
    AnalyticWeights,
    construct_fixed_weights,
)
from .wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    VMState,
    encode_program,
    validate_program,
)

SCHEMA = "visuaml.exact-i32-bitsliced-neural-vm.v49"
WIDTH = 32
LIMIT = (1 << 32) - 1
CARRY_THRESHOLDS = np.arange(1., 33., dtype=np.float64) * 2.


class BitVMAdmissionError(ValueError):
    """Unsupported WASM effect, malformed binary register or token encoding."""


@dataclass(frozen=True)
class BitCircuitWeights:
    router: AnalyticWeights
    and_kernel: np.ndarray
    xor_kernel: np.ndarray
    carry_thresholds: np.ndarray
    adder_thresholds: np.ndarray

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        h.update(self.router.digest().encode())
        for tensor in (self.and_kernel, self.xor_kernel, self.carry_thresholds,
                       self.adder_thresholds):
            h.update(np.ascontiguousarray(tensor, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class BitProgramTokens:
    opcodes: np.ndarray
    immediates: np.ndarray
    local_selectors: np.ndarray
    instruction_count: int
    local_count: int
    original_wasm_sha256: str

    def digest(self) -> str:
        h = hashlib.sha256(SCHEMA.encode())
        for a in (self.opcodes, self.immediates, self.local_selectors):
            h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
        h.update(f"{self.instruction_count}:{self.local_count}".encode())
        return h.hexdigest()


@dataclass(frozen=True)
class BitMachineState:
    pc: np.ndarray
    depth: np.ndarray
    stack_bits: np.ndarray
    local_bits: np.ndarray
    halted: bool = False


def construct_bit_weights() -> BitCircuitWeights:
    """Shared matrices and ReLU gate weights are calculated, never learned."""
    return BitCircuitWeights(
        construct_fixed_weights(),
        np.array([1., 1., -1.], dtype=np.float64),
        np.array([[1., -1.], [-1., 1.]], dtype=np.float64),
        CARRY_THRESHOLDS.copy(),
        np.array([1., 2.], dtype=np.float64),
    )


def _bits(value: int) -> np.ndarray:
    """External program/input encoding, not part of the neural instruction path."""
    if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
        raise BitVMAdmissionError("Expected signed i32 external value")
    unsigned = value & LIMIT
    return np.array([(unsigned >> i) & 1 for i in range(WIDTH)], dtype=np.float64)


def _int(bits: np.ndarray) -> int:
    if not isinstance(bits, np.ndarray) or bits.shape != (WIDTH,):
        raise BitVMAdmissionError("Cannot decode non-i32 bitvector")
    if not np.isin(bits, [0., 1.]).all():
        raise BitVMAdmissionError("Cannot decode nonbinary i32 register")
    unsigned = sum(int(bit) << i for i, bit in enumerate(bits))
    return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned


def _admit_program(program: I32Program) -> None:
    validate_program(program)
    if program.local_count > MAX_LOCALS or len(program.operations) > MAX_OPS:
        raise BitVMAdmissionError("i32 machine operation/local bounds exceeded")
    depth = 0
    for op in program.operations:
        if op.opcode not in INDEX:
            raise BitVMAdmissionError("Unsupported opcode")
        if op.opcode in ("i32.const", "local.get"):
            depth += 1
        elif op.opcode in (
            "local.set", "drop", "i32.add", "i32.sub", "i32.mul",
            "i32.lt_s", "i32.gt_s"
        ):
            depth -= 1
        if not 0 <= depth <= MAX_STACK:
            raise BitVMAdmissionError("Bit-machine stack bound exceeded")


def tokenize(program: I32Program) -> BitProgramTokens:
    _admit_program(program)
    opcode = np.zeros((PC_WIDTH, len(OPCODES)), dtype=np.float64)
    immediate = np.zeros((PC_WIDTH, WIDTH), dtype=np.float64)
    local = np.zeros((PC_WIDTH, LOCAL_WIDTH), dtype=np.float64)
    for index, inst in enumerate(program.operations):
        opcode[index, INDEX[inst.opcode]] = 1.
        if inst.opcode == "i32.const":
            immediate[index] = _bits(inst.immediate)
        if inst.opcode.startswith("local."):
            local[index, inst.immediate] = 1.
    return BitProgramTokens(
        opcode, immediate, local, len(program.operations), program.local_count,
        hashlib.sha256(encode_program(program)).hexdigest(),
    )


def _admit_binary(a: np.ndarray, shape: tuple[int, ...]) -> None:
    if (not isinstance(a, np.ndarray) or a.dtype != np.float64
            or a.shape != shape or not is_exact_binary(a)):
        raise BitVMAdmissionError("Noncanonical binary neural tensor")


def decode_tokens(tokens: BitProgramTokens) -> I32Program:
    if not isinstance(tokens, BitProgramTokens):
        raise BitVMAdmissionError("Unknown bit-token program type")
    if (type(tokens.instruction_count) is not int or
            not 2 <= tokens.instruction_count <= MAX_OPS
            or type(tokens.local_count) is not int or
            not 0 <= tokens.local_count <= MAX_LOCALS):
        raise BitVMAdmissionError("Bit-token schema or instruction bounds violated")
    _admit_binary(tokens.opcodes, (PC_WIDTH, len(OPCODES)))
    _admit_binary(tokens.immediates, (PC_WIDTH, WIDTH))
    _admit_binary(tokens.local_selectors, (PC_WIDTH, LOCAL_WIDTH))
    output = []
    for i in range(tokens.instruction_count):
        row = tokens.opcodes[i]
        if float(np.sum(row)) != 1.:
            raise BitVMAdmissionError("Opcode selector is not one-hot")
        name = OPCODES[int(np.argmax(row))]
        immediate = tokens.immediates[i]
        local = tokens.local_selectors[i]
        if name == "i32.const":
            if np.any(local):
                raise BitVMAdmissionError("Constant has spurious local selector")
            output.append(VMOperation(name, _int(immediate)))
        elif name.startswith("local."):
            if np.any(immediate) or float(np.sum(local)) != 1.:
                raise BitVMAdmissionError("Noncanonical local selector")
            idx = int(np.argmax(local))
            if idx > tokens.local_count:
                raise BitVMAdmissionError("Local access outside declarations")
            output.append(VMOperation(name, idx))
        else:
            if np.any(immediate) or np.any(local):
                raise BitVMAdmissionError("Spurious opcode operands")
            output.append(VMOperation(name))
    count = tokens.instruction_count
    if (np.any(tokens.opcodes[count:]) or np.any(tokens.immediates[count:])
            or np.any(tokens.local_selectors[count:])):
        raise BitVMAdmissionError("Noncanonical nonzero instruction padding")
    program = I32Program(tokens.local_count, tuple(output))
    _admit_program(program)
    if hashlib.sha256(encode_program(program)).hexdigest() != tokens.original_wasm_sha256:
        raise BitVMAdmissionError("Tokenized program/source WASM digest mismatch")
    return program


def _and(a, b, fixed: BitCircuitWeights | None = None):
    """Actual fixed-weight ReLU(a+b-1) AND gate on binary activations."""
    kernel = (fixed.and_kernel if fixed is not None else
              np.array([1., 1., -1.], dtype=np.float64))
    return np.maximum(0., kernel[0]*a + kernel[1]*b + kernel[2])


def _xor(a, b, fixed: BitCircuitWeights | None = None):
    """Two constructed ReLU gates read their actual coefficient matrix."""
    kernel = (fixed.xor_kernel if fixed is not None else
              np.array([[1., -1.], [-1., 1.]], dtype=np.float64))
    return (np.maximum(0., kernel[0, 0]*a + kernel[0, 1]*b)
            + np.maximum(0., kernel[1, 0]*a + kernel[1, 1]*b))


def _or(a, b, fixed: BitCircuitWeights | None = None):
    return a + b - _and(a, b, fixed)


def _add(a: np.ndarray, b: np.ndarray, cin: float = 0.,
         fixed: BitCircuitWeights | None = None) -> np.ndarray:
    """Ripple carry constructed from ReLU integer threshold cells."""
    out = np.empty(WIDTH, dtype=np.float64)
    carry = np.float64(cin)
    thresholds = (fixed.adder_thresholds if fixed is not None else
                  np.array([1., 2.], dtype=np.float64))
    for i in range(WIDTH):
        n = a[i] + b[i] + carry
        carry = (np.maximum(0., n - thresholds[0])
                 - np.maximum(0., n - thresholds[1]))
        out[i] = n - 2. * carry
    return out


def _sub(a: np.ndarray, b: np.ndarray,
         fixed: BitCircuitWeights | None = None) -> np.ndarray:
    return _add(a, 1. - b, 1., fixed)


def _mul(a: np.ndarray, b: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """32x32 ReGLU bit products + a fixed ReLU carry staircase.

    For nonnegative integer column sum n, floor(n/2) is exactly the sum of
    step(n>=2j), each represented as ReLU(n-(2j-1))-ReLU(n-2j).
    Discard the carry past bit 31 for WASM modulo 2**32 semantics.
    """
    partial = np.maximum(0., a[:, None]) * b[None, :]
    out = np.empty(WIDTH, dtype=np.float64)
    carry = np.float64(0.)
    for k in range(WIDTH):
        # No unknown-width integer multiplication and no modulo operation.
        n = np.sum([partial[i, k-i] for i in range(k+1)]) + carry
        step = np.maximum(0., n - (thresholds - 1.)) - np.maximum(
            0., n - thresholds
        )
        carry = np.sum(step)
        out[k] = n - 2. * carry
    return out


def _eqz(a: np.ndarray, fixed: BitCircuitWeights | None = None):
    """All bits zero, via a product of binary ReGLU/AND gates."""
    result = np.float64(1.)
    for i in range(WIDTH):
        result = _and(result, 1. - a[i], fixed)
    return result


def _unsigned_lt(a: np.ndarray, b: np.ndarray,
                 fixed: BitCircuitWeights | None = None):
    same = np.float64(1.)
    less = np.float64(0.)
    for i in reversed(range(WIDTH)):
        less = _or(less, _and(same, _and(1. - a[i], b[i], fixed), fixed), fixed)
        same = _and(same, 1. - _xor(a[i], b[i], fixed), fixed)
    return less


def _signed_lt(a: np.ndarray, b: np.ndarray,
               fixed: BitCircuitWeights | None = None):
    different_sign = _xor(a[31], b[31], fixed)
    return _or(
        _and(a[31], 1. - b[31], fixed),
        _and(1. - different_sign, _unsigned_lt(a, b, fixed), fixed),
        fixed,
    )


def initial_state(program: I32Program, value: int) -> BitMachineState:
    _admit_program(program)
    pc = np.zeros(PC_WIDTH, dtype=np.float64)
    pc[0] = 1.
    depth = np.zeros(DEPTH_WIDTH, dtype=np.float64)
    depth[0] = 1.
    local = np.zeros((LOCAL_WIDTH, WIDTH), dtype=np.float64)
    local[0] = _bits(value)
    return BitMachineState(
        pc, depth, np.zeros((MAX_STACK, WIDTH), dtype=np.float64), local,
    )


def _admit_state(state: BitMachineState) -> None:
    if not isinstance(state, BitMachineState) or state.halted:
        raise BitVMAdmissionError("Invalid or halted machine state")
    _admit_binary(state.pc, (PC_WIDTH,))
    _admit_binary(state.depth, (DEPTH_WIDTH,))
    _admit_binary(state.stack_bits, (MAX_STACK, WIDTH))
    _admit_binary(state.local_bits, (LOCAL_WIDTH, WIDTH))
    if np.sum(state.pc) != 1. or np.sum(state.depth) != 1.:
        raise BitVMAdmissionError("PC/depth not one-hot")


def bit_step(
    state: BitMachineState, tokens: BitProgramTokens, fixed: BitCircuitWeights
) -> BitMachineState:
    """Single fixed neural transition, no Python instruction dispatch.

    Every arithmetic candidate is computed from the same gate topology.
    Opcode selectors are tensor data, not Python control branches.
    """
    _admit_state(state)
    flags = state.pc @ tokens.opcodes
    immediate = state.pc @ tokens.immediates
    local_selector = state.pc @ tokens.local_selectors
    if not np.isin(flags, [0., 1.]).all() or np.sum(flags) != 1.:
        raise BitVMAdmissionError("Invalid fetched opcode vector")
    control = flags @ fixed.router.operation_control
    push, pop, binary, unary, localwrite, halt = control

    top_sel = state.depth @ fixed.router.stack_top
    second_sel = state.depth @ fixed.router.stack_second
    push_sel = state.depth @ fixed.router.stack_push
    top = top_sel @ state.stack_bits
    second = second_sel @ state.stack_bits
    local_val = local_selector @ state.local_bits

    candidate = np.zeros((len(OPCODES), WIDTH), dtype=np.float64)
    candidate[INDEX["i32.const"]] = immediate
    candidate[INDEX["local.get"]] = local_val
    candidate[INDEX["i32.add"]] = _add(second, top, fixed=fixed)
    candidate[INDEX["i32.sub"]] = _sub(second, top, fixed=fixed)
    candidate[INDEX["i32.mul"]] = _mul(second, top, fixed.carry_thresholds)
    candidate[INDEX["i32.eqz"], 0] = _eqz(top, fixed)
    candidate[INDEX["i32.lt_s"], 0] = _signed_lt(second, top, fixed)
    candidate[INDEX["i32.gt_s"], 0] = _signed_lt(top, second, fixed)
    write_bits = flags @ candidate
    write_sel = push * push_sel + binary * second_sel + unary * top_sel
    clear_sel = pop * top_sel
    new_stack = (
        state.stack_bits * (1. - write_sel[:, None])
        + write_bits[None, :] * write_sel[:, None]
    ) * (1. - clear_sel[:, None])
    new_locals = (
        state.local_bits * (1. - localwrite * local_selector[:, None])
        + top[None, :] * (localwrite * local_selector[:, None])
    )
    new_depth = (
        (1. - push - pop) * state.depth
        + push * (state.depth @ fixed.router.depth_push)
        + pop * (state.depth @ fixed.router.depth_pop)
    )
    new_pc = state.pc @ fixed.router.program_next
    result = BitMachineState(new_pc, new_depth, new_stack, new_locals,
                             bool(halt == 1.))
    _admit_binary(new_stack, (MAX_STACK, WIDTH))
    _admit_binary(new_locals, (LOCAL_WIDTH, WIDTH))
    _admit_binary(new_depth, (DEPTH_WIDTH,))
    _admit_binary(new_pc, (PC_WIDTH,))
    if np.sum(new_pc) != 1. or np.sum(new_depth) != 1.:
        raise BitVMAdmissionError("Gate transition lost one-hot machine state")
    return result


def decode_state(state: BitMachineState, program: I32Program) -> VMState:
    pc = int(np.argmax(state.pc))
    depth = int(np.argmax(state.depth))
    return VMState(
        pc, tuple(_int(state.stack_bits[i]) for i in range(depth)),
        tuple(_int(state.local_bits[i]) for i in range(program.local_count+1)),
        state.halted,
    )


def run_bit_vm(
    program: I32Program, argument: int, *,
    fixed: BitCircuitWeights | None = None,
) -> dict:
    _admit_program(program)
    fixed = construct_bit_weights() if fixed is None else fixed
    tokens = tokenize(program)
    if decode_tokens(tokens) != program:
        raise BitVMAdmissionError("Program token admission failed")
    state = initial_state(program, argument)
    trace = []
    for index, operation in enumerate(program.operations):
        state = bit_step(state, tokens, fixed)
        trace.append({
            "step": index,
            "opcode": operation.opcode,
            "decoded": decode_state(state, program),
        })
    final = decode_state(state, program)
    if not state.halted or len(final.stack) != 1:
        raise BitVMAdmissionError("Fixed bit-gated machine failed to return")
    return {
        "schema": SCHEMA, "input": argument,
        "original_wasm_sha256": tokens.original_wasm_sha256,
        "fixed_bit_weights_sha256": fixed.digest(),
        "program_token_sha256": tokens.digest(),
        "result": final.stack[0],
        "trace": trace,
        "source_program_trained_into_weights": False,
        "all_signed_i32_inputs_admitted": True,
        "full_domain_exhaustively_executed": False,
        "full_wasm_supported": False,
        "standard_softmax_transformer_executed": False,
        "arbitrary_neural_program_decompilation": False,
    }
