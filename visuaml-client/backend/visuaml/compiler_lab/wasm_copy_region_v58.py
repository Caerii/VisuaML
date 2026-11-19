"""Recognize and lower a bounded family of i32 word-move regions."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from hashlib import sha256

from .bit_neural_vm_v49 import _validate_i32
from .neural_control_routing_v52 import TERMINAL, compile_route_tokens
from .neural_microstep_executor_v58 import (
    MicrostepAdmissionError, PreparedMicrostepProgram, _validate_run_options,
)
from .wasm_control_flow_v51 import MEMORY_BYTES, ControlOperation, ControlProgram, MemoryArgument, _syntax, encode_control, uses_memory


def copy_program() -> ControlProgram:
    """run(count) moves words using the source and destination stored at bytes 0 and 4."""
    def op(name, value=None):
        return ControlOperation(name, value)

    load = op("i32.load", MemoryArgument())
    store = op("i32.store", MemoryArgument())

    def address(local):
        return op("local.get", local), op("local.get", 3), op("i32.const", 4), op("i32.mul"), op("i32.add")

    transfer = (*address(2), *address(1), load, store)
    countdown = (
        op("local.get", 0), op("local.set", 3), op("block"), op("loop"),
        op("local.get", 3), op("i32.eqz"), op("br_if", 1),
        op("local.get", 3), op("i32.const", 1), op("i32.sub"), op("local.set", 3),
        *transfer, op("br", 0), op("end"), op("end"),
    )
    countup = (
        op("block"), op("loop"), op("local.get", 3), op("local.get", 0), op("i32.lt_s"),
        op("i32.eqz"), op("br_if", 1), *transfer,
        op("local.get", 3), op("i32.const", 1), op("i32.add"), op("local.set", 3),
        op("br", 0), op("end"), op("end"),
    )
    return ControlProgram(3, (
        op("local.get", 0), op("i32.const", 0), op("i32.gt_s"), op("if"),
        op("i32.const", 0), load, op("local.set", 1),
        op("i32.const", 4), load, op("local.set", 2),
        op("local.get", 1), op("local.get", 2), op("i32.lt_s"), op("if"),
        *countdown, op("else"), *countup, op("end"), op("end"), op("local.get", 0), op("end"),
    ))


@dataclass(frozen=True)
class CopyPlan:
    source_descriptor: int
    destination_descriptor: int
    empty_steps: int
    forward_base: int
    forward_iteration: int
    backward_base: int
    backward_iteration: int

    def steps(self, count: int, *, backward: bool) -> int:
        if count <= 0:
            return self.empty_steps
        if backward:
            return self.backward_base + count * self.backward_iteration
        return self.forward_base + count * self.forward_iteration


def _normalize(program: ControlProgram) -> list[ControlOperation]:
    """Recognize only exact commutations of the reviewed stack expressions."""
    ops = []
    for op in program.operations:
        if op.opcode == "nop":
            continue
        if (op.opcode == "i32.gt_s" and len(ops) >= 2
                and all(item.opcode == "local.get" for item in ops[-2:])):
            ops[-2], ops[-1] = ops[-1], ops[-2]
            op = ControlOperation("i32.lt_s")
        if op.opcode == "i32.add" and len(ops) >= 4:
            left, scale, mul, base = ops[-4:]
            if (left.opcode == base.opcode == "local.get" and scale == ControlOperation("i32.const", 4)
                    and mul.opcode == "i32.mul"):
                ops[-4:] = base, left, scale, mul
        if isinstance(op.immediate, MemoryArgument):
            op = ControlOperation(op.opcode, MemoryArgument(op.immediate.offset))
        ops.append(op)
    return ops


def _fuel_schedule(program, descriptors):
    route = compile_route_tokens(program)
    sequential, alternate = route.sequential.argmax(axis=1), route.alternate.argmax(axis=1)
    outer, direction = [i for i, op in enumerate(program.operations) if op.opcode == "if"]
    exits = {i for i, op in enumerate(program.operations) if op.opcode == "br_if"}

    def cost(count, backward):
        pc = trips = 0
        # A matched loop has one exit test and one fixed back edge. One and
        # two iterations identify their constant path costs, including nops.
        for tick in range(1, len(program.operations) * 4 + 1):
            if pc in exits:
                branch = trips == count
                trips += 1
            else:
                branch = (pc == outer and count == 0) or (pc == direction and not backward)
                branch = branch or route.guard_kind[pc, 0] == 1.
            pc = int((alternate if branch else sequential)[pc])
            if pc == TERMINAL:
                return tick
        raise RuntimeError("Matched copy control graph has unexpected fuel schedule")

    empty = cost(0, False)
    forward, backward = cost(1, False), cost(1, True)
    f_inc, b_inc = cost(2, False) - forward, cost(2, True) - backward
    return CopyPlan(*descriptors, empty, forward - f_inc, f_inc, backward - b_inc, b_inc)


def recognize_copy_region(program: ControlProgram) -> CopyPlan | None:
    """Match the reviewed move grammar and derive descriptor roles and source fuel."""
    _syntax(program)
    ops = _normalize(program)
    expected = copy_program().operations
    if len(ops) != len(expected):
        return None
    descriptors = []
    for index in (4, 7):
        const, load = ops[index:index + 2]
        if const.opcode != "i32.const" or load.opcode != "i32.load":
            return None
        address = (const.immediate & 0xffffffff) + load.immediate.offset
        if address > MEMORY_BYTES - 4:
            return None
        descriptors.append(address)
        ops[index:index + 2] = expected[index:index + 2]

    locals_by_role = {0: 0}
    for actual, pattern in zip(ops, expected, strict=True):
        if actual.opcode != pattern.opcode:
            return None
        if actual.opcode in ("local.get", "local.set"):
            role, slot = pattern.immediate, actual.immediate
            if role not in locals_by_role:
                if slot in locals_by_role.values():
                    return None
                locals_by_role[role] = slot
            elif locals_by_role[role] != slot:
                return None
        elif actual.immediate != pattern.immediate:
            return None
    return _fuel_schedule(program, descriptors)


@dataclass(frozen=True)
class CopyRegionResult:
    result_i32: int
    steps: int
    trace: tuple[dict, ...]
    fused: bool


class PreparedCopyRegion:
    """A recognized move grammar with guarded bulk lowering and source replay."""

    def __init__(self, program: ControlProgram, **overrides):
        raw = encode_control(program)
        if not uses_memory(program):
            raise MicrostepAdmissionError("No certified copy-region lowering for this WASM source")
        self.source_wasm_sha256 = sha256(raw).hexdigest()
        self.plan = recognize_copy_region(program)
        self._specialize = not overrides
        self._reference = PreparedMicrostepProgram(
            program, use_native_loop=True, use_packed_scalar=True, use_structured_control=True, **overrides,
        )

    def run_memory(self, argument: int, page: bytearray, *, fuel=400, capture_trace=True) -> CopyRegionResult:
        _validate_run_options(fuel, capture_trace)
        _validate_i32(argument)
        if type(page) is not bytearray or len(page) != MEMORY_BYTES:
            raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
        plan = self.plan
        if plan is not None and self._specialize and not capture_trace:
            if argument <= 0 and fuel >= plan.empty_steps:
                return CopyRegionResult(argument, plan.empty_steps, (), True)
            if argument > 0:
                # Hold the buffer length fixed while reading the descriptor and
                # moving bytes. The slice expression snapshots overlapping input.
                with memoryview(page) as memory:
                    if len(memory) != MEMORY_BYTES:
                        raise MicrostepAdmissionError("Expected a bytearray with one WASM memory page")
                    src, = struct.unpack_from("<I", memory, plan.source_descriptor)
                    dst, = struct.unpack_from("<I", memory, plan.destination_descriptor)
                    length = 4 * argument
                    steps = plan.steps(argument, backward=src < dst)
                    if (steps <= fuel and src + length <= MEMORY_BYTES and dst + length <= MEMORY_BYTES):
                        memory[dst:dst + length] = memory[src:src + length].tobytes()
                        return CopyRegionResult(argument, steps, (), True)
        reference = self._reference.run_memory(argument, page, fuel=fuel, capture_trace=capture_trace)
        return CopyRegionResult(reference.result_i32, reference.steps, reference.trace, False)
