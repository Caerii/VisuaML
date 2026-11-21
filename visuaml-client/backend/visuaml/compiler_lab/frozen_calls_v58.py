"""Direct calls for a frozen bounded i32 recurrent machine."""
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import torch
import wasmtime
from torch import Tensor

from .frozen_recurrent_v58 import (
    MachineState, RecurrentStep, _function_rows, initial_state,
)
from .wasm_control_flow_v51 import (
    MAX_LOCALS, MAX_STACK, SCALED_MAX_OPERATIONS, SCALED_SCHEMA,
    ControlAdmissionError, ControlOperation, ControlProgram, WasmBinaryError,
    _decode_control_body, _memory_section, _read_u32, _read_s32, _TYPE_PAYLOAD, _syntax, _u32,
    parse_section_envelope,
)

MAX_FUNCTIONS = 8
FRAME_WIDTH = 2 + MAX_STACK + MAX_LOCALS + 1


@dataclass(frozen=True)
class CallImage:
    rows: Tensor
    entry: int
    memory_pages: int
    functions: tuple[tuple[str, ...], ...]
    starts: tuple[int, ...]
    binary_sha256: str
    image_sha256: str
    data_segments: tuple[tuple[int, bytes], ...] = ()


def _active_data(payload, capacity):
    count, cursor = _read_u32(payload, 0)
    segments = []
    for _ in range(count):
        mode, cursor = _read_u32(payload, cursor)
        if mode != 0 or payload[cursor:cursor+1] != b"\x41":
            raise ControlAdmissionError("Expected active memory-zero data with an i32.const offset")
        address, cursor = _read_s32(payload, cursor+1)
        if payload[cursor:cursor+1] != b"\x0b":
            raise ControlAdmissionError("Expected end of data offset")
        length, cursor = _read_u32(payload, cursor+1)
        end = cursor+length
        address &= 0xffffffff
        if address+length > capacity or end > len(payload):
            raise ControlAdmissionError("Active data lies outside replay memory")
        segments.append((address, payload[cursor:end]))
        cursor = end
    if cursor != len(payload):
        raise ControlAdmissionError("Trailing active data bytes")
    return tuple(segments)


def load_call_image(raw: bytes, *, memory_pages: int = 1) -> CallImage:
    """Validate fixed-memory, acyclic (lower-index) unary i32 calls."""
    try:
        wasmtime.Module.validate(wasmtime.Engine(), raw)
        sections = {item.section_id: item.payload for item in parse_section_envelope(raw) if item.section_id}
        extras = set(sections) - {1, 3, 5, 7, 10, 11}
        if extras:
            raise ControlAdmissionError(f"Unsupported WASM core sections: {sorted(extras)}")
        if not {1, 3, 5, 7, 10}.issubset(sections) or sections[1] != _TYPE_PAYLOAD:
            raise ControlAdmissionError("Expected unary i32 functions with fixed memory")
        if sections[5] != _memory_section(memory_pages):
            raise ControlAdmissionError("Unexpected fixed memory capacity")
        data_segments = _active_data(sections[11], memory_pages*65536) if 11 in sections else ()
        count, pos = _read_u32(sections[3], 0)
        if not 1 <= count <= MAX_FUNCTIONS or sections[3][pos:] != bytes(count):
            raise ControlAdmissionError("Only bounded unary i32 functions admitted")
        entry = count - 1
        run = b"\x03run\x00" + _u32(entry)
        memory = b"\x06memory\x02\x00"
        if sections[7] not in (b"\x02" + run + memory, b"\x02" + memory + run):
            raise ControlAdmissionError("Expected memory and last function exported as run")
        data = sections[10]
        bodies, pos = _read_u32(data, 0)
        if bodies != count:
            raise ControlAdmissionError("Function and code counts disagree")

        functions, starts, prepared = [], [], []
        offset = 0
        for function in range(count):
            size, pos = _read_u32(data, pos)
            locals_, ops = _decode_control_body(
                data[pos:pos + size], True, SCALED_MAX_OPERATIONS, direct_calls=True,
            )
            pos += size
            calls = {i: op.immediate for i, op in enumerate(ops) if op.opcode == "call"}
            if any(target >= function for target in calls.values()):
                raise ControlAdmissionError("Calls must target lower-index functions")
            # A unary i32 call replaces the top operand with one i32.
            lexical = tuple(ControlOperation("nop") if i in calls else op
                            for i, op in enumerate(ops))
            program = ControlProgram(locals_, lexical, SCALED_SCHEMA)
            _syntax(program)
            rows = _function_rows(program, calls)[:-1].clone()
            prepared.append((rows, calls, ops))
            starts.append(offset)
            functions.append(tuple(op.opcode for op in ops))
            offset += len(ops)
        if pos != len(data):
            raise ControlAdmissionError("Unexpected trailing function bytes")
    except (wasmtime.WasmtimeError, WasmBinaryError, IndexError) as error:
        raise ControlAdmissionError("Invalid direct-call WASM module") from error

    segments = []
    for start, (rows, calls, ops) in zip(starts, prepared, strict=True):
        routes = rows[:, 3:5]
        routes[:] = torch.where(routes == len(ops), offset, routes + start)
        call_targets = torch.full((len(ops), 1), -1, dtype=torch.int64)
        for pc, target in calls.items():
            call_targets[pc, 0] = starts[target]
        returns = torch.tensor([op.opcode == "return" or i == len(ops) - 1
                                for i, op in enumerate(ops)], dtype=torch.int64).unsqueeze(1)
        segments.append(torch.cat((rows, call_targets, returns), dim=1))
    segments.append(torch.tensor([[0, 0, 0, offset, offset, 0, 0, 0, 0, -1, 0]]))
    image = torch.cat(segments)
    image_hash = sha256(image.numpy().astype("<i8", copy=False).tobytes())
    if data_segments:
        image_hash.update(sections[11])
    return CallImage(image, starts[entry], memory_pages, tuple(functions), tuple(starts),
                     sha256(raw).hexdigest(), image_hash.hexdigest(), data_segments)


class CallState(NamedTuple):
    vm: MachineState
    depth: Tensor
    frames: Tensor


class CallTick(NamedTuple):
    state: CallState
    access: Tensor


def initialize(image: CallImage, argument: int, page: bytes | bytearray | None, fuel: int) -> CallState:
    if page is None:
        page = bytearray(image.memory_pages*65536)
        for address, data in image.data_segments:
            page[address:address+len(data)] = data
    vm = initial_state(argument, page, fuel, memory_pages=image.memory_pages)
    vm = vm._replace(pc=torch.tensor(image.entry, dtype=torch.int64))
    return CallState(vm, torch.zeros((), dtype=torch.int64),
                     torch.zeros((MAX_FUNCTIONS - 1, FRAME_WIDTH), dtype=torch.int64))


class CallStep(torch.nn.Module):
    """Recurrent call and return transitions own both caller and callee state."""

    __constants__ = ["reuse_row"]

    def __init__(self, reuse_row: bool = False, reuse_state: bool = False,
                 select_result: bool = False, select_memory: bool = False):
        super().__init__()
        self.reuse_row = reuse_row
        self.base = RecurrentStep(bounded_store=True, reuse_state=reuse_state, select_result=select_result,
                                  select_memory=select_memory)
        self.register_buffer("stack_slots", torch.arange(MAX_STACK))
        self.stack_width = MAX_STACK

    def forward(self, image: Tensor, state: CallState) -> CallTick:
        row = torch.index_select(image, 0, state.vm.pc.reshape(1))[0]
        return self.step_row(image, state, row)

    def step_row(self, image: Tensor, state: CallState, row: Tensor) -> CallTick:
        vm = state.vm
        enabled = bool((vm.status == 0) & (vm.fuel > 0))
        if enabled and bool(row[9] >= 0):
            top = torch.gather(vm.stack, 0, (vm.height - 1).reshape(1))
            frame = torch.cat((row[3].reshape(1), vm.height.reshape(1), vm.stack, vm.locals))
            frames = torch.index_copy(state.frames, 0, state.depth.reshape(1), frame.unsqueeze(0))
            locals_ = torch.cat((top, torch.zeros_like(vm.locals[1:])))
            callee = MachineState(row[9], torch.zeros_like(vm.height), torch.zeros_like(vm.stack),
                                  locals_, vm.page, vm.fuel - 1, vm.status)
            return CallTick(CallState(callee, state.depth + 1, frames), torch.zeros_like(row[:4]))
        if enabled and bool((row[10] != 0) & (state.depth > 0)):
            depth = state.depth - 1
            frame = torch.index_select(state.frames, 0, depth.reshape(1))[0]
            result = torch.gather(vm.stack, 0, (vm.height - 1).reshape(1))[0]
            stack = torch.where(self.stack_slots == frame[1] - 1, result,
                                frame[2:2 + self.stack_width])
            caller = MachineState(frame[0], frame[1], stack, frame[2 + self.stack_width:],
                                  vm.page, vm.fuel - 1, vm.status)
            frames = torch.index_copy(state.frames, 0, depth.reshape(1),
                                      torch.zeros_like(frame).unsqueeze(0))
            return CallTick(CallState(caller, depth, frames), torch.zeros_like(row[:4]))
        tick = self.base.step_row(image, vm, row) if self.reuse_row else self.base(image, vm)
        return CallTick(CallState(tick.state, state.depth, state.frames), tick.access)


def freeze_call_machine():
    return torch.jit.freeze(torch.jit.script(CallStep().eval()))
