"""A frozen exact-tensor recurrent step for the bounded structured i32 dialect."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import torch
import wasmtime
from torch import Tensor

from .neural_control_routing_v52 import compile_route_coordinates
from .wasm_control_flow_v51 import (
    MAX_LOCALS, MAX_STACK, MEMORY_BYTES, SCALED_MAX_FUEL, ControlAdmissionError,
    MemoryArgument, decode_control,
)

# The placement of each entry is part of the fixed gate graph's ABI.
OPCODES = (
    "nop", "block", "loop", "if", "else", "end", "br", "br_if", "return",
    "drop", "local.get", "local.set", "local.tee", "i32.const", "i32.eqz",
    "i32.lt_s", "i32.gt_s", "i32.add", "i32.sub", "i32.mul", "i32.load", "i32.store",
)
POPS = (0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 1, 2)
PUSHES = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0)


class MachineState(NamedTuple):
    pc: Tensor
    height: Tensor
    stack: Tensor
    locals: Tensor
    page: Tensor
    fuel: Tensor
    status: Tensor  # running, halted, exhausted, memory trap


class Tick(NamedTuple):
    state: MachineState
    access: Tensor  # [0 none / 1 load / 2 store / 3 fault, address, before, after]


@dataclass(frozen=True)
class ProgramImage:
    rows: Tensor
    binary_sha256: str
    image_sha256: str
    local_count: int
    memory_pages: int


def load_program(raw: bytes, *, memory_pages: int = 1) -> ProgramImage:
    """Admit unchanged WASM and resolve lexical edges and branch restore heights."""
    if type(memory_pages) is not int or not 1 <= memory_pages <= 256:
        raise ControlAdmissionError("Fixed memory must have between 1 and 256 pages")
    program = decode_control(raw, scaled=True, memory_pages=memory_pages)
    try:
        wasmtime.Module.validate(wasmtime.Engine(), raw)
    except wasmtime.WasmtimeError as error:
        raise ControlAdmissionError("Invalid WASM operand types") from error
    image = _function_rows(program)
    return ProgramImage(image, sha256(raw).hexdigest(),
                        sha256(image.numpy().astype("<i8", copy=False).tobytes()).hexdigest(),
                        program.local_count, memory_pages)


def _function_rows(program, call_sites=frozenset()) -> Tensor:
    ops = program.operations
    count = len(ops)
    coordinates, guards = compile_route_coordinates(program, count)
    indexed = [OPCODES.index(op.opcode) for op in ops]
    parents = []
    labels = []
    for pc, op in enumerate(ops):
        parents.append(labels[-1 - op.immediate] if op.opcode in ("br", "br_if") else -1)
        if op.opcode in ("block", "loop", "if"):
            labels.append(pc)
        elif op.opcode == "end" and labels:
            labels.pop()

    heights = {0: 0}
    restore = {}
    pending = deque([0])
    while pending:
        pc = pending.popleft()
        height = heights[pc]
        if pc == count:
            if height != 1:
                raise ControlAdmissionError("Terminal must have exactly one result")
            continue
        op = ops[pc]
        code = indexed[pc]
        required = max(POPS[code], int(op.opcode == "local.tee" or pc in call_sites))
        after = height - POPS[code] + PUSHES[code]
        if height < required or after > MAX_STACK:
            raise ControlAdmissionError("Operand height outside the frozen stack")
        normal, alternate = map(int, coordinates[pc])
        if parents[pc] >= 0:
            parent = parents[pc]
            restore[pc] = heights[parent] - POPS[indexed[parent]]
        edges = [(normal, after)] if not guards[pc, 0] else []
        if guards[pc].any():
            edges.append((alternate, restore.get(pc, after)))
        for target, depth in edges:
            if target not in heights:
                heights[target] = depth
                pending.append(target)
            elif heights[target] != depth:
                raise ControlAdmissionError("Inconsistent operand height at a lexical join")

    rows = []
    for pc, op in enumerate(ops):
        immediate = op.immediate
        rows.append([
            indexed[pc], immediate & 0xffffffff if isinstance(immediate, int) else 0,
            immediate.offset if isinstance(immediate, MemoryArgument) else 0,
            *map(int, coordinates[pc]), *map(int, guards[pc]), restore.get(pc, 0),
        ])
    rows.append([0, 0, 0, count, count, 0, 0, 0, 0])
    return torch.tensor(rows, dtype=torch.int64)

def initial_state(argument: int, page: bytes | bytearray, fuel: int, *,
                  memory_pages: int = 1) -> MachineState:
    """Allocate new invocation-private live memory, including for mutable inputs."""
    if (type(argument) is not int or not -(1 << 31) <= argument < 1 << 31
            or type(fuel) is not int or not 0 <= fuel <= SCALED_MAX_FUEL
            or type(memory_pages) is not int or not 1 <= memory_pages <= 256
            or type(page) not in (bytes, bytearray) or len(page) != MEMORY_BYTES * memory_pages):
        raise ControlAdmissionError("Invalid initial recurrent state")
    zero = torch.zeros((), dtype=torch.int64)
    locals_ = torch.zeros(MAX_LOCALS + 1, dtype=torch.int64)
    locals_[0] = argument & 0xffffffff
    return MachineState(zero, zero.clone(), torch.zeros(MAX_STACK, dtype=torch.int64),
                        locals_, torch.frombuffer(bytearray(page), dtype=torch.uint8),
                        torch.tensor(fuel, dtype=torch.int64), zero.clone())


def snapshot_page(state: MachineState) -> bytes:
    """Publish an immutable byte copy, independently of later live stores."""
    return state.page.numpy().tobytes()


class RecurrentStep(torch.nn.Module):
    """All instruction/state decisions are in the scripted graph."""

    __constants__ = ["bounded_store", "reuse_state"]

    def __init__(self, bounded_store: bool = False, reuse_state: bool = False):
        super().__init__()
        self.bounded_store = bounded_store
        self.reuse_state = reuse_state
        self.register_buffer("opcodes", torch.arange(len(OPCODES), dtype=torch.int64))
        self.register_buffer("pops", torch.tensor(POPS, dtype=torch.int64))
        self.register_buffer("pushes", torch.tensor(PUSHES, dtype=torch.int64))
        self.register_buffer("stack_slots", torch.arange(MAX_STACK, dtype=torch.int64))
        self.register_buffer("local_slots", torch.arange(MAX_LOCALS + 1, dtype=torch.int64))
        self.register_buffer("lanes", torch.arange(4, dtype=torch.int64))
        self.register_buffer("shifts", torch.arange(4, dtype=torch.int64) * 8)

    def forward(self, image: Tensor, state: MachineState) -> Tick:
        row = torch.index_select(image, 0, state.pc.reshape(1))[0]
        return self.step_row(image, state, row)

    def step_row(self, image: Tensor, state: MachineState, row: Tensor) -> Tick:
        gates = (self.opcodes == row[0]).to(torch.int64)
        right = torch.gather(state.stack, 0, (state.height - 1).clamp(min=0).reshape(1))[0]
        left = torch.gather(state.stack, 0, (state.height - 2).clamp(min=0).reshape(1))[0]
        signed_left = (left ^ 2147483648) - 2147483648
        signed_right = (right ^ 2147483648) - 2147483648
        address = torch.where(gates[21] == 1, left, right) + row[2]
        indices = (address + self.lanes).clamp(max=state.page.numel() - 1)
        loaded = (torch.gather(state.page, 0, indices).to(torch.int64) << self.shifts).sum()
        local_index = row[1].clamp(max=self.local_slots.numel() - 1)
        local = torch.gather(state.locals, 0, local_index.reshape(1))[0]
        z = torch.zeros_like(right)
        candidates = torch.stack((
            z, z, z, z, z, z, z, z, z, z, local, z, z, row[1],
            (right == 0).to(torch.int64), (signed_left < signed_right).to(torch.int64),
            (signed_left > signed_right).to(torch.int64), (left + right) & 4294967295,
            (left - right) & 4294967295, (signed_left * signed_right) & 4294967295, loaded, z,
        ))
        result = (gates * candidates).sum()
        take = (row[5] + row[6] * (right != 0).to(torch.int64)
                + row[7] * (right == 0).to(torch.int64)) != 0
        next_pc = torch.where(take, row[4], row[3])
        base = state.height - (gates * self.pops).sum()
        pushes = (gates * self.pushes).sum()
        height = torch.where(take & ((gates[6] + gates[7]) == 1), row[8], base + pushes)
        running = state.status == 0
        has_fuel = state.fuel > 0
        memory = (gates[20] + gates[21]) == 1
        fault = memory & (address > state.page.numel() - 4)
        commit = running & has_fuel & ~fault
        store = commit & (gates[21] == 1)
        page = state.page
        if bool(store):
            value = ((right >> self.shifts) & 255).to(torch.uint8)
            if self.bounded_store:
                page.scatter_(0, indices, value)
            else:
                page = torch.scatter(page, 0, indices, value)
        status = torch.where(
            running,
            torch.where(~has_fuel, 2, torch.where(fault, 3, (next_pc == image.size(0) - 1).to(torch.int64))),
            state.status,
        )
        access_kind = torch.where(running & has_fuel & fault, 3,
                                  torch.where(commit & memory, gates[20] + 2 * gates[21], z))
        access = torch.stack((access_kind, address, loaded, torch.where(store, right, loaded)))
        if self.reuse_state:
            cells, locals_ = state.stack, state.locals
            if bool(commit):
                if bool((pushes != 0) | (height < state.height)):
                    cells = torch.where((self.stack_slots == base) & (pushes == 1), result, cells)
                    cells = torch.where(self.stack_slots < height, cells, z)
                if bool((gates[11] + gates[12]) == 1):
                    locals_ = torch.index_copy(locals_, 0, local_index.reshape(1), right.reshape(1))
                return Tick(MachineState(next_pc, height, cells, locals_, page, state.fuel - 1, status), access)
            return Tick(MachineState(state.pc, state.height, cells, locals_, page, state.fuel, status), access)
        cells = torch.where((self.stack_slots == base) & (pushes == 1), result, state.stack)
        cells = torch.where(self.stack_slots < height, cells, z)
        locals_ = torch.where((self.local_slots == local_index) & ((gates[11] + gates[12]) == 1),
                              right, state.locals)
        return Tick(MachineState(
            torch.where(commit, next_pc, state.pc), torch.where(commit, height, state.height),
            torch.where(commit, cells, state.stack), torch.where(commit, locals_, state.locals),
            page, state.fuel - commit.to(torch.int64), status,
        ), access)


def freeze_machine(*, bounded_store: bool = False):
    """Freeze one graph and its constant tensors without any guest input."""
    return torch.jit.freeze(torch.jit.script(RecurrentStep(bounded_store).eval()))
