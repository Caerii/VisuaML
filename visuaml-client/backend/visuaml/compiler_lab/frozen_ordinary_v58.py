"""Frozen portfolio with prepared ordinary effects and optional lazy operands."""
from typing import NamedTuple

import torch
from torch import Tensor

from .frozen_calls_v58 import CallState, CallStep, CallTick
from .frozen_recurrent_v58 import MAX_LOCALS, MAX_STACK, MachineState


class PortfolioTick(NamedTuple):
    state: CallState
    access: Tensor
    fused_pairs: int
    fused_loads: int
    fused_arithmetic: int


class OrdinaryPortfolio(torch.nn.Module):
    __constants__ = ["lazy_fields", "native_runs"]
    def __init__(self, lazy_fields: bool = False, native_runs: bool = False):
        super().__init__()
        self.capacity = MAX_STACK
        self.lazy_fields = lazy_fields
        self.native_runs = native_runs
        self.register_buffer("stack_slots", torch.arange(MAX_STACK))
        self.register_buffer("local_slots", torch.arange(MAX_LOCALS + 1))
        self.step = CallStep(reuse_row=True, reuse_state=True, select_result=True, select_memory=True)
        self.register_buffer("pair_offsets", torch.arange(2, dtype=torch.int64))
        self.register_buffer("byte_offsets", torch.arange(4, dtype=torch.int64))
        self.register_buffer("shifts", torch.arange(4, dtype=torch.int64) * 8)

    def forward(self, image: Tensor, state: CallState, budget: int) -> PortfolioTick:
        if budget < 0 or budget > 64:
            raise RuntimeError("Expected a transition budget from 0 to 64")
        access = torch.zeros(4, dtype=torch.int64)
        pairs = 0
        loads = 0
        arithmetic = 0
        while budget > 0 and bool(state.vm.status == 0):
            vm = state.vm
            if not bool(vm.fuel > 0):
                vm = MachineState(vm.pc, vm.height, vm.stack, vm.locals, vm.page,
                                  vm.fuel, torch.full_like(vm.status, 2))
                state = CallState(vm, state.depth, state.frames)
                access = torch.zeros_like(access)
                break
            row = torch.index_select(image, 0, vm.pc.reshape(1))[0]
            if self.native_runs and bool(row[17]):
                pc, height, stack, locals_, fuel, status, steps = torch.ops.visuaml_ordinary.run(
                    image, vm.pc, vm.height, vm.stack, vm.locals, vm.fuel, budget)
                if steps > 0:
                    vm = MachineState(pc, height, stack, locals_, vm.page, fuel, status)
                    state = CallState(vm, state.depth, state.frames)
                    access = torch.zeros_like(access)
                    budget -= steps
                    continue
            if (budget >= 2 and bool(row[11] >= 0)
                    and bool((vm.fuel >= 2) & (vm.height <= self.capacity - 2))):
                kind = int(row[12])
                if kind >= 0:
                    values = torch.gather(vm.locals, 0, row[11:13])
                    stack = torch.scatter(vm.stack, 0, vm.height + self.pair_offsets, values)
                    vm = MachineState(vm.pc + 2, vm.height + 2, stack, vm.locals, vm.page,
                                      vm.fuel - 2, vm.status)
                    state = CallState(vm, state.depth, state.frames)
                    access = torch.zeros_like(access)
                    budget -= 2
                    pairs += 1
                    continue
                if kind == -2:
                    local = torch.gather(vm.locals, 0, row[11:12])[0]
                    address = local + row[13]
                    if bool(address <= vm.page.numel() - 4):
                        bytes_ = torch.gather(vm.page, 0, address + self.byte_offsets)
                        word = (bytes_.to(torch.int64) << self.shifts).sum()
                        stack = torch.scatter(vm.stack, 0, vm.height.reshape(1), word.reshape(1))
                        vm = MachineState(vm.pc + 2, vm.height + 1, stack, vm.locals, vm.page,
                                          vm.fuel - 2, vm.status)
                        state = CallState(vm, state.depth, state.frames)
                        access = torch.stack((torch.ones_like(word), address, word, word))
                        budget -= 2
                        loads += 1
                        continue
                if kind == -3 and bool(vm.height > 0):
                    index = (vm.height - 1).reshape(1)
                    left = torch.gather(vm.stack, 0, index)[0]
                    word = self.step.base.selected_result(int(row[11]), row[13], left, row[13],
                                                          left, left, torch.zeros_like(left))
                    stack = torch.scatter(vm.stack, 0, index, word.reshape(1))
                    vm = MachineState(vm.pc + 2, vm.height, stack, vm.locals, vm.page,
                                      vm.fuel - 2, vm.status)
                    state = CallState(vm, state.depth, state.frames)
                    access = torch.zeros_like(access)
                    budget -= 2
                    arithmetic += 1
                    continue
            if bool(row[16]) and (bool(row[9] >= 0) or bool(state.depth > 0)):
                tick = self.step.step_row(image, state, row)
            else:
                tick = self.ordinary(image, state, row)
            state, access = tick.state, tick.access
            budget -= 1
        return PortfolioTick(state, access, pairs, loads, arithmetic)

    def ordinary(self, image: Tensor, state: CallState, row: Tensor) -> CallTick:
        vm = state.vm
        op = int(row[0])
        pops, pushes = row[14], row[15]
        zero = torch.zeros_like(vm.height)
        right, left, local = zero, zero, zero
        if not self.lazy_fields or op == 12 or bool(pops != 0):
            right = torch.gather(vm.stack, 0, (vm.height - 1).clamp(min=0).reshape(1))[0]
        if not self.lazy_fields or bool(pops == 2):
            left = torch.gather(vm.stack, 0, (vm.height - 2).clamp(min=0).reshape(1))[0]
        index = row[1]
        if not self.lazy_fields or op == 10:
            local = torch.gather(vm.locals, 0, index.clamp(max=self.local_slots.numel() - 1).reshape(1))[0]

        memory = op == 20 or op == 21
        word, address, indices = zero, zero, self.byte_offsets
        if memory:
            address = (left if op == 21 else right) + row[2]
            indices = (address + self.byte_offsets).clamp(max=vm.page.numel() - 1)
            word = (torch.gather(vm.page, 0, indices).to(torch.int64) << self.shifts).sum()
            if bool(address > vm.page.numel() - 4):
                trap = MachineState(vm.pc, vm.height, vm.stack, vm.locals, vm.page, vm.fuel,
                                    torch.full_like(vm.status, 3))
                return CallTick(CallState(trap, state.depth, state.frames),
                                torch.stack((torch.full_like(word, 3), address, word, word)))

        if self.lazy_fields:
            take = bool(right == 0) if op == 3 else (bool(right != 0) if op == 7 else bool(row[5]))
            next_pc = row[4] if take else row[3]
        else:
            take_t = (row[5] + row[6] * (right != 0).to(torch.int64)
                      + row[7] * (right == 0).to(torch.int64)) != 0
            next_pc = torch.where(take_t, row[4], row[3])
            take = bool(take_t)
        base = vm.height - pops
        height = row[8] if (op == 6 or op == 7) and take else base + pushes
        cells, locals_ = vm.stack, vm.locals
        if bool((pushes != 0) | (height < vm.height)):
            if bool(pushes != 0):
                value = self.step.base.selected_result(op, row[1], left, right, local, word, zero)
                cells = torch.scatter(cells, 0, base.reshape(1), value.reshape(1))
            if bool(height < vm.height):
                cells = torch.where(self.stack_slots < height, cells, zero)
        if op == 11 or op == 12:
            locals_ = torch.index_copy(locals_, 0, index.reshape(1), right.reshape(1))
        access = torch.zeros(4, dtype=torch.int64)
        if memory:
            if op == 21:
                vm.page.scatter_(0, indices, ((right >> self.shifts) & 255).to(torch.uint8))
            access = torch.stack((torch.full_like(word, 2 if op == 21 else 1),
                                  address, word, right if op == 21 else word))
        status = (next_pc == image.size(0) - 1).to(torch.int64)
        next_vm = MachineState(next_pc, height, cells, locals_, vm.page, vm.fuel - 1, status)
        return CallTick(CallState(next_vm, state.depth, state.frames), access)


def freeze_ordinary(*, lazy_fields: bool = False, native_runs: bool = False):
    return torch.jit.freeze(torch.jit.script(OrdinaryPortfolio(lazy_fields, native_runs).eval()))
