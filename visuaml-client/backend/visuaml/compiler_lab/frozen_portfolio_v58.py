"""Frozen lowering for a checked portfolio of two-source-step regions."""
import torch
from torch import Tensor

from .frozen_calls_v58 import CallState, CallStep
from .frozen_recurrent_v58 import MAX_STACK, MachineState
from .frozen_region_v58 import RegionTick


class PortfolioStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.capacity = MAX_STACK
        self.step = CallStep(reuse_row=True, reuse_state=True, select_result=True, select_memory=True)
        self.register_buffer("pair_offsets", torch.arange(2, dtype=torch.int64))
        self.register_buffer("byte_offsets", torch.arange(4, dtype=torch.int64))
        self.register_buffer("shifts", torch.arange(4, dtype=torch.int64) * 8)

    def forward(self, image: Tensor, state: CallState, budget: int) -> RegionTick:
        if budget < 0 or budget > 64:
            raise RuntimeError("Expected a transition budget from 0 to 64")
        access = torch.zeros(4, dtype=torch.int64)
        pairs = 0
        others = 0
        while budget > 0 and bool(state.vm.status == 0):
            vm = state.vm
            if not bool(vm.fuel > 0):
                vm = MachineState(vm.pc, vm.height, vm.stack, vm.locals, vm.page,
                                  vm.fuel, torch.full_like(vm.status, 2))
                state = CallState(vm, state.depth, state.frames)
                access = torch.zeros_like(access)
                break
            row = torch.index_select(image, 0, vm.pc.reshape(1))[0]
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
                        others += 1
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
                    others += 1
                    continue
            tick = self.step.step_row(image, state, row)
            state, access = tick.state, tick.access
            budget -= 1
        return RegionTick(state, access, pairs, others)


def freeze_portfolio():
    return torch.jit.freeze(torch.jit.script(PortfolioStep().eval()))
