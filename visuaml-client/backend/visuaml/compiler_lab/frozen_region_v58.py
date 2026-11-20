"""Lowering for proof-admitted ordered local-get pairs."""
from typing import NamedTuple

import torch
from torch import Tensor

from .frozen_calls_v58 import CallState, CallStep
from .frozen_recurrent_v58 import MAX_STACK, MachineState


class RegionTick(NamedTuple):
    state: CallState
    access: Tensor
    fused_pairs: int


class RegionStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.capacity = MAX_STACK
        self.step = CallStep(reuse_row=True, reuse_state=True, select_result=True, select_memory=True)
        self.register_buffer("pair_offsets", torch.arange(2, dtype=torch.int64))

    def forward(self, image: Tensor, state: CallState, budget: int) -> RegionTick:
        if budget < 0 or budget > 64:
            raise RuntimeError("Expected a transition budget from 0 to 64")
        access = torch.zeros(4, dtype=torch.int64)
        fused = 0
        while budget > 0 and bool(state.vm.status == 0):
            vm = state.vm
            row = torch.index_select(image, 0, vm.pc.reshape(1))[0]
            if (budget >= 2 and bool(row[11] >= 0)
                    and bool((vm.fuel >= 2) & (vm.height <= self.capacity - 2))):
                values = torch.gather(vm.locals, 0, row[11:13])
                stack = torch.scatter(vm.stack, 0, vm.height + self.pair_offsets, values)
                vm = MachineState(vm.pc + 2, vm.height + 2, stack, vm.locals,
                                  vm.page, vm.fuel - 2, vm.status)
                state = CallState(vm, state.depth, state.frames)
                access = torch.zeros_like(access)
                budget -= 2
                fused += 1
            else:
                tick = self.step.step_row(image, state, row)
                state, access = tick.state, tick.access
                budget -= 1
        return RegionTick(state, access, fused)


def freeze_region_runner():
    return torch.jit.freeze(torch.jit.script(RegionStep().eval()))
