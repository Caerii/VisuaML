"""Bounded recurrence over the frozen direct-call WASM transition."""
import torch
from torch import Tensor

from .frozen_calls_v58 import CallState, CallStep, CallTick


class MultiStep(torch.nn.Module):
    def __init__(self, reuse_row: bool = False, reuse_state: bool = False,
                 select_result: bool = False, select_memory: bool = False):
        super().__init__()
        self.step = CallStep(reuse_row, reuse_state, select_result, select_memory)

    def forward(self, image: Tensor, state: CallState, budget: int) -> CallTick:
        if budget < 0 or budget > 64:
            raise RuntimeError("Expected a transition budget from 0 to 64")
        # Access describes the last attempted transition, or none at budget zero.
        access = torch.zeros(4, dtype=torch.int64)
        remaining = budget
        while remaining > 0 and bool(state.vm.status == 0):
            tick = self.step(image, state)
            state, access = tick.state, tick.access
            remaining -= 1
        return CallTick(state, access)


def freeze_multistep(*, reuse_row: bool = False, reuse_state: bool = False,
                     select_result: bool = False, select_memory: bool = False):
    return torch.jit.freeze(torch.jit.script(MultiStep(reuse_row, reuse_state, select_result, select_memory).eval()))
