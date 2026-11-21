"""Adapt validated source rows to the archived ordinary CPU runner."""
from pathlib import Path
from subprocess import CalledProcessError

import torch

from visuaml.compiler_lab.frozen_recurrent_v58 import POPS, PUSHES
from visuaml.compiler_lab.ordinary_native_v58 import register

GRAPH = Path(__file__).resolve().parents[3] / "voxel-replay/evidence/compiled-ordinary-cpu/machine.pt"
MAX_BATCH = 64


def prepare(rows):
    # Use the ordinary lane only. Per-program fusion metadata is optional in
    # the research runner and this portable profile doesn't supply it.
    opcodes = rows[:, 0].tolist()
    effects = torch.tensor([(POPS[op], PUSHES[op]) for op in opcodes], dtype=torch.int64)
    frames = ((rows[:, 9] >= 0) | (rows[:, 10] != 0)).to(torch.int64)
    eligible = ((rows[:, 0] < 20) & (frames == 0)).to(torch.int64)
    eligible[-1] = 0
    unused = torch.full((len(rows), 3), -1, dtype=torch.int64)
    return torch.cat((rows, unused, effects, frames[:, None], eligible[:, None]), dim=1)


class CompiledRunner:
    def __init__(self, rows):
        try:
            register()
        except (OSError, CalledProcessError) as exc:
            raise ValueError("Compiled runner setup failed. Install a C++17 compiler or use --runner reference.") from exc
        self.graph = torch.jit.load(str(GRAPH), map_location="cpu")
        self.rows = prepare(rows)

    def advance(self, state, budget):
        return self.graph(self.rows, state, budget).state
