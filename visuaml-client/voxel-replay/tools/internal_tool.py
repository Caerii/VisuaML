"""One learned router and a frozen recurrent hard-attention i32 circuit."""
import argparse
from pathlib import Path

import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1] / "evidence/internal-tool"


def prepare_program(rows):
    """Validate the program-data ABI before handing it to the fixed attention core."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 48:
        raise ValueError("Tool needs 1–48 instructions")
    for opcode, register, operand in rows:
        if (type(opcode) is not int or not 0 <= opcode <= 4
                or type(register) is not int or not 0 <= register < 2
                or type(operand) is not int):
            raise ValueError("Invalid tool instruction")
        if opcode in (1, 2) and not 0 <= operand < 4:
            raise ValueError("Memory operand out of range")
        if opcode == 3 and not -(1 << 31) <= operand < (1 << 31):
            raise ValueError("Immediate is outside i32 range")
        if opcode == 4 and not 0 <= operand < len(rows):
            raise ValueError("Branch target out of range")
    if rows[-1][0] != 0:
        raise ValueError("Tool image must end in halt")
    return torch.tensor(rows, dtype=torch.int64)


class Circuit(nn.Module):
    def read(self, address: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Exact, unique-winner positional attention. Addresses are validated
        # against the image contract before entering the inference module.
        key = torch.arange(values.size(0), device=values.device)
        winner = (key == address).to(torch.int64).argmax()
        return torch.index_select(values, 0, winner.reshape(1))[0]

    def forward(self, program: torch.Tensor, operands: torch.Tensor):
        pc = torch.zeros((), dtype=torch.int64)
        running = torch.ones((), dtype=torch.bool)
        memory = torch.cat((operands[:1], torch.zeros(3, dtype=torch.int64)))
        registers = torch.cat((torch.zeros(1, dtype=torch.int64), operands[1:]))
        trace = torch.jit.annotate(list[torch.Tensor], [])
        for _ in range(48):
            op, reg, arg = self.read(pc, program).unbind()
            old = self.read(reg, registers)
            computed = torch.where(op == 1, self.read(arg, memory), (old + arg) & 4294967295)
            rmask = (torch.arange(2) == reg) & running & ((op == 1) | (op == 3))
            registers = torch.where(rmask, computed, registers)
            mmask = (torch.arange(4) == arg) & running & (op == 2)
            memory = torch.where(mmask, old, memory)
            running = running & (op != 0)
            next_pc = torch.where((op == 4) & (old != 0), arg, pc + 1)
            pc = torch.where(running, next_pc, pc)
            trace.append(torch.cat((pc.reshape(1), running.to(torch.int64).reshape(1), registers, memory)))
        return memory[0], torch.stack(trace)


class InternalTool(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(6, 8)
        self.query = nn.Parameter(torch.randn(8))
        self.route = nn.Linear(8, 3)
        self.circuit = Circuit()

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.tokens(tokens)
        attention = torch.softmax(values @ self.query, dim=-1)
        return self.route((attention.unsqueeze(-1) * values).sum(-2))

    def forward(self, tokens: torch.Tensor, operands: torch.Tensor,
                program: torch.Tensor, enabled: torch.Tensor, perturbation: torch.Tensor):
        route = self.logits(tokens).argmax(-1)
        use_tool = (route != 0) & enabled
        chosen = torch.where(route == 2, operands.flip((0,)), operands)
        if bool(use_tool):
            exact, trace = self.circuit(program, chosen)
            return exact + perturbation, route, chosen, trace
        return operands[0], route, chosen, torch.empty((0, 8), dtype=torch.int64)


def freeze():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = InternalTool()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # Train on routing labels only. No program image or arithmetic labels.
    tokens = torch.tensor([[i, filler] for i in range(3) for filler in (3, 4)])
    labels = torch.arange(3).repeat_interleave(2)
    for _ in range(100):
        loss = nn.functional.cross_entropy(model.logits(tokens), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.equal(model.logits(tokens).argmax(-1), labels)
    frozen = torch.jit.freeze(torch.jit.script(model.eval()))
    frozen.save(str(OUT / "internal-tool.pt"))


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--freeze", action="store_true", required=True)
    arg.parse_args()
    freeze()
