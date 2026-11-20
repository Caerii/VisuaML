"""Compile bounded greedy decoding and exact tool dispatch into one TorchScript graph."""
import argparse
from pathlib import Path

import torch
from torch import nn

from synthesize_attention_vm import OUT as EXECUTOR, SynthesizedVM
from two_call_controller import OUT as REFERENCE

OUT = REFERENCE.parent / "composed-inference"


class Inference(nn.Module):
    """Integer ABI. Tool images are prevalidated and indexed by token ID minus two."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.executor = SynthesizedVM()

    def forward(self, prompt: torch.Tensor, programs: torch.Tensor, roles: torch.Tensor,
                fuel: torch.Tensor, override: int = -1,
                registry_ids: torch.Tensor | None = None):
        prefix = prompt
        requests = torch.full((3, 6), -1, dtype=torch.long)
        # status, raw output, feedback value, number of states published
        feedback = torch.full((2, 4), -1, dtype=torch.long)
        traces = torch.zeros((2, 48, 8), dtype=torch.long)
        outcome, answer = 2, -1
        for turn in range(3):
            length = 0
            for _ in range(6):
                if prefix.numel() >= 64:
                    break
                token = self.controller(prefix.unsqueeze(0))[0, -1].argmax().reshape(1)
                requests[turn, length] = token[0]
                prefix = torch.cat((prefix, token))
                length += 1
                if int(token[0]) == 7:
                    break
            row = requests[turn]
            kind = int(row[0])
            if length == 2 and kind == 20 and int(row[1]) == 7:
                outcome = 1
                break
            if length == 3 and kind == 16 and 22 <= int(row[1]) < 54 and int(row[2]) == 7:
                outcome, answer = 0, int(row[1]) - 22
                break
            if turn == 2 or length != 5 or kind != 14 or int(row[4]) != 7:
                break
            slot = int(row[1]) - 2
            if int(row[1]) == 54:
                slot = 2
            known = int(row[1]) in (2, 3, 54)
            if registry_ids is not None:
                matches = (registry_ids == row[1]).nonzero()
                known = matches.numel() == 1
                slot = int(matches[0, 0]) if known else -1
            value, count = int(row[2]) - 22, int(row[3]) - 22
            status, raw, reported, steps = 18, -1, -1, 0
            if 0 <= slot < programs.size(0) and known and 0 <= value <= 31 and 1 <= count <= 9:
                if int(roles[slot]) != 0 or count == 1:
                    budget = int(fuel[slot])
                    if budget < 0:
                        status = 17
                    elif budget == 0:
                        status = 19
                    else:
                        operands = torch.stack((row[2] - 22, row[3] - 22))
                        number, trace = self.executor(programs[slot], operands, min(48, budget))
                        steps = trace.size(0)
                        halted = (trace[:, 1] == 0).nonzero()
                        if halted.numel() == 0:
                            status = 19
                        else:
                            steps = int(halted[0, 0]) + 1
                            raw = int(number)
                            reported = override if turn == 0 and override >= 0 else raw
                            status = 15 if 0 <= reported <= 31 else 18
                        traces[turn, :steps] = trace[:steps]
            feedback[turn] = torch.tensor([status, raw, reported, steps])
            if status == 15:
                prefix = torch.cat((prefix, torch.tensor([status, reported + 22, 13])))
            else:
                prefix = torch.cat((prefix, torch.tensor([status, 13])))
        return requests, feedback, traces, torch.tensor([outcome, answer])


def build(controller: Path, output: Path):
    model = Inference(torch.jit.load(str(controller))).eval()
    graph = torch.jit.freeze(torch.jit.script(model))
    kinds = sorted({node.kind() for node in walk(graph.inlined_graph)})
    assert "prim::PythonOp" not in kinds
    assert "prim::CallMethod" not in kinds
    output.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(output))
    output.with_suffix(".ir").write_text(str(graph.inlined_graph) + "\n")


def walk(block):
    for node in block.nodes():
        yield node
        for child in node.blocks():
            yield from walk(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=REFERENCE / "controller.pt")
    parser.add_argument("--output", type=Path, default=OUT / "reference.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    build(args.controller, args.output)
