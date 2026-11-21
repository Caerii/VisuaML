"""Compile wrapping word arithmetic into the cached causal assembly decoder."""
from typing import List, Tuple

import torch
from torch import nn

from causal_kv_memory import CausalMemory, OUT as MEMORY_OUT

OUT = MEMORY_OUT.parent / "word-decoder"
FRAME = 12
COMMIT, HALT, EXEC, CODE, NONE, PAD = range(261, 267)
OPERATIONS = {"add": ("i32.add", 32), "test": ("i32.eqz", 32)}


class WordCircuit(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        if declaration != {"add": ("i32.add", 32), "test": ("i32.eqz", 32)}:
            raise ValueError("Unsupported operation declaration")
        self.distances = [1, 2, 4, 8, 16]

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        generate = torch.relu(left + right - 1)
        xor = left + right - 2*generate
        propagate = xor
        for distance in self.distances:
            generate = torch.cat((generate[:, :distance], generate[:, distance:] +
                                  torch.relu(propagate[:, distance:] + generate[:, :-distance] - 1)), dim=1)
            propagate = torch.cat((propagate[:, :distance],
                                   torch.relu(propagate[:, distance:] + propagate[:, :-distance] - 1)), dim=1)
        carry = torch.cat((torch.zeros_like(generate[:, :1]), generate[:, :-1]), dim=1)
        total = xor + carry - 2*torch.relu(xor + carry - 1)
        zero = torch.relu(1-left.sum(dim=1, keepdim=True))
        return total, zero


class WordDecoder(CausalMemory):
    def __init__(self):
        super().__init__()
        self.alu = WordCircuit(OPERATIONS)
        self.register_buffer("lags", torch.arange(1, 12, dtype=torch.float64))
        self.register_buffer("phases", torch.arange(512) % 12)
        self.register_buffer("pc_increment", torch.eye(16, dtype=torch.float64).roll(1, dims=1))
        self.register_buffer("weights", 2 ** torch.arange(8))
        symbols = torch.arange(267)
        identifiers = torch.nn.functional.one_hot(symbols.clamp(max=16), 17)[:, :16]
        opcodes = torch.nn.functional.one_hot((symbols-256).clamp(0, 5), 6)[:, :5]
        opcodes = opcodes * (symbols >= 256)[:, None]
        bits = ((symbols[:, None] // self.weights) % 2) * (symbols < 256)[:, None]
        self.register_buffer("embedding", torch.cat((identifiers, opcodes, bits), dim=1).double())

    def gate(self, value: torch.Tensor, flag: torch.Tensor):
        return torch.relu(value + flag - 1)

    def forward(self, tokens: torch.Tensor, cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        start = 0 if not cache else cache[0][0].size(1)
        count = tokens.numel()
        torch._assert(start + count <= 512, "Context exhausted")
        pos = self.positions[start:start+count]
        phase = self.phases[start:start+count]
        target = pos[None] - self.lags[:, None]
        q = torch.stack((48*target, -24*torch.ones_like(target)), dim=-1)
        positional = torch.stack((pos, pos.square()), dim=-1)
        gathered, c1 = self.attend(q, positional[None].expand(11, -1, -1),
                                   self.embedding[tokens][None].expand(11, -1, -1), cache[0] if cache else None)
        fields = self.recover(gathered)
        pc = fields[10, :, :16]
        reg = fields[6:10, :, 21:].flip([0]).transpose(0, 1).reshape(count, 32)
        word = fields[:4, :, 21:].flip([0]).transpose(0, 1).reshape(count, 32)
        recency = 16*pos[:, None]
        bias = torch.ones_like(recency)
        arg_key = fields[4, :, :16]
        code = (tokens == 264).double()[:, None]
        key = self.gate(fields[9, :, :16], code)
        instruction = self.gate(torch.cat((fields[8, :, 16:21], arg_key, word), dim=-1), code)
        fetched, c2 = self.attend(torch.cat((8192*pc, bias), dim=-1)[None],
                                   torch.cat((key, recency), dim=-1)[None], instruction[None],
                                   cache[1] if cache else None)
        fetched = self.recover(fetched[0])
        opcode, arg, immediate = fetched[:, :5], fetched[:, 5:21], fetched[:, 21:]
        commit = (tokens == 261).double()[:, None]
        key = self.gate(arg_key[:, :4], commit)
        payload = self.gate(word, commit)
        read, c3 = self.attend(torch.cat((8192*arg[:, :4], bias), dim=-1)[None],
                                 torch.cat((key, recency), dim=-1)[None], payload[None],
                                 cache[2] if cache else None)
        loaded = self.recover(read[0])
        added, zero = self.alu(reg, immediate)
        halt, load, store, add, branch = opcode.split(1, dim=1)
        taken = self.gate(1-zero, branch)
        next_pc = (self.gate(pc @ self.pc_increment, 1-taken-halt)
                   + self.gate(arg, taken) + self.gate(pc, halt))
        next_reg = self.gate(loaded, load) + self.gate(added, add) + self.gate(reg, 1-load-add)
        completion = torch.cat((next_pc, next_reg, self.gate(arg[:, :4], store),
                                store, self.gate(reg, store), halt), dim=-1)
        completion = self.gate(completion, (tokens == 263).double()[:, None])
        target = pos - phase.double()
        q = torch.stack((48*target, -24*torch.ones_like(target)), dim=-1)[None]
        result, c4 = self.attend(q, positional[None], completion[None], cache[3] if cache else None)
        result = self.recover(result[0])
        reg_bytes = (result[:, 16:48].reshape(-1, 4, 8).long() * self.weights).sum(-1)
        store_bytes = (result[:, 53:85].reshape(-1, 4, 8).long() * self.weights).sum(-1)
        address = torch.where(result[:, 52] == 1, result[:, 48:52].argmax(dim=1), 265)
        options = torch.cat((result[:, :16].argmax(dim=1)[:, None], reg_bytes, address[:, None],
                             store_bytes, (261+result[:, 85].long())[:, None],
                             torch.full((count, 1), 263, dtype=torch.long)), dim=1)
        chosen = options.gather(1, phase[:, None]).squeeze(1)
        return torch.nn.functional.one_hot(chosen, 267).double(), [c1, c2, c3, c4]


def freeze():
    model = torch.jit.freeze(torch.jit.script(WordDecoder().eval()))
    ir = str(model.inlined_graph)
    assert "prim::PythonOp" not in ir and "prim::CallMethod" not in ir
    model.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(ir.rstrip()+"\n")


if __name__ == "__main__":
    freeze()
