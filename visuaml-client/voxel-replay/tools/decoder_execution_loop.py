"""A fixed causal decoder for the one-bit, one-register assembly subset."""
from typing import List, Tuple

import torch

from causal_kv_memory import CausalMemory, OUT as MEMORY_OUT

OUT = MEMORY_OUT.parent / "decoder-execution-loop"


class LoopDecoder(CausalMemory):
    def __init__(self):
        super().__init__()
        self.register_buffer("lags", torch.arange(1, 6, dtype=torch.float64))
        self.register_buffer("phases", torch.arange(512) % 6)
        self.register_buffer("pc_increment", torch.eye(16, dtype=torch.float64).roll(1, dims=1))

    def gate(self, value: torch.Tensor, flag: torch.Tensor):
        return torch.relu(value + flag - 1)

    def forward(self, tokens: torch.Tensor, cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        start = 0 if not cache else cache[0][0].size(1)
        count = tokens.numel()
        torch._assert(start + count <= 512, "Context exhausted")
        pos = self.positions[start:start+count]
        phase = self.phases[start:start+count]
        emb = torch.nn.functional.one_hot(tokens, 29).double()
        target = pos[None] - self.lags[:, None]
        q = torch.stack((48*target, -24*torch.ones_like(target)), dim=-1)
        positional = torch.stack((pos, pos.square()), dim=-1)
        gathered, c1 = self.attend(q, positional[None].expand(5, -1, -1),
                                   emb[None].expand(5, -1, -1), cache[0] if cache else None)
        fields = self.recover(gathered)
        pc, reg = fields[4, :, :16], fields[3, :, 22:23]
        recency = 16*pos[:, None]
        bias = torch.ones_like(recency)

        code = emb[:, 25:26]
        key = self.gate(fields[3, :, :16], code)
        payload = self.gate(torch.cat((fields[2, :, 16:21], fields[1, :, :16]), dim=-1), code)
        fetched, c2 = self.attend(torch.cat((8192*pc, bias), dim=-1)[None],
                                   torch.cat((key, recency), dim=-1)[None], payload[None],
                                   cache[1] if cache else None)
        opcode, arg = self.recover(fetched[0])[:, :5], self.recover(fetched[0])[:, 5:]

        commit = emb[:, 26:27]
        key = self.gate(fields[1, :, :4], commit)
        payload = self.gate(fields[0, :, 22:23], commit)
        read, c3 = self.attend(torch.cat((8192*arg[:, :4], bias), dim=-1)[None],
                                 torch.cat((key, recency), dim=-1)[None], payload[None],
                                 cache[2] if cache else None)
        loaded = self.recover(read[0])

        halt, load, store, branch = opcode[:, :1], opcode[:, 1:2], opcode[:, 2:3], opcode[:, 4:5]
        taken = self.gate(reg, branch)
        next_pc = (self.gate(pc @ self.pc_increment, 1-taken-halt)
                   + self.gate(arg, taken) + self.gate(pc, halt))
        next_reg = self.gate(loaded, load) + self.gate(reg, 1-load)
        write = self.gate(reg, store)
        completion = torch.cat((next_pc, next_reg, self.gate(arg[:, :4], store),
                                store, write, halt), dim=-1)
        completion = self.gate(completion, emb[:, 24:25])
        target = pos - phase.double()
        q = torch.stack((48*target, -24*torch.ones_like(target)), dim=-1)[None]
        result, c4 = self.attend(q, positional[None], completion[None], cache[3] if cache else None)
        result = self.recover(result[0])

        # Fixed phase-specific vocabulary projections.
        zeros = torch.zeros_like(emb)
        pc_logits = torch.cat((result[:, :16], zeros[:, 16:]), dim=-1)
        reg_logits = torch.cat((zeros[:, :21], 1-result[:, 16:17], result[:, 16:17], zeros[:, 23:]), dim=-1)
        addr_logits = torch.cat((result[:, 17:21], zeros[:, 4:23], 1-result[:, 21:22], zeros[:, 24:]), dim=-1)
        data_logits = torch.cat((zeros[:, :21], 1-result[:, 22:23], result[:, 22:23], zeros[:, 23:]), dim=-1)
        markers = torch.cat((zeros[:, :26], 1-result[:, 23:24], result[:, 23:24], zeros[:, 28:]), dim=-1)
        begin = zeros + torch.nn.functional.one_hot(torch.tensor(24), 29)
        logits = torch.stack((pc_logits, reg_logits, addr_logits, data_logits, markers, begin), dim=1)
        return logits.gather(1, phase[:, None, None].expand(-1, 1, 29)).squeeze(1), [c1,c2,c3,c4]


def freeze():
    model = torch.jit.freeze(torch.jit.script(LoopDecoder().eval()))
    ir = str(model.inlined_graph)
    assert "prim::PythonOp" not in ir and "prim::CallMethod" not in ir
    model.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(ir.rstrip()+"\n")


if __name__ == "__main__":
    freeze()
