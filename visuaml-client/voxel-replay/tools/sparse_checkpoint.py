"""Frozen decoder with default-zero memory and an ordered sparse checkpoint."""
from typing import List, Tuple

import torch
from torch import nn

from word_decoder import OUT as WORD, WordDecoder

OUT = WORD.parent / "sparse-checkpoint"


class SparseCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = WordDecoder(sparse=True)
        self.register_buffer("addresses", torch.eye(16, dtype=torch.float64))
        self.register_buffer("offsets", torch.arange(10, 5, -1, dtype=torch.float64))
        self.register_buffer("weights", 2**torch.arange(8))

    def forward(self, tokens: torch.Tensor, cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        return self.decoder(tokens, cache)

    def recover(self, weighted: torch.Tensor):
        return 2*(torch.relu(weighted-0.25)-torch.relu(weighted-0.75))

    @torch.jit.export
    def scan(self, cache: List[Tuple[torch.Tensor, torch.Tensor]], address_range: int):
        torch._assert(1 <= address_range <= 16, "Address range must be 1 through 16")
        keys, values = cache[2]
        bias = torch.ones((address_range, 1), dtype=torch.float64)
        queries = torch.cat((16384*self.addresses[:address_range], 8192*bias, bias), dim=1)
        lanes = self.recover(torch.softmax(queries @ keys[0].T, dim=1) @ values[0])
        return (lanes.reshape(address_range, 4, 8).long()*self.weights).sum(-1)

    @torch.jit.export
    def emit(self, cache: List[Tuple[torch.Tensor, torch.Tensor]], boundary: int,
             words: torch.Tensor, capacity: int):
        keys, values = cache[0]
        length = keys.size(1)
        torch._assert(boundary == 261 and length % 12 == 0, "Live COMMIT required")
        torch._assert(0 <= capacity <= 16 and length+12*(capacity+1) <= 512, "Unreserved checkpoint")
        retained = words.any(dim=1).nonzero().flatten()
        torch._assert(len(retained) <= capacity, "Sparse checkpoint overflow")
        preamble = torch.tensor([[266,0,0,0,0,0]]).expand(len(retained), -1)
        memory = torch.cat((preamble, retained.unsqueeze(1), words[retained],
                            torch.full((len(retained), 1), 261)), dim=1)
        target = length-1-self.offsets
        query = torch.stack((48*target, -24*torch.ones_like(target)), dim=1)
        lanes = self.recover(torch.softmax(query @ keys[0].T, dim=1) @ values[0])
        pc = lanes[0, :16].argmax().reshape(1)
        register = (lanes[1:, 21:].long()*self.weights).sum(-1)
        state = torch.cat((torch.tensor([266]), pc, register, torch.tensor([265,0,0,0,0,261])))
        return torch.cat((memory.flatten(), state))

    @torch.jit.export
    def checkpoint(self, cache: List[Tuple[torch.Tensor, torch.Tensor]], boundary: int,
                   address_range: int, capacity: int):
        return self.emit(cache, boundary, self.scan(cache, address_range), capacity)


def freeze():
    model = torch.jit.freeze(torch.jit.script(SparseCheckpoint().eval()),
                            preserved_attrs=["checkpoint", "scan", "emit"])
    ir = "\n".join((str(model.inlined_graph), str(model.checkpoint.graph)))
    assert "prim::PythonOp" not in ir
    model.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(ir.rstrip()+"\n")


if __name__ == "__main__":
    freeze()
