"""Add an attention readout for checkpoint tokens to the frozen word decoder."""
from typing import List, Tuple

import torch
from torch import nn

from word_decoder import OUT as WORD

OUT = WORD.parent / "decoder-emitted-checkpoint"
CHECKPOINT_TOKENS = 60


class CheckpointDecoder(nn.Module):
    def __init__(self, word_decoder):
        super().__init__()
        self.word_decoder = word_decoder
        self.register_buffer("byte_weights", 2**torch.arange(8))
        self.register_buffer("word_addresses", torch.eye(4, dtype=torch.float64))
        self.register_buffer("state_offsets", torch.arange(10, 5, -1, dtype=torch.float64))

    def forward(self, tokens: torch.Tensor, cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        return self.word_decoder(tokens, cache)

    def bits(self, weighted: torch.Tensor):
        return 2*(torch.relu(weighted-0.25)-torch.relu(weighted-0.75))

    @torch.jit.export
    def checkpoint(self, cache: List[Tuple[torch.Tensor, torch.Tensor]], boundary_token: int):
        token_keys, token_values = cache[0]
        length = token_keys.size(1)
        torch._assert(boundary_token == 261 and length % 12 == 0, "Checkpoint needs a live COMMIT")
        torch._assert(length + 60 <= 512, "Checkpoint phase exceeds context")
        # Four binary64 memory queries, using the existing address and recency scores.
        memory_keys, memory_values = cache[2]
        queries = torch.cat((8192*self.word_addresses, torch.ones((4, 1), dtype=torch.float64)), dim=1)
        memory_bits = self.bits(torch.softmax(queries @ memory_keys[0].T, dim=1) @ memory_values[0])
        words = (memory_bits.reshape(4, 4, 8).long()*self.byte_weights).sum(-1)

        # Read PC and four register bytes from the last committed frame.
        target = (length-1) - self.state_offsets
        position_query = torch.stack((48*target, -24*torch.ones_like(target)), dim=1)
        state = self.bits(torch.softmax(position_query @ token_keys[0].T, dim=1) @ token_values[0])
        pc = state[0, :16].argmax().reshape(1)
        register = (state[1:, 21:].long()*self.byte_weights).sum(-1)
        reserved = torch.tensor([[266, 0, 0, 0, 0, 0]]).expand(4, -1)
        memory = torch.cat((reserved, torch.arange(4)[:, None], words, torch.full((4, 1), 261)), dim=1)
        live = torch.cat((torch.tensor([266]), pc, register, torch.tensor([265, 0, 0, 0, 0, 261])))
        return torch.cat((memory.flatten(), live))


def freeze():
    word = torch.jit.load(str(WORD / "decoder.pt"))
    module = torch.jit.freeze(torch.jit.script(CheckpointDecoder(word).eval()),
                              preserved_attrs=["checkpoint"])
    ir = str(module.inlined_graph) + "\n" + str(module.checkpoint.graph)
    assert "prim::PythonOp" not in ir and "prim::CallMethod" not in ir
    module.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(ir.rstrip()+"\n")


if __name__ == "__main__":
    freeze()
