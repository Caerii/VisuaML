"""Two causal attention layers implementing latest committed u32 memory."""
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1] / "evidence/causal-kv-memory"
CONTEXT = 512
# Address 0..3, payload bit 4..5, commit 6, read 7, separator 8.


class CausalMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("positions", torch.arange(512, dtype=torch.float64))
        self.register_buffer("offsets", torch.tensor([33, 1] + list(range(32, 0, -1)), dtype=torch.float64))

    def attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               past: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        start = 0
        if past is not None:
            old_key, old_value = past
            start = old_key.size(1)
            key = torch.cat((old_key, key), dim=1)
            value = torch.cat((old_value, value), dim=1)
        mask = self.positions[:key.size(1)].unsqueeze(0) > self.positions[start:key.size(1)].unsqueeze(1)
        scores = (query @ key.transpose(-2, -1)).masked_fill(mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ value, (key, value)

    def recover(self, value: torch.Tensor):
        return 2 * (torch.relu(value - 0.25) - torch.relu(value - 0.75))

    def forward(self, tokens: torch.Tensor,
                first: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                second: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        start = 0 if first is None else first[0].size(1)
        size = tokens.numel()
        torch._assert(start + size <= 512, "Context exhausted")
        pos = self.positions[start:start+size]
        embedding = torch.nn.functional.one_hot(tokens, 9).double()
        target = pos.unsqueeze(0) - self.offsets.unsqueeze(1)
        # Dot product equals -24*(position-target)^2 up to a query constant.
        q1 = torch.stack((48*target, -24*torch.ones_like(target)), dim=2)
        k1 = torch.stack((pos, pos.square()), dim=1).unsqueeze(0).expand(34, -1, -1)
        v1 = torch.cat((embedding[:, :4], embedding[:, 5:6]), dim=1).unsqueeze(0).expand(34, -1, -1)
        gathered, first_cache = self.attend(q1, k1, v1, first)
        data = self.recover(gathered)
        marker = embedding[:, 6:7]
        address_key = torch.relu(data[0, :, :4] + marker - 1)
        payload = torch.relu(data[2:, :, 4].transpose(0, 1) + marker - 1)
        # A matching commit gains 8192 and the latest token wins ties.
        q2 = torch.cat((8192*data[1, :, :4], torch.ones_like(pos[:, None])), dim=1)[None]
        k2 = torch.cat((address_key, 16*pos[:, None]), dim=1)[None]
        read, second_cache = self.attend(q2, k2, payload[None], second)
        return self.recover(read[0]), first_cache, second_cache


def freeze():
    model = torch.jit.freeze(torch.jit.script(CausalMemory().eval()))
    model.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(str(model.inlined_graph).rstrip()+"\n")


if __name__ == "__main__":
    freeze()
