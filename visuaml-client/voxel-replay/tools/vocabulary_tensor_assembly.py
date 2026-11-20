"""Lower structured causal assembly to an autoregressive byte vocabulary."""
import argparse
import hashlib
from pathlib import Path

import torch
from torch import nn

from causal_tensor_assembly import OUT as STRUCTURED

OUT = Path(__file__).resolve().parents[1] / "evidence/vocabulary-tensor-assembly"
FRAME = 21
CONTEXT = 64 * FRAME
COMMIT, HALT = 256, 257


class VocabularyDecoder(nn.Module):
    def __init__(self, transition):
        super().__init__()
        self.transition = transition
        weights = 2 ** torch.arange(8, dtype=torch.long)
        self.register_buffer("weights", weights)
        self.register_buffer("embedding", ((torch.arange(256)[:, None] // weights) % 2).double())

    def committed(self, tokens: torch.Tensor):
        frames = tokens[:tokens.numel() // 21 * 21].reshape(-1, 21)
        markers = frames[:, -1]
        torch._assert(bool(((markers == 256) | (markers == 257)).all()), "Missing commit marker")
        return self.embedding[frames[:, :20]].reshape(-1, 160)

    def forward(self, tokens: torch.Tensor):
        torch._assert(21 <= tokens.numel() < 1344, "Token context exhausted")
        phase = tokens.numel() % 21
        record = self.transition(self.committed(tokens))
        if phase == 20:
            token = 257 - record[-1].long()
        else:
            token = (record.reshape(20, 8)[phase].long() * self.weights).sum()
        return torch.nn.functional.one_hot(token, 258).to(torch.float64)


def encode_image(image):
    weights = 2 ** torch.arange(8, dtype=torch.long)
    bytes_ = (image.reshape(-1, 20, 8).long() * weights).sum(-1)
    return torch.cat((bytes_, torch.full((len(image), 1), COMMIT)), dim=1).flatten()


def generate(model, initial, token_budget):
    if not 0 <= token_budget <= CONTEXT - len(initial):
        raise ValueError("Token budget exceeds context")
    prefix = initial
    for _ in range(token_budget):
        token = model(prefix).argmax().reshape(1)
        prefix = torch.cat((prefix, token))
        if int(token[0]) == HALT:
            break
    return prefix[len(initial):]


def freeze():
    transition = torch.jit.load(str(STRUCTURED / "decoder.pt"))
    model = torch.jit.freeze(torch.jit.script(VocabularyDecoder(transition).eval()),
                             preserved_attrs=["committed"])
    ir = str(model.inlined_graph)
    assert "prim::PythonOp" not in ir and "prim::CallMethod" not in ir
    path = OUT / "decoder.pt"
    model.save(str(path))
    path.with_suffix(".ir").write_text(ir.rstrip() + "\n")
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    torch.set_num_threads(1)
    freeze()
