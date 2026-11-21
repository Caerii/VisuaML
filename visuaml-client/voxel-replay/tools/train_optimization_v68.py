"""Freeze two synthetic future-reuse estimators."""
from random import Random

import torch
from torch import nn

from frozen_call_checkpoint import canonical
from optimization_histories_v68 import OUT


def train(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = Random(seed)
    x,y = [],[]
    for _ in range(5200):
        persistence = rng.uniform(.05,.98)
        recent = [float(rng.random()<persistence) for _ in range(4)]
        y.append(float(rng.random()<persistence))
        x.append(recent)
    x,y = torch.tensor(x),torch.tensor(y)
    model = nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(520):
        idx = torch.randint(len(x),(64,))
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx]).flatten(),y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.jit.freeze(torch.jit.trace(model.eval(),x[:1])).save(str(OUT/f"reuse-{seed}.pt"))
    (OUT/f"reuse-{seed}.json").write_bytes(canonical(dict(seed=seed,synthetic_rows=len(x),
        target="next same-layout indicator from four visible edits"))+b"\n")


if __name__=="__main__":
    for seed in (681,682):
        train(seed)
