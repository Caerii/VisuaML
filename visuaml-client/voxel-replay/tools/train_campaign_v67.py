"""Freeze small reuse-aware proposal rankers on synthetic quote tuples."""
import torch
from torch import nn

from frozen_call_checkpoint import canonical
from incremental_optimization_v67 import OUT


def train(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # Features are anticipated repeated savings, cold proof cost, reused
    # obligations, deployment estimate, and specialization fraction.
    x = torch.rand(4500,5)
    y = x[:,0]-x[:,1]*(1-x[:,2])-x[:,3]*.3
    model = nn.Sequential(nn.Linear(5,24),nn.GELU(),nn.Linear(24,24),nn.GELU(),nn.Linear(24,1))
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(650):
        sample = torch.randint(len(x),(64,))
        loss = nn.functional.mse_loss(model(x[sample]).flatten(),y[sample])
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.jit.freeze(torch.jit.trace(model.eval(),x[:4])).save(str(OUT/f"planner-{seed}.pt"))
    (OUT/f"planner-{seed}.json").write_bytes(canonical(dict(seed=seed,synthetic_rows=len(x),
        updates=650,target="quoted savings minus non-reused proof and deployment costs"))+b"\n")


if __name__=="__main__":
    for seed in (671,672):
        train(seed)
