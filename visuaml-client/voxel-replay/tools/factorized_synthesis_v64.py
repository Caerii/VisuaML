"""Compositional version spaces and frozen query guidance for lane programs."""
from dataclasses import dataclass
from functools import lru_cache
from math import log2
from random import Random

import torch
from torch import nn

from constrained_synthesis_v63 import OUT as V63
from frozen_call_checkpoint import canonical

OUT = V63.parent/"compositional-synthesis-v64"
MOD = 1<<32
POINTS = (-2147483648,-17,-2,-1,0,1,4,11,2147483647)
# Component ABI is (slope, bias, signed-negative input adjustment).
LIBRARY = tuple((a,b,g) for g in (0,5,13) for a in (1,2,3,7) for b in (0,3,19))[:32]


def signed(n):
    return (n+(MOD//2))%MOD-MOD//2


def component(module, value):
    a,b,g = module
    return signed(a*value+b+(g if value<0 else 0))


@lru_cache(maxsize=4096)
def bucket(choices,point):
    grouped = {}
    for index in choices:
        output = component(LIBRARY[index],POINTS[point])
        grouped[output] = (*grouped.get(output,()),index)
    return grouped


@dataclass(frozen=True)
class Product:
    """A product of canonical interned leaf domains; lanes do not alias."""
    leaves: tuple

    @classmethod
    def initial(cls,depth,size):
        if not 1<=depth<=32 or not 2<=size<=len(LIBRARY):
            raise ValueError("unsupported lane count or library size")
        return cls((tuple(range(size)),)*depth)

    @property
    def count(self):
        from math import prod
        return prod(map(len,self.leaves))

    def refine(self,lane,point,reply):
        if not 0<=lane<len(self.leaves) or not 0<=point<len(POINTS):
            raise ValueError("undeclared probe operand")
        subset = bucket(self.leaves[lane],point).get(reply)
        if not subset:
            raise ValueError("observation contradicts grammar")
        return Product(self.leaves[:lane]+(subset,)+self.leaves[lane+1:])

    def solution(self):
        if any(len(domain)!=1 for domain in self.leaves):
            raise ValueError("program space is ambiguous")
        return tuple(domain[0] for domain in self.leaves)


def queries(space,costs,budget):
    for lane,choices in enumerate(space.leaves):
        if len(choices)<=1:
            continue
        for point in range(len(POINTS)):
            parts = bucket(choices,point)
            charge = costs[lane][point]
            if len(parts)<=1 or charge>budget:
                continue
            split = [len(group)/len(choices) for group in parts.values()]
            entropy = -sum(p*log2(p) for p in split)
            features = [entropy/5, max(split), min(split), len(parts)/32,
                        len(choices)/32, charge/12, budget/120, point/8]
            yield lane,point,charge,features


class Ranker(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind = kind
        self.encode = nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,32))
        self.attn = nn.MultiheadAttention(32,4,batch_first=True)
        self.out = nn.Linear(32,1)

    def forward(self,data,memory):
        features = self.encode(data)
        if self.kind=="recurrent":
            features = features+memory.unsqueeze(1)
            features,_ = self.attn(features,features,features,need_weights=False)
        return self.out(features).squeeze(-1),features.mean(1)


def exact_value(row):
    return row[3][0]/row[2]


def train(kind,seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = Random(8421)
    examples,labels = [],[]
    for _ in range(720):
        size = rng.randint(8,18)
        state = Product.initial(rng.randint(2,5),size)
        costs = [[rng.randint(1,6) for _ in POINTS] for _ in state.leaves]
        keys = [rng.randrange(size) for _ in state.leaves]
        for step in range(4):
            candidates = list(queries(state,costs,40-step*4))
            if len(candidates)<12:
                break
            subset = rng.sample(candidates,12)
            examples.append([item[3] for item in subset])
            labels.append(max(range(12),key=lambda i:exact_value(subset[i])))
            lane,point,_,_ = max(candidates,key=exact_value)
            state = state.refine(lane,point,component(LIBRARY[keys[lane]],POINTS[point]))
    x,y = torch.tensor(examples),torch.tensor(labels)
    model = Ranker(kind)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(700):
        idx = torch.randint(len(x),(32,))
        optimizer.zero_grad()
        logits,_ = model(x[idx],torch.zeros(32,32))
        loss = nn.functional.cross_entropy(logits,y[idx])
        loss.backward()
        optimizer.step()
    frozen = torch.jit.freeze(torch.jit.trace(model.eval(),(torch.zeros(1,12,8),torch.zeros(1,32)),
                                            check_trace=False))
    frozen.save(str(OUT/f"{kind}-{seed}.pt"))
    (OUT/f"{kind}-{seed}.json").write_bytes(canonical(dict(seed=seed,kind=kind,rows=len(y),
        train_lanes=[2,5],train_library=[8,18],steps=700,
        objective="entropy-per-cost ranking; one-step supervision"))+b"\n")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("kind",choices=("mlp","recurrent"))
    parser.add_argument("seed",type=int)
    args = parser.parse_args()
    train(args.kind,args.seed)
