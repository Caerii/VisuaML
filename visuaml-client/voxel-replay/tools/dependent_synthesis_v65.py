"""Exact tree-factor propagation and stateful query policy for shared modules."""
from dataclasses import dataclass
from functools import lru_cache
from math import log2, prod
from random import Random

import torch
from torch import nn

from factorized_synthesis_v64 import OUT as PRIOR, signed
from frozen_call_checkpoint import canonical

OUT = PRIOR.parent/"dependent-synthesis-v65"
TESTS = ((1,1),(1,2),(2,1),(-1,1),(0,1),(1,0))
FEATURES = 8


@dataclass(frozen=True)
class Space:
    size: int
    library: int
    # Each edge stores allowed ordered coefficient pairs.
    factors: tuple

    @classmethod
    def build(cls,size,library,edges):
        if not 2<=library<=12 or size<2 or len(edges)!=size-1:
            raise ValueError("expected bounded tree")
        edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        if len(set(edges))!=len(edges) or any(not 0<=a<b<size for a,b in edges):
            raise ValueError("invalid factor endpoints")
        seen = {0}
        for _ in range(size):
            seen.update(v for a,b in edges if a in seen or b in seen for v in (a,b))
        if len(seen)!=size:
            raise ValueError("cyclic or disconnected factor layout")
        pairs = tuple((a,b) for a in range(1,library+1) for b in range(1,library+1))
        return cls(size,library,tuple((a,b,pairs) for a,b in edges))

    def observe(self,edge,test,result):
        a,b,pairs = self.factors[edge]
        x,y = TESTS[test]
        kept = tuple((p,q) for p,q in pairs if signed(p*x+q*y)==result)
        return Space(self.size,self.library,self.factors[:edge]+((a,b,kept),)+self.factors[edge+1:])

    def posterior(self):
        links = [[] for _ in range(self.size)]
        allowed = {}
        values = range(1,self.library+1)
        for a,b,pairs in self.factors:
            links[a].append(b)
            links[b].append(a)
            allowed[a,b] = set(pairs)
            allowed[b,a] = {(q,p) for p,q in pairs}

        @lru_cache(None)
        def cavity(node,excluded):
            incoming = [(other,cavity(other,node)) for other in links[node] if other!=excluded]
            return tuple(prod(sum(weight for q,weight in zip(values,branch,strict=True)
                         if (p,q) in allowed[node,other]) for other,branch in incoming)
                         for p in values)

        singles = [cavity(i,-1) for i in range(self.size)]
        edges = []
        for a,b,pairs in self.factors:
            left,right = cavity(a,b),cavity(b,a)
            edges.append(tuple((p,q,left[p-1]*right[q-1]) for p,q in pairs
                               if left[p-1] and right[q-1]))
        return sum(singles[0]),singles,edges


def choices(space,costs,budget,posterior):
    total,_,edges = posterior
    if total<=1:
        return []
    options = []
    for i,pairs in enumerate(edges):
        for j,(x,y) in enumerate(TESTS):
            if costs[i][j]>budget:
                continue
            distribution = {}
            for a,b,count in pairs:
                key = signed(a*x+b*y)
                distribution[key] = distribution.get(key,0)+count
            if len(distribution)<2:
                continue
            probs = [count/total for count in distribution.values()]
            entropy = -sum(p*log2(p) for p in probs)
            options.append((i,j,costs[i][j],[entropy/6,max(probs),len(probs)/24,
                costs[i][j]/12,budget/200,len(pairs)/144,i/max(1,len(edges)-1),j/5]))
    return options


def utility(row):
    return row[3][0]/row[2]


class Policy(nn.Module):
    def __init__(self,history):
        super().__init__()
        self.history = history
        self.encode = nn.Sequential(nn.Linear(FEATURES,32),nn.GELU(),nn.Linear(32,32))
        self.attn = nn.MultiheadAttention(32,4,batch_first=True)
        self.cell = nn.GRUCell(32,32)
        self.score = nn.Linear(32,1)

    def forward(self,x,memory):
        encoded = self.encode(x)
        pooled = encoded.mean(1)
        if self.history:
            summary = self.cell(pooled,memory)
        else:
            summary = pooled
        token = summary.unsqueeze(1)
        context,_ = self.attn(encoded,torch.cat((encoded,token),1),
                              torch.cat((encoded,token),1),need_weights=False)
        return self.score(encoded+context).squeeze(-1),summary


def train(mode,seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = Random(650234)
    episodes = []
    for _ in range(530):
        n,k = rng.randint(3,7),rng.randint(3,6)
        space = Space.build(n,k,[(i,i+1) for i in range(n-1)])
        keys = [rng.randint(1,k) for _ in range(n)]
        costs = [[rng.randint(1,5) for _ in TESTS] for _ in space.factors]
        budget = 24
        data = []
        for _ in range(12):
            valid = choices(space,costs,budget,space.posterior())
            if not valid:
                break
            top = max(valid,key=utility)
            subset = [top]+rng.sample(valid,min(11,len(valid)))
            subset = (subset*12)[:12]
            rng.shuffle(subset)
            edge,test,charge,features = top
            a,b,_ = space.factors[edge]
            x,y = TESTS[test]
            outcome = keys[a]*x+keys[b]*y
            memory = [*features]
            memory[4],memory[7] = -1,outcome/24
            data.append((torch.tensor([row[3] for row in subset]),subset.index(top),memory))
            space = space.observe(edge,test,outcome)
            budget -= charge
        episodes.append(data)
    x = torch.zeros(len(episodes),12,12,FEATURES)
    y = torch.full((len(episodes),12),-100,dtype=torch.long)
    seen = torch.zeros(len(episodes),12,FEATURES)
    for i,episode in enumerate(episodes):
        for j,(features,index,observed) in enumerate(episode):
            x[i,j],y[i,j],seen[i,j] = features,index,torch.tensor(observed)
    model = Policy(mode in ("full","truncated"))
    opt = torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(480):
        batch = torch.randint(len(y),(24,))
        memory = torch.zeros(24,32)
        logits = []
        for t in range(12):
            current = x[batch,t]
            if mode=="full-history":
                current = torch.cat((current,seen[batch,:t]),1)
            output,memory = model(current,memory)
            output = output[:,:12]
            if mode=="truncated" and t%3==2:
                memory = memory.detach()
            logits.append(output)
        loss = nn.functional.cross_entropy(torch.stack(logits,1).flatten(0,1),y[batch].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    torch.jit.freeze(torch.jit.trace(model,(torch.zeros(1,12,FEATURES),torch.zeros(1,32)),
                                    check_trace=False)).save(str(OUT/f"{mode}-{seed}.pt"))
    (OUT/f"{mode}-{seed}.json").write_bytes(canonical(dict(mode=mode,seed=seed,episodes=len(y),
        objective="teacher entropy/credit, full ordered teacher trajectories",updates=480))+b"\n")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("full","truncated","stateless","full-history"))
    parser.add_argument("seed",type=int)
    args = parser.parse_args()
    train(args.mode,args.seed)
