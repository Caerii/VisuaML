"""Exact version space and typed action interface for bounded synthesis."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import log2
from random import Random

import torch
from torch import nn

from adaptive_synthesis_v62 import OUT as V62
from frozen_call_checkpoint import canonical

OUT = V62.parent/"constrained-synthesis-v63"
MAX_POINTS = 14
CANDIDATES = 4
STOP = MAX_POINTS+CANDIDATES
WIDTH = 32
FEATURES = 10


@lru_cache(maxsize=32)
def grammar(width, runs):
    if not 1<=width<=MAX_POINTS or not 1<=runs<=3:
        raise ValueError("unsupported grammar bounds")
    words = [0]
    for k in range(1,runs+1):
        for endpoints in combinations(range(width+1),2*k):
            words.append(sum(((1<<(hi-lo))-1)<<lo
                             for lo,hi in zip(endpoints[::2],endpoints[1::2],strict=True)))
    return tuple(sorted(words,key=lambda x:(segments(x),x)))


def segments(word):
    return (word & ~(word<<1)).bit_count()


@dataclass(frozen=True)
class VersionSpace:
    width: int
    runs: int
    zero: int = 0
    one: int = 0

    def __post_init__(self):
        grammar(self.width,self.runs)
        if (self.zero & self.one or min(self.zero,self.one)<0
                or (self.zero|self.one)>=(1<<self.width)):
            raise ValueError("inconsistent observation masks")

    def observe(self,point,value):
        if not 0<=point<self.width or type(value) is not int or value not in (0,1):
            raise ValueError("invalid observation")
        bit = 1<<point
        return VersionSpace(self.width,self.runs,self.zero|(bit if not value else 0),
                            self.one|(bit if value else 0))

    def contains(self,word):
        return word in grammar(self.width,self.runs) and not word&self.zero and word&self.one==self.one

    @property
    def viable(self):
        return tuple(word for word in grammar(self.width,self.runs)
                     if not word&self.zero and word&self.one==self.one)


@dataclass(frozen=True)
class Action:
    kind: str
    operand: int
    charge: int


def disagreement(a,b):
    bits = a^b
    return (bits&-bits).bit_length()-1


def state(space, costs, proof_cost, remaining, proof):
    viable = space.viable
    count = len(viable)
    options,features = [],[]

    def append(action,kind,gain,balance,legal):
        options.append(action)
        features.append([float(kind==0),float(kind==1),float(kind==2), gain/12,balance,
                         action.charge/12,remaining/24,log2(max(1,count))/12,
                         float(legal),space.width/MAX_POINTS])

    for point in range(MAX_POINTS):
        available = point<space.width and not (space.zero|space.one)&(1<<point)
        charge = costs[point] if point<space.width else 99
        zeros = sum(not (word>>point)&1 for word in viable) if available else 0
        ones = count-zeros if available else 0
        gain = entropy((zeros,ones))
        legal = count>1 and available and min(zeros,ones)>0 and charge<=remaining
        append(Action("probe",point,charge),0,gain,min(zeros,ones)/max(1,count),legal)
    for rank in range(CANDIDATES):
        word = viable[rank] if rank<count else -1
        histogram = [0]*(space.width+1)
        if word>=0:
            for other in viable:
                histogram[disagreement(word,other)+1] += 1
        legal = word>=0 and proof and proof_cost<=remaining
        append(Action("prove",word,proof_cost),1,entropy(histogram),1/max(1,count),legal)
    append(Action("stop",0,0),2,0,0,True)
    return options,torch.tensor(features)


def entropy(counts):
    total = sum(counts)
    return -sum((n/total)*log2(n/total) for n in counts if n)


def teacher(options, tokens):
    legal = [i for i,r in enumerate(tokens) if r[8]]
    if len(legal)==1:
        return STOP
    if tokens[0,7]==0:
        return next((i for i in legal if options[i].kind=="prove"),STOP)
    return max((i for i in legal if i!=STOP),
               key=lambda i:(float(tokens[i,3])/options[i].charge,-i))


class Router(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.project = nn.Linear(FEATURES,WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.score = nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,1))
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self,tokens,memory):
        encoded = self.project(tokens)
        if self.mode!="mlp":
            if self.mode=="recurrent":
                encoded = encoded+memory.unsqueeze(1)
            attended,_ = self.attention(encoded,encoded,encoded,need_weights=False)
            encoded = self.norm(encoded+attended)
        return self.score(encoded).squeeze(-1),encoded.mean(1)


def train(mode,seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = Random(91029)
    episodes = []
    for _ in range(1800):
        width,runs = rng.randint(5,9),rng.choice((1,2))
        space = VersionSpace(width,runs)
        target = rng.choice(space.viable)
        costs = [rng.randint(1,5) for _ in range(width)]
        proof,charge,budget = rng.random()>.1,rng.randint(2,7),rng.randint(5,22)
        rows = []
        for _ in range(8):
            options,tokens = state(space,costs,charge,budget,proof)
            index = teacher(options,tokens)
            rows.append((tokens,index))
            request = options[index]
            if request.kind=="stop" or request.kind=="prove" and request.operand==target:
                break
            point = request.operand if request.kind=="probe" else disagreement(target,request.operand)
            space = space.observe(point,(target>>point)&1)
            budget -= request.charge
        episodes.append(rows)
    x = torch.zeros(len(episodes),8,STOP+1,FEATURES)
    y = torch.full((len(episodes),8),-100,dtype=torch.long)
    for i,episode in enumerate(episodes):
        for t,(tokens,index) in enumerate(episode):
            x[i,t],y[i,t] = tokens,index
    model = Router(mode)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(560):
        batch = torch.randint(len(episodes),(32,))
        memory = torch.zeros(32,WIDTH)
        logits = []
        for t in range(8):
            values,memory = model(x[batch,t],memory)
            logits.append(values)
        loss = nn.functional.cross_entropy(torch.stack(logits,1).flatten(0,1),y[batch].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    frozen = torch.jit.freeze(torch.jit.trace(model.eval(),
             (torch.zeros(1,STOP+1,FEATURES),torch.zeros(1,WIDTH)),check_trace=False))
    frozen.save(str(OUT/f"{mode}-{seed}.pt"))
    (OUT/f"{mode}-{seed}.json").write_bytes(canonical(dict(seed=seed,episodes=len(episodes),
          training_width=[5,9],updates=560,mode=mode,teacher="entropy per cost"))+b"\n")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("mlp","attention","recurrent"))
    parser.add_argument("seed",type=int)
    args = parser.parse_args()
    train(args.mode,args.seed)
