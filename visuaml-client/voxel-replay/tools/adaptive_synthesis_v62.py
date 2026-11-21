"""Active synthesis of a signed-i32 interval predicate from examples."""
from functools import lru_cache
from random import Random

import torch
from torch import nn

from frozen_call_checkpoint import canonical
from recurrent_evidence_v61 import OUT as PRIOR

OUT = PRIOR.parent/"adaptive-synthesis-v62"
POINTS = 8
WIDTH = 32
PROGRAMS = [()] + [(lo,hi) for lo in range(POINTS) for hi in range(lo+1,POINTS+1)]
HALT = POINTS+len(PROGRAMS)
MASKS = [sum(1<<i for i,p in enumerate(PROGRAMS) if p and p[0]<=x<p[1]) for x in range(POINTS)]
ALL = (1<<len(PROGRAMS))-1


def possible(observations):
    viable = ALL
    for x,result in enumerate(observations):
        if result:
            viable &= MASKS[x] if result==1 else ALL^MASKS[x]
    return viable


@lru_cache(maxsize=100000)
def optimal(mask, costs, remaining):
    """Uniform-prior bounded search maximizing approved programs, then conserving queries."""
    if mask.bit_count()==1:
        return 1.,0.,POINTS+(mask.bit_length()-1)
    if not mask or remaining<=0:
        return 0.,0.,HALT
    best = (0.,0.,HALT)
    for x,partition in enumerate(MASKS):
        yes,no = mask&partition,mask&~partition
        cost = costs[x]
        if yes and no and cost<=remaining:
            left = optimal(yes,costs,remaining-cost)
            right = optimal(no,costs,remaining-cost)
            p = yes.bit_count()/mask.bit_count()
            choice = (p*left[0]+(1-p)*right[0],p*left[1]+(1-p)*right[1]-cost,x)
            if choice[:2]>best[:2]:
                best = choice
    return best


def encode(observed,costs,budget,proof):
    return torch.tensor([[observed[x],costs[x]/4,budget/16,float(proof),
                        x/POINTS,*[float(i==x) for i in range(POINTS)]]
                         for x in range(POINTS)],dtype=torch.float32)


class Policy(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Linear(13,WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(104,64),nn.GELU(),nn.Linear(64,WIDTH))
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH,HALT+1)

    def forward(self,data,state):
        if self.kind=="mlp":
            memory = self.mlp(data.flatten(1))
        else:
            tokens = self.embedding(data)
            memory = state if self.kind=="recurrent" else tokens.mean(1)
            joined = torch.cat((tokens,memory.unsqueeze(1)),1)
            updated,_ = self.attention(joined[:,8:9],joined,joined,need_weights=False)
            memory = self.norm(updated[:,0]+memory)
        return self.head(memory),memory


def train():
    torch.manual_seed(76291)
    torch.set_num_threads(1)
    rng = Random(76291)
    trajectories = []
    for _ in range(4800):
        program = PROGRAMS[rng.randrange(len(PROGRAMS))]
        # Two cost patterns keep bounded teacher construction tractable.
        costs = rng.choice(((1,)*8,(1,2,1,2,1,2,1,2)))
        budget,proof = rng.randint(4,11),rng.random()>.08
        observed = [0]*8
        rows = []
        for _ in range(9):
            choice = optimal(possible(observed),costs,budget-1)[2] if proof else HALT
            rows.append((encode(observed,costs,budget,proof),choice))
            if choice>=POINTS:
                break
            budget -= costs[choice]
            observed[choice] = 1 if program and program[0]<=choice<program[1] else -1
        trajectories.append(rows)
    x = torch.zeros(len(trajectories),9,8,13)
    y = torch.full((len(trajectories),9),-100,dtype=torch.long)
    for i,rows in enumerate(trajectories):
        for j,(features,action) in enumerate(rows):
            x[i,j],y[i,j]=features,action
    for kind in ("recurrent","attention","mlp"):
        model = Policy(kind)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
        for _ in range(1350):
            selected = torch.randint(len(x),(48,))
            memory = torch.zeros(48,WIDTH)
            outputs = []
            for step in range(9):
                logits,memory = model(x[selected,step],memory)
                outputs.append(logits)
            loss = nn.functional.cross_entropy(torch.stack(outputs,1).flatten(0,1),y[selected].flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        traced = torch.jit.trace(model.eval(),(torch.zeros(1,8,13),torch.zeros(1,WIDTH)),check_trace=False)
        torch.jit.freeze(traced).save(str(OUT/(kind+".pt")))
    (OUT/"training.json").write_bytes(canonical(dict(seed=76291,trajectories=4800,
        updates_per_model=1350,cost_patterns=[[1]*8,[1,2]*4],teacher="finite version-space dynamic program"))+b"\n")


if __name__=="__main__":
    train()
