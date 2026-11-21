"""Structured typed assembly with recurrent learned proposals."""
import json
from random import Random

import torch
from torch import nn

from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import affine
from reusable_evidence_v59 import OUT as V59

OUT = V59.parent/"assembled-loop-v61"
MODULES = ("add","sub","mul","halt")
VERSION = "affine-assembly-loop-v1"
WIDTH = 32


def sample(rng, holdout=False):
    length = rng.randint(4,7) if holdout else rng.randint(1,4)
    start = rng.randrange(4)
    constant = rng.randrange(4)
    rows = []
    for t in range(length):
        op = [2,1,0][t%3] if holdout and t<3 else rng.randrange(3)
        # The goal describes a recipe over x, two constants, and accumulator.
        left = rng.choice((0,3)) if t else 0
        right = rng.choice((1,2))
        if (start+t)%2==0:
            left,right = right,left
        rows.append([op,left,right])
    return dict(constants=[rng.randint(-12,12) for _ in range(2)],recipe=rows,
                argument=rng.randint(-2**31,2**31-1),budget=length+1,seed_feature=constant)


def observation(task, pc, remaining, previous=None):
    tokens = torch.zeros((4,16))
    for i,module in enumerate(MODULES):
        tokens[i,i] = 1
    op,left,right = task["recipe"][pc] if pc<len(task["recipe"]) else (3,0,0)
    tokens[:,4+op] = 1
    tokens[:,8+left] = 1
    tokens[:,12+right] = 1
    tokens[:,0:4] += torch.eye(4) * (remaining/32)
    if previous is not None:
        tokens[3,0:4] += previous
    return tokens


class AttentionPolicy(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.projection = nn.Linear(16,WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.block = nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,WIDTH),nn.GELU())
        self.head = nn.Linear(WIDTH,12)

    def forward(self, tokens, history):
        x = self.projection(tokens)
        if self.recurrent:
            x = x+history.unsqueeze(1)
        selected,_ = self.attention(x,x,x,need_weights=False)
        state = self.block(selected[:,0]+x[:,0])
        return self.head(state),state


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,WIDTH),nn.GELU())
        self.head = nn.Linear(WIDTH,12)

    def forward(self, tokens, history):
        state = self.encoder(tokens.flatten(1))
        return self.head(state),state


def target(task,pc):
    return task["recipe"][pc] if pc<len(task["recipe"]) else [3,0,0]


def train():
    rng = Random(81161)
    torch.set_num_threads(1)
    torch.manual_seed(81161)
    tasks = [sample(rng) for _ in range(2800)]
    for kind,model in (("recurrent",AttentionPolicy(True)),("attention",AttentionPolicy(False)),("mlp",MLPPolicy())):
        opt = torch.optim.AdamW(model.parameters(),lr=.002)
        for _ in range(640):
            ids = torch.randint(len(tasks),(32,))
            batch = [tasks[i] for i in ids]
            opt.zero_grad()
            latent = torch.zeros((32,WIDTH))
            loss = torch.zeros(())
            for pc in range(5):
                states = torch.stack([observation(task,pc,max(0,task["budget"]-pc)) for task in batch])
                actions = torch.tensor([target(task,pc) for task in batch])
                logits,latent = model(states,latent)
                loss = loss+nn.functional.cross_entropy(logits.reshape(-1,4),actions.flatten())
            loss.backward()
            opt.step()
        model.eval()
        traced = torch.jit.trace(model,(torch.zeros(1,4,16),torch.zeros(1,WIDTH)),check_trace=False)
        torch.jit.freeze(traced).save(str(OUT/(kind+".pt")))


def initial(task):
    return dict(pc=0,remaining=task["budget"],latent=[0.0]*WIDTH,
                registers=[[("x",0)],[("const",task["constants"][0])],
                           [("const",task["constants"][1])],[("const",0)]],
                journal=[],status="running")


def expression(task):
    regs = initial(task)["registers"]
    for op,a,b in task["recipe"]:
        regs[3] = [*regs[a],*regs[b],(MODULES[op],0)]
    return regs[3]


def propose(task,state,model):
    if model is None:
        return target(task,state["pc"]),state["latent"]
    reg = affine(state["registers"][3])
    feedback = torch.tensor([reg[0]%5/50,reg[1]%7/50,0.,0.])
    tokens = observation(task,state["pc"],state["remaining"],feedback)
    with torch.inference_mode():
        logits,latent = model(tokens.unsqueeze(0),torch.tensor([state["latent"]]))
    return logits.reshape(3,4).argmax(1).tolist(),latent[0].tolist()


def step(task,state,model):
    if state["status"]!="running":
        return
    if state["remaining"]==0:
        state["status"]="fuel"
        return
    action,latent = propose(task,state,model)
    op,a,b = action
    state["remaining"] -= 1
    state["latent"] = latent
    row = dict(module=MODULES[op],operands=[a,b],remaining=state["remaining"])
    state["journal"].append(row)
    if op==3:
        state["status"]="proposed"
        return
    regs = state["registers"]
    value = [*regs[a],*regs[b],(MODULES[op],0)]
    try:
        row["normal_form"] = affine(value)
    except ValueError:
        state["status"]="invalid-operands"
        return
    regs[3] = value
    state["pc"] += 1


def checkpoint(task,state,model_hash):
    binding = dict(version=VERSION,task=digest(canonical(task)),model=model_hash,state=state)
    return canonical(binding)


def restore(raw,task,model,model_hash):
    data = json.loads(raw)
    state = initial(task)
    completed = len(data["state"]["journal"])
    if completed>task["budget"]:
        raise ValueError("checkpoint exceeds fuel")
    for _ in range(completed):
        step(task,state,model)
    if checkpoint(task,state,model_hash)!=raw:
        raise ValueError("checkpoint state or identity differs from replay")
    return state


if __name__=="__main__":
    train()
