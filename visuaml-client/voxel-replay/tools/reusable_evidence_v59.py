"""Exact evidence scopes and a budgeted, three-candidate query policy."""
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from frozen_call_checkpoint import canonical, digest
from pocket_world_inference import OUT as WORLD

OUT = WORLD.parent/"reusable-evidence-v59"
# Status values are unknown, screen inconclusive, rejected, and certified.
UNKNOWN, SCREENED, REJECTED, CERTIFIED = range(4)


@dataclass(frozen=True)
class Scope:
    domain: str
    semantics: str
    subject: str
    assumptions: tuple
    query: str

    def identity(self):
        return digest(canonical([self.domain,self.semantics,self.subject,self.assumptions,self.query]))


class EvidenceStore:
    """Invocation-local trust store. External JSON is an audit log."""
    def __init__(self):
        self.entries = {}

    def query(self, scope, execute):
        key = scope.identity()
        if key in self.entries:
            return self.entries[key], True
        result = execute()
        assert type(result) is bool
        self.entries[key] = result
        return result, False


@dataclass(frozen=True)
class Candidate:
    replay: int
    proof: int
    saving: int
    uses: int
    valid: float
    detectable: float

    @property
    def payoff(self):
        return self.saving*self.uses


def actions(candidates, states, budget):
    for i,(candidate,state) in enumerate(zip(candidates,states,strict=True)):
        if state==UNKNOWN and candidate.replay<=budget:
            yield 1+2*i,i,candidate.replay
        if state in (UNKNOWN,SCREENED) and candidate.proof<=budget:
            yield 2+2*i,i,candidate.proof


def chance(candidate, state, query):
    if query%2:
        return ((candidate.detectable,REJECTED),(1-candidate.detectable,SCREENED))
    p = candidate.valid if state==UNKNOWN else candidate.valid/max(1e-9,1-candidate.detectable)
    return ((p,CERTIFIED),(1-p,REJECTED))


def earned(candidates, states):
    return max((plan.payoff for plan,status in zip(candidates,states,strict=True)
                if status==CERTIFIED),default=0)


def optimal(candidates, states, budget, horizon=6):
    """Exact ex-ante optimum under the declared independent mixture model."""
    @lru_cache(None)
    def value(statuses, remaining, depth):
        best = (float(earned(candidates,statuses)),0)
        if depth:
            for query,i,cost in actions(candidates,statuses,remaining):
                proposed = -cost + sum(prob*value(statuses[:i]+(result,)+statuses[i+1:],
                                                  remaining-cost,depth-1)[0]
                                       for prob,result in chance(candidates[i],statuses[i],query) if prob)
                if proposed>best[0]+1e-8:
                    best = (proposed,query)
        return best
    return value(states,budget,horizon)


def encode(candidates, states, budget):
    features = [float(budget)/150]
    for candidate,state in zip(candidates,states,strict=True):
        features.extend([candidate.replay/150,candidate.proof/150,candidate.payoff/150,
                         candidate.valid,candidate.detectable])
        features.extend(float(state==value) for value in range(4))
        features.extend([float(candidate.replay<=budget),float(candidate.proof<=budget)])
    return features


def train():
    from random import Random
    rng = Random(91303)
    torch.set_num_threads(1)
    torch.manual_seed(91303)
    data, labels = [], []
    for _ in range(9000):
        candidates = []
        for _ in range(3):
            p = rng.choice((.2,.4,.6,.8))
            candidates.append(Candidate(rng.randint(1,9),rng.randint(5,46),
                              rng.randint(1,8),rng.randint(1,14),p,rng.choice((0,.5,1))*(1-p)))
        candidates = tuple(candidates)
        states = (UNKNOWN,)*3 if rng.random()<.5 else tuple(rng.randrange(4) for _ in candidates)
        budget = rng.randint(2,100)
        data.append(encode(candidates,states,budget))
        labels.append(optimal(candidates,states,budget)[1])
    x,y = torch.tensor(data),torch.tensor(labels)
    network = nn.Sequential(nn.Linear(x.shape[1],96),nn.ReLU(),nn.Linear(96,96),
                            nn.ReLU(),nn.Linear(96,7))
    optimizer = torch.optim.AdamW(network.parameters(),lr=.002)
    for _ in range(3800):
        batch = torch.randint(len(y),(128,))
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(network(x[batch]),y[batch])
        loss.backward()
        optimizer.step()
    torch.jit.freeze(torch.jit.script(network.eval())).save(str(OUT/"policy.pt"))


if __name__=="__main__":
    train()
