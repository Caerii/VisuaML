"""Cost-aware proposals with independent exact authorization."""

import torch
from torch import nn

from checked_decisions import certify
from composed_inference import walk
from pocket_world_inference import OUT as WORLD

OUT = WORLD.parent/"evidence-economics"


@torch.jit.script
def expected_choice(cheap: int, proof: int, payoff: int, remaining: int,
                    valid: int, rejects: int, inspected: bool) -> int:
    # 0 stop, 1 cheap check, 2 universal/full check. Probabilities use percent.
    if inspected:
        return 2 if proof<=remaining and valid*payoff>(100-rejects)*proof else 0
    direct = valid*payoff-100*proof if proof<=remaining else 0
    staged = valid*payoff-(100-rejects)*proof-100*cheap if cheap+proof<=remaining else 0
    if staged>0 and staged>direct:
        return 1
    return 2 if direct>0 else 0


@torch.jit.script
def features(cheap: int, proof: int, payoff: int, remaining: int,
             valid: int, rejects: int, inspected: bool) -> torch.Tensor:
    size = float(max(1,cheap,proof,payoff,remaining))
    return torch.tensor([cheap/size,proof/size,payoff/size,remaining/size,
                         valid/100.,rejects/100.,float(inspected),
                         float(proof<=remaining),float(cheap+proof<=remaining)])


class Economy(nn.Module):
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.policy = nn.Sequential(nn.Linear(9,64),nn.ReLU(),nn.Linear(64,64),
                                    nn.ReLU(),nn.Linear(64,3))

    def forward(self, budget: int, estimates: torch.Tensor, actual: torch.Tensor,
                payoff: int, valid_prior: int, rejection_prior: int, method: int,
                domain: int, evidence: torch.Tensor, walls: torch.Tensor,
                signal: torch.Tensor, target: int, horizon: int):
        assert budget>=0 and bool((estimates>0).all()) and bool((actual>0).all())
        assert 0<=valid_prior<=100 and 0<=rejection_prior<=100-valid_prior
        # Reference always available. Candidate cheaper only if eligible.
        lower,upper = torch.tensor([2,1]),torch.tensor([2,4])
        journal = torch.full((3,6),-1,dtype=torch.long)
        remaining,inspected = budget,False
        for turn in range(3):
            certificate = certify(lower,upper,4)
            if certificate>=0:
                journal[turn,0],journal[turn,4] = 3+certificate,certificate
                return journal,certificate,budget-remaining
            cheap,proof = int(estimates[0]),int(estimates[1])
            if method==0:
                proposal = int(self.policy(features(cheap,proof,payoff,remaining,
                                                    valid_prior,rejection_prior,inspected)).argmax())
            elif method==1:
                proposal = 2
            elif method==2:
                proposal = 2 if inspected else 1
            elif method==3:
                proposal = expected_choice(cheap,proof,payoff,remaining,
                                            valid_prior,rejection_prior,inspected)
            else:
                proposal = 4  # Test-only unsupported adoption request.
            journal[turn,0],journal[turn,1] = proposal,remaining
            if proposal==0 or proposal>2 or (proposal==1 and inspected):
                break
            charge = int(actual[proposal-1])
            if charge>remaining:
                journal[turn,5] = 1
                break
            remaining -= charge
            journal[turn,2] = charge
            if domain==0:
                # Compiler evidence: validated ordered local-pair selectors
                # plus the values they select at the targeted replay.
                positive = int(evidence[0])==int(evidence[1]) if proposal==2 else int(evidence[2])==int(evidence[3])
            else:
                if proposal==2:
                    result,_,_ = self.simulator.forecast(walls,signal,0,target,0,horizon)
                    positive = bool(result)
                else:
                    # A complete wall slice between the only source and the
                    # target blocks cardinal motion through this checkpoint.
                    width = walls.size(3)
                    start = int(torch.nonzero(signal.flatten())[0,0])
                    axis, cut = int(evidence[0]),int(evidence[1])
                    assert int(torch.count_nonzero(signal))==1
                    assert 0<=axis<=1 and 0<=cut<walls.size(2+axis)
                    left = start//width if axis==0 else start%width
                    right = target//width if axis==0 else target%width
                    line = walls[:,:,cut,:] if axis==0 else walls[:,:,:,cut]
                    positive = not (min(left,right)<cut<max(left,right) and bool((line==1).all()))
            journal[turn,3] = int(positive)
            if not positive:
                lower[1],upper[1] = 4,4
            elif proposal==2:
                lower[1],upper[1] = 1,1
            inspected = True
        return journal,-1,budget-remaining


def train():
    from random import Random
    rng = Random(4083)
    torch.manual_seed(4083)
    torch.set_num_threads(1)
    model = Economy(torch.jit.load(str(WORLD/"inference.pt")))
    rows, labels = [], []
    for _ in range(34000):
        cheap,proof = rng.randint(1,12),rng.randint(2,30)
        payoff,remaining = rng.randint(0,75),rng.randint(0,42)
        valid = rng.randint(10,85)
        rejects = rng.randint(0,100-valid)
        for inspected in (False,True):
            args = cheap,proof,payoff,remaining,valid,rejects,inspected
            rows.append(features(*args))
            labels.append(expected_choice(*args))
    x,y = torch.stack(rows),torch.tensor(labels)
    weights = torch.bincount(y).float().reciprocal()[y]
    optimizer = torch.optim.AdamW(model.policy.parameters(),lr=.003)
    for _ in range(4200):
        batch = torch.multinomial(weights,128,replacement=True)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model.policy(x[batch]),y[batch])
        loss.backward()
        optimizer.step()
    graph = torch.jit.freeze(torch.jit.script(model.eval()))
    assert not {"prim::PythonOp","prim::CallMethod"} & {node.kind() for node in walk(graph.inlined_graph)}
    graph.save(str(OUT/"inference.pt"))


if __name__=="__main__":
    train()
