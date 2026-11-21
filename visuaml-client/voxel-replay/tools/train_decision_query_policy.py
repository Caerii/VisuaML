"""Train a proposal network with balanced supervision on exact stopping states."""
import json
from itertools import combinations, product

import torch
from torch import nn

from checked_decisions import CheckedInference, Controller, certify
from composed_inference import walk
from frozen_call_checkpoint import digest
from pocket_world_inference import OUT as WORLD

OUT = WORLD.parent/"decision-query-policy"


def examples():
    states = {}

    def add(lower, upper, queried, budget, failure, label):
        features = tuple(value/failure for value in lower+upper)+tuple(queried)+(budget,)
        previous = states.setdefault(features,label)
        assert previous == label

    for min_steps, edits in product(range(1,9),(False,True)):
        floor = [min_steps]+[min_steps+9*int(edits)]*2
        for budget in (1,2):
            for used in range(budget+1):
                for actions in combinations(range(3),used):
                    values = [list(range(floor[i],9+9*int(edits and i!=0)))+[32] for i in actions]
                    for observed in product(*values):
                        lower,upper = floor.copy(),[32]*3
                        queries = [0]*3
                        for action,value in zip(actions,observed):
                            lower[action]=upper[action]=value
                            queries[action]=1
                        answer = certify(torch.tensor(lower),torch.tensor(upper),32)
                        unqueried = [i for i in range(3) if not queries[i]]
                        label = 3+answer if answer>=0 else unqueried[0] if used<budget else 7
                        add(lower,upper,queries,budget-used,32,label)
    for budget in (1,2):
        # With equal query costs, universal equivalence dominates a replay:
        # it can both admit the candidate and disprove it in one call.
        add([2,1,0],[2,4,4],[0,0,0],budget,4,1)
        for cost,label in ((1,4),(4,3)):
            add([2,cost,0],[2,cost,4],[0,1,0],budget-1,4,label)
    return states


def train():
    torch.set_num_threads(1)
    torch.manual_seed(19071)
    states = examples()
    tokens = torch.tensor(list(states))
    labels = torch.tensor(list(states.values()))
    controller = Controller()
    optimizer = torch.optim.AdamW(controller.parameters(),lr=.003)
    # Give rare certificate and query classes the same chance to appear.
    weights = torch.bincount(labels).clamp_min(1).float().reciprocal()[labels]
    for _ in range(6000):
        batch = torch.multinomial(weights,128,replacement=True)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(controller.network(tokens[batch]),labels[batch])
        loss.backward()
        optimizer.step()
    simulator = WORLD/"inference.pt"
    graph = torch.jit.freeze(torch.jit.script(
        CheckedInference(controller,torch.jit.load(str(simulator))).eval()))
    assert not {"prim::PythonOp","prim::CallMethod"} & {node.kind() for node in walk(graph.inlined_graph)}
    graph.save(str(OUT/"inference.pt"))
    accuracy = float((controller.network(tokens).argmax(1)==labels).float().mean())
    (OUT/"freeze.json").write_text(json.dumps(dict(seed=19071, updates=6000, training_states=len(labels),
        training_accuracy=accuracy, simulator_sha256=digest(simulator.read_bytes()),
        graph_sha256=digest((OUT/"inference.pt").read_bytes()),
        training_source="synthetic interval states only"),indent=2)+"\n")


if __name__=="__main__":
    train()
