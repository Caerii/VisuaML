"""One frozen graph for language decoding, private world queries and feedback."""
import json

import torch
from torch import nn
from torch.nn import functional as F

from composed_inference import OUT as COMPOSED, walk
from replay_cellular_automata import Neighborhood, RULE
import two_call_controller as base

OUT = COMPOSED.parent / "pocket-world-inference"
VOCAB = base.VOCAB + ["gamma","below","otherwise","choose","if","open","wait","gate","for","ticks"]


class WorldInference(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.rule = Neighborhood(RULE)

    def forecast(self, walls: torch.Tensor, signal: torch.Tensor, gate: int, target: int,
                 action: int, horizon: int):
        terrain = walls.clone()
        if action == 1:
            terrain.flatten()[gate] = 0
        frames = []
        for _ in range(horizon):
            signal = self.rule(terrain,signal)
            frames.append(signal)
        return int(signal.flatten()[target]), torch.cat(frames), terrain

    def forward(self, prompt: torch.Tensor, walls: torch.Tensor, signal: torch.Tensor,
                gate: int, target: int, provided: bool, coverage: int,
                disabled: bool = False, substitute: int = -1, precomputed: int = -1):
        prefix = prompt
        requests = torch.full((3,6),-1,dtype=torch.long)
        feedback = torch.full((2,4),-1,dtype=torch.long)
        frames = torch.zeros((2,8,walls.size(2),walls.size(3)))
        observed_action,observed_horizon = -1,0
        outcome,answer = 2,-1
        size = walls.numel()
        for turn in range(3):
            length = 0
            for _ in range(6):
                if prefix.numel() >= 64:
                    break
                token = self.controller(prefix.unsqueeze(0))[0,-1].argmax().reshape(1)
                requests[turn,length] = token[0]
                prefix = torch.cat((prefix,token))
                length += 1
                if int(token[0]) == 7:
                    break
            row = requests[turn]
            kind = int(row[0])
            if length == 2 and kind == 20 and int(row[1]) == 7:
                outcome = 1
                break
            if length == 3 and kind == 16 and 22 <= int(row[1]) <= 23 and int(row[2]) == 7:
                outcome,answer = 0,int(row[1])-22
                break
            if turn == 2 or length != 5 or kind != 14 or int(row[4]) != 7:
                break
            tool,left,right = int(row[1]),int(row[2])-22,int(row[3])-22
            status,raw,reported,steps = 18,-1,-1,0
            if turn == 0 and tool == 2 and 0 <= left <= 1 and 1 <= right <= 8:
                if not provided or disabled or right > coverage or not (0 <= gate < size and 0 <= target < size):
                    status = 17
                else:
                    observed_action,observed_horizon = left,right
                    if precomputed >= 0:
                        raw = precomputed
                    else:
                        raw, history, _ = self.forecast(walls,signal,gate,target,left,right)
                        steps = right
                        frames[turn,:steps] = history[:,0]
                    reported = substitute if substitute >= 0 else raw
                    status = 15 if 0 <= reported <= 1 else 18
            elif turn == 1 and tool in (3,54) and 0 <= left <= 1 and right == 1 and observed_horizon > 0:
                if tool == 3:
                    # Compare with leaving the gate closed at the same horizon.
                    raw, history, _ = self.forecast(walls,signal,gate,target,0,observed_horizon)
                    steps = observed_horizon
                    frames[turn,:steps] = history[:,0]
                else:
                    # Check the straight relay corridor after the proposed edit.
                    terrain = walls.clone()
                    if observed_action == 1:
                        terrain.flatten()[gate] = 0
                    width = terrain.size(3)
                    raw = int(gate//width == target//width and gate <= target and
                              bool((terrain.flatten()[gate:target+1] == 0).all()))
                reported = raw
                status = 15
            feedback[turn] = torch.tensor([status,raw,reported,steps])
            if status == 15:
                prefix = torch.cat((prefix,torch.tensor([status,reported+22,13])))
            else:
                prefix = torch.cat((prefix,torch.tensor([status,13])))
        return requests,feedback,frames,torch.tensor([outcome,answer])


def words(action, horizon, low, high, polite):
    return (f"{polite} {'open' if action else 'wait'} gate for {horizon} ticks "
            f"then choose {low} if below 1 otherwise {high} . assistant")


def train():
    torch.set_num_threads(1)
    torch.manual_seed(286)
    base.VOCAB[:] = VOCAB
    base.IDS.update({word:i for i,word in enumerate(VOCAB)})
    examples = []
    for action in (0,1):
        for horizon in range(1,9):
            for low,high in (("beta","gamma"),("gamma","beta")):
                for polite in ("please","kindly"):
                    # Reserve the final evaluation wording as an unseen combination.
                    if (action,horizon,low,polite) == (1,6,"beta","please"):
                        continue
                    for result in (-1,0,1):
                        for answer in (0,1):
                            parts = [(words(action,horizon,low,high,polite),False),
                                     (f"call alpha {action} {horizon} .",True)]
                            if result < 0:
                                parts += [("unavailable assistant",False),("stop .",True)]
                            else:
                                parts += [(f"result {result} assistant",False),
                                          (f"call {low if result == 0 else high} {result} 1 .",True),
                                          (f"result {answer} assistant",False),(f"answer {answer} .",True)]
                            ids,labels = [],[]
                            for text,learn in parts:
                                tokens = base.encode(text)
                                ids.extend(tokens)
                                labels.extend(tokens if learn else [-100]*len(tokens))
                            examples.append((ids,labels))
    length = max(len(ids) for ids,_ in examples)
    inputs = torch.zeros((len(examples),length-1),dtype=torch.long)
    outputs = torch.full_like(inputs,-100)
    for i,(ids,labels) in enumerate(examples):
        inputs[i,:len(ids)-1] = torch.tensor(ids[:-1])
        outputs[i,:len(ids)-1] = torch.tensor(labels[1:])
    model = base.Controller()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(700):
        rows = torch.randint(len(examples),(32,))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs[rows]).flatten(0,1),outputs[rows].flatten())
        loss.backward()
        optimizer.step()
    controller = torch.jit.freeze(torch.jit.trace(model.eval(),(inputs[:1],),check_trace=False))
    controller.save(str(OUT/"controller.pt"))
    freeze(torch.jit.load(str(OUT/"controller.pt")))
    (OUT/"training.json").write_text(json.dumps(dict(vocabulary=VOCAB,seed=286,updates=700,
        examples=len(examples),scope="synthetic text and bit replies; no world layouts"),indent=2)+"\n")


def freeze(controller):
    graph = torch.jit.freeze(torch.jit.script(WorldInference(controller).eval()), preserved_attrs=["forecast"])
    assert not {"prim::PythonOp","prim::CallMethod"} & {node.kind() for node in walk(graph.inlined_graph)}
    graph.save(str(OUT/"inference.pt"))
    (OUT/"inference.ir").write_text(str(graph.inlined_graph).rstrip()+"\n")



if __name__ == "__main__":
    train()
