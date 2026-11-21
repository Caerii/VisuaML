"""Goal-conditioned language controller over the frozen Pocket forecast."""
import json

import torch
from torch import nn
from torch.nn import functional as F

from composed_inference import walk
from pocket_world_inference import OUT as WORLD, VOCAB as PRIOR
import two_call_controller as base

OUT = WORLD.parent / "pocket-goal-planning"
VOCAB = PRIOR + ["reach","target","by","prefer","avoid","edits","if","possible"]
# Reuse the existing "if" token rather than introducing two encodings for it.
VOCAB = list(dict.fromkeys(VOCAB))


def goal(horizon, polite):
    return f"{polite} reach target by {horizon} ticks . prefer avoid edits if possible . assistant"


class Planner(nn.Module):
    def __init__(self, controller, world):
        super().__init__()
        self.controller = controller
        self.world = world

    def forward(self, prompt: torch.Tensor, walls: torch.Tensor, signal: torch.Tensor,
                gate: int, target: int, coverage: int, enabled: bool = True,
                precomputed: torch.Tensor | None = None, intervene_action: int = -1,
                replacement: int = -1):
        prefix = prompt
        requests = torch.full((3,6),-1,dtype=torch.long)
        feedback = torch.full((2,4),-1,dtype=torch.long)
        frames = torch.zeros((2,8,walls.size(2),walls.size(3)))
        outcome,answer = 2,-1
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
            if length == 3 and kind == 16 and 22 <= int(row[1]) <= 24 and int(row[2]) == 7:
                outcome,answer = 0,int(row[1])-22
                break
            if turn == 2 or length != 5 or kind != 14 or int(row[1]) != 2 or int(row[4]) != 7:
                break
            action,horizon = int(row[2])-22,int(row[3])-22
            status,raw,reported,steps = 18,-1,-1,0
            if 0 <= action <= 1 and 1 <= horizon <= 8:
                # The graph does not choose call order, stop conditions, or action.
                if not enabled or coverage < horizon or not (0 <= gate < walls.numel() and
                                                               0 <= target < walls.numel()):
                    status = 17
                else:
                    if precomputed is None:
                        raw,history,_ = self.world.forecast(walls,signal,gate,target,action,horizon)
                        frames[turn,:horizon] = history[:,0]
                        steps = horizon
                    else:
                        raw = int(precomputed[action,horizon])
                    reported = replacement if action == intervene_action else raw
                    status = 15 if 0 <= reported <= 1 else 18
            feedback[turn] = torch.tensor([status,raw,reported,steps])
            prefix = torch.cat((prefix,torch.tensor([status,reported+22,13]) if status == 15
                                else torch.tensor([status,13])))
        return requests,feedback,frames,torch.tensor([outcome,answer])


def train():
    torch.set_num_threads(1)
    torch.manual_seed(972)
    base.VOCAB[:] = VOCAB
    base.IDS.update({word:i for i,word in enumerate(VOCAB)})
    rows = []
    for deadline in range(1,9):
        for polite in ("please","kindly"):
            if (deadline,polite) == (6,"kindly"):
                continue
            for bits in ((0,0),(0,1),(1,1),None):
                # Wait-first is the majority of demonstration traces, but
                # Open-first traces also teach the controller to check Wait.
                for first in (0,0,0,1):
                    steps = [(goal(deadline,polite),False)]
                    tested = {}
                    for action in (first,1-first):
                        steps.append((f"call alpha {action} {deadline} .",True))
                        if bits is None:
                            steps.extend([("unavailable assistant",False),("stop .",True)])
                            break
                        bit = bits[action]
                        tested[action] = bit
                        steps.append((f"result {bit} assistant",False))
                        if tested.get(0) == 1 or len(tested)==2:
                            answer = 0 if tested[0] else 1 if tested.get(1) else 2
                            steps.append((f"answer {answer} .",True))
                            break
                    tokens, labels = [], []
                    for text,learn in steps:
                        ids = base.encode(text)
                        tokens.extend(ids)
                        labels.extend(ids if learn else [-100]*len(ids))
                    rows.append((tokens,labels))
    width = max(len(token) for token,_ in rows)-1
    prefixes = torch.zeros((len(rows),width),dtype=torch.long)
    labels = torch.full_like(prefixes,-100)
    for i,(ids,ys) in enumerate(rows):
        prefixes[i,:len(ids)-1] = torch.tensor(ids[:-1])
        labels[i,:len(ids)-1] = torch.tensor(ys[1:])
    model = base.Controller()
    opt = torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(650):
        selection = torch.randint(len(rows),(32,))
        opt.zero_grad()
        loss = F.cross_entropy(model(prefixes[selection]).flatten(0,1),labels[selection].flatten())
        loss.backward()
        opt.step()
    lang_path = OUT/"controller.pt"
    torch.jit.freeze(torch.jit.trace(model.eval(),(prefixes[:1],),check_trace=False)).save(str(lang_path))
    graph = torch.jit.freeze(torch.jit.script(Planner(torch.jit.load(str(lang_path)),
                                    torch.jit.load(str(WORLD/"inference.pt"))).eval()))
    assert not {"prim::PythonOp","prim::CallMethod"} & {node.kind() for node in walk(graph.inlined_graph)}
    graph.save(str(OUT/"inference.pt"))
    (OUT/"inference.ir").write_text(str(graph.inlined_graph).rstrip()+"\n")
    (OUT/"training.json").write_text(json.dumps(dict(seed=972,updates=650,transcripts=len(rows),
        vocabulary=VOCAB,training_worlds=0,scope="synthetic search demonstrations from Wait/Open outcome bits"),
        indent=2)+"\n")


if __name__ == "__main__":
    train()
