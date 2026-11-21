"""Fixed synchronous neighborhood lowering and learned local update controls."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[1] / "evidence/pocket-cellular-automata"
RULE = dict(operation="neighborhood_any", offsets=[[0,0],[-1,0],[1,0],[0,-1],[0,1]],
            boundary="sealed", commit="synchronous", wall="forces_zero")


class Neighborhood(nn.Module):
    """NEIGHBORHOOD von Neumann radius one, sealed, synchronous."""
    def __init__(self, declaration):
        super().__init__()
        if any(declaration[key] != RULE[key] for key in ("operation", "boundary", "commit", "wall")):
            raise ValueError("Unsupported neighborhood semantics")
        kernel = torch.zeros((1,1,3,3))
        for dx,dy in declaration["offsets"]:
            if abs(dx)+abs(dy) > 1:
                raise ValueError("Offset outside declared radius")
            kernel[0,0,dy+1,dx+1] = 1
        self.register_buffer("rule", kernel)

    def forward(self, walls, previous):
        occupied = F.conv2d(previous, self.rule, padding=1).clamp(0,1)
        return occupied*(1-walls)


def initial():
    script = """import { passage } from './cellular-automata.mjs';
const s = passage();
console.log(JSON.stringify({...s,walls:[...s.walls],signal:[...s.signal]}));"""
    return json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script],
                                              cwd=OUT.parents[1], text=True))


def scalar(walls, signal, width, height):
    next_ = []
    for i, wall in enumerate(walls):
        x, y = i % width, i // width
        reaches = signal[i]
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            a,b = x+dx,y+dy
            if 0 <= a < width and 0 <= b < height:
                reaches |= signal[b*width+a]
        next_.append(int(not wall and reaches))
    return next_


def tensor(values, h, w):
    return torch.tensor(values, dtype=torch.float32).reshape(1,1,h,w)


def exact(model, scene):
    width, height, gate = (scene[name] for name in ("width","height","gate"))
    histories = {}
    for name, edit in (("dig",True),("omitted",False)):
        walls, state = scene["walls"].copy(), scene["signal"]
        frames = [state]
        if edit:
            walls[gate] = 0
        for _ in range(16):
            value = model(tensor(walls,height,width), tensor(state,height,width))
            state = scalar(walls,state,width,height)
            assert value.flatten().int().tolist() == state
            frames.append(state)
        histories[name] = frames
    open_walls = scene["walls"].copy()
    open_walls[gate] = 0
    walls_batch = torch.cat([tensor(values,height,width) for values in (open_walls, scene["walls"])])
    state_batch = tensor(scene["signal"],height,width).repeat(2,1,1,1)
    for tick in range(1, 17):
        state_batch = model(walls_batch, state_batch)
        assert state_batch.reshape(2,-1).int().tolist() == [histories[key][tick] for key in ("dig","omitted")]
    differences = []
    for index, (left,right) in enumerate(zip(histories["dig"],histories["omitted"])):
        for i,(a,b) in enumerate(zip(left,right)):
            if a != b:
                dist = abs(i%width-gate%width)+abs(i//width-gate//width)
                assert dist < index
                differences.append([index,i])
    assert differences[0] == [1,gate]
    # Retain only the backward cone of (14,8) at tick 6. A rectangular
    # halo suffices and the narrower halo loses the nearest initial signal.
    walls = tensor(scene["walls"],height,width)
    walls.flatten()[gate] = 0
    start = tensor(scene["signal"],height,width)
    target_x,target_y,horizon = 14,8,6
    answers = {}
    for halo in (6,5):
        x0,x1 = max(0,target_x-halo),min(width,target_x+halo+1)
        y0,y1 = max(0,target_y-halo),min(height,target_y+halo+1)
        cropped = start[:,:,y0:y1,x0:x1].clone()
        for _ in range(horizon):
            cropped = model(walls[:,:,y0:y1,x0:x1],cropped)
        answers[str(halo)] = int(cropped[0,0,target_y-y0,target_x-x0])
    assert answers == {"6":1,"5":0}
    assert histories["dig"][horizon][target_y*width+target_x] == 1
    return dict(rule=RULE, scene=scene, frames=histories, first_signal_difference=differences[0],
                halo=dict(target=[target_x,target_y], tick=horizon, full=1, cropped=answers))


def cross(previous):
    windows = F.unfold(previous, 3, padding=1)
    return windows[:,[1,3,4,5,7],:].transpose(1,2)


def propose(model, walls, previous, patch=None):
    features = cross(previous)
    activation = torch.relu(model.input(features))
    if patch is not None:
        activation[:,patch,:] = torch.relu(model.input(torch.zeros(5)))
    proposed = (model.output(activation).transpose(1,2).reshape_as(previous) >= 0).float()
    reached = features.any(-1).reshape_as(previous).float()
    return (1-walls)*torch.maximum(previous, proposed*reached)


class LearnedRule(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Linear(5,width)
        self.output = nn.Linear(width,1)

    def forward(self, features):
        return self.output(torch.relu(self.input(features)))


def train(width, seed):
    torch.manual_seed(seed)
    model = LearnedRule(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    # Random training neighborhoods. Downstream tests use full unfamiliar layouts.
    for _ in range(180):
        inputs = torch.randint(2,(128,5)).float()
        targets = inputs.any(1,keepdim=True).float()
        loss = F.binary_cross_entropy_with_logits(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def learned(exact_model, scene, measure):
    cases = []
    rng = torch.Generator().manual_seed(1401)
    for i in range(24):
        height,width = ((19,31),(31,53),(33,67))[i%3]
        walls = (torch.rand((1,1,height,width),generator=rng) < .28).float()
        signal = (torch.rand((1,1,height,width),generator=rng) < .025).float()*(1-walls)
        cases.append((walls,signal))
    results = {}
    for width in (4,12):
        for seed in (14,81):
            stem = f"learned-{width}-{seed}.pt"
            model = train(width,seed) if measure else torch.jit.load(str(OUT / stem))
            if measure:
                torch.jit.script(model).save(str(OUT / stem))
            record = dict(layouts=len(cases), exact_layouts=0, first_failures=[])
            with torch.inference_mode():
                patterns = ((torch.arange(32)[:,None] // (2**torch.arange(5))) % 2).float()
                raw = model(patterns)[:,0] >= 0
                record["raw_correct_patterns"] = int((raw == patterns.any(dim=1)).sum())
                record["raw_zero_pattern_active"] = bool(raw[0])
                for index,(walls,start) in enumerate(cases):
                    a,b = start,start
                    for step in range(1,13):
                        a,b = exact_model(walls,a),propose(model,walls,b)
                        diff = (a!=b).flatten().nonzero()
                        if len(diff):
                            record["first_failures"].append([index,step,int(diff[0])])
                            break
                    else:
                        record["exact_layouts"] += 1
                h,w,gate = (scene[name] for name in ("height","width","gate"))
                walls = tensor(scene["walls"],h,w)
                walls.flatten()[gate] = 0
                start = tensor(scene["signal"],h,w)
                a,b = propose(model,walls,start),propose(model,walls,start,patch=gate)
                delta = (a!=b).flatten().nonzero().flatten().tolist()
                # Repeat without modifying the bank anywhere else; its spatial
                # effects must remain in the future cone of the patched cell.
                assert all(cell == gate for cell in delta)
                patch_valid = delta == [gate] and int(a.flatten()[gate]) == 1
                divergence = []
                for tick in range(1,9):
                    diff = (a!=b).flatten().nonzero().flatten().tolist()
                    assert all(abs(i//w-gate//w)+abs(i%w-gate%w) < tick for i in diff)
                    divergence.append(len(diff))
                    a,b = propose(model,walls,a),propose(model,walls,b)
                record.update(patch_prediction_met=patch_valid, patch_first_tick_cells=delta,
                              divergence_counts=divergence)
            results[f"{width}-{seed}"] = record
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    scene = initial()
    rule_path = OUT/"neighborhood.pt"
    if args.write:
        torch.jit.freeze(torch.jit.script(Neighborhood(RULE).eval())).save(str(rule_path))
    model = torch.jit.load(str(rule_path))
    with torch.inference_mode():
        output = exact(model,scene)
    output["learned"] = learned(model,scene,args.write)
    output["profile"] = "visuaml-pocket-cellular-v1"
    archive = OUT/"replay.json"
    if args.write:
        archive.write_text(json.dumps(output, separators=(",",":"))+"\n")
    else:
        assert output == json.loads(archive.read_text())
    print("Verified 32 tensor/scalar ticks, spatial crop controls, and learned intervention replays.")
