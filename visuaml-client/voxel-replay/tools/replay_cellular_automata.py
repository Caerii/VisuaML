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
    target, horizon = [14,8], 6
    portable = {}
    full_target = [frame[target[1]*width+target[0]] for frame in histories["dig"][:horizon+1]]
    for halo in (6,5):
        packet = export_region(scene, target, halo, horizon)
        # Serialization round trip drops any reference to the full grid.
        restored = json.loads(json.dumps(packet))
        dig = resume_region(model, restored, True)
        omitted = resume_region(model, restored, False)
        first = next((i for i,(a,b) in enumerate(zip(full_target,dig)) if a != b), None)
        portable[str(halo)] = dict(checkpoint=packet, target_with_edit=dig, target_without_edit=omitted,
                                   first_different_tick=first)
    assert portable["6"]["first_different_tick"] is None
    assert portable["5"]["first_different_tick"] == 6
    assert portable["6"]["target_without_edit"] == [0]*7
    # An explicit shortest witness from excluded pre-tick signal to the target.
    path = [[x, target[1]] for x in range(target[0]-horizon,target[0]+1)]
    assert scene["signal"][path[0][1]*width+path[0][0]] == 1
    assert path[0][0] < portable["5"]["checkpoint"]["origin"][0]
    assert all(histories["dig"][i][y*width+x] for i,(x,y) in enumerate(path))
    return dict(rule=RULE, scene=scene, frames=histories, first_signal_difference=differences[0],
                halo=dict(target=target, tick=horizon, full=full_target, cropped=portable,
                          dependency=[dict(cell=cell,tick=i) for i,cell in enumerate(path)]))


def export_region(scene, target, halo, horizon):
    width, height = scene["width"], scene["height"]
    x,y = target
    x0,x1 = max(0,x-halo),min(width,x+halo+1)
    y0,y1 = max(0,y-halo),min(height,y+halo+1)
    indices = [row*width+col for row in range(y0,y1) for col in range(x0,x1)]
    return dict(format="signal-relay-region-v1", rule=RULE["operation"], tick=0, horizon=horizon,
                origin=[x0,y0], width=x1-x0, height=y1-y0, target=target,
                gate=[scene["gate"]%width,scene["gate"]//width],
                walls=[scene["walls"][i] for i in indices], signal=[scene["signal"][i] for i in indices])


def resume_region(model, packet, dig):
    if packet["format"] != "signal-relay-region-v1" or packet["rule"] != RULE["operation"]:
        raise ValueError("Unsupported spatial checkpoint")
    width,height = packet["width"],packet["height"]
    origin_x,origin_y = packet["origin"]
    target_x,target_y = packet["target"]
    gate_x,gate_y = packet["gate"]
    walls = tensor(packet["walls"],height,width)
    state = tensor(packet["signal"],height,width)
    if dig:
        walls[0,0,gate_y-origin_y,gate_x-origin_x] = 0
    answers = [int(state[0,0,target_y-origin_y,target_x-origin_x])]
    for _ in range(packet["horizon"]):
        state = model(walls,state)
        answers.append(int(state[0,0,target_y-origin_y,target_x-origin_x]))
    return answers


def cross(previous):
    windows = F.unfold(previous, 3, padding=1)
    return windows[:,[1,3,4,5,7],:].transpose(1,2)


def propose(model, walls, previous, patch=None):
    features = cross(previous)
    activation = torch.relu(model.input(features))
    if patch is not None:
        cell, replacement = patch
        activation[:,cell,:] = replacement
    raw = (model.output(activation).transpose(1,2).reshape_as(previous) >= 0).float()
    reached = features.any(-1).reshape_as(previous).float()
    guarded = (1-walls)*torch.maximum(previous, raw*reached)
    return raw, guarded


def decisions(raw, guarded):
    return dict(accepted_on=int(((raw == 1) & (guarded == 1)).sum()),
                accepted_off=int(((raw == 0) & (guarded == 0)).sum()),
                changed_to_on=int(((raw == 0) & (guarded == 1)).sum()),
                rejected_on=int(((raw == 1) & (guarded == 0)).sum()))


def patches(model, walls, initial, gate, width):
    features = cross(initial)
    encode = lambda sample: torch.relu(model.input(sample))
    current = encode(features[0,gate])
    donor = encode(torch.zeros(5))
    same_features = features[0,gate].clone()
    same_features[(same_features == 0).nonzero()[0]] = 1
    same_side = encode(same_features)
    base_raw, base = propose(model,walls,initial)
    rng = torch.Generator().manual_seed(727)
    directions = torch.randn((16, len(current)),generator=rng)
    norm = (donor-current).norm()
    random = current + directions/directions.norm(dim=1,keepdim=True)*norm
    variants = {"empty":donor, "same_side":same_side}
    variants.update({f"random_{i}":value for i,value in enumerate(random)})
    target = 8*width+14
    summary = {}
    for kind, bank in variants.items():
        raw, changed = propose(model,walls,initial,(gate,bank))
        raw_delta = (raw!=base_raw).flatten().nonzero().flatten().tolist()
        cell_delta = (changed!=base).flatten().nonzero().flatten().tolist()
        assert all(i == gate for i in raw_delta + cell_delta)
        a,b = base,changed
        first_target = None
        counts = []
        target_values = []
        for tick in range(1,9):
            diff = (a!=b).flatten().nonzero().flatten().tolist()
            assert all(abs(i//width-gate//width)+abs(i%width-gate%width) < tick for i in diff)
            if target in diff and first_target is None:
                first_target = tick
            counts.append(len(diff))
            target_values.append([int(a.flatten()[target]),int(b.flatten()[target])])
            a,b = propose(model,walls,a)[1],propose(model,walls,b)[1]
        summary[kind] = dict(raw_changed_cells=raw_delta, committed_changed_cells=cell_delta,
                             gate_raw=[int(base_raw.flatten()[gate]),int(raw.flatten()[gate])],
                             gate_committed=[int(base.flatten()[gate]),int(changed.flatten()[gate])],
                             first_target_difference=first_target, target_values=target_values,
                             divergence_counts=counts)
        assert first_target == (6 if cell_delta else None)
    return summary


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
                totals = dict(accepted_on=0, accepted_off=0, changed_to_on=0, rejected_on=0)
                for index,(walls,start) in enumerate(cases):
                    a,b = start,start
                    for step in range(1,13):
                        a = exact_model(walls,a)
                        raw,b = propose(model,walls,b)
                        for key,count in decisions(raw,b).items():
                            totals[key] += count
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
                record["guard_actions"] = totals
                record["patches"] = patches(model,walls,start,gate,w)
                closed = walls.clone()
                closed.flatten()[gate] = 1
                donor = torch.relu(model.input(torch.zeros(5)))
                raw, committed = propose(model,closed,start)
                patched_raw, patched_commit = propose(model,closed,start,(gate,donor))
                assert torch.equal(committed, patched_commit)
                record["closed_gate_control"] = dict(
                    raw=[int(raw.flatten()[gate]), int(patched_raw.flatten()[gate])],
                    committed=[int(committed.flatten()[gate]), int(patched_commit.flatten()[gate])])
            results[f"{width}-{seed}"] = record
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--region", type=Path)
    parser.add_argument("--omit-edit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rule_path = OUT/"neighborhood.pt"
    if args.write:
        torch.jit.freeze(torch.jit.script(Neighborhood(RULE).eval())).save(str(rule_path))
    model = torch.jit.load(str(rule_path))
    if args.region:
        with torch.inference_mode():
            packet = json.loads(args.region.read_text())
            print(json.dumps(resume_region(model, packet, not args.omit_edit)))
        raise SystemExit(0)
    scene = initial()
    with torch.inference_mode():
        output = exact(model,scene)
    output["learned"] = learned(model,scene,args.write)
    output["profile"] = "visuaml-pocket-cellular-v1"
    archive = OUT/"replay.json"
    if args.write:
        archive.write_text(json.dumps(output, separators=(",",":"))+"\n")
        for halo, record in output["halo"]["cropped"].items():
            (OUT / f"region-halo-{halo}.json").write_text(
                json.dumps(record["checkpoint"], separators=(",",":"))+"\n")
    else:
        assert output == json.loads(archive.read_text())
    print("Verified 32 tensor/scalar ticks, spatial crop controls, and learned intervention replays.")
