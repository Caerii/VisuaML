"""Audit private forecasts, feedback interventions and learned tool switches."""
import argparse
import json
from time import perf_counter_ns

import torch

from frozen_call_checkpoint import canonical,digest
from pocket_world_inference import OUT,VOCAB,words
from replay_cellular_automata import export_region,initial,scalar,tensor

PROMPT = words(1,6,"beta","gamma","please")
STEPS = 6


def author():
    near = initial()
    delayed = dict(near,signal=[value if index%near["width"] != 8 else 0
                                for index,value in enumerate(near["signal"])])
    return dict(prompt=PROMPT, worlds={name:dict(full=scene,checkpoint=export_region(scene,[14,8],6,6),
                expected=expected) for name,scene,expected in (
                    ("near",near,dict(value=1,tool="gamma",answer=1)),
                    ("delayed",delayed,dict(value=0,tool="beta",answer=0)))},
                predictions=dict(near_substitute=dict(value=0,tool="beta",answer=0),
                                 delayed_substitute=dict(value=1,tool="gamma",answer=1)))


def inputs(packet):
    width,height = packet["width"],packet["height"]
    ox,oy = packet["origin"]
    def offset(location):
        x,y = location
        if not (ox <= x < ox+width and oy <= y < oy+height):
            return -1
        return (y-oy)*width+x-ox
    return tensor(packet["walls"],height,width), tensor(packet["signal"],height,width), (
        offset(packet["gate"])), offset(packet["target"])


def oracle(scene, action):
    width,height,gate = scene["width"],scene["height"],scene["gate"]
    walls = scene["walls"].copy()
    if action:
        walls[gate] = 0
    out = []
    state = scene["signal"]
    for _ in range(STEPS):
        state = scalar(walls,state,width,height)
        out.append(state)
    return out


def region_oracle(opened, packet, width, action):
    _,_,gate,target = inputs(packet)
    local = oracle(dict(width=packet["width"],height=packet["height"],gate=gate,
                        walls=packet["walls"],signal=packet["signal"]),action)
    ox,oy = packet["origin"]
    tx,ty = packet["target"]
    for tick,(regional,full) in enumerate(zip(local,opened),1):
        assert regional[target] == full[ty*width+tx]
        for i,value in enumerate(regional):
            x,y = ox+i%packet["width"],oy+i//packet["width"]
            if abs(x-tx)+abs(y-ty) <= STEPS-tick:
                assert value == full[y*width+x]
    return local


def reconstruct(name, model=None):
    """Rebuild the selected tool frames from its saved checkpoint and request."""
    fixture = json.loads((OUT/"holdout.json").read_text())["worlds"][name]
    archive = json.loads((OUT/"replay.json").read_text())
    path = OUT/"inference.pt"
    assert archive["inference_sha256"] == digest(path.read_bytes())
    recorded = archive["cases"][name]
    request = recorded["requests"][0]
    action,horizon = map(int, request[2:4])
    packet = fixture["checkpoint"]
    assert digest(canonical(packet)) == recorded["checkpoint_sha256"]
    model = model or torch.jit.load(str(path))
    walls,signal,gate,target = inputs(packet)
    result,frames,_ = model.forecast(walls,signal,gate,target,action,horizon)
    history = frames.flatten(1).int().tolist()
    assert result == recorded["computed"] and history == recorded["world_frames"][0]
    return dict(label="reconstructed", scene=name, request=request, world_frames=history,
                checkpoint=packet, result=result, world_effects_delivered=0)


def verify(simulation, spec):
    prompt = torch.tensor([VOCAB.index(word) for word in spec["prompt"].split()])
    journal, timings = {}, {}
    for name, fixture in spec["worlds"].items():
        packet,scene,expected = (fixture[field] for field in ("checkpoint","full","expected"))
        opened,closed = (oracle(scene,a) for a in (1,0))
        source_target = packet["target"][1]*scene["width"]+packet["target"][0]
        assert opened[-1][source_target] == expected["value"]
        region_opened = region_oracle(opened,packet,scene["width"],1)
        region_closed = region_oracle(closed,packet,scene["width"],0)
        first_local = next((dict(tick=i+1,cell=[packet["origin"][0]+j%packet["width"],
                                              packet["origin"][1]+j//packet["width"]])
            for i,(a,b) in enumerate(zip(region_opened,region_closed)) for j,(left,right) in enumerate(zip(a,b))
            if left!=right),None)
        first_difference = next((dict(tick=i+1,cell=[j%scene["width"],j//scene["width"]])
            for i,(a,b) in enumerate(zip(opened,closed)) for j,(left,right) in enumerate(zip(a,b))
            if left!=right),None)
        modes = ("normal","substitute","precomputed","disabled","missing","short_halo")
        for mode in modes:
            label = name if mode == "normal" else f"{name}_{mode}"
            applied = export_region(scene,[14,8],5,5) if mode == "short_halo" else packet
            w,s,g,t = inputs(applied)
            coverage = applied["horizon"]
            if mode == "missing":
                w,s,g,t = torch.ones((1,1,1,1)),torch.zeros((1,1,1,1)),-1,-1
            desired = spec["predictions"][name+"_substitute"] if mode == "substitute" else expected
            options = (prompt,w,s,g,t,mode != "missing",coverage,
                       mode == "disabled",desired["value"] if mode == "substitute" else -1,
                       expected["value"] if mode == "precomputed" else -1)
            requests,feedback,frames,result = simulation(*options)
            warm = []
            for _ in range(8):
                began = perf_counter_ns()
                simulation(*options)
                warm.append(perf_counter_ns()-began)
            tokens = [[VOCAB[i] for i in row if i>=0] for row in requests.tolist()]
            statuses = feedback.tolist()
            frame_lists = [frames[j,:n].flatten(1).int().tolist() if n > 0 else []
                           for j,(_,_,_,n) in enumerate(statuses)]
            stopped = mode in ("disabled","missing","short_halo")
            validity = dict(
                first_tool=tokens[0][0:2] == ["call","alpha"],
                first_arguments=tokens[0][2:5] == ["1","6","."],
                world_execution=(statuses[0] == [17,-1,-1,0] and not frame_lists[0]) if stopped
                    else statuses[0] == [15,expected["value"],desired["value"],0 if mode=="precomputed" else 6]
                    and frame_lists[0] == ([] if mode == "precomputed" else region_opened),
                followup_tool=(tokens[1] == ["stop","."] and not tokens[2]) if stopped
                    else tokens[1][0:2] == ["call",desired["tool"]],
                followup_arguments=(statuses[1] == [-1]*4) if stopped else
                    tokens[1][2:5] == [str(desired["value"]),"1","."],
                result_use=result.tolist() == ([1,-1] if stopped else [0,desired["answer"]]))
            if not stopped:
                expected_second = (closed[-1][source_target] if desired["tool"] == "beta" else
                                   int(not any(scene["walls"][scene["gate"]+1:source_target+1])))
                assert expected_second == desired["answer"]
                steps = 6 if desired["tool"] == "beta" else 0
                validity["second_execution"] = (statuses[1] == [15,expected_second,expected_second,steps]
                        and frame_lists[1] == (region_closed if steps else []))
            assert all(validity.values()), (label,validity,tokens,statuses,result.tolist())
            journal[label] = dict(scene=name,mode=mode,requests=tokens,feedback=statuses,
                                  computed=statuses[0][1],reported=statuses[0][2],world_frames=frame_lists,
                                  outcome=result.tolist(), checkpoint_sha256=None if mode == "missing" else digest(canonical(applied)),
                                  first_affected=first_local if frame_lists[0] else None,
                                  full_world_first_affected=first_difference if frame_lists[0] else None,
                                  checks=validity)
            timings[label] = warm
        integrated, pre = journal[name], journal[name+"_precomputed"]
        assert integrated["requests"] == pre["requests"] and integrated["outcome"] == pre["outcome"]
    assert journal["near"]["requests"][0] == journal["delayed"]["requests"][0]
    return journal,timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-holdout",action="store_true")
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--reconstruct",choices=("near","delayed"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.author_holdout:
        with (OUT/"holdout.json").open("xb") as handle:
            handle.write(canonical(author())+b"\n")
        raise SystemExit
    with torch.inference_mode():
        if args.reconstruct:
            print(json.dumps(reconstruct(args.reconstruct)))
            raise SystemExit
        path = OUT/"inference.pt"
        spec = json.loads((OUT/"holdout.json").read_text())
        cases,timing = verify(torch.jit.load(str(path)),spec)
    replay = dict(profile="visuaml-pocket-world-inference-v1", inference_sha256=digest(path.read_bytes()),
                  prompt=spec["prompt"], cases=cases)
    if args.write:
        (OUT/"replay.json").write_bytes(canonical(replay)+b"\n")
        (OUT/"timings.json").write_bytes(canonical(timing)+b"\n")
    else:
        assert replay == json.loads((OUT/"replay.json").read_text())
    print("Verified learned switch, exact full-world forecasts, interventions and unavailable controls.")
