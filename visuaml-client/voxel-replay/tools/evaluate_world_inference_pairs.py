"""Paired held-out inference evaluation using the unchanged world graph."""
import argparse
from collections import Counter
import json
from math import sqrt
from random import Random

import torch

from frozen_call_checkpoint import canonical, digest
from pocket_world_inference import OUT as FROZEN, VOCAB, words
from replay_cellular_automata import export_region, scalar
from replay_pocket_world_inference import inputs

OUT = FROZEN.parent / "pocket-world-statistics"
FAMILIES = ("straight","elbow","branched","blocked-ingress")
EXPLORATORY_SEED, HOLDOUT_SEED = 12119, 48151
MODES = ("actual","swapped","irrelevant","precomputed","disabled","missing","short","polite","free")
STOP = {"disabled","missing","short"}


def build_pairs(seed, per_family):
    rng = Random(seed)
    pairs = []
    for family in FAMILIES:
        for number in range(per_family):
            h = 5+number%4
            width,height = rng.randrange(23,29),rng.randrange(19,25)
            gx,y,sign = rng.randrange(8,12),rng.randrange(6,height-5),rng.choice((-1,1))
            distance = {"straight":1,"elbow":2,"branched":3,"blocked-ingress":2}[family]
            tx = gx+h-distance
            walls = [1]*(width*height)
            w = lambda x,y: y*width+x
            for x in range(gx+1,tx+1):
                walls[w(x,y)] = 0
            if family == "straight":
                near,far = (gx-1,y),(gx-2,y)
                for x in (gx-1,gx-2):
                    walls[w(x,y)] = 0
            elif family == "blocked-ingress":
                near = far = (gx-2,y)
                for x,z in ((gx-2,y),(gx-1,y),(gx-2,y+sign),(gx-1,y+sign),(gx,y+sign)):
                    walls[w(x,z)] = 0
            else:
                branch_y = y+sign*(distance-1)
                near,far = (gx-1,branch_y),(gx-2,branch_y)
                for step in range(distance):
                    walls[w(gx-1,y+sign*step)] = 0
                walls[w(*far)] = 0
                if family == "branched":
                    walls[w(gx-1,y-sign)] = walls[w(gx-2,y-sign)] = 0
            # Unseeded pockets beyond a sealed row vary the layout while
            # leaving the intended arrival time independent of these draws.
            for j in range(width*2):
                walls[j] = rng.randrange(2)
            worlds = []
            for bit in (0,1):
                terrain = walls.copy()
                signal = [0]*len(terrain)
                signal[w(*far)] = 1
                if family == "blocked-ingress" and bit == 0:
                    terrain[w(gx-1,y)] = 1
                elif bit:
                    signal[w(*near)] = 1
                worlds.append(dict(width=width,height=height,gate=w(gx,y),target=[tx,y],
                                   walls=terrain,signal=signal))
            assert sum(sum(a!=b for a,b in zip(worlds[0][key],worlds[1][key])) for key in ("walls","signal")) == 1
            pair = dict(id=f"{family}-{number:02}",family=family,horizon=h, worlds=worlds,
                        low="beta" if number%2==0 else "gamma",
                        high="gamma" if number%2==0 else "beta")
            for bit,scene in enumerate(worlds):
                actual_arrival = arrival(scene)
                expect = h if bit else h+2 if family=="blocked-ingress" else h+1
                assert actual_arrival == expect,(pair["id"],bit,actual_arrival,expect)
            pairs.append(pair)
    assert len({digest(canonical(x["worlds"])) for x in pairs}) == len(pairs)
    return pairs


def simulate(scene, action, horizon):
    terrain = scene["walls"].copy()
    if action == 1:
        terrain[scene["gate"]] = 0
    previous, frames = scene["signal"], []
    for _ in range(horizon):
        previous = scalar(terrain,previous,scene["width"],scene["height"])
        frames.append(previous)
    return frames


def arrival(scene):
    x,y = scene["target"]
    target = y*scene["width"]+x
    return next((i+1 for i,frame in enumerate(simulate(scene,1,12)) if frame[target]),None)


def expected(pair, bit, mode):
    reported = 1-bit if mode == "swapped" else bit
    tool = pair["high"] if reported else pair["low"]
    return dict(tool=tool, reported=reported, answer=int(tool=="gamma"))


def first_error(stages):
    return next((stage for stage,correct in stages.items() if not correct),None)


def verification_frames(scene, packet, action, n, observed):
    if observed == []:
        return False
    width,height = packet["width"],packet["height"]
    ox,oy = packet["origin"]
    _,_,gate,target = inputs(packet)
    local = simulate(dict(width=width,height=height,gate=gate,walls=packet["walls"],signal=packet["signal"]),action,n)
    full = simulate(scene,action,n)
    tx,ty = scene["target"]
    w = scene["width"]
    if local != observed:
        return False
    for tick,(regional,whole) in enumerate(zip(local,full),1):
        if regional[target] != whole[ty*w+tx]:
            return False
        for j,value in enumerate(regional):
            x,y = ox+j%width,oy+j//width
            if abs(x-tx)+abs(y-ty) <= n-tick and value != whole[y*w+x]:
                return False
    return True


def run(model,pair,bit,mode,prompt_override=None):
    scene = pair["worlds"][bit]
    horizon,low,high = pair["horizon"],pair["low"],pair["high"]
    prompt = words(1,horizon,low,high,"kindly" if mode=="polite" else "please")
    prediction = expected(pair,bit,mode)
    if mode == "free":
        prompts = (
          f"Could you open the gate and see what happens after {horizon} updates? Use {low} if the target stays dark; otherwise use {high}.",
          f"Imagine opening this passage. At step {horizon}, check the destination. Choose {high} for a lit cell or {low} for a dark cell.",
          f"Try digging the barrier, run the world for {horizon} beats, and call {low} if the signal hasn't reached the target or {high} if it has.",
        )
        prompt = prompts[(int(pair["id"].rsplit("-",1)[-1])+bit)%len(prompts)]
    if prompt_override is not None:
        prompt = prompt_override
    # The actual tokenizer is the frozen whitespace vocabulary codec.
    unknown = sorted(set(prompt.split())-set(VOCAB))
    if unknown:
        return dict(prompt=prompt,prediction=prediction,success=False,decision_correct=False,
                    stages={"input_vocabulary":False},first_wrong=dict(stage="input_vocabulary",event=0,tokens=unknown),
                    requests=[],forecast=[],verified_forecasts=0,execution_mismatches=0)
    if mode == "irrelevant":
        walls = scene["walls"].copy()
        walls[0] ^= 1
        scene = dict(scene,walls=walls)
    width = scene["width"]
    tx,ty = scene["target"]
    halo = horizon-1 if mode == "short" else horizon
    packet = export_region(scene,[tx,ty],halo,halo)
    walls,signal,gate,target = inputs(packet)
    if mode == "missing":
        walls,signal,gate,target = torch.ones((1,1,1,1)),torch.zeros((1,1,1,1)),-1,-1
    requests, feedback, frames, outcome = model(torch.tensor([VOCAB.index(w) for w in prompt.split()]),
        walls,signal,gate,target,mode!="missing",halo,mode=="disabled",
        prediction["reported"] if mode=="swapped" else -1,bit if mode=="precomputed" else -1)
    tokens = [[VOCAB[t] for t in row if t>=0] for row in requests.tolist()]
    protocol = feedback.tolist()
    stages = dict(first_tool=tokens[0][:2]==["call","alpha"],
                  first_arguments=tokens[0][2:]==["1",str(horizon),"."])
    expected_status = [17,-1,-1,0] if mode in STOP else [15,bit,prediction["reported"],0 if mode=="precomputed" else horizon]
    executed,mismatches,emitted = 0,0,[]
    for j,(status,raw,_,steps) in enumerate(protocol):
        history = frames[j,:steps].flatten(1).int().tolist() if steps > 0 else []
        emitted.append(history)
        if steps > 0:
            executed += 1
            action = int(tokens[0][2]) if j==0 else 0
            consistent = verification_frames(scene,packet,action,steps,history)
            consistent = consistent and raw == history[-1][target]
            mismatches += not consistent
    stages["exact_execution"] = mismatches==0
    if mode in STOP:
        stages["failure_handling"] = protocol[0]==expected_status and tokens[1]==["stop","."] and not tokens[2] and outcome.tolist()==[1,-1]
        decision = False
    else:
        stages["first_feedback"] = protocol[0]==expected_status
        stages["followup_choice"] = tokens[1][:2]==["call",prediction["tool"]]
        stages["followup_arguments"] = tokens[1][2:]==[str(prediction["reported"]),"1","."]
        stages["second_feedback"] = protocol[1]==[15,prediction["answer"],prediction["answer"],horizon if prediction["tool"]=="beta" else 0]
        stages["result_use"] = tokens[2]==["answer",str(prediction["answer"]),"."] and outcome.tolist()==[0,prediction["answer"]]
        decision = all(stages[k] for k in ("first_tool","first_arguments","first_feedback","followup_choice"))
    failed = first_error(stages)
    events = dict(first_tool=1,first_arguments=1,exact_execution=2,first_feedback=2,
                  followup_choice=3,followup_arguments=3,second_feedback=4,result_use=5,failure_handling=3)
    return dict(prompt=prompt,prediction=prediction,success=all(stages.values()),decision_correct=decision,
                stages=stages,first_wrong=dict(stage=failed,event=events[failed]) if failed else None,
                requests=tokens,feedback=protocol,outcome=outcome.tolist(),
                forecast=[int(row[target]) for row in emitted[0]],verified_forecasts=executed,
                execution_mismatches=int(mismatches),
                checkpoint_sha256=None if mode=="missing" else digest(canonical(packet)))


def wilson(success, total):
    p,z = success/total,1.959963985
    middle = (p+z*z/(2*total))/(1+z*z/total)
    margin = z*sqrt((p*(1-p)+z*z/(4*total))/total)/(1+z*z/total)
    return [max(0,middle-margin), min(1,middle+margin)]


def report(records):
    summary = {}
    for mode in MODES:
        rows = [x["runs"][mode][str(bit)] for x in records for bit in (0,1)]
        denominator = Counter()
        passed = Counter()
        for row in rows:
            denominator.update(row["stages"].keys())
            passed.update([stage for stage,value in row["stages"].items() if value])
        summary[mode] = dict(assigned=len(rows), success=sum(r["success"] for r in rows),
            decision_correct=sum(r["decision_correct"] for r in rows),
            answered=sum(r.get("outcome",[-1])[0]==0 for r in rows),
            first_failures=dict(Counter(r["first_wrong"]["stage"] for r in rows if r["first_wrong"])),
            stage_counts={stage:[passed[stage],count] for stage,count in denominator.items()},
            verified_forecasts=sum(r["verified_forecasts"] for r in rows),
            execution_mismatches=sum(r["execution_mismatches"] for r in rows))
    def joint(mode):
        return [all(row["decision_correct"] for row in x["runs"][mode].values()) for x in records]
    actual = joint("actual")
    swap = [a and b for a,b in zip(actual,joint("swapped"))]
    def stable(mode):
        return sum(all(x["runs"][mode][str(b)]["decision_correct"] and
            x["runs"][mode][str(b)]["requests"][1] == x["runs"]["actual"][str(b)]["requests"][1]
            for b in (0,1)) for x in records)
    paired = dict(actual=[sum(actual),len(records)],actual_wilson95=wilson(sum(actual),len(records)),
                  crossed_all_four=[sum(swap),len(records)],irrelevant_stable=[stable("irrelevant"),len(records)],
                  controlled_polite_stable=[stable("polite"),len(records)],
                  precomputed_stable=[stable("precomputed"),len(records)])
    return dict(by_mode=summary,paired=paired,
        deterministic_policy=dict(actual_decisions=2*len(records),actual_pairs=len(records),
                                  swapped_decisions=2*len(records),unavailable_stops=6*len(records)),
        paired_task_completion=dict(enabled=sum(all(r["success"] for r in x["runs"]["actual"].values()) for x in records),
            disabled=sum(all(r.get("outcome",[-1])[0]==0 for r in x["runs"]["disabled"].values()) for x in records),
            pairs=len(records)),by_family={
        name:dict(pairs=sum(x["family"]==name for x in records),
                  both_actual=sum(a for x,a in zip(records,actual) if x["family"]==name),
                  all_crossed=sum(a for x,a in zip(records,swap) if x["family"]==name))
        for name in FAMILIES})


def evaluate(model,pairs):
    output = []
    for pair in pairs:
        rows = {mode:{str(bit):run(model,pair,bit,mode) for bit in (0,1)} for mode in MODES}
        output.append(dict(id=pair["id"],family=pair["family"],horizon=pair["horizon"],runs=rows))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore",action="store_true")
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        artifact = FROZEN/"inference.pt"
        model = torch.jit.load(str(artifact))
        seed = EXPLORATORY_SEED if args.explore else HOLDOUT_SEED
        pairs = build_pairs(seed,2 if args.explore else 25)
        data = evaluate(model,pairs)
    outcome = dict(seed=seed, graph_sha256=digest(artifact.read_bytes()),
                   controller_sha256=digest((FROZEN/"controller.pt").read_bytes()),
                   pairs=data, summary=report(data))
    path = OUT/("exploratory.json" if args.explore else "outcomes.json")
    if args.write:
        path.write_bytes(canonical(outcome)+b"\n")
    else:
        assert outcome == json.loads(path.read_text())
    print(json.dumps(outcome["summary"],indent=2))
