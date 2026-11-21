"""Audit paired goal interventions and budget-limited evidence."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path

import torch

from evaluate_world_inference_pairs import verification_frames
from frozen_call_checkpoint import canonical, digest
from pocket_paired_goals import OUT, WORLD, VOCAB, goal
from replay_cellular_automata import export_region
from replay_pocket_world_inference import inputs


def arrival(world, action, deadline):
    """Independent shortest path on the full terrain after the action."""
    width, height = world["width"], world["height"]
    walls = world["walls"].copy()
    if action:
        walls[world["gate"]] = 0
    start, = [i for i, value in enumerate(world["signal"]) if value]
    x, y = world["target"]
    target = y*width+x
    assert start != target
    queue = deque([(start,0)])
    seen = {start}
    while queue:
        index, steps = queue.popleft()
        if index == target:
            return steps
        if steps == deadline:
            continue
        x,y = index%width,index//width
        for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = ny*width+nx
                if not walls[neighbor] and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor,steps+1))
    return 0


def optimal(objective, times):
    available = [a for a in (0,1) if times[a]]
    return min(available,key=lambda a: (times[a],a) if objective=="speed" else (a,)) if available else 2


def evidence(objective, deadline, known):
    outcomes = set()
    for wait in range(deadline+1):
        for opened in range(deadline+1):
            times = (wait,opened)
            if wait and (not opened or opened > wait):
                continue
            if all(times[a]==value for a,value in known.items()):
                outcomes.add(optimal(objective,times))
    return next(iter(outcomes)) if len(outcomes)==1 else None


def score(objective, deadline, truth, calls, state, pick):
    known = {}
    unnecessary = 0
    for call in calls:
        unnecessary += evidence(objective,deadline,known) is not None or call["action"] in known
        if call["status"]==15:
            known[call["action"]] = call["shown"]
    supported = evidence(objective,deadline,known)
    return dict(state=state,action=pick,oracle=truth,justified_action=supported,calls=calls,
                objective_success=state==0 and pick==truth,
                unsupported=state==0 and pick!=supported,
                missed=int(state!=0 and supported is not None),
                unnecessary=unnecessary,
                appropriate=state==0 and pick==supported or state==1 and supported is None)


def baseline(objective, deadline, times, budget, method):
    order = (0,1) if method!="open_first" else (1,0)
    known, calls = {}, []
    for action in order[:budget]:
        if method!="exhaustive" and evidence(objective,deadline,known) is not None:
            break
        known[action] = times[action]
        calls.append(dict(action=action,shown=times[action],status=15))
    pick = evidence(objective,deadline,known)
    return score(objective,deadline,optimal(objective,times),calls,int(pick is None),pick)


def evaluate(model, world, objective, budget, mode="actual", intervention=(-1,-1)):
    deadline = world["deadline"]
    packet = export_region(world,world["target"],deadline,deadline)
    walls,signal,gate,target = inputs(packet)
    table = [[0]+[arrival(world,a,h) for h in range(1,9)] for a in (0,1)]
    times = [row[deadline] for row in table]
    prompt = goal(deadline,objective,budget)
    request,feedback,frames,answer = model(
        torch.tensor([VOCAB.index(t) for t in prompt.split()]),walls,signal,gate,target,deadline,budget,
        mode!="disabled",torch.tensor(table) if mode=="precomputed" else None,*intervention)
    requests = [[VOCAB[t] for t in row if t>=0] for row in request.tolist()]
    calls, errors = [], []
    for turn,(status,raw,shown,steps) in enumerate(feedback.tolist()):
        if status == -1:
            continue
        tokens = requests[turn]
        action,horizon = int(tokens[2]),int(tokens[3])
        if horizon!=deadline or action not in (0,1):
            errors.append("arguments")
        history = frames[turn,:steps].flatten(1).int().tolist() if steps else []
        target_bits = [frame[target] for frame in history]
        if steps and (not verification_frames(world,packet,action,steps,history)
                      or raw!=table[action][horizon]
                      or raw!=next((i for i,bit in enumerate(target_bits,1) if bit),0)):
            errors.append("exact_execution")
        if status==15 and shown!=(intervention[1] if action==intervention[0] else raw):
            errors.append("feedback")
        calls.append(dict(action=action,horizon=horizon,raw=raw,shown=shown,status=status,
                          exact_steps=steps,target_bits=target_bits))
    state, pick = answer.tolist()
    if any(row[:1]==["call"] for row in requests[len(calls):]):
        errors.append("unexecuted_request")
    result = score(objective,deadline,optimal(objective,times),calls,state,pick if state==0 else None)
    return dict(result,prompt=prompt,requests=requests,errors=errors,
                missed_available_forecast=int(state==1 and result["justified_action"] is None
                    and bool(calls) and all(c["status"]==15 for c in calls)
                    and len({c["action"] for c in calls}) < budget),
                checkpoint_sha256=digest(canonical(packet)))


def worlds():
    from random import Random

    rng = Random(62831)
    for family in ("shortcut","deadline","tie","blocked"):
        for variant in range(24):
            size = 25
            distance = 4+variant%3
            deadline = distance+1 if family=="deadline" else distance+2
            direction = variant%4
            dx,dy = ((1,0),(0,1),(-1,0),(0,-1))[direction]
            ox,oy = rng.randrange(8,17),rng.randrange(8,17)
            side = rng.choice((-1,1))
            def point(x,y):
                return ox+dx*x-side*dy*y,oy+dy*x+side*dx*y
            def offset(x,y):
                px,py = point(x,y)
                return py*size+px
            gate = offset(2,0)
            walls = [1]*(size*size)
            for x in range(distance+1):
                walls[offset(x,0)] = 0
            for x in range(1,4):
                walls[offset(x,1)] = 0
            walls[gate] = int(family!="tie")
            if family=="blocked":
                # Break both routes between the start and target.
                walls[offset(distance,0)] = 1
            signal = [0]*len(walls)
            signal[offset(0,0)] = 1
            for index in range(size):  # isolated random terrain behind a sealed row
                walls[index] = rng.randrange(2)
            world = dict(width=size,height=size,gate=gate,target=list(point(distance,0)),walls=walls,
                         signal=signal,deadline=deadline)
            expected = {"shortcut":[distance+2,distance],"deadline":[0,distance],
                        "tie":[distance,distance],"blocked":[0,0]}[family]
            assert [arrival(world,a,deadline) for a in (0,1)]==expected
            yield f"{family}-{variant:02}",world,expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--replay",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    artifact = OUT/"inference.pt"
    graph_hash = digest(artifact.read_bytes())
    model = torch.jit.load(str(artifact))
    if args.replay:
        bundle = json.loads(args.replay.read_text())
        assert bundle["graph_sha256"]==graph_hash
        with torch.inference_mode():
            for example in bundle["examples"]:
                world = bundle["worlds"][example["id"]]
                assert evaluate(model,world,example["objective"],example["budget"],example["mode"],
                                tuple(example["intervention"]))==example["run"]
        print(f"Verified {len(bundle['examples'])} saved runs")
        return

    rows, examples, example_worlds = [], [], {}
    with torch.inference_mode():
        for name,world,times in worlds():
            deadline = world["deadline"]
            distant_walls = world["walls"].copy()
            distant_walls[0] ^= 1
            distant = dict(world,walls=distant_walls)
            for objective in ("edits","speed"):
                for budget in (1,2):
                    for method in ("actual","precomputed","disabled","irrelevant","wait_first","open_first","exhaustive","intervention"):
                        if method=="intervention" and (not name.startswith("shortcut") or budget!=2):
                            continue
                        # Predeclared predictions for the shortcut: edit objective rechecks Open
                        # after seeing Wait fail; speed objective stops after seeing Open fail.
                        intervention = (int(objective=="speed"),0) if method=="intervention" else (-1,-1)
                        predicted = 1 if objective=="edits" else 2
                        run = (baseline(objective,deadline,times,budget,method) if method in
                               {"wait_first","open_first","exhaustive"} else
                               evaluate(model,distant if method=="irrelevant" else world,
                                        objective,budget,method,intervention))
                        row = dict(id=name,objective=objective,budget=budget,mode=method,
                                   intervention=list(intervention), run=run)
                        if method=="intervention":
                            row["prediction"] = dict(first_action=intervention[0],
                                second_action=1 if objective=="edits" else None,action=predicted)
                            row["prediction_match"] = ([c["action"] for c in run["calls"]]==
                                ([0,1] if objective=="edits" else [1]) and run["action"]==predicted)
                        rows.append(row)
                        if name in ("shortcut-00","blocked-00") and method in ("actual","intervention"):
                            example_worlds[name] = world
                            examples.append(row)
    summary = {}
    for row in rows:
        run = row["run"]
        key = f"{row['mode']}/{row['objective']}/{row['budget']}"
        counts = summary.setdefault(key,Counter())
        counts.update(dict(assigned=1, appropriate=int(run["appropriate"]), objective_success=int(run["objective_success"]),
            stops=int(run["state"]==1), unsupported=int(run["unsupported"]), unnecessary=run["unnecessary"],
            missed_decisions=run["missed"], missed_forecasts=run.get("missed_available_forecast",0),
            forecasts=len(run["calls"]), first_wait=int(bool(run["calls"]) and not run["calls"][0]["action"]),
            executed=sum(c.get("exact_steps",0)>0 for c in run["calls"]),
            errors=len(run.get("errors",[])), prediction_matches=int(row.get("prediction_match",False))))
    assert not any(c["errors"] for c in summary.values()), summary
    paired = {(r["id"],r["budget"],r["objective"],r["mode"]):r["run"] for r in rows}
    for row in rows:
        if row["mode"] in ("precomputed","irrelevant"):
            original = paired[row["id"],row["budget"],row["objective"],"actual"]
            assert row["run"]["requests"]==original["requests"]
            assert row["run"]["action"]==original["action"]
    data = dict(graph_sha256=graph_hash,simulator_sha256=digest((WORLD/"inference.pt").read_bytes()),
                rows=rows,summary=summary)
    bundle = dict(graph_sha256=graph_hash,worlds=example_worlds,examples=examples)
    for filename,value in (("outcomes.json",data),("replays.json",bundle)):
        path = OUT/filename
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert value==json.loads(path.read_text()), filename
    for key,counts in sorted(summary.items()):
        print(key,dict(counts))


if __name__=="__main__":
    main()
