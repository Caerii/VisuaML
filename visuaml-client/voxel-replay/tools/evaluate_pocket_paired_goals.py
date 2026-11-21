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
