"""Run both domains against independent full-state references."""
import argparse
from collections import Counter
from itertools import product
import json
from random import Random

import torch

from checked_decisions import certify
from evaluate_pocket_paired_goals import arrival
from evaluate_world_inference_pairs import verification_frames
from frozen_call_checkpoint import canonical, digest, restore
from pocket_world_inference import OUT as WORLD
from replay_cellular_automata import export_region
from replay_guest_region_plan import prove
from replay_pocket_world_inference import inputs
from search_compiler_counterexample import Experiment

OUT = WORLD.parent/"checked-decisions"


def worlds():
    rng = Random(704233)
    for kind in ("direct","bound","invalid_shortcut","blocked"):
        for number in range(16):
            size = 25
            distance = 4+number%3
            ox,oy = rng.randrange(8,17),rng.randrange(8,17)
            dx,dy = ((1,0),(0,1),(-1,0),(0,-1))[number%4]
            def position(x,y):
                return ox+x*dx-y*dy,oy+x*dy+y*dx
            def idx(x,y):
                px,py = position(x,y)
                return py*size+px
            walls,signal = [1]*(size**2),[0]*(size**2)
            for x in range(distance+1):
                walls[idx(x,0)] = walls[idx(x,1)] = 0
            signal[idx(0,0)]=1
            edits = [idx(2,0),idx(2,1)]
            if kind!="direct":
                for edit in edits:
                    walls[edit]=1
                # Isolate the corridors to prevent crossing around an unopened gate.
                for x in range(1,distance):
                    walls[idx(x,1)] = 1
                    walls[idx(x,2)] = 0
                walls[idx(0,1)] = walls[idx(distance,1)] = 0
                walls[idx(0,2)] = walls[idx(distance,2)] = 0
                edits[1] = idx(2,2)
                walls[edits[1]]=1
            if kind=="invalid_shortcut":
                # A is the detour and B opens the direct route.
                edits.reverse()
            if kind=="blocked":
                walls[idx(distance,0)]=1
            for index in range(size):
                walls[index] = rng.randrange(2)
            target = list(position(4+number%3,0))
            yield kind,number,dict(width=size,height=size,gate=edits[0],target=target,
                                  walls=walls,signal=signal,horizon=8),edits


def check_shared_contract():
    intervals = [(lo,hi) for lo in range(4) for hi in range(lo,4)]
    checked = 0
    for count in (2,3):
        for box in product(intervals,repeat=count):
            outcomes = set()
            for costs in product(*(range(lo,hi+1) for lo,hi in box)):
                choices = [(cost,i) for i,cost in enumerate(costs) if cost<3]
                outcomes.add(min(choices)[1] if choices else count)
            expected = next(iter(outcomes)) if len(outcomes)==1 else -1
            lo,hi = zip(*box)
            assert certify(torch.tensor(lo),torch.tensor(hi),3)==expected
            checked += 1
    return checked


def world_run(graph,world,edits,budget,penalty,policy,coverage):
    horizon = world["horizon"]
    packet = export_region(world,world["target"],horizon,horizon)
    walls,signal,_,target = inputs(packet)
    ox,oy = packet["origin"]
    size = world["width"]
    regional_edits = [(edit//size-oy)*packet["width"]+edit%size-ox for edit in edits]
    empty = torch.empty(0,dtype=torch.long)
    journal,frames,decision,accepted = graph(0,walls,signal,torch.tensor(regional_edits),target,
                                            horizon,budget,penalty,empty,empty,empty,policy,coverage)
    costs, expected_times = [], []
    for action in range(3):
        scene = dict(world,gate=edits[max(0,action-1)])
        tick = arrival(scene,int(action!=0),horizon)
        expected_times.append(tick)
        costs.append(tick+(9 if penalty and action else 0) if tick else 32)
    optimum = min(range(3),key=lambda i:(costs[i],i)) if min(costs)<32 else 3
    entries = journal[journal[:,0]>=0].tolist()
    forecasts = 0
    for i,row in enumerate(entries):
        action,reported = row[0],row[8]
        if reported < 0:
            continue
        assert 0<=action<=2 and reported==expected_times[action] and row[9]==horizon
        selected_world = dict(world,gate=edits[max(0,action-1)])
        regional = dict(packet,gate=[edits[max(0,action-1)]%size,edits[max(0,action-1)]//size])
        observed = frames[i,:horizon].flatten(1).int().tolist()
        assert verification_frames(selected_world,regional,int(action!=0),horizon,observed)
        forecasts += 1
    assert decision<0 or decision==optimum
    assert forecasts<=budget
    # Remove the geometric prior while keeping the identical observed replies.
    loose_lower = [1]+[10 if penalty else 1]*2
    loose_upper = [32]*3
    for action,*rest in entries:
        if rest[7]>=0:
            loose_lower[action] = loose_upper[action] = costs[action]
    without_distance = certify(torch.tensor(loose_lower),torch.tensor(loose_upper),32)
    return dict(decision=decision,accepted=accepted,oracle=optimum,arrivals=expected_times,
                journal=entries,forecasts=forecasts,without_distance=without_distance,checkpoint=digest(canonical(packet)),
                raw_claim=entries[-1][0]-3 if 3<=entries[-1][0]<=6 else None)


def compiler_runs(graph):
    experiment = Experiment("passage")
    initial = experiment.initial()
    state = restore(initial,experiment.identity,experiment.image)
    assert int(state.vm.pc)==experiment.site
    assert int(state.vm.height)+2<=state.vm.stack.numel() and int(state.vm.fuel)>=2
    walls = torch.zeros((1,1,1,1))
    source = torch.tensor(experiment.slots)
    records = []
    for bad in (False,True):
        words = experiment.bad if bad else experiment.good
        slots = torch.tensor([words[3],words[2]])
        proof = prove(experiment.slots,words)
        assert [p["result"] for p in proof]==(["sat","unsat"] if bad else ["unsat","unsat"])
        _,real_replay = experiment.tick(initial,1,bad)
        assert bool(real_replay["first_difference"])==bad
        # This witness varies the first-boundary preview only. The exit
        # selectors, source fuel debits, and reviewed lowering are fixed.
        assert words[0]==1 and tuple(words[1:3])==experiment.slots and tuple(words[4:])==(1,2)
        assert not experiment.image.rows[experiment.site:experiment.site+2,5:8].any()
        for budget in (1,2):
            rows,_,decision,accepted = graph(1,walls,walls,torch.empty(0),0,0,budget,False,
                                             source,slots,state.vm.locals)
            assert decision in (-1,0) if bad else decision in (-1,1)
            records.append(dict(bad=bad,budget=budget,decision=decision,accepted=accepted,
                                journal=rows[rows[:,0]>=0].tolist(),smt=proof,replay=real_replay,
                                program_sha256=digest(experiment.raw),site=experiment.site,plan=list(words)))
    # Equal live locals hide the wrong preview from this targeted replay.
    # The one-input result cannot authorize adoption. Universal equality
    # still fails on the second query, so the graph keeps the reference.
    rows,_,decision,accepted = graph(1,walls,walls,torch.empty(0),0,0,2,False,
                                     source,slots,torch.zeros_like(state.vm.locals))
    journal = rows[rows[:,0]>=0].tolist()
    assert [r[0] for r in journal]==[8,1,3] and journal[0][8]==0
    assert decision==0 and accepted
    records.append(dict(bad=True,masked_replay=True,budget=2,decision=decision,accepted=accepted,journal=journal))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = OUT/"inference.pt"
    graph = torch.jit.load(str(model))
    runs,examples = [],{}
    with torch.inference_mode():
        for kind,number,world,edits in worlds():
            for penalty,budget,policy in product((False,True),(1,2),(0,1)):
                run = world_run(graph,world,edits,budget,penalty,policy,8)
                row = dict(kind=kind,number=number,penalty=penalty,budget=budget,policy=policy,**run)
                runs.append(row)
                if number==0 and not penalty and budget==2 and policy==0:
                    examples[kind] = dict(world=world,edits=edits,run=row)
        world,edits = examples["invalid_shortcut"]["world"],examples["invalid_shortcut"]["edits"]
        unavailable = world_run(graph,world,edits,2,False,0,7)
        assert unavailable["forecasts"]==0 and unavailable["decision"]==-1
        invalid = examples["invalid_shortcut"]["run"]
        assert invalid["arrivals"]==[0,8,4] and invalid["decision"]==-1
        bound = examples["bound"]["run"]
        assert bound["decision"]==1 and bound["without_distance"]==-1
        compiler = compiler_runs(graph)
    summary = {}
    for row in runs:
        counters = summary.setdefault(f'{row["policy"]}/{row["budget"]}/{row["penalty"]}',Counter())
        counters.update(dict(assigned=1,certified=int(row["decision"]>=0),
                             abstained=int(row["decision"]<0),forecasts=row["forecasts"],
                             bounds_needed=int(row["decision"]>=0 and row["without_distance"]!=row["decision"]),
                             missed_certificates=int(row["decision"]<0 and row["journal"][-1][1]>=0),
                             rejected_claims=int(row["raw_claim"] is not None and not row["accepted"])))
    data = dict(graph_sha256=digest(model.read_bytes()),simulator_sha256=digest((WORLD/"inference.pt").read_bytes()),
                checked_interval_boxes=check_shared_contract(),summary=summary,runs=runs,
                examples=examples,unavailable=unavailable,compiler=compiler)
    if args.write:
        (OUT/"outcomes.json").write_bytes(canonical(data)+b"\n")
    else:
        assert data==json.loads((OUT/"outcomes.json").read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
