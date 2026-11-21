"""Compare raw policies under the frozen checked-decision contract."""
from collections import Counter
from itertools import permutations, product
import argparse
import json
import struct
from random import Random

import torch

from checked_decisions import certify
from evaluate_checked_decisions import OUT as PRIOR, check_shared_contract, world_run
from frozen_call_checkpoint import canonical, digest, restore
from replay_guest_region_plan import prove
from search_compiler_counterexample import Experiment
from train_decision_query_policy import OUT


def holdout():
    rng = Random(317093)
    for kind in ("direct","open_a","open_b","blocked"):
        for i in range(24):
            distance = 3+i%5
            width = 31
            x,y = rng.randrange(10,21),rng.randrange(10,21)
            dx,dy = ((1,0),(0,-1),(-1,0),(0,1))[i%4]
            side = rng.choice((-1,1))

            def point(along,across):
                return x+dx*along-dy*across*side,y+dy*along+dx*across*side

            def cell(along,across):
                a,b = point(along,across)
                return a+width*b

            walls,signal = [1]*(width**2),[0]*(width**2)
            for step in range(distance+1):
                walls[cell(step,0)] = walls[cell(step,2)] = 0
            walls[cell(0,1)] = walls[cell(distance,1)] = 0
            signal[cell(0,0)] = 1
            gates = [cell(1+i%(distance-1),0),cell(distance//2,2)]
            walls[gates[1]] = 1
            walls[gates[0]] = int(kind!="direct")
            if kind=="open_b":
                gates.reverse()
            if kind=="blocked":
                walls[cell(distance,0)] = 1
            # Disconnected background keeps coordinate/terrain variation reproducible.
            walls[:width] = [rng.randrange(2) for _ in range(width)]
            yield f"{kind}-{i:02}",dict(width=width,height=width,gate=gates[0],
                target=list(point(distance,0)),walls=walls,signal=signal,horizon=8),gates


def summary(row, compiler=False):
    journal = row["journal"]
    final = journal[-1]
    claim,recommendation = final[:2]
    return dict(assigned=1, certificates=int(row["decision"]>=0),
                calls=row["calls"] if compiler else row["forecasts"],
                unsupported_claims=int(3<=claim<=6 and claim-3!=recommendation),
                appropriate_abstentions=int(claim==7 and recommendation==-1 and row["accepted"]),
                missed_certificates=int(claim==7 and recommendation>=0),
                protocol_stops=int(claim<3 or claim==8))


def bounded_oracle(run, budget, penalty):
    """Clairvoyant diagnostic for available certificates under the call cap."""
    lower = run["journal"][0][2:5]
    upper = run["journal"][0][5:8]
    costs = [tick+9*int(penalty and i!=0) if tick else 32
             for i,tick in enumerate(run["arrivals"])]
    for length in range(budget+1):
        winning = []
        for order in permutations(range(3),length):
            lo,hi = lower.copy(),upper.copy()
            for action in order:
                lo[action]=hi[action]=costs[action]
            if certify(torch.tensor(lo),torch.tensor(hi),32)>=0:
                winning.append(order)
        if winning:
            return dict(minimum_calls=length,first_queries=sorted({order[0] for order in winning if order}))
    return dict(minimum_calls=None,first_queries=[])


def compiler_holdout(graphs):
    archive = []
    empty = torch.empty(0)
    walls = torch.zeros((1,1,1,1))
    for guest in ("passage","terrain"):
        experiment = Experiment(guest)
        initial = experiment.initial()
        state = restore(initial,experiment.identity,experiment.image)
        # Reach the same site with the same recorded input as Experiment.tick.
        if guest=="terrain":
            struct.pack_into("<i",state.vm.page.numpy(),experiment.argument+320,4)
        else:
            struct.pack_into("<i",state.vm.page.numpy(),experiment.argument+4,1)
        while int(state.vm.pc)!=experiment.site:
            assert int(state.vm.status)==0
            state = experiment.one(experiment.image.rows,state).state
        assert int(state.vm.height)+2<=state.vm.stack.numel() and int(state.vm.fuel)>=2
        source = torch.tensor(experiment.slots)
        for bad in (False,True):
            words = experiment.bad if bad else experiment.good
            assert words[0]==1 and tuple(words[1:3])==experiment.slots and tuple(words[4:])==(1,2)
            candidate = torch.tensor([words[3],words[2]])
            proof = prove(experiment.slots,words)
            assert [p["result"] for p in proof]==(["sat","unsat"] if bad else ["unsat","unsat"])
            _,real_replay = experiment.tick(initial,1,bad)
            values = state.vm.locals
            for mask in (False,True):
                locals_ = torch.zeros_like(values) if mask else values
                for budget in (1,2):
                    cases = {}
                    for name,graph,policy in graphs:
                        rows,_,decision,accepted = graph(1,walls,walls,empty,0,0,budget,False,
                                                         source,candidate,locals_,policy)
                        journal = rows[rows[:,0]>=0].tolist()
                        calls = sum(row[8]>=0 for row in journal)
                        assert calls<=budget and decision in (-1,int(not bad))
                        cases[name] = dict(journal=journal,decision=decision,accepted=accepted,calls=calls)
                    archive.append(dict(guest=guest,program_sha256=digest(experiment.raw),
                        site=experiment.site,plan=list(words),bad=bad,equal_locals=mask,budget=budget,
                        oracle=int(not bad),exhaustive_calls=2,proof=proof,exact_replay=real_replay,runs=cases))
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    old_path,new_path = PRIOR/"inference.pt",OUT/"inference.pt"
    old,new = [torch.jit.load(str(path)) for path in (old_path,new_path)]
    # The fixed path uses the same exact checker and asks unresolved world
    # questions in index order. For the compiler it requests universal evidence.
    graphs = [("original",old,0),("adapted",new,0),("fixed",new,1)]
    rows,worlds,stats = [],{},{}
    with torch.inference_mode():
        for name,world,edits in holdout():
            worlds[name] = dict(scene=world,edits=edits)
            for budget,penalty in product((1,2),(False,True)):
                runs = {}
                for policy,graph,fixed in graphs:
                    run = world_run(graph,world,edits,budget,penalty,fixed,8)
                    runs[policy] = run
                    stats.setdefault(f"world/{budget}/{penalty}/{policy}",Counter()).update(summary(run))
                bound = bounded_oracle(runs["fixed"],budget,penalty)
                assert (runs["adapted"]["decision"]>=0)==(bound["minimum_calls"] is not None)
                rows.append(dict(id=name,budget=budget,prefer_no_edit=penalty,
                    exhaustive=dict(choice=runs["fixed"]["oracle"],calls=3),bounded_oracle=bound,runs=runs))
        compiler = compiler_holdout(graphs)
    for case in compiler:
        for policy,row in case["runs"].items():
            stats.setdefault(f'compiler/{case["budget"]}/{policy}',Counter()).update(summary(row,True))
    assert all(not row["unsupported_claims"] for key,row in stats.items() if key.endswith("/adapted"))
    assert stats["world/1/False/adapted"]["certificates"]==24
    assert stats["world/1/False/original"]["missed_certificates"]>0
    assert len({digest(canonical(x)) for x in worlds.values()})==len(worlds)
    data = dict(old_graph_sha256=digest(old_path.read_bytes()),new_graph_sha256=digest(new_path.read_bytes()),
                checked_interval_boxes=check_shared_contract(),summary=stats,worlds=worlds,world_runs=rows,
                compiler_runs=compiler)
    if args.write:
        (OUT/"outcomes.json").write_bytes(canonical(data)+b"\n")
    else:
        assert data==json.loads((OUT/"outcomes.json").read_bytes())
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
