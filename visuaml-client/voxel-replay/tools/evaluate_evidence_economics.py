"""Paired cost schedules over exact world and reviewed compiler witnesses."""
from argparse import ArgumentParser
from collections import Counter
from itertools import product
import json
from random import Random

import torch

from evaluate_pocket_paired_goals import arrival
from evidence_economics import OUT, WORLD
from frozen_call_checkpoint import canonical, digest

POLICIES = ("learned","prove_first","cheap_first","expected_value")
COMPILER = OUT.parent/"decision-query-policy"/"outcomes.json"
# Regime parameters are shared by both domains. Priors are mixture frequencies,
# independent of the hidden outcome of each individual case.
REGIMES = {
    "screen": (6,11,2,18,61,24),
    "prove": (16,2,3,12,49,25),
    "low_payoff": (8,5,4,16,8,23),
    "tight_budget": (12,6,2,18,59,12),
}


def compiler_case(archive, kind, index):
    bad = kind!="valid"
    guest = "passage" if index%2==0 else "terrain"
    # The independent archive contains the exact same template obligations
    # used by the graph and the frozen execution/Wasmtime witness.
    rows = [r for r in archive if r["guest"]==guest and r["bad"]==bad and
            r["equal_locals"]==(kind=="masked") and r["budget"]==2]
    case, = rows
    assert [p["result"] for p in case["proof"]]==(["sat","unsat"] if bad else ["unsat","unsat"])
    delta = case["exact_replay"]["first_difference"]
    if bad and kind=="detected":
        assert delta and delta["reference"]!=delta["candidate"]
    # Equality is the observable for both checks; the masked control sets
    # both selected live values equal while preserving distinct selectors.
    plan = case["plan"]
    candidate,source = plan[3],plan[1]
    if kind=="detected":
        reference_value = delta["reference"]["stack"][-1]
        candidate_value = delta["candidate"]["stack"][-1]
    else:
        reference_value,candidate_value = 0,0
    dummy = torch.zeros((1,1,1,1))
    return (0,torch.tensor([source,candidate,reference_value,candidate_value]),
            dummy,dummy,0,0), dict(guest=guest,site=case["site"],plan=plan,
                                    program_sha256=case["program_sha256"],
                                    live_preview=[reference_value,candidate_value])


def world_case(kind, index, rng):
    width,height = 17,17
    start = rng.randint(2,4)
    distance = rng.randint(4,7)
    y = rng.randint(7,9)
    goal,cut = start+distance,start+2
    walls,signal = [0]*(width*height),[0]*(width*height)
    signal[y*width+start] = 1
    if kind!="valid":
        for row in range(height):
            walls[row*width+cut] = 1
        if kind=="masked":
            walls[cut] = 0  # Gap outside the horizon's reachable cone.
    scene = dict(width=width,height=height,walls=walls,signal=signal,
                 target=[goal,y],gate=0)
    assert bool(arrival(scene,0,distance))==(kind=="valid")
    assert all(walls[row*width+cut] for row in range(height))==(kind=="detected")
    shape = (1,1,height,width)
    return (1,torch.tensor([1,cut]),torch.tensor(walls,dtype=torch.float32).reshape(shape),
            torch.tensor(signal,dtype=torch.float32).reshape(shape),y*width+goal,distance), dict(
                scene=scene,cut=cut,deadline=distance)


def run(graph, policy, budget, estimates, actual, expected_payoff, priors, args, actual_payoff, kind):
    journal,decision,cost = graph(budget,torch.tensor(estimates),torch.tensor(actual),
                                 expected_payoff,*priors,policy,*args)
    entries = journal[journal[:,0]>=0].tolist()
    actual_cost = sum(row[2] for row in entries if row[2]>=0)
    assert cost==actual_cost<=budget and len(entries)<=3
    assert all(row[3]==int(kind!="detected") for row in entries if row[0]==1 and row[2]>=0)
    assert all(row[3]==int(kind=="valid") for row in entries if row[0]==2 and row[2]>=0)
    if decision==1:
        assert kind=="valid"
        assert any(row[0]==2 and row[3]==1 and row[2]>=0 for row in entries)
    if decision==0:
        assert kind!="valid"
    gross = actual_payoff if decision==1 else 0
    return dict(raw_first=entries[0][0],decision=decision,cost=cost,gross=gross,net=gross-cost,
                paid_back=decision==1 and gross>cost, budget_denied=any(row[5]==1 for row in entries),
                raw_unsupported=any(row[0]==4 and row[4]!=1 for row in entries),
                journal=entries)


def evaluate(graph,archive):
    rng = Random(1170933)
    summary,rows = {},[]
    for domain,regime in product(("compiler","world"),REGIMES):
        valid,rejections,cheap,proof,value,budget = REGIMES[regime]
        outcomes = ["valid"]*valid+["detected"]*rejections+["masked"]*(20-valid-rejections)
        rng.shuffle(outcomes)
        for index,kind in enumerate(outcomes):
            adapter = compiler_case(archive,kind,index) if domain=="compiler" else world_case(kind,index,rng)
            args,fixture = adapter
            # Gross value amortizes an assumed per-use saving over workload.
            saving = rng.randint(2,5)
            workload = max(1,(value+rng.randrange(-4,6))//saving)
            gross = saving*workload
            limit = budget+rng.choice((-1,0,2))
            actual = [cheap,proof+rng.choice((-1,0,2))]
            for quotes in ("accurate","underpriced","overpriced"):
                estimated = actual.copy()
                expected_payoff = gross
                if quotes=="underpriced":
                    estimated[1] = max(1,proof-5)
                    expected_payoff += 12
                if quotes=="overpriced":
                    estimated[1] = proof+6
                    expected_payoff = max(0,gross-13)
                tests = {}
                for policy,name in enumerate(POLICIES):
                    row = run(graph,policy,limit,estimated,actual,expected_payoff,
                              (5*valid,5*rejections),args,gross,kind)
                    tests[name] = row
                    count = summary.setdefault(f"{domain}/{quotes}/{name}",Counter())
                    count.update(assigned=1, cost=row["cost"], net=row["net"],
                        adopted=int(row["decision"]==1), paid_back=int(row["paid_back"]),
                        adoption_without_payback=int(row["decision"]==1 and not row["paid_back"]),
                        refuted=int(row["decision"]==0), unresolved=int(row["decision"]==-1),
                        budget_denied=int(row["budget_denied"]), unsupported=int(row["raw_unsupported"]),
                        first_stop=int(row["raw_first"]==0),first_cheap=int(row["raw_first"]==1),
                        first_proof=int(row["raw_first"]==2),
                        missed_profitable=int(kind=="valid" and row["decision"]!=1 and gross>actual[1]
                                              and actual[1]<=limit))
                rows.append(dict(domain=domain,regime=regime,quotes=quotes,index=index,kind=kind,
                    budget=limit,estimates=estimated,actual=actual,expected_payoff=expected_payoff,
                    workload=workload,saving_per_use=saving,fixture=fixture,runs=tests))
    return rows,summary


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    opts = parser.parse_args()
    torch.set_num_threads(1)
    artifact = OUT/"inference.pt"
    graph = torch.jit.load(str(artifact))
    archive = json.loads(COMPILER.read_text())["compiler_runs"]
    with torch.inference_mode():
        rows,summary = evaluate(graph,archive)
        args,_ = world_case("valid",0,Random(0))
        rejected = run(graph,4,30,[2,10],[2,10],60,(50,30),args,60,"valid")
        assert rejected["raw_unsupported"] and rejected["decision"]==-1 and rejected["cost"]==0
        for domain in ("compiler","world"):
            accurate = summary[f"{domain}/accurate/learned"]
            assert all(accurate[f"first_{choice}"] > 0 for choice in ("stop","cheap","proof"))
            assert summary[f"{domain}/underpriced/learned"]["budget_denied"]>0
            assert summary[f"{domain}/overpriced/learned"]["missed_profitable"]>0
            assert not any(row["unsupported"] for key,row in summary.items() if key.startswith(domain+"/"))
    result = dict(graph_sha256=digest(artifact.read_bytes()),proof_archive_sha256=digest(COMPILER.read_bytes()),
        simulator_sha256=digest((WORLD/"inference.pt").read_bytes()),rows=rows,summary=summary,
        rejected_unsupported_claim=rejected)
    path = OUT/"outcomes.json"
    if opts.write:
        path.write_bytes(canonical(result)+b"\n")
    else:
        assert result==json.loads(path.read_text())
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
