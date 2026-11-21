"""Local, independently replayable evaluation of evidence economics."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import product
import json
from random import Random
from statistics import median
from time import perf_counter_ns

import torch

from evaluate_evidence_economics import COMPILER, compiler_case, world_case
from frozen_call_checkpoint import canonical, digest
from replay_guest_region_plan import prove
from reusable_evidence_v59 import (
    OUT, WORLD, CERTIFIED, REJECTED, SCREENED, UNKNOWN, Candidate, EvidenceStore,
    Scope, actions, earned, encode, optimal,
)

MODES = ("learned","greedy_value","ex_ante_dp","proof_first","replay_first","fixed_order")
KINDS = ("valid","detected","masked")


class Task:
    def __init__(self):
        self.simulator = torch.jit.load(str(WORLD/"inference.pt"))
        self.archive = json.loads(COMPILER.read_text())["compiler_runs"]
        self.compiler_id = digest(COMPILER.read_bytes())
        self.world_id = digest((WORLD/"inference.pt").read_bytes())

    def input(self, domain, kind, number, rng):
        if domain=="compiler":
            return compiler_case(self.archive,kind,number)
        return world_case(kind,number,rng)

    def evidence(self, args, fixture, screen):
        domain,evidence,walls,signal,target,horizon = args
        if domain==0:
            site = (fixture["program_sha256"],fixture["site"])
            plan = fixture["plan"]
            if screen:
                a,b = fixture["live_preview"]
                scope = Scope("compiler",self.compiler_id,digest(canonical(site)),
                              (tuple(plan),a,b),"boundary-preview")
                return scope,lambda: a!=b
            source = tuple(plan[1:3])
            scope = Scope("compiler",self.compiler_id,digest(canonical(site)),
                          (tuple(plan),"ordered-local-pair"),"full-equivalence")
            return scope,lambda: all(row["result"]=="unsat" for row in prove(source,plan))

        width,height = walls.size(3),walls.size(2)
        cut = fixture["cut"]
        start = int(signal.flatten().argmax())
        if screen:
            line = tuple(int(v) for v in walls[0,0,:,cut].tolist())
            scope = Scope("world",self.world_id,"cardinal-cut",
                          (width,height,start%width,target%width,cut,line),"cut")
            return scope,lambda: min(start%width,target%width)<cut<max(start%width,target%width) and all(line)
        # A cell outside the horizon cannot influence the target under the
        # unchanged synchronous cardinal-neighborhood rule.
        xy = (target%width,target//width)
        cells = tuple((i,int(walls.flatten()[i]),int(signal.flatten()[i]))
                      for i in range(width*height)
                      if abs(i%width-xy[0])+abs(i//width-xy[1])<=horizon)
        scope = Scope("world",self.world_id,"target-cone",(width,height,target,horizon,cells),"full-forecast")
        return scope,lambda: bool(self.simulator.forecast(walls,signal,0,target,0,horizon)[0])


def calibrate(task):
    rng = Random(7322)
    raw = defaultdict(list)
    for domain,kind,repeat,screen in product(("compiler","world"),KINDS,range(8),(True,False)):
        args,fixture = task.input(domain,kind,repeat,rng)
        _,query = task.evidence(args,fixture,screen)
        started = perf_counter_ns()
        query()
        raw[f"{domain}/{'screen' if screen else 'strong'}"].append(perf_counter_ns()-started)
    return dict(raw)


def select(mode,model,candidates,states,budget):
    options = list(actions(candidates,states,budget))
    if mode=="learned":
        with torch.inference_mode():
            return int(model(torch.tensor(encode(candidates,states,budget))).argmax())
    if mode=="ex_ante_dp":
        return optimal(candidates,states,budget)[1]
    if mode=="greedy_value":
        best = (0,0)
        incumbent = earned(candidates,states)
        for query,i,cost in options:
            plan = candidates[i]
            gain = max(0,plan.payoff-incumbent)
            if query%2:
                if plan.proof+cost>budget:
                    continue
                score = plan.valid*gain-cost-(1-plan.detectable)*plan.proof
            else:
                p = plan.valid if states[i]==UNKNOWN else plan.valid/(1-plan.detectable)
                score = p*gain-cost
            if score>best[0]:
                best = score,query
        return best[1]
    if mode=="proof_first":
        options = [row for row in options if row[0]%2==0]
    if mode=="replay_first":
        options.sort(key=lambda row:0 if row[0]%2 else 1)
    return options[0][0] if options else 0


def trial(task,model,mode,candidates,kinds,fixtures,budget,clock):
    stores = [EvidenceStore() for _ in candidates]
    states = [UNKNOWN]*3
    journal = []
    remaining = budget
    for _ in range(7):
        started = perf_counter_ns()
        proposal = select(mode,model,candidates,tuple(states),remaining)
        clock["proposal/"+mode].append(perf_counter_ns()-started)
        options = {query:(i,cost) for query,i,cost in actions(candidates,states,remaining)}
        record = dict(proposal=proposal,remaining=remaining,states=states.copy())
        journal.append(record)
        if proposal==0:
            break
        if proposal not in options:
            record["denied"] = True
            break
        i,charge = options[proposal]
        screen = bool(proposal%2)
        scope,query = task.evidence(*fixtures[i],screen)
        started = perf_counter_ns()
        answer,hit = stores[i].query(scope,query)
        clock["evidence/"+scope.domain+("/hit" if hit else "/miss")].append(perf_counter_ns()-started)
        assert answer==(kinds[i]==("detected" if screen else "valid"))
        cost = 0 if hit else charge
        remaining -= cost
        states[i] = (REJECTED if answer else SCREENED) if screen else (CERTIFIED if answer else REJECTED)
        record.update(scope=scope.identity(),query=scope.query,cache_hit=hit,cost=cost,result=answer)
    approved = [i for i,status in enumerate(states) if status==CERTIFIED]
    chosen = max(approved,key=lambda i:(candidates[i].payoff,-i),default=-1)
    assert chosen==-1 or kinds[chosen]=="valid"
    spent = budget-remaining
    gross = candidates[chosen].payoff if chosen>=0 else 0
    clairvoyant = max([0]+[plan.payoff-plan.proof for plan,kind in zip(candidates,kinds)
                            if kind=="valid" and plan.proof<=budget])
    final_value = optimal(candidates,tuple(states),remaining)[0]-gross
    return dict(chosen=chosen,spent=spent,gross=gross,net=gross-spent,journal=journal,
                clairvoyant_net=clairvoyant, justified_abstention=chosen==-1 and final_value<=1e-8,
                missed_profitable=chosen==-1 and clairvoyant>0,
                unsupported_proposals=sum(bool(row.get("denied")) for row in journal),
                unsupported_accepted=0)


def reuse_and_relations(task,clock):
    rng = Random(189)
    store = EvidenceStore()
    results = {}
    for domain in ("compiler","world"):
        args,fixture = task.input(domain,"valid",0,rng)
        scope,execute = task.evidence(args,fixture,False)
        warm = EvidenceStore()
        for attempt in range(16):
            for name, cache in (("cold",EvidenceStore()),("reused",warm)):
                started = perf_counter_ns()
                reply,hit = cache.query(scope,execute)
                clock.setdefault(f"amortization/{domain}/{name}",[]).append(perf_counter_ns()-started)
                assert reply and hit==(name=="reused" and attempt>0)
    rng = Random(189)
    for domain in ("compiler","world"):
        args,fixture = task.input(domain,"valid",0,rng)
        scope,execute = task.evidence(args,fixture,False)
        assert store.query(scope,execute)==(True,False)
        changed_assumptions = replace(scope,semantics=scope.semantics+"-changed")
        assert changed_assumptions.identity()!=scope.identity()
        if domain=="world":
            widgets = list(args)
            widgets[2] = args[2].clone()
            widgets[2][0,0,0,0] = 1
            related,related_query = task.evidence(tuple(widgets),fixture,False)
            assert related.identity()==scope.identity()
            assert store.query(related,related_query)==(True,True)
            widgets[2].flatten()[args[4]]=1
            changed,changed_query = task.evidence(tuple(widgets),fixture,False)
            assert changed.identity()!=scope.identity()
            assert store.query(changed,changed_query)==(False,False)
        else:
            assert store.query(scope,execute)==(True,True)
            bad,bad_fixture = task.input(domain,"masked",0,rng)
            changed,changed_query = task.evidence(bad,bad_fixture,False)
            assert changed.identity()!=scope.identity()
            assert store.query(changed,changed_query)==(False,False)
        assert store.query(changed_assumptions,execute)==(True,False)
        results[domain] = dict(reused=scope.identity(),invalidated=changed.identity(),
                               artifact=asdict(scope),result=True,fixture=fixture,
                               requests=16,executions_with_reuse=1)
    # Terrain inclusion implies earliest(A)<=earliest(B). Both share the
    # source, target, time horizon, and cardinal transition.
    args,_ = task.input("world","valid",0,rng)
    a, b = args[2].clone(),args[2].clone()
    b.flatten()[0]=1
    assert bool((a<=b).all())
    assert task.simulator.forecast(b,args[3],0,args[4],0,args[5])[0]==1
    boxes = [(x,y) for x,y in product(range(1,args[5]+1),repeat=2)]
    best = lambda pairs: sorted({0 if x<=y else 1 for x,y in pairs})
    independent,related = best(boxes),best([(x,y) for x,y in boxes if x<=y])
    assert independent==[0,1] and related==[0]
    a.flatten()[args[4]]=1
    assert not bool((a<=b).all())
    assert task.simulator.forecast(a,args[3],0,args[4],0,args[5])[0]==0
    results["relation"] = dict(independent=independent,checked_inclusion=related,
                               invalidated_by_blocking_target=True)
    return results


def evaluate(task,model,calibration):
    rows,clock = [],defaultdict(list)
    rng = Random(829031)
    for domain,split,index in product(("compiler","world"),("in_range","shifted"),range(64)):
        candidates,fixtures,kinds = [],[],[]
        budget = rng.randint(14,85) if split=="in_range" else rng.randint(30,160)
        strong_ns = median(calibration[domain+"/strong"])
        cheap_ns = median(calibration[domain+"/screen"])
        unit_ns = max(1,strong_ns/12)
        for candidate in range(3):
            prior = rng.choice((.2,.4,.6,.8))
            detectable = rng.choice((0,.5,.85))*(1-prior)
            selection = rng.random()
            p = prior if split=="in_range" else prior*.7
            kind = "valid" if selection<p else "detected" if selection<p+detectable else "masked"
            replay = max(1,round(cheap_ns/unit_ns*rng.uniform(.6,2.6)))
            proof = max(replay+1,round(strong_ns/unit_ns*rng.uniform(.55,3.4 if split=="in_range" else 6.2)))
            plan = Candidate(replay,proof,rng.randint(1,8),rng.randint(1,14 if split=="in_range" else 22),
                             prior,detectable)
            candidates.append(plan)
            kinds.append(kind)
            fixtures.append(task.input(domain,kind,index+candidate,rng))
        runs = {}
        for mode in MODES:
            started = perf_counter_ns()
            runs[mode] = trial(task,model,mode,tuple(candidates),kinds,fixtures,budget,clock)
            clock["end_to_end/"+mode].append(perf_counter_ns()-started)
        rows.append(dict(domain=domain,split=split,case=index,budget=budget,kinds=kinds,
                         tariff_unit_ns=unit_ns,candidates=[vars(c) for c in candidates],
                         fixtures=[item[1] for item in fixtures],runs=runs))
    summary = {}
    for row in rows:
        for mode,run in row["runs"].items():
            key = "/".join((row["domain"],row["split"],mode))
            sums = summary.setdefault(key,Counter())
            sums.update(assigned=1,net=run["net"],spent=run["spent"],calls=sum("cost" in j for j in run["journal"]),
                        payoff=run["gross"],certificates=int(run["chosen"]>=0),
                        justified_abstentions=int(run["justified_abstention"]),
                        missed_profitable=int(run["missed_profitable"]),
                        denied=run["unsupported_proposals"],unsupported_accepted=run["unsupported_accepted"],
                        paid_back=int(run["chosen"]>=0 and run["net"]>0),oracle_net=run["clairvoyant_net"])
    return rows,summary,dict(clock)


def main():
    parser = ArgumentParser()
    parser.add_argument("--calibrate",action="store_true")
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    task = Task()
    calibration_path = OUT/"calibration.json"
    assert args.write or not args.calibrate, "Recalibration requires a new outcomes archive"
    if args.calibrate:
        calibration_path.write_bytes(canonical(calibrate(task))+b"\n")
    calibration = json.loads(calibration_path.read_bytes())
    policy_path = OUT/"policy.pt"
    model = torch.jit.load(str(policy_path))
    rows,summary,timings = evaluate(task,model,calibration)
    reuse = reuse_and_relations(task,timings)
    result = dict(graph=digest(policy_path.read_bytes()),simulator=task.world_id,compiler=task.compiler_id,
                  calibration=digest(calibration_path.read_bytes()),rows=rows,summary=summary,
                  reuse=reuse)
    if args.write:
        (OUT/"outcomes.json").write_bytes(canonical(result)+b"\n")
        (OUT/"timings.json").write_bytes(canonical(timings)+b"\n")
    else:
        assert canonical(result)==canonical(json.loads((OUT/"outcomes.json").read_bytes()))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
