"""Paired allocation evaluation with auditable typed state and exact gating."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import replace
import json
from random import Random
from time import perf_counter_ns

import torch

from evaluate_reusable_evidence_v59 import Task, select
from frozen_call_checkpoint import canonical, digest
from recurrent_evidence_v61 import OUT, WIDTH, encode, mask
from reusable_evidence_v59 import (
    OUT as V59, Candidate, EvidenceStore, CERTIFIED, REJECTED, SCREENED, UNKNOWN,
    actions, earned,
)
from stress_metareasoning_v60 import risk_greedy

METHODS = ("recurrent","reset_attention","v59_mlp","quoted_dp","calibrated_dp","robust_dp",
           "risk_greedy","proof_first","replay_first")


def recurrent(model,plans,states,remaining,memory):
    costs = torch.tensor([[[item.replay,item.proof] for item in plans]],dtype=torch.float32)
    payoff = torch.tensor([[item.payoff for item in plans]])
    priors = torch.tensor([[[item.valid,item.detectable] for item in plans]])
    status = torch.tensor([states])
    budget = torch.tensor([remaining])
    with torch.inference_mode():
        logits,_,memory = model(encode(costs,payoff,priors,status,budget),memory)
        admissible = mask(costs,status,budget)
        raw = int(logits.argmax())
        proposal = int(logits.masked_fill(~admissible,-1e9).argmax())
    return raw,proposal,memory


def run(name,policy,mlp,plans,truth,states,budget,result,clock):
    states = list(states)
    memory = torch.zeros((1,WIDTH))
    remaining,journal = budget,[]
    used = earned(plans,tuple(states))
    for _ in range(7):
        begin = perf_counter_ns()
        if name in ("recurrent","reset_attention"):
            if name=="reset_attention":
                memory = torch.zeros_like(memory)
            raw,query,memory = recurrent(policy,plans,states,remaining,memory)
        elif name=="risk_greedy":
            raw = query = risk_greedy(plans,tuple(states),remaining)
        else:
            if name=="calibrated_dp":
                information = truth
            elif name=="robust_dp":
                information = tuple(replace(p,valid=max(.02,p.valid-.22)) for p in plans)
            else:
                information = plans
            raw = query = select({"v59_mlp":"learned","quoted_dp":"ex_ante_dp","robust_dp":"ex_ante_dp",
                "calibrated_dp":"ex_ante_dp"}.get(name,name),mlp,information,tuple(states),remaining)
        clock["assembly"].append(perf_counter_ns()-begin)
        row = dict(raw=raw,scheduled=query,remaining=remaining,states=states.copy())
        journal.append(row)
        if query==0:
            break
        afford = {q:(i,c) for q,i,c in actions(plans,states,remaining)}
        if query not in afford:
            row["denied"] = True
            break
        i,charge = afford[query]
        screen = bool(query%2)
        reply = result(i,screen,clock)
        states[i] = REJECTED if (reply if screen else not reply) else SCREENED if screen else CERTIFIED
        remaining -= charge
        row.update(answer=reply,cost=charge,module="screen" if screen else "strong")
    gross = earned(plans,tuple(states))
    charge = budget-remaining
    assert 0<=charge<=budget
    return dict(net=gross-used-charge,gross=gross,cost=charge,steps=sum("cost" in r for r in journal),
                journal=journal,authorized=[i for i,s in enumerate(states) if s==CERTIFIED])


def statistics(summary,key,run,kinds,plans,budget,initial):
    assert all(kinds[i]=="valid" for i in run["authorized"])
    incumbent = earned(plans,tuple(initial))
    oracle = max([0]+[p.payoff-incumbent-p.proof for i,p in enumerate(plans)
                  if kinds[i]=="valid" and initial[i]==UNKNOWN and p.proof<=budget])
    assert run["net"]<=oracle
    summary.setdefault(key,Counter()).update(assigned=1,net=run["net"],regret=oracle-run["net"],
        calls=run["steps"],certified=int(bool(run["authorized"])),
        raw_filtered=sum(r["raw"]!=r["scheduled"] for r in run["journal"]),
        denied=sum("denied" in r for r in run["journal"]),
        missed=int(run["gross"]==incumbent and oracle>0),unsupported_accepted=0)


def scenarios():
    rng = Random(217663)
    for family in ("calibrated","shared_shift","rank_reversal","warm_cache"):
        for case in range(96):
            plans,truth,kinds,initial = [],[],[],[]
            shared = rng.choice((-.28,.22))
            for i in range(3):
                prior = rng.choice((.18,.34,.57,.76))
                detect = rng.choice((.12,.55))*(1-prior)
                candidate = Candidate(rng.randint(2,15),rng.randint(7,63),rng.randint(2,9),
                                      rng.randint(3,16),prior,detect)
                delta = {"calibrated":0,"shared_shift":shared,
                         "rank_reversal":.29 if candidate.payoff<55 else -.31,
                         "warm_cache":-.08}[family]
                probability = min(1-detect,max(.02,prior+delta))
                draw = rng.random()
                kind = "valid" if draw<probability else "detected" if draw<probability+detect else "masked"
                plans.append(candidate)
                truth.append(replace(candidate,valid=probability))
                kinds.append(kind)
                initial.append(CERTIFIED if family=="warm_cache" and i==0 and kind=="valid" else UNKNOWN)
            yield family,case,tuple(plans),tuple(truth),kinds,initial,rng.randint(13,114)


def tensor_adapter(kind,machine):
    from unittest.mock import patch
    from persistent_evidence_v60 import prepare, smt
    from replay_frozen_recurrent import PACKED, wasmtime_run
    from replay_guest_rewrite import invoke

    source = [("x",0),("const",3),("add",0),("const",3),("sub",0)]
    candidate = [("x",0)] if kind=="valid" else [("x",0),("const",2),("mul",0)]
    argument = 1 if kind=="detected" else 0
    scope = prepare(source,candidate)
    def query(screen):
        if not screen:
            return smt(source,candidate)[0]
        observed = []
        with patch(PACKED,side_effect=AssertionError("packed fallback")):
            for binary in scope[2]:
                page,reply = invoke(machine,binary,argument,bytearray(65536))
                assert wasmtime_run(binary,argument,bytearray(65536))==(reply["return_i32"],page)
                observed.append(reply["return_i32"])
        return observed[0]!=observed[1]
    return dict(source=source,candidate=candidate,argument=argument),query


def evaluation(policy,mlp):
    summary,rows,live_rows,timings = {},[],[],defaultdict(list)
    for family,case,plans,truth,kinds,initial,budget in scenarios():
        def reply(i,screen,_):
            return kinds[i]==("detected" if screen else "valid")
        output = {}
        for name in METHODS:
            output[name] = run(name,policy,mlp,plans,truth,initial,budget,reply,timings)
            statistics(summary,family+"/"+name,output[name],kinds,plans,budget,initial)
        rows.append(dict(family=family,case=case,plans=[vars(p) for p in plans],truth=[p.valid for p in truth],
                         kinds=kinds,initial=initial,budget=budget,runs=output))
    task = Task()
    rng = Random(620029)
    from replay_guest_rewrite import EXECUTOR
    machine = torch.jit.load(str(EXECUTOR/"machine.pt"))
    for domain in ("compiler","world","tensor"):
        for case in range(6):
            family,_,plans,truth,kinds,initial,budget = next(row for row in scenarios() if row[1]==case)
            fixtures = [tensor_adapter(kind,machine) if domain=="tensor" else task.input(domain,kind,case+i,rng)
                        for i,kind in enumerate(kinds)]
            output = {}
            for name in ("recurrent","quoted_dp","robust_dp","replay_first","proof_first"):
                store = EvidenceStore()
                def real_reply(i,screen,clock):
                    start = perf_counter_ns()
                    if domain=="tensor":
                        callback = lambda: fixtures[i][1](screen)
                        scope = None
                    else:
                        scope,callback = task.evidence(*fixtures[i],screen)
                    clock["scope"].append(perf_counter_ns()-start)
                    start = perf_counter_ns()
                    observed = []
                    def execute():
                        execute_start = perf_counter_ns()
                        value = callback()
                        observed.append(perf_counter_ns()-execute_start)
                        return value
                    value = execute() if scope is None else store.query(scope,execute)[0]
                    if scope is not None:
                        clock["retrieval"].append(perf_counter_ns()-start-sum(observed))
                    clock["verification" if domain in ("compiler","tensor") and not screen else "execution"].extend(observed)
                    assert value==(kinds[i]==("detected" if screen else "valid"))
                    return value
                phases = defaultdict(list)
                start = perf_counter_ns()
                output[name] = run(name,policy,mlp,plans,truth,initial,budget,real_reply,phases)
                phases["end_to_end"].append(perf_counter_ns()-start)
                for key,values in phases.items():
                    timings[f"{domain}/{name}/{key}"].extend(values)
                statistics(summary,"live_"+domain+"/"+name,output[name],kinds,plans,budget,initial)
            live_rows.append(dict(domain=domain,case=case,fixtures=[f[0] if domain=="tensor" else f[1] for f in fixtures],runs=output))
    return summary,rows,live_rows,dict(timings)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [OUT/"policy.pt",V59/"policy.pt"]
    policy,mlp = [torch.jit.load(str(p)) for p in paths]
    summary,rows,live,timings = evaluation(policy,mlp)
    # Recreate a held-out trajectory in a fresh scheduler state.
    first = rows[0]
    same = run("recurrent",policy,mlp,tuple(Candidate(**p) for p in first["plans"]),(),
               first["initial"],first["budget"],
               lambda i,screen,_: first["kinds"][i]==("detected" if screen else "valid"),
               defaultdict(list))
    assert same==first["runs"]["recurrent"]
    record = dict(hashes=[digest(p.read_bytes()) for p in paths],summary=summary,rows=rows,live=live)
    if args.write:
        (OUT/"outcomes.json").write_bytes(canonical(record)+b"\n")
        (OUT/"timings.json").write_bytes(canonical(timings)+b"\n")
    else:
        assert canonical(record)==canonical(json.loads((OUT/"outcomes.json").read_bytes()))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
