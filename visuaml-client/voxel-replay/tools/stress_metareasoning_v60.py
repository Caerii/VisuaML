"""Matched prior shifts on the frozen v59 allocator (accounting simulation)."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
import json
from random import Random

import torch

from evaluate_reusable_evidence_v59 import select
from frozen_call_checkpoint import canonical, digest
from reusable_evidence_v59 import (
    OUT as V59, Candidate, CERTIFIED, REJECTED, SCREENED, UNKNOWN, actions, earned, optimal,
)

OUT = V59.parent/"persistent-evidence-v60"
REGIMES = ("accurate","optimistic","pessimistic","rank_shift")
POLICIES = ("learned","quoted_dp","calibrated_dp","calibrated_greedy","robust_dp","risk_greedy")


def risk_greedy(plans,states,budget):
    incumbent = earned(plans,states)
    best = (0.,0)
    for request,i,charge in actions(plans,states,budget):
        plan = plans[i]
        gain = max(0,plan.payoff-incumbent)
        p = plan.valid if states[i]==UNKNOWN else plan.valid/(1-plan.detectable)
        if request%2:
            if charge+plan.proof>budget:
                continue
            expected_cost = charge+(1-plan.detectable)*plan.proof
        else:
            expected_cost = charge
        score = p*gain-expected_cost-.25*gain*p*(1-p)
        if score>best[0]:
            best = score,request
    return best[1]


def attempt(model,plans,calibrated,kinds,budget,policy):
    if policy.startswith("calibrated"):
        information = calibrated
    elif policy=="robust_dp":
        # Rectangular ambiguity set. Its lower endpoint also minimizes
        # post-screen validity because the detectable rate is held fixed.
        information = tuple(replace(c,valid=max(.02,c.valid-.2)) for c in plans)
    else:
        information = plans
    states,remaining,trace = [UNKNOWN]*3,budget,[]
    executors = {"learned":"learned","quoted_dp":"ex_ante_dp","calibrated_dp":"ex_ante_dp",
                 "calibrated_greedy":"greedy_value","robust_dp":"ex_ante_dp"}
    for _ in range(7):
        before = tuple(states)
        query = (risk_greedy(information,before,remaining) if policy=="risk_greedy" else
                 select(executors[policy],model,information,before,remaining))
        row = dict(raw_query=query,budget=remaining,states=states.copy())
        trace.append(row)
        if query==0:
            break
        available = {q:(i,c) for q,i,c in actions(plans,states,remaining)}
        if query not in available:
            row["denied"] = True
            break
        i,cost = available[query]
        status = (REJECTED if kinds[i]==1 else SCREENED) if query%2 else (
            CERTIFIED if kinds[i]==0 else REJECTED)
        states[i] = status
        remaining -= cost
        row.update(result=status,charged=cost)
    gross = earned(plans,tuple(states))
    assert all(kinds[i]==0 for i,s in enumerate(states) if s==CERTIFIED)
    net = gross-(budget-remaining)
    oracle = max([0]+[p.payoff-p.proof for p,k in zip(plans,kinds) if k==0 and p.proof<=budget])
    assert net<=oracle
    return dict(net=net,regret=oracle-net,trace=trace,missed=int(gross==0 and oracle>0),
                certified=int(gross>0),denied=sum("denied" in row for row in trace),
                cost=budget-remaining)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model_path = V59/"policy.pt"
    model = torch.jit.load(str(model_path))
    schedule = V59/"outcomes.json"
    archive = json.loads(schedule.read_bytes())
    rng = Random(445818)
    summary,rows = {},[]
    for fixture in archive["rows"]:
        plans = tuple(Candidate(**p) for p in fixture["candidates"])
        draws = [rng.random() for _ in plans]
        top = max(range(3),key=lambda i:plans[i].payoff)
        for regime in REGIMES:
            def shifted(i,p):
                delta = dict(accurate=0,optimistic=-.2,pessimistic=.2,
                             rank_shift=-.35 if i==top else .12)[regime]
                return min(1-p.detectable,max(.02,p.valid+delta))
            calibrated = tuple(replace(p,valid=shifted(i,p)) for i,p in enumerate(plans))
            kinds = [0 if draw<p.valid else 1 if draw<p.valid+p.detectable else 2
                     for draw,p in zip(draws,calibrated)]
            row = dict(domain=fixture["domain"],split=fixture["split"],case=fixture["case"],
                       budget=fixture["budget"],regime=regime,kinds=kinds,runs={})
            for policy in POLICIES:
                run = attempt(model,plans,calibrated,kinds,row["budget"],policy)
                row["runs"][policy] = run
                counts = summary.setdefault(f"{regime}/{policy}",Counter())
                counts.update({key:value for key,value in run.items() if key!="trace"})
            rows.append(row)
    result = dict(seed=445818,policy_sha256=digest(model_path.read_bytes()),
                  schedule_sha256=digest(schedule.read_bytes()),summary=summary,rows=rows)
    path = OUT/"prior-stress.json"
    if args.write:
        path.write_bytes(canonical(result)+b"\n")
    else:
        assert result==json.loads(path.read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
