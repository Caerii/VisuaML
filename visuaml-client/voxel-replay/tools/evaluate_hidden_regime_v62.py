"""Counterfactual and multi-seed cross-task evidence evaluation."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from random import Random
from statistics import mean
from time import perf_counter_ns

import torch

from frozen_call_checkpoint import canonical, digest
from hidden_regime_v62 import OUT, TICKS, WIDTH, features

SEEDS = (721,722)
P_LOW,P_HIGH = .16,.84


def posterior(good,bad):
    odds = (good-bad)*math.log(P_HIGH/P_LOW)
    return P_LOW+(P_HIGH-P_LOW)/(1+math.exp(-odds))


def belief_action(gross,cost,budget,horizon,good,bad,robust=False):
    @lru_cache(None)
    def value(slots,left,wins,losses):
        if slots<=0 or left<=0:
            return 0.,0
        p = max(0.,posterior(wins,losses)-(.12 if robust else 0))
        work = p*gross-cost+p*value(slots-1,left-1,wins+1,losses)[0]+(
            1-p)*value(slots-1,left-1,wins,losses+1)[0]
        return (work,1) if work>0 else (0.,0)
    return value(horizon,budget//cost,good,bad)[1]


class History:
    def __init__(self,model,mode):
        self.model,self.mode = model,mode
        self.tokens = torch.zeros(1,TICKS,WIDTH)
        self.missing = torch.ones(1,TICKS,dtype=torch.bool)

    def action(self,gross,cost,budget,last,cached):
        current = features(*map(torch.tensor,([gross],[cost],[budget],[last],[cached])))
        if self.mode=="reset":
            self.tokens.zero_()
            self.missing.fill_(True)
        with torch.inference_mode():
            logits,memory = self.model(current,self.tokens,self.missing)
            if self.mode=="history":
                self.tokens = torch.cat((self.tokens[:,1:],self.model.project(current).unsqueeze(1)),1)
                self.missing = torch.cat((self.missing[:,1:],torch.zeros(1,1,dtype=torch.bool)),1)
            elif self.mode=="recurrent":
                self.tokens = torch.cat((self.tokens[:,:-1],memory.unsqueeze(1)),1)
                self.missing[:,-1] = False
        return int(logits[0,0]>=0)


def run(task,method,model=None,mode=None,evaluate=None):
    gross,cost,budget = task["gross"],task["cost"],task["budget"]
    good,bad = task["pilots"].count(1),task["pilots"].count(-1)
    runner = History(model,mode) if model is not None else None
    # Observational prefix is common to all methods and already acquired.
    if runner:
        for outcome in task["pilots"][:-1]:
            runner.action(gross,cost,budget,outcome,False)
    last = task["pilots"][-1]
    trace,net,spend = [],0,0
    start = perf_counter_ns()
    for t,(valid,cached) in enumerate(zip(task["outcomes"],task["cache"],strict=True)):
        begun = perf_counter_ns()
        if runner:
            raw = runner.action(gross,cost,budget,last,cached)
        elif method in ("bayes_dp","robust_dp"):
            horizon = TICKS-t
            raw = belief_action(gross,cost,budget,horizon,good,bad,method=="robust_dp")
        elif method=="risk_greedy":
            p = posterior(good,bad)
            raw = int(gross*(p-.3*p*(1-p))>cost)
        elif method=="quoted":
            raw = int(gross*.5>cost)
        else:
            raw = int(gross*task["truth"][t]>cost)
        decision_ns = perf_counter_ns()-begun
        # Cached authority comes from a prevalidated certificate in this
        # accounting model; otherwise the typed proof must execute.
        query = raw and cost<=budget and not cached
        checked = evaluate(t,cached) if evaluate is not None and (query or cached) else valid
        assert checked==valid
        authorized = bool(cached or query and checked)
        assert not cached or valid
        assert not authorized or valid
        if query:
            budget -= cost
            spend += cost
            good += int(valid)
            bad += int(not valid)
        net += gross*authorized-cost*query
        last = (1 if valid else -1) if query else 0
        trace.append(dict(raw=raw,proof=bool(query),cached=cached,authorized=authorized,
                          reply=last,budget=budget,decision_ns=decision_ns))
    total = perf_counter_ns()-start
    for entry in trace:
        entry.pop("decision_ns")
    best = sum(task["cache"])*gross + min(task["budget"]//cost,sum(
        valid and not cached for valid,cached in zip(task["outcomes"],task["cache"])))*max(0,gross-cost)
    assert net<=best and spend<=task["budget"]
    return dict(net=net,regret=best-net,trace=trace,spent=spend),total


def tasks():
    rng = Random(37862)
    for family in ("stable","weaker_signal","regime_switch"):
        for index in range(80):
            strength = .84 if family!="weaker_signal" else .67
            p = strength if index%2 else 1-strength
            truth = [p if family!="regime_switch" or t<6 else 1-p for t in range(TICKS)]
            outcomes = [rng.random()<prob for prob in truth]
            cache = [valid and rng.random()<.12 for valid in outcomes]
            yield dict(family=family,index=index,domain="compiler" if index%4<2 else "world",
                       gross=rng.randint(13,35),cost=rng.randint(4,14),budget=rng.randint(25,62),
                       pilots=[1 if rng.random()<p else -1 for _ in range(3)],
                       truth=truth,outcomes=outcomes,cache=cache)


def replay_live(task,models,evidence):
    from evaluate_reusable_evidence_v59 import Task
    from reusable_evidence_v59 import EvidenceStore
    runtime = Task()
    rng = Random(671)
    rows = []
    for schedule in task:
        domain = schedule["domain"]
        fixtures = [runtime.input(domain,"valid" if valid else "masked",t,rng)
                    for t,valid in enumerate(schedule["outcomes"])]
        runs = {}
        for name,model,mode in (("bayes_dp",None,None),("recurrent",models["recurrent",721],"recurrent")):
            cache = EvidenceStore()
            for t,previous in enumerate(schedule["cache"]):
                if previous:
                    scope,execute = runtime.evidence(*fixtures[t],False)
                    assert cache.query(scope,execute)[0]
            def execute(t,cached):
                start = perf_counter_ns()
                scope,query = runtime.evidence(*fixtures[t],False)
                evidence[f"{domain}/{name}/scope_ns"].append(perf_counter_ns()-start)
                start = perf_counter_ns()
                value,hit = cache.query(scope,query)
                evidence[f"{domain}/{name}/evidence_ns"].append(perf_counter_ns()-start)
                assert not cached or hit
                return value
            result,timing = run(schedule,name,model,mode,execute)
            evidence[f"{domain}/{name}/total_ns"].append(timing)
            runs[name] = result
        rows.append(dict(domain=domain,case=schedule["index"],fixtures=[f[1] for f in fixtures],runs=runs))
    return rows


def contrasts(models):
    rows = []
    for (kind,seed),model in models.items():
        positive,negative,shuffled,changed = 0,0,0,0
        for gross in range(14,31):
            for cost in range(5,13):
                flags = []
                for pilots in ([1,1,-1],[-1,-1,-1],[1,-1,-1],[-1,1,-1]):
                    history = History(model,kind)
                    for outcome in pilots[:-1]:
                        history.action(gross,cost,30,outcome,False)
                    flags.append(history.action(gross,cost,30,pilots[-1],False))
                positive += flags[0]
                negative += flags[1]
                changed += flags[0]!=flags[1]
                shuffled += flags[2]!=flags[3]
        rows.append(dict(kind=kind,seed=seed,paired_inputs=17*8,positive_proof=positive,
                         negative_proof=negative,history_switches=changed,shuffle_switches=shuffled))
    assert posterior(2,1)>.5 and posterior(0,3)<.5
    return rows


def interval(values,seed):
    rng = Random(seed)
    boot = sorted(mean(rng.choices(values,k=len(values))) for _ in range(1000))
    return [boot[25],boot[974]]


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {(kind,seed):OUT/f"belief-{kind}-{seed}.pt" for kind in ("recurrent","history") for seed in SEEDS}
    models = {key:torch.jit.load(str(path)) for key,path in paths.items()}
    summary,elapsed,rows,diffs = {},{},[],{}
    for task in tasks():
        runs = {}
        for name in ("bayes_dp","robust_dp","risk_greedy","quoted","regime_oracle"):
            runs[name],timing = run(task,name)
            elapsed.setdefault(f"{task['family']}/{name}",[]).append(timing)
        for (kind,seed),model in models.items():
            for mode in ((kind,"reset") if kind=="recurrent" else (kind,)):
                name = f"{mode}-{seed}"
                runs[name],timing = run(task,name,model,mode)
                elapsed.setdefault(f"{task['family']}/{name}",[]).append(timing)
        for name,result in runs.items():
            key = f"{task['family']}/{name}"
            summary.setdefault(key,Counter()).update(net=result["net"],regret=result["regret"],
                certified=sum(r["authorized"] for r in result["trace"]),
                proofs=sum(r["proof"] for r in result["trace"]),cost=result["spent"],assigned=1)
            diffs.setdefault(key,[]).append(result["net"]-runs["bayes_dp"]["net"])
        rows.append(dict(task=task,runs=runs))
    live_times = defaultdict(list)
    live = replay_live([entry["task"] for entry in rows[:4]],models,live_times)
    result = dict(hashes={f"{k}-{s}":digest(p.read_bytes()) for (k,s),p in paths.items()},live=live,
                  summary=summary,rows=rows,counterfactuals=contrasts(models),
                  bootstrap_vs_bayes={key:interval(values,i) for i,(key,values) in enumerate(diffs.items())})
    path = OUT/"hidden-regime.json"
    if args.write:
        path.write_bytes(canonical(result)+b"\n")
        (OUT/"hidden-timings.json").write_bytes(canonical(dict(accounting=elapsed,live=dict(live_times)))+b"\n")
    else:
        assert canonical(result)==canonical(json.loads(path.read_bytes()))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
