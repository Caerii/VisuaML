"""Conservative online deployment with separate performance and proof identities."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from math import ceil
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from frozen_call_checkpoint import canonical, digest
from measure_uncertainty_v69 import METHOD, OUT, VARIANTS
from optimization_histories_v68 import admit, binary
from persistent_evidence_v60 import Node, Store

METHODS = ("greedy","bayes","robust","dp","beam","mpc","fixed_measure","adaptive_measure","oracle")


def saving(row,mode,series="samples"):
    batches = row[series]
    return median(a-b for a,b in zip(batches["reload"],batches[mode],strict=True))/1000


def group(row):
    return row["hardware_id"],row["width"],row["profile"]


def quantile(values,q):
    return sorted(values)[min(len(values)-1,max(0,ceil(q*len(values))-1))]


def proof_key(row,mode):
    return tuple(row["keys"]),mode


def installation(row,mode):
    return proof_key(row,mode),row["hardware_id"]


def stage_cost(row,mode,key):
    return row["stages"][mode][key+"_ns"]/1000


@dataclass(frozen=True)
class Forecast:
    center: float
    radius: float
    proof: float
    reuse: float
    deploy: float
    proof_upper: float
    deploy_upper: float
    evidence: tuple
    scope: tuple

    @property
    def limits(self):
        return self.center-self.radius,self.center+self.radius


class Calibration:
    def __init__(self,rows):
        training = defaultdict(list)
        for row in rows:
            if row["split"]=="train":
                training[group(row)].append(row)
        residual = defaultdict(list)
        for row in rows:
            if row["split"]=="calibrate":
                for mode in VARIANTS:
                    values = training[group(row)]
                    center = median(saving(x,mode) for x in values)
                    residual[mode].append(abs(saving(row,mode)-center))
        self.bench = {scope:median(row["quick_ns"]/1000 for row in cohort)
                      for scope,cohort in training.items()}
        self.options = {}
        for scope,values in training.items():
            for mode in VARIANTS:
                self.options[scope,mode] = Forecast(
                    median(saving(row,mode) for row in values),quantile(residual[mode],.8),
                    median(stage_cost(row,mode,"verification") for row in values),
                    median(stage_cost(row,mode,"retrieval") for row in values),
                    median(stage_cost(row,mode,"deployment") for row in values),
                    max(stage_cost(row,mode,"verification") for row in values),
                    max(stage_cost(row,mode,"deployment") for row in values),
                    tuple(digest(canonical(row)) for row in rows
                          if row["split"]=="calibrate" or row in values),scope)

    def for_row(self,row,mode):
        return self.options[group(row),mode]


def require_scope(quote,row):
    if quote.scope!=group(row):
        raise ValueError("performance quote belongs to another execution environment")


def observation_key(row):
    return (tuple(row["keys"]),row["hardware_id"],row["profile"],row["distribution"])


def campaigns(rows):
    rng = Random(696711)
    held = [row for row in rows if row["split"]=="holdout"]
    for family in ("steady","footprint_shift","compiler_shift"):
        for case in range(8):
            current = rng.choice(held)
            revisions = []
            for i in range(6):
                if i and family!="steady":
                    key = current["keys"]
                    options = [row for row in held if row["keys"]==key and
                               (row["hardware_id"]==current["hardware_id"] if family=="footprint_shift"
                                else row["profile"]==current["profile"])]
                    current = rng.choice(options)
                uses = rng.randrange(500,2500) if (i+case)%3==0 else rng.randrange(7000,44000)
                ratio = rng.choice((.67,.8,1.,1.27,1.55))
                if family=="compiler_shift" and (i+case)%4==0:
                    ratio = .48 if i%2 else 1.9
                hint = round(uses*ratio)
                revisions.append(dict(observation=current["id"],uses=uses,hint=hint))
            yield family,case,revisions


def priced(row,mode,proved,active,quote=None,robust=False):
    if quote:
        proof = quote.reuse if proof_key(row,mode) in proved else quote.proof_upper if robust else quote.proof
        deployment = 0 if installation(row,mode)==active else quote.deploy_upper if robust else quote.deploy
    else:
        proof = stage_cost(row,mode,"retrieval" if proof_key(row,mode) in proved else "verification")
        deployment = 0 if installation(row,mode)==active else stage_cost(row,mode,"deployment")
    return proof+deployment


def result(row,mode,uses,proved,active):
    if mode=="keep":
        return 0,proved,None,0
    cost = priced(row,mode,proved,active)
    return uses*saving(row,mode)-cost,proved|{proof_key(row,mode)},installation(row,mode),cost


def clairvoyant(tasks,rows):
    @lru_cache(None)
    def dp(i,proved,active):
        if i==len(tasks):
            return 0,()
        task = tasks[i]
        row = rows[task["observation"]]
        best = (-float("inf"),())
        for action in ("keep",*VARIANTS):
            gain,then,installed,_ = result(row,action,task["uses"],proved,active)
            suffix,path = dp(i+1,then,installed)
            if gain+suffix>best[0]:
                best = gain+suffix,(action,*path)
        return best
    return dp(0,frozenset(),None)


def expected_profit(row,mode,quote,uses,proved,active,obs,method):
    if mode=="keep":
        return 0.
    require_scope(quote,row)
    if obs is None:
        slower,faster = quote.limits
        signal = quote.center
    else:
        signal = obs[mode]
        # Independent quick batches inform the mean, with a retained noise allowance.
        slower,faster = signal-quote.radius/2,signal+quote.radius/2
    if method=="robust":
        count = max(0,uses-.5)/1.55 if slower>=0 else (uses+.5)/.67
        return count*slower-priced(row,mode,proved,active,quote,True)
    if method=="bayes":
        points = [uses*s*ratio-priced(row,mode,proved,active,quote)
                  for s in (slower,signal,faster) for ratio in (.75,1.,1.26)]
        # Risk-sensitive Bayesian expected utility under the discrete quote prior.
        return mean(gain if gain>=0 else gain*1.8 for gain in points)
    return uses*signal-priced(row,mode,proved,active,quote)


def choose(row,uses,cal,proved,active,obs,method,i,repeat):
    def value(mode,proofs,installed):
        quote = None if mode=="keep" else cal.for_row(row,mode)
        return expected_profit(row,mode,quote,uses,proofs,installed,obs,method)
    modes = ("keep",*VARIANTS)
    if method not in ("dp","beam","mpc","adaptive_measure"):
        return max(modes,key=lambda mode:value(mode,proved,active))
    horizon = min(3 if method=="beam" else 2,6-i)
    frontier = [(0,(),proved,active)]
    for k in range(horizon):
        expanded = []
        for net,path,p,a in frontier:
            for mode in modes:
                candidate = p if mode=="keep" else p|{proof_key(row,mode)}
                installed = None if mode=="keep" else installation(row,mode)
                expanded.append((net+(repeat**k)*value(mode,p,a),(*path,mode),candidate,installed))
        frontier = sorted(expanded,key=lambda item:-item[0])[:4 if method=="beam" else len(expanded)]
    return frontier[0][1][0]


def info_value(row,uses,cal,proved,active,repeat):
    estimates = []
    for direction in (-1,1):
        alternatives = [0]
        for mode in VARIANTS:
            quote = cal.for_row(row,mode)
            alternatives.append(uses*(quote.center+direction*quote.radius)-priced(row,mode,proved,active,quote))
        estimates.append(alternatives)
    return repeat*(mean(max(alt) for alt in estimates)-max(mean(v) for v in zip(*estimates,strict=True)))


def live_policy(tasks,rows,method,cal,path,admitted):
    proved,active = frozenset(),None
    observed,events,elapsed = {},[],[]
    factors = []
    for i,task in enumerate(tasks):
        start = perf_counter_ns()
        row = rows[task["observation"]]
        tag = observation_key(row)
        uses = task["hint"] if method!="mpc" or not factors else task["hint"]*median(factors)
        repeat = .65
        if i:
            previous = rows[tasks[i-1]["observation"]]
            repeat = .8 if observation_key(previous)==tag else .35
        observation = observed.get(tag)
        raw = path[i] if method=="oracle" else choose(row,uses,cal,proved,active,observation,method,i,repeat)
        cost = 0.
        counterfactual = raw
        alternate_net = result(row,raw,task["uses"],proved,active)[0]
        benchmark = method=="fixed_measure" and observation is None and i%2==0 or (
            method=="adaptive_measure" and observation is None and
            info_value(row,uses,cal,proved,active,repeat)>(
                cal.bench[group(row)]+sum(priced(row,mode,proved,None,cal.for_row(row,mode)) for mode in VARIANTS)))
        if benchmark:
            for mode in VARIANTS:
                assert row["powers"][mode]["root"] in admitted
                cost += priced(row,mode,proved,None)
                proved |= {proof_key(row,mode)}
            cost += row["quick_ns"]/1000
            observed[tag] = {mode:saving(row,mode,"quick") for mode in VARIANTS}
            raw = choose(row,uses,cal,proved,active,observed[tag],method,i,repeat)
        quote = None if raw=="keep" else cal.for_row(row,raw)
        interval = None if quote is None else quote.limits
        expected = expected_profit(row,raw,quote,uses,proved,active,observed.get(tag),method)
        reused = raw!="keep" and proof_key(row,raw) in proved
        if raw!="keep":
            assert row["powers"][raw]["root"] in admitted
        earned,proved,active,charge = result(row,raw,task["uses"],proved,active)
        actual_saving = saving(row,raw) if raw!="keep" else 0.
        elapsed.append(perf_counter_ns()-start)
        events.append(dict(observation=row["id"],forecast_scope=row["hardware_id"],action=raw,
            modeled_uses=round(uses,2),actual_uses=task["uses"],expected_net_us=round(expected-cost,3),
            realized_net_us=round(earned-cost,3),charges_us=round(charge+cost,3),
            observed_saving_us=round(actual_saving,6),interval_us=interval,benchmark=bool(benchmark),
            without_benchmark=counterfactual,
            benchmark_increment_us=round(earned-cost-alternate_net,3) if benchmark else None,
            reused=bool(reused),
            root=None if raw=="keep" else row["powers"][raw]["root"]))
        factors.append(task["uses"]/task["hint"])
    return events,elapsed


def import_proofs(archive):
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for key,row in archive["nodes"].items():
            node = Node(row["kind"],row["payload"],tuple(row["parents"]))
            if any(p not in archive["nodes"] for p in node.parents) or store.add(node)!=key:
                raise ValueError("broken proof provenance")
        admitted,cache = {},set()
        for root in archive["roots"]:
            raw = admit(store,root["root"],root["keys"],root["mode"],cache)
            admitted[root["root"]] = digest(raw)
        sample = archive["roots"][0]
        certificate = store.load(sample["root"])
        forged = store.add(Node(certificate.kind,dict(certificate.payload,authority="migration"),certificate.parents))
        try:
            admit(store,forged,sample["keys"],sample["mode"],set())
        except ValueError:
            pass
        else:
            raise AssertionError("forged authority accepted")
    return admitted


def summarize(rows,cal):
    calibrated,transfer = defaultdict(list),[]
    for row in rows:
        if row["split"]!="holdout":
            continue
        for mode in VARIANTS:
            quote = cal.for_row(row,mode)
            error = saving(row,mode)-quote.center
            tag = f"{row['hardware']['optimization']}/{mode}"
            calibrated[tag].append(abs(error)<=quote.radius)
            calibrated["all"].append(abs(error)<=quote.radius)
            other = next(item for item in rows if item["split"]=="train" and item["width"]==row["width"]
                         and item["profile"]==row["profile"] and item["hardware_id"]!=row["hardware_id"])
            stale = cal.for_row(other,mode)
            try:
                require_scope(stale,row)
            except ValueError:
                pass
            else:
                raise AssertionError("stale performance estimate admitted across compiler settings")
            transfer.append(dict(observation=row["id"],mode=mode,cross_environment_allowed=False,
                                 naive_covered=abs(saving(row,mode)-stale.center)<=stale.radius))
    return {key:dict(covered=sum(flags),total=len(flags)) for key,flags in calibrated.items()},transfer


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-only",action="store_true")
    args = parser.parse_args()
    performance = OUT/"performance.json"
    proof_path = OUT/"proofs.json"
    values = json.loads(performance.read_bytes())
    admitted = import_proofs(json.loads(proof_path.read_bytes()))
    if args.import_only:
        print(f"Imported {len(admitted)} exact roots")
        return
    for row in values:
        if digest(canonical(row["hardware"]))!=row["hardware_id"] or row["method"]!=METHOD:
            raise ValueError("performance scope provenance differs")
        for mode in VARIANTS:
            info = row["powers"][mode]
            assert admitted[info["root"]]==info["binary"]==digest(binary(row["keys"],mode))
    cal = Calibration(values)
    data,walls,counts = [],[],defaultdict(Counter)
    for family,index,campaign in campaigns(values):
        best,oracle = clairvoyant(campaign,values)
        runs,timings = {},{}
        for method in METHODS:
            events,wall = live_policy(campaign,values,method,cal,oracle,admitted)
            value = sum(row["realized_net_us"] for row in events)
            assert value<=best+.01
            row = dict(events=events,net_us=round(value,3),regret_us=round(best-value,3))
            runs[method],timings[method] = row,wall
            counts[family+"/"+method].update(assigned=1,
                net_us=row["net_us"],regret_us=row["regret_us"],
                losses=sum(e["realized_net_us"]<0 for e in events),
                benchmarks=sum(e["benchmark"] for e in events),
                benchmark_choice_changes=sum(e["benchmark"] and e["action"]!=e["without_benchmark"] for e in events),
                deployed=sum(e["action"]!="keep" for e in events),
                reused=sum(e["reused"] for e in events))
        data.append(dict(family=family,index=index,oracle_us=round(best,3),revisions=campaign,runs=runs))
        walls.append(timings)
    coverage,transfer = summarize(values,cal)
    rng = Random(691)
    intervals = {}
    for method in METHODS:
        delta = [r["runs"][method]["net_us"]-r["runs"]["greedy"]["net_us"] for r in data]
        boot = sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(1200))
        intervals[method] = [boot[30],boot[1169]]
    cost_checks,usage_checks,revisit = Counter(),Counter(),defaultdict(list)
    for row in values:
        if row["split"]=="holdout":
            for mode in VARIANTS:
                quote = cal.for_row(row,mode)
                cost_checks.update(total=1,proof_upper=int(stage_cost(row,mode,"verification")<=quote.proof_upper),
                                  deploy_upper=int(stage_cost(row,mode,"deployment")<=quote.deploy_upper))
    for history in data:
        before = None
        for current in history["revisions"]:
            usage_checks.update(total=1,contained=int((current["hint"]-.5)/1.55<=current["uses"]<=(current["hint"]+.5)/.67))
            if before is not None:
                revisit[history["family"]].append(
                    observation_key(values[before["observation"]])==observation_key(values[current["observation"]]))
            before = current
    output = dict(cost_coverage=cost_checks,usage_assumption_coverage=usage_checks,
        scope_reuse={key:dict(matches=sum(flags),total=len(flags)) for key,flags in revisit.items()},
        performance_sha256=digest(performance.read_bytes()),proofs_sha256=digest(proof_path.read_bytes()),
        coverage=coverage,portability=transfer,summary=counts,rows=data,paired_vs_greedy_us=intervals,
        trained_forecasts=[dict(mode=mode,**vars(quote)) for (_,mode),quote in sorted(cal.options.items())])
    if args.write:
        (OUT/"outcomes.json").write_bytes(canonical(output)+b"\n")
        (OUT/"policy-timings.json").write_bytes(canonical(walls)+b"\n")
    else:
        assert canonical(output)==canonical(json.loads((OUT/"outcomes.json").read_bytes()))
    print(f"Replayed {len(data)} uncertainty histories with {len(admitted)} independently admitted roots")


if __name__=="__main__":
    main()
