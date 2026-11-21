"""Replay actual validated deployments and paired online economic comparisons."""
from argparse import ArgumentParser
from collections import defaultdict
import ctypes
from itertools import product
import json
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from evaluate_dependency_v65 import expected, native_c
from frozen_call_checkpoint import canonical, digest
from incremental_optimization_v67 import Campaign
from optimization_histories_v68 import (
    OUT, MODES, Authority, Branch, admit, advance, binary, decisions, optimize, pointer_binary, pointer_node, pointer_proof,
)
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from verify_dependency_v65 import PAGE, SCRATCH

METHODS = ("greedy","lookahead","beam","value_of_information","learned-681","learned-682","clairvoyant")
BENCHMARK_ESTIMATE_US = 650


def measure(fn):
    start = perf_counter_ns()
    value = fn()
    return value,perf_counter_ns()-start


def calibration():
    rng = Random(341068)
    rows = []
    for width,profile in product((8,16),("short","full")):
        keys = [1 if i%5==0 else rng.randrange(2,12) for i in range(width)]
        n = 48 if profile=="short" else SCRATCH//(4*width)
        initial = rng.randbytes(PAGE)
        funcs,init = {},{}
        engine = wasmtime.Engine()
        for mode in ("reload",*MODES):
            start = perf_counter_ns()
            room = wasmtime.Store(engine)
            exports = wasmtime.Instance(room,wasmtime.Module(engine,binary(keys,mode)),[]).exports(room)
            init[mode] = perf_counter_ns()-start
            fn,memory = exports["run"],exports["memory"]
            memory.write(room,initial)
            assert fn(room,n)==n and memory.read(room,0,PAGE)==expected(keys,n,initial)[1]
            funcs[mode] = lambda f=fn,r=room: f(r,n)
        with TemporaryDirectory() as folder:
            kernel,init["native"],compiler = native_c(folder,keys)
            page = (ctypes.c_uint32*(PAGE//4)).from_buffer_copy(initial)
            assert kernel(page,n)==n and bytes(page)==expected(keys,n,initial)[1]
            funcs["native"] = lambda: kernel(page,n)
            samples = defaultdict(list)
            for _ in range(24):
                branches = sorted(funcs)
                rng.shuffle(branches)
                for name in branches:
                    fn = funcs[name]
                    assert fn()==n
                    start = perf_counter_ns()
                    for _ in range(64):
                        fn()
                    samples[name].append((perf_counter_ns()-start)/64)
        rows.append(dict(width=width,profile=profile,keys=keys,count=n,compiler=compiler,
                         setup_ns=init,run_ns=dict(samples)))
    return rows


def deployment(raw,keys,profile):
    engine = wasmtime.Engine()
    room = wasmtime.Store(engine)
    exports = wasmtime.Instance(room,wasmtime.Module(engine,raw),[]).exports(room)
    n = 48 if profile=="short" else SCRATCH//(4*len(keys))
    page = Random(9843).randbytes(PAGE)
    exports["memory"].write(room,page)
    assert exports["run"](room,n)==n
    assert exports["memory"].read(room,0,PAGE)==expected(keys,n,page)[1]
    return exports["run"],room,n


def histories():
    rng = Random(7841602)
    for kind in ("repeat","bursty","shift"):
        for index in range(5):
            width = 8 if index%2==0 else 16
            keys = [rng.randint(1,12) for _ in range(width)]
            original = keys.copy()
            episode = []
            hidden = rng.choice(("short","full"))
            for t in range(6):
                if t and (kind=="shift" or kind=="bursty" and t in (2,3,5) or kind=="repeat" and t==2):
                    keys = original.copy() if t==5 and kind=="bursty" else keys.copy()
                    for _ in range(width//3 if kind=="shift" else 1):
                        keys[rng.randrange(width)] = rng.randint(1,12)
                if rng.random()<(.55 if kind=="shift" else .22):
                    hidden = "full" if hidden=="short" else "short"
                described = hidden if rng.random()<.66 else ("full" if hidden=="short" else "short")
                uses = rng.randrange(150,2400) if (t+index)%3==0 else rng.randrange(3700,15000)
                episode.append(dict(keys=keys.copy(),uses=uses,budget=rng.randint(4200,10200),
                                    profile=hidden,hint=described))
            yield kind,index,episode


def forecast(task,table,profile):
    return table[len(task["keys"]),profile]


def remaining_guess(task,repeat_chance):
    result = dict(task)
    if repeat_chance<.47:
        result["keys"] = [(task["keys"][0]%12)+1,*task["keys"][1:]]
    return result


def choose(method,task,quotes,state,recent,models):
    known,proven,installed = state
    if method=="greedy":
        return optimize([task],[quotes],known,proven,installed)[1][0]
    chance = (1+sum(recent))/(2+len(recent))
    if method.startswith("learned"):
        with torch.inference_mode():
            chance = float(models[method](torch.tensor([recent])).sigmoid().flatten()[0])
    next_ = remaining_guess(task,chance)
    path = [task,next_,next_] if method=="beam" else [task,next_]
    return optimize(path,[quotes]*len(path),known,proven,installed,
                    beam=4 if method=="beam" else None)[1][0]


def information_value(task,table,state,recent):
    """A bounded one-revisit baseline comparing expected actions with perfect profile information."""
    options = list(decisions(task,*state))
    profiles = [forecast(task,table,profile) for profile in ("short","full")]
    payoff = [[advance(task,act,mode,*state,estimate.get(mode,0))[0] for act,mode in options]
              for estimate in profiles]
    informed = mean(max(scores) for scores in payoff)
    uninformed = max(mean(scores) for scores in zip(*payoff,strict=True))
    repeat = (1+sum(recent))/(2+len(recent))
    return repeat*(informed-uninformed)


def live_run(method,episode,table,models,authority,store,full_roots,oracle_path):
    recent = [1.,1.,0.,0.]
    state = frozenset(),frozenset(),None
    events,wall = [],[]
    previous = mode_used = None
    branch = Branch(store)
    measured_profiles = {}
    resident = None
    for t,task in enumerate(episode):
        revision,assembly = measure(lambda: branch.edit(task["keys"]))
        if previous is not None:
            recent = [*recent[1:],float(previous==task["keys"])]
        previous = task["keys"]
        signature = tuple(task["keys"])
        profile = measured_profiles.get(signature,task["hint"])
        quotes = forecast(task,table,profile)
        actual = forecast(task,table,task["profile"])
        (action,mode),planning = measure(lambda: oracle_path[t] if method=="clairvoyant"
                                         else choose(method,task,quotes,state,recent,models))
        # Measurement is a separately offered action. It trades this
        # revision's deployment against a noisy profile prediction later.
        benchmarked = method in ("beam","learned-681","learned-682") and action=="keep" and t<len(episode)-1 and (
            task["hint"]=="short" and sum(recent)>=2 and signature not in measured_profiles)
        if method=="value_of_information":
            benchmarked = (action=="keep" and t<len(episode)-1 and signature not in measured_profiles
                           and information_value(task,table,state,recent)>BENCHMARK_ESTIMATE_US)
        costs = dict(assembly_ns=assembly,planner_ns=planning,verification_ns=0,deploy_ns=0,benchmark_ns=0,native_call_ns=0)
        event = dict(revision=revision,action=action,mode=mode,quote_profile=profile)
        if benchmarked:
            measured_profiles[signature] = task["profile"]
            # Reproduce a real sample from the currently revealed profile.
            start = perf_counter_ns()
            gold = binary(task["keys"],"reload")
            fn,env,n = deployment(gold,task["keys"],task["profile"])
            for _ in range(24):
                fn(env,n)
            costs["benchmark_ns"] = perf_counter_ns()-start
            event.update(action="benchmark",realized=-BENCHMARK_ESTIMATE_US,predicted=-BENCHMARK_ESTIMATE_US,
                         charge=BENCHMARK_ESTIMATE_US,root=None)
            state = state[0],state[1],None
            resident = mode_used = None
        else:
            predicted = advance(task,action,mode,*state,quotes.get(mode,0))
            result = advance(task,action,mode,*state,actual.get(mode,0))
            assert predicted is not None and result is not None
            modeled,_,_,_,_ = predicted
            realized,known,proven,installed,charge = result
            state = known,proven,installed
            root = None
            if action!="keep":
                (root,raw,work),costs["verification_ns"] = measure(lambda: authority.prove(task["keys"],mode))
                key = (tuple(task["keys"]),mode)
                if key not in full_roots:
                    # Fresh regeneration and independent revalidation are
                    # controls, kept outside the deployed path timing.
                    campaign = Campaign(store,False)
                    fresh,_,_ = campaign.revise(task["keys"],"rolling" if mode=="pointer" else mode)
                    if mode=="pointer":
                        fresh = store.add(pointer_node(task["keys"],fresh))
                    admit(store,fresh,task["keys"],mode,set())
                    full_roots[key] = fresh,mode,task["keys"]
                assert root==full_roots[key][0]
                event["work"] = work
                event["transition"] = store.add(Node("optimization-transition",
                    dict(action=action,mode=mode,binary=digest(raw)),(revision,root)))
                if action=="apply":
                    placement = (signature,mode)
                    if mode_used!=placement:
                        resident,costs["deploy_ns"] = measure(lambda: deployment(raw,task["keys"],task["profile"]))
                        mode_used = placement
                    fn,env,_ = resident
                    n = 48 if task["profile"]=="short" else SCRATCH//(4*len(task["keys"]))
                    returned,costs["native_call_ns"] = measure(lambda: fn(env,n))
                    assert returned==n
            if action!="apply":
                resident = mode_used = None
            event.update(root=root,realized=realized,predicted=modeled,charge=charge)
        assert event["charge"]<=task["budget"]
        events.append(event)
        wall.append(costs)
    return events,wall


def branch_check():
    keys = [2,3,5,7,11,8,4,1]
    cases = [keys,[9,*keys[1:]],[*keys[:3],2,*keys[4:]],keys]
    traces,wall = [],{}
    for sharing in (False,True):
        with TemporaryDirectory() as folder:
            store = Store(folder)
            root = Branch(store)
            ancestor = root.edit(keys)
            left,right = root.fork(),root.fork()
            renderer = Authority(store)
            steps = []
            for branch,layout,mode in zip((root,left,right,left),cases,
                                          ("rolling","pointer","specialized","rolling"),strict=True):
                active = branch.edit(layout)
                worker = renderer if sharing else Authority(store)
                (proof,_,work),total = measure(lambda: worker.prove(layout,mode))
                steps.append(dict(revision=active,binary_proof=proof,keys=layout,mode=mode,work=work))
                wall.setdefault("shared_ns" if sharing else "isolated_ns",[]).append(total)
            assert steps[0]["revision"]!=steps[-1]["revision"]
            assert steps[0]["binary_proof"]==steps[-1]["binary_proof"]
            try:
                left.rollback(right.head)
            except ValueError:
                pass
            else:
                raise AssertionError("sibling rollback accepted")
            left.rollback(ancestor)
            assert left.head==ancestor and right.head!=ancestor
            traces.append(steps)
            if sharing:
                nodes = {path.name:json.loads(path.read_bytes()) for path in store.root.iterdir()}
    assert sum(row["work"]["fresh"] for row in traces[1])<sum(row["work"]["fresh"] for row in traces[0])
    return dict(isolated=traces[0],shared=traces[1],sibling_rollback="rejected"),wall,nodes


def runtime_controls(store,authority):
    keys = [2,3,5,7,11,8,4,1]
    root,raw,_ = authority.prove(keys,"pointer")
    parent, = store.load(root).parents
    counter = pointer_proof(len(keys),True)
    assert counter and pointer_proof(len(keys)) is None
    false = store.add(pointer_node(keys,parent,wrong=True))
    claimed = store.load(parent)
    policy_changed = store.add(Node(claimed.kind,dict(claimed.payload,authority="checkpoint"),claimed.parents))
    other = store.add(pointer_node(keys,policy_changed))
    failures = []
    for name,r,k in (("wrong-stride",false,keys),("proof-policy",other,keys),("stale-lanes",root,keys[::-1])):
        try:
            admit(store,r,k,"pointer",set())
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError("invalid certificate granted authority")
    original = Random(884).randbytes(PAGE)
    wrong = wasmtime_run(pointer_binary(keys,True),2,bytearray(original))[1]
    correct = expected(keys,2,original)[1]
    first = next(i for i,(a,b) in enumerate(zip(wrong,correct,strict=True)) if a!=b)
    graph = torch.jit.load(str(EXECUTOR/"machine.pt"))
    records = []
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for n in (-1,0,1,2,SCRATCH//(4*len(keys))+1):
            memory,record = invoke(graph,raw,n,bytearray(original),fuel=2500)
            assert (record["return_i32"],memory)==expected(keys,n,original)==wasmtime_run(raw,n,bytearray(original))
            records.append(dict(n=n,return_value=record["return_i32"],page=digest(memory),steps=record["source_steps"]))
    return dict(failed=failures,counterexample=counter,first_wrong_byte=first,tensor=records,
                root=root,binary=digest(raw),keys=keys)


def migration_cost():
    from incremental_optimization_v67 import OUT as V67
    from migrate_snapshot_v67 import resume, step
    data = json.loads((V67/"migration.json").read_bytes())
    packet = data["saved"]
    original = Random(data["seed"]).randbytes(PAGE)
    keys,n = packet["layout"],packet["records"]
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for key,row in data["nodes"].items():
            assert store.add(Node(row["kind"],row["payload"],tuple(row["parents"])))==key
        samples = []
        fixed,reversed_ = [],[]
        for i in range(24):
            value,elapsed = measure(lambda: resume(packet,original,keys,n,packet["root"],store,data["total"]))
            assert list(value)==data["expected"]
            samples.append(elapsed)
            for kind in (("rolling","reverse") if i%2 else ("reverse","rolling")):
                state = dict(packet["state"])
                page = bytearray.fromhex(packet["page"])
                start = perf_counter_ns()
                while step(state,page,keys,n,kind):
                    pass
                (fixed if kind=="rolling" else reversed_).append(perf_counter_ns()-start)
                assert [state,digest(page)]==data["expected"]
    return dict(resume_ns=samples,rolling_suffix_ns=fixed,reverse_suffix_ns=reversed_,
                economically_justified=median(samples)<median(fixed))


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--calibrate",action="store_true")
    parser.add_argument("--import-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.import_bundle:
        data = json.loads((OUT/"proofs.json").read_bytes())
        with TemporaryDirectory() as folder:
            store = Store(folder)
            for key,r in data["nodes"].items():
                node = Node(r["kind"],r["payload"],tuple(r["parents"]))
                if any(p not in data["nodes"] for p in node.parents) or store.add(node)!=key:
                    raise ValueError("invalid DAG")
            cache = set()
            for row in data["roots"]:
                admit(store,row["root"],row["keys"],row["mode"],cache)
            roots = {row["root"]:row for row in data["roots"]}
            for key in data["transitions"]:
                record = store.load(key)
                revision,root = record.parents
                source = store.load(revision)
                declared = roots[root]
                if (record.kind!="optimization-transition" or source.kind!="revision" or
                    source.payload["keys"]!=declared["keys"] or record.payload["mode"]!=declared["mode"] or
                    record.payload["action"] not in ("apply","prove") or
                    record.payload["binary"]!=digest(binary(declared["keys"],declared["mode"]))):
                    raise ValueError("transition exceeds admitted scope")
            for row in data["branch_revisions"]:
                revision = store.load(row["revision"])
                if revision.kind!="revision" or revision.payload["keys"]!=row["keys"]:
                    raise ValueError("branch revision does not bind approved code")
                admit(store,row["binary_proof"],row["keys"],row["mode"],cache)
        print(f"Imported {len(data['roots'])} deployment roots and {len(data['branch_revisions'])} fork events")
        return
    if args.calibrate and not args.write:
        raise ValueError("calibration changes the replay identity, use --write")
    calibration_path = OUT/"calibration.json"
    if args.calibrate:
        calibration_path.write_bytes(canonical(calibration())+b"\n")
    measured = json.loads(calibration_path.read_bytes())
    table = {}
    for row in measured:
        base = median(row["run_ns"]["reload"])
        table[row["width"],row["profile"]] = {mode:round((base-median(row["run_ns"][mode]))/1000,3)
                                            for mode in MODES}
    paths = {f"learned-{seed}":OUT/f"reuse-{seed}.pt" for seed in (681,682)}
    models = {name:torch.jit.load(str(path)) for name,path in paths.items()}
    rows,timings,summary = [],[],defaultdict(list)
    with TemporaryDirectory() as folder:
        store = Store(folder)
        full_roots = {}
        for family,index,episode in histories():
            truth = [forecast(task,table,task["profile"]) for task in episode]
            best,path = optimize(episode,truth,frozenset(),frozenset(),None)
            record,clock = {},{}
            for name in METHODS:
                events,wall = live_run(name,episode,table,models,Authority(store),store,full_roots,path)
                realized = sum(e["realized"] for e in events)
                assert realized<=best+1e-6
                record[name] = dict(events=events,realized=round(realized,3),regret=round(best-realized,3))
                summary[family+"/"+name].append(realized)
                clock[name] = wall
            rows.append(dict(family=family,index=index,revisions=episode,oracle=round(best,3),runs=record))
            timings.append(clock)
        control = runtime_controls(store,Authority(store))
        full_roots[(tuple(control["keys"]),"pointer")] = control["root"],"pointer",control["keys"]
        roots = [dict(root=root,mode=mode,keys=keys) for root,mode,keys in full_roots.values()]
        transitions = sorted({event["transition"] for row in rows for run in row["runs"].values()
                              for event in run["events"] if "transition" in event})
        pending = [row["root"] for row in roots]+transitions
        nodes = {}
        while pending:
            key = pending.pop()
            if key not in nodes:
                node = store.load(key)
                nodes[key] = json.loads(node.bytes())
                pending.extend(node.parents)
    branch,branch_wall,branch_nodes = branch_check()
    nodes.update(branch_nodes)
    rng = Random(633)
    intervals = {}
    for name in METHODS:
        delta = [row["runs"][name]["realized"]-row["runs"]["greedy"]["realized"] for row in rows]
        samples = sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(1000))
        intervals[name] = [samples[25],samples[974]]
    workload_intervals = {}
    for profile in measured:
        key = f"{profile['width']}/{profile['profile']}"
        workload_intervals[key] = {}
        for mode in MODES:
            savings = [base-cost for base,cost in zip(profile["run_ns"]["reload"],
                                                     profile["run_ns"][mode],strict=True)]
            boot = sorted(mean(rng.choices(savings,k=len(savings))) for _ in range(1000))
            workload_intervals[key][mode] = [boot[25],boot[974]]
    calibration_errors = {name:[] for name in (*models,"empirical")}
    for row in rows:
        recent = [1.,1.,0.,0.]
        tasks = row["revisions"]
        for i,task in enumerate(tasks[:-1]):
            if i:
                recent = [*recent[1:],float(tasks[i-1]["keys"]==task["keys"])]
            observed = float(tasks[i+1]["keys"]==task["keys"])
            calibration_errors["empirical"].append(((1+sum(recent))/(2+len(recent))-observed)**2)
            with torch.inference_mode():
                for name,model in models.items():
                    forecast_ = float(model(torch.tensor([recent])).sigmoid().flatten()[0])
                    calibration_errors[name].append((forecast_-observed)**2)
    data = dict(rows=rows,control=control,branches=branch,total={key:round(sum(v),3) for key,v in summary.items()},
                paired_mean_vs_greedy_ci95_us=intervals,calibration_saving_ci95_ns=workload_intervals,
                recurrence_brier={name:mean(errors) for name,errors in calibration_errors.items()},
                calibration_sha256=digest(calibration_path.read_bytes()),
                models={name:digest(path.read_bytes()) for name,path in paths.items()})
    for name,value in (("histories.json",data),("proofs.json",dict(
            roots=roots,nodes=nodes,branch_revisions=branch["shared"],transitions=transitions))):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes())),name
    if args.write:
        (OUT/"timings.json").write_bytes(canonical(dict(online=timings,branches=branch_wall,
             migration=migration_cost()))+b"\n")
    print(f"Replayed {len(rows)} histories and {len(roots)} independently admitted binaries")


if __name__=="__main__":
    main()
