"""Replay repeated proof campaigns and measure verified deployment costs."""
from argparse import ArgumentParser
from collections import defaultdict
import ctypes
import json
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from dependent_synthesis_v65 import Space
from evaluate_dependency_v65 import expected, native_c
from frozen_call_checkpoint import canonical, digest
from incremental_optimization_v67 import OUT, Campaign, RevisionIndex, Slot, compile_stencil, verify
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from structural_decomposition_v66 import CachedTree, decomposition, graph, order
from verify_dependency_v65 import PAGE, SCRATCH


def revisions(family,width):
    rng = Random(617+width)
    keys = [rng.randint(1,12) for _ in range(width)]
    for i in range(16):
        if i:
            edit = range(width) if family=="global" else [(5*i)%width]
            for lane in edit:
                keys[lane] = (keys[lane]+rng.randrange(1,12)-1)%12+1
        yield keys.copy(), "rolling" if i%8<4 else "specialized"


def factor_adapter():
    base = Space.build(9,4,[(i,i+1) for i in range(8)])
    index = RevisionIndex()
    for i,(_,_,rows) in enumerate(base.factors):
        factor = Slot("inference","factor",i)
        index.get(factor,(),lambda: digest(canonical(rows)))
    # Cache dependencies for a query at the rightmost node.
    prior = ()
    for i in range(8):
        key = Slot("inference","message",i)
        index.get(key,(Slot("inference","factor",i),*prior),lambda: "typed-message")
        prior = (key,)
    cache = CachedTree(base)
    cache.marginal(8)
    changed = base.observe(6,0,5)
    affected = index.invalidate([Slot("inference","factor",6)])
    actual = cache.edit(*changed.factors[6])
    assert cache.marginal(8)==changed.posterior()[1][-1]
    assert len(actual)==sum(slot.kind=="message" for slot in affected)==2
    return dict(factor=6,invalidated_arrows=actual,marginal=list(cache.marginal(8)))


def valid_proposals(keys,seen):
    fraction = keys.count(1)/len(keys)
    # Public scenario estimates in normalized modeled units.
    proposals = [
        ("keep",[0,0,0,0,0]),
        ("rolling",[.68,len(keys)/32,seen,.27,0]),
        ("specialized",[.65+fraction*.3,len(keys)/32,seen*.9,.32,fraction]),
    ]
    return proposals


def compare_planning(keys,seen,models):
    results = {}
    proposals = valid_proposals(keys,seen)
    values = [f[0]-f[1]*(1-f[2])-.3*f[3] for _,f in proposals]
    tensor = torch.tensor([f for _,f in proposals])
    for name,model in models.items():
        start = perf_counter_ns()
        with torch.inference_mode():
            chosen = int(model(tensor).flatten().argmax())
        results[name] = dict(choice=proposals[chosen][0],modeled_value=values[chosen],
                             ns=perf_counter_ns()-start)
    # Greedy prices one revision; bounded lookahead prices reuse if the
    # same recipe is revisited on the next revision.
    for name in ("greedy","bounded"):
        start = perf_counter_ns()
        if name=="greedy":
            chosen = max(range(3),key=values.__getitem__)
        else:
            # Enumerate both recipe choices over a two-revision stationary
            # quote. Repeating a verified recipe can reuse its leaf charges.
            def horizon(first):
                second = [value + (proposals[j][1][1]*(1-proposals[j][1][2])
                                   if j==first and j!=0 else 0)
                          for j,value in enumerate(values)]
                return values[first]+max(second)
            chosen = max(range(3),key=horizon)
        results[name] = dict(choice=proposals[chosen][0],modeled_value=values[chosen],
                             ns=perf_counter_ns()-start)
    return results


def benchmark(keys,mode,cpu):
    rng = Random(46172)
    initial = rng.randbytes(PAGE)
    candidates = {variant:compile_stencil(keys,variant) for variant in ("reload",mode)}
    checks = []
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for n in (-1,0,1,3,SCRATCH//(4*len(keys))+1):
            want = expected(keys,n,initial)
            row = []
            for variant,raw in candidates.items():
                memory,evidence = invoke(cpu,raw,n,bytearray(initial),fuel=250+len(keys)*230)
                assert (evidence["return_i32"],memory)==want and wasmtime_run(raw,n,bytearray(initial))==want
                row.append((variant,evidence["source_steps"]))
            checks.append(dict(argument=n,steps=row,page=digest(want[1])))
    largest = SCRATCH//(4*len(keys))
    wanted = expected(keys,largest,initial)[1]
    deployments,funcs = {},{}
    engine = wasmtime.Engine()
    for variant,raw in candidates.items():
        start = perf_counter_ns()
        runtime = wasmtime.Store(engine)
        exports = wasmtime.Instance(runtime,wasmtime.Module(engine,raw),[]).exports(runtime)
        fn,memory = exports["run"],exports["memory"]
        deployments[variant] = perf_counter_ns()-start
        memory.write(runtime,initial)
        assert fn(runtime,largest)==largest and memory.read(runtime,0,PAGE)==wanted
        funcs[variant] = lambda n,f=fn,env=runtime:f(env,n)
    with TemporaryDirectory() as folder:
        runner,deployments["native"],cc = native_c(folder,keys)
        page = (ctypes.c_uint32*(PAGE//4)).from_buffer_copy(initial)
        assert runner(page,largest)==largest and bytes(page)==wanted
        funcs["native"] = lambda n: runner(page,n)
        samples = defaultdict(list)
        for _ in range(22):
            names = sorted(funcs)
            rng.shuffle(names)
            for name in names:
                fn = funcs[name]
                assert fn(largest)==largest
                start = perf_counter_ns()
                for _ in range(64):
                    fn(largest)
                samples[name].append((perf_counter_ns()-start)/64)
    saving = [a-b for a,b in zip(samples["reload"],samples[mode],strict=True)]
    boot = sorted(mean(rng.choices(saving,k=len(saving))) for _ in range(1000))
    return dict(raw_hashes={k:digest(b) for k,b in candidates.items()},checks=checks,records=largest,compiler=cc),dict(
        samples=dict(samples),deployment_ns=deployments,median_saving_ns=median(saving),
        paired_mean_ci95_ns=[boot[25],boot[974]])


def run(models):
    rows,phases,roots,examples = [],[],[],[]
    graph_path = EXECUTOR/"machine.pt"
    cpu = torch.jit.load(str(graph_path))
    with TemporaryDirectory() as folder:
        store = Store(Path(folder)/"incremental")
        full_store = Store(Path(folder)/"full")
        per_revision_store = Store(Path(folder)/"one-revision")
        for family in ("local","global"):
            for width in (8,24):
                modes = dict(incremental=Campaign(store,True),fresh=Campaign(full_store,False))
                caches = dict(incremental=set())
                seen = 0
                for revision,(keys,mode) in enumerate(revisions(family,width)):
                    verified,measures,results = {},{},{}
                    methods = dict(modes,per_revision=Campaign(per_revision_store,True))
                    order_ = tuple(methods) if revision%2==0 else tuple(reversed(methods))
                    for policy in order_:
                        engine = methods[policy]
                        root,work,wall = engine.revise(keys,mode)
                        start = perf_counter_ns()
                        assert verify(engine.store,root,keys,mode,caches.get(policy,set()))
                        elapsed = perf_counter_ns()-start
                        wall["independent_ns"] = elapsed
                        wall["total_ns"] = wall["analysis_ns"]+wall["publication_ns"]+elapsed
                        verified[policy] = root
                        measures[policy] = wall
                        results[policy] = work
                    assert len(set(verified.values()))==1
                    root = verified["incremental"]
                    roots.append(dict(root=root,keys=keys,mode=mode))
                    planned = compare_planning(keys,seen,models)
                    planning_time = {label:item.pop("ns") for label,item in planned.items()}
                    seen = results["incremental"]["reused"]/width
                    rows.append(dict(family=family,width=width,revision=revision,keys=keys,mode=mode,
                                     root=root,analysis=results,planning=planned))
                    phases.append(dict(family=family,width=width,revision=revision,timing=measures,
                                       planning_ns=planning_time))
                    if family=="local" and revision in (0,12):
                        observed,timing = benchmark(keys,mode,cpu)
                        upfront = measures["incremental"]["total_ns"]+timing["deployment_ns"][mode]
                        saving = timing["median_saving_ns"]
                        timing.update(upfront_ns=upfront,break_even=int(upfront//saving)+1 if saving>0 else None)
                        examples.append(dict(width=width,revision=revision,mode=mode,observed=observed))
                        phases[-1]["native"] = timing
        nodes = {p.name:json.loads(p.read_bytes()) for p in store.root.iterdir()}
        row = roots[0]
        base = store.load(row["root"])
        proof_id = base.parents[1]
        proof = store.load(proof_id)
        forged_check = store.add(Node(proof.kind,dict(proof.payload,result="sat"),proof.parents))
        forged_result = store.add(Node(base.kind,base.payload,(base.parents[0],forged_check,*base.parents[2:])))
        forged_code = store.add(Node(base.kind,dict(base.payload,emitted="0"*64),base.parents))
        negatives = {}
        for label,root,keys in (
            ("stale-lane-order",row["root"],row["keys"][::-1]),
            ("rehashed-check",forged_result,row["keys"]),
            ("changed-binary",forged_code,row["keys"]),
        ):
            try:
                verify(store,root,keys,row["mode"],set())
            except ValueError:
                negatives[label] = "rejected"
            else:
                raise AssertionError("invalid campaign certificate accepted")
    base_graph = graph(10,[(i,(i+1)%10) for i in range(10)]+[(0,5),(2,7)])
    widths = {kind:decomposition(base_graph,order(base_graph,kind))[2] for kind in ("min_fill","min_degree")}
    # Reuse prior learned order models as frozen structural controls.
    from structural_decomposition_v66 import OUT as PREV
    for seed in (661,662):
        model = torch.jit.load(str(PREV/f"order-{seed}.pt"))
        widths[f"order-{seed}"] = decomposition(base_graph,order(base_graph,"learned",model))[2]
    data = dict(revisions=rows,benchmarks=examples,adapter=factor_adapter(),ordering_control=widths,
                graph_sha256=digest(graph_path.read_bytes()),negative_controls=negatives)
    return data,phases,dict(roots=roots,nodes=nodes)


def import_bundle(path):
    archive = json.loads(path.read_bytes())
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for key,raw in archive["nodes"].items():
            if store.add(Node(raw["kind"],raw["payload"],tuple(raw["parents"])))!=key or any(
                    parent not in archive["nodes"] for parent in raw["parents"]):
                raise ValueError("missing or changed proof identity")
        cache = set()
        for root in archive["roots"]:
            verify(store,root["root"],root["keys"],root["mode"],cache)
    return len(archive["roots"]),len(cache)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.import_bundle:
        print(import_bundle(OUT/"proofs.json"))
        return
    paths = {f"learned-{seed}":OUT/f"planner-{seed}.pt" for seed in (671,672)}
    data,timing,bundle = run({name:torch.jit.load(str(path)) for name,path in paths.items()})
    data["policies"] = {name:digest(path.read_bytes()) for name,path in paths.items()}
    for name,value in (("campaigns.json",data),("proofs.json",bundle)):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes()))
    if args.write:
        (OUT/"timings.json").write_bytes(canonical(timing)+b"\n")
    print(f"Verified {len(data['revisions'])} revisions against full regeneration")


if __name__=="__main__":
    main()
