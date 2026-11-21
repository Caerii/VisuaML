"""Replay factorized search, proof reuse, exact execution, and native timings."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import ctypes
from itertools import product
from functools import lru_cache
import json
from pathlib import Path
from random import Random
from statistics import median, mean
import struct
import subprocess
from sys import getsizeof
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime
import z3

from factorized_synthesis_v64 import OUT, LIBRARY, POINTS, Product, bucket, component, exact_value, queries
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from verify_compositional_v64 import (
    PAGE, compile_module, component_node, composition, import_bundle, local_proof,
    publish, semantics, validate,
)


def timing(function):
    start = perf_counter_ns()
    result = function()
    return result,perf_counter_ns()-start


@lru_cache(maxsize=32768)
def minimax(domain,cost):
    if len(domain)<=1:
        return 0
    return min(cost[p]+max(minimax(group,cost) for group in bucket(domain,p).values())
               for p in range(len(POINTS)) if len(bucket(domain,p))>1)


def search(keys,costs,budget,library,method,model,clock):
    space = Product.initial(len(keys),library)
    memory = torch.zeros(1,32)
    trace = []
    while space.count>1:
        available,ns = timing(lambda: list(queries(space,costs,budget)))
        clock["query_construction"].append(ns)
        if not available:
            break
        start = perf_counter_ns()
        if model is not None:
            with torch.inference_mode():
                features = torch.tensor([[row[3] for row in available]])
                choice,memory = model(features,torch.zeros_like(memory) if method=="reset" else memory)
            answer = available[int(choice[0].argmax())]
        elif method=="minimax":
            def value(row):
                lane,point,charge,_ = row
                local_costs = tuple(costs[lane])
                worst = charge+max(minimax(group,local_costs)
                                   for group in bucket(space.leaves[lane],point).values())
                return (minimax(space.leaves[lane],local_costs)-worst,exact_value(row))
            answer = max(available,key=value)
        elif method=="coverage":
            answer = max(available,key=lambda row:(row[3][3]/row[2],exact_value(row)))
        else:
            answer = max(available,key=exact_value)
        clock["proposal"].append(perf_counter_ns()-start)
        lane,point,charge,_ = answer
        reply = component(LIBRARY[keys[lane]],POINTS[point])
        new,ns = timing(lambda: space.refine(lane,point,reply))
        clock["refinement"].append(ns)
        assert new.count<space.count
        trace.append(dict(lane=lane,point=point,reply=reply,cost=charge,before=space.count,after=new.count))
        space,budget = new,budget-charge
    answer = space.solution() if space.count==1 else None
    assert answer is None or answer==keys
    return dict(solution=answer,trace=trace,remaining=budget)


def scale():
    rng = Random(34264)
    records,costs = [],[]
    for lanes,size in ((3,8),(6,8),(12,16),(24,32),(32,32)):
        initial,construct = timing(lambda: Product.initial(lanes,size))
        keys = tuple(rng.randrange(size) for _ in range(lanes))
        flat,flat_ns = timing(lambda: list(product(range(size),repeat=lanes))) if initial.count<=300000 else (None,None)
        flat_bytes = getsizeof(flat)+sum(getsizeof(row) for row in flat) if flat is not None else None
        unique = {id(leaf):leaf for leaf in initial.leaves}
        py_bytes = getsizeof(initial.leaves)+sum(getsizeof(leaf) for leaf in unique.values())
        space = initial
        seen = []
        flat_filter,product_filter = [],[]
        while space.count>1:
            q = list(queries(space,[[1]*len(POINTS)]*lanes,1000))
            lane,point,_,_ = max(q,key=exact_value)
            reply = component(LIBRARY[keys[lane]],POINTS[point])
            space,elapsed = timing(lambda: space.refine(lane,point,reply))
            product_filter.append(elapsed)
            seen.append((lane,point,reply))
            if flat is not None:
                flat,elapsed = timing(lambda: [row for row in flat
                         if component(LIBRARY[row[lane]],POINTS[point])==reply])
                flat_filter.append(elapsed)
                assert len(flat)==space.count
        assert space.solution()==keys
        start = perf_counter_ns()
        vars_ = [z3.Int(f"lane_{i}") for i in range(lanes)]
        solver = z3.Solver()
        solver.add(*(z3.And(v>=0,v<size) for v in vars_))
        for lane,point,reply in seen:
            solver.add(z3.Or(*(vars_[lane]==v for v in bucket(tuple(range(size)),point)[reply])))
        assert solver.check()==z3.sat
        solution = tuple(solver.model().eval(v).as_long() for v in vars_)
        assert solution==keys
        solver.add(z3.Or(*(var!=key for var,key in zip(vars_,keys,strict=True))))
        assert solver.check()==z3.unsat
        records.append(dict(lanes=lanes,library=size,candidates=initial.count,queries=len(seen),
                            flat_measured=flat_bytes is not None,flat_container_bytes=flat_bytes,
                            product_container_bytes=py_bytes,smt2_bytes=len(solver.to_smt2())))
        costs.append(dict(lanes=lanes,factorized_initial_ns=construct,flat_initial_ns=flat_ns,
                          smt_construct_and_uniqueness_ns=perf_counter_ns()-start,
                          flat_filter_ns=flat_filter,product_filter_ns=product_filter))
    return records,costs


def tasks():
    rng = Random(664130)
    for family,lanes,size,budget in (("in_range",4,14,21),("deep",12,24,65),("wide_shift",24,32,165)):
        for i in range(20):
            keys = tuple(rng.randrange(size) for _ in range(lanes))
            costs = [[rng.randint(3,11) if family=="wide_shift" else rng.randint(1,5) for _ in POINTS]
                     for _ in keys]
            yield family,i,keys,costs,budget,size


def expected(modules,n,page):
    result = bytearray(page)
    if not 0<=n<=PAGE//(4*len(modules)):
        return -1,result
    for row in range(n):
        for lane,mod in enumerate(modules):
            at = (row*len(modules)+lane)*4
            struct.pack_into("<i",result,at,component(mod,struct.unpack_from("<i",page,at)[0]))
    return n,result


def compile_native(folder,modules):
    ops = "\n".join(f"{{ uint32_t x=p[{i}]; p[{i}]=x*{a}u+{b}u+{g}u*(x>>31); }}"
                    for i,(a,b,g) in enumerate(modules))
    source = Path(folder)/"fused.c"
    source.write_text(f"""#include <stdint.h>
      int transform(uint32_t *p, int n) {{
       if(n<0 || n>{PAGE//(4*len(modules))}) return -1;
       for(int i=0;i<n;++i,p+={len(modules)}) {{ {ops} }}
       return n;
      }}
    """)
    library = Path(folder)/"fused.so"
    _,compile_ns = timing(lambda: subprocess.run(["cc","-O3","-shared","-fPIC",str(source),
                                           "-o",str(library)],check=True))
    invoke_native = ctypes.CDLL(str(library)).transform
    invoke_native.argtypes = [ctypes.POINTER(ctypes.c_uint32),ctypes.c_int]
    invoke_native.restype = ctypes.c_int
    return invoke_native,compile_ns


def execution(graph,store,modules):
    (root,baseline,candidate),publish_ns = timing(lambda: publish(store,modules))
    accepted,check_ns = timing(lambda: validate(store,root,modules))
    assert accepted==candidate
    rng = Random(971)
    page = rng.randbytes(PAGE)
    limits = [-1,0,1,3,PAGE//(4*len(modules))+1]
    frozen = []
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for n in limits:
            wanted = expected(modules,n,page)
            observed = []
            for binary in (baseline,candidate):
                output,row = invoke(graph,binary,n,bytearray(page),fuel=100+len(modules)*220)
                assert (row["return_i32"],output)==wanted
                assert wasmtime_run(binary,n,bytearray(page))==wanted
                observed.append(row["source_steps"])
            frozen.append(dict(n=n,source_steps=observed,return_value=wanted[0],page=digest(wanted[1])))
    engine = wasmtime.Engine()
    methods,deploy = {},{}
    for name,binary in (("cranelift_staged",baseline),("cranelift_fused",candidate)):
        start = perf_counter_ns()
        room = wasmtime.Store(engine)
        extern = wasmtime.Instance(room,wasmtime.Module(engine,binary),[]).exports(room)
        memory,run = extern["memory"],extern["run"]
        deploy[name] = perf_counter_ns()-start
        memory.write(room,page)
        assert run(room,1)==1 and memory.read(room,0,PAGE)==expected(modules,1,page)[1]
        cap = PAGE//(4*len(modules))
        memory.write(room,page)
        assert run(room,cap)==cap and memory.read(room,0,PAGE)==expected(modules,cap,page)[1]
        methods[name] = lambda n,mem=memory,s=room,f=run: f(s,n)
    with TemporaryDirectory() as folder:
        native,deploy["cc_O3_fused"] = compile_native(folder,modules)
        values = (ctypes.c_uint32*(PAGE//4)).from_buffer_copy(page)
        assert native(values,1)==1 and bytes(values)==expected(modules,1,page)[1]
        cap = PAGE//(4*len(modules))
        ctypes.memmove(values,page,PAGE)
        assert native(values,cap)==cap and bytes(values)==expected(modules,cap,page)[1]
        methods["cc_O3_fused"] = lambda n: native(values,n)
        native_times = defaultdict(list)
        maximum = PAGE//(4*len(modules))
        for n in (maximum//4,maximum):
            for i in range(20):
                for name in (list(methods) if i%2==0 else reversed(methods)):
                    code = methods[name]
                    assert code(n)==n
                    start = perf_counter_ns()
                    for _ in range(64):
                        code(n)
                    native_times[f"{n}/{name}"].append((perf_counter_ns()-start)/64)
    return dict(root=root,modules=modules,frozen=frozen,wasm=[digest(baseline),digest(candidate)]),dict(
        proof_and_lowering_ns=publish_ns,independent_check_ns=check_ns,
        deployment_ns=deploy,native_ns=dict(native_times))


def proof_controls(store,roots):
    first = roots[0]
    original = tuple(tuple(mod) for mod in first["modules"])
    truth = store.load(first["root"])
    semantic,*children = truth.parents
    changed = next(mod for mod in LIBRARY if mod!=original[0])
    forged = store.add(component_node(original[0],changed,semantic))
    ref = compile_module(original,False)
    wrong = compile_module((changed,*original[1:]),True)
    false_root = store.add(composition(semantic,[forged,*children[1:]],ref,wrong))
    changed_lane = (changed,*original[1:])
    for root,scope in ((false_root,original),(first["root"],changed_lane),(first["root"],original[:-1])):
        try:
            validate(store,root,scope)
        except ValueError:
            pass
        else:
            raise AssertionError("forged or invalidated proof was accepted")
    assert first["root"] in store.affected([children[0]])
    assert not local_proof(original[0],changed)
    return dict(forged_rejected=True,scope_rejected=True,dependent_count=len(store.affected([children[0]])))


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.import_bundle:
        with TemporaryDirectory() as folder:
            print(import_bundle(OUT/"proofs.json",folder))
        return
    fps = {name:OUT/(name+".pt") for name in ("mlp-641","recurrent-641","recurrent-642")}
    nets = {name:torch.jit.load(str(path)) for name,path in fps.items()}
    methods = dict(nets,entropy=None,coverage=None,minimax=None,reset=nets["recurrent-641"])
    results,summary,phase,roots = [],{},defaultdict(list),[]
    scaling,scale_times = scale()
    graph_path = EXECUTOR/"machine.pt"
    graph = torch.jit.load(str(graph_path))
    measured,samples,cache = [],[],{}
    seen_components = set()
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for family,i,keys,costs,budget,size in tasks():
            runs = {}
            for name,model in methods.items():
                sample = defaultdict(list)
                start = perf_counter_ns()
                row = search(keys,costs,budget,size,name,model,sample)
                sample["search_ns"].append(perf_counter_ns()-start)
                for metric,values in sample.items():
                    phase[f"{family}/{name}/{metric}"].extend(values)
                runs[name] = row
                summary.setdefault(family+"/"+name,Counter()).update(assigned=1,
                    identified=int(row["solution"] is not None),credits=budget-row["remaining"],
                    probes=len(row["trace"]))
            root = None
            if any(row["solution"] is not None for row in runs.values()):
                modules = tuple(LIBRARY[key] for key in keys)
                for mod in set(modules)-seen_components:
                    assert local_proof(mod,mod)
                    seen_components.add(mod)
                (root,baseline,alternative),elapsed = timing(lambda: publish(store,modules))
                value,retrieval = timing(lambda: validate(store,root,modules,cache))
                assert value==alternative
                phase["certificate/publication"].append(elapsed)
                phase["certificate/recheck_with_components"].append(retrieval)
                roots.append(dict(root=root,modules=modules))
                if i in (0,1):
                    certified,wall = execution(graph,store,modules)
                    measured.append(dict(family=family,index=i,**certified))
                    search_ns = phase[f"{family}/entropy/search_ns"][-1]
                    wall["search_ns"] = search_ns
                    max_n = PAGE//(4*len(modules))
                    base = wall["native_ns"][f"{max_n}/cranelift_staged"]
                    repl = wall["native_ns"][f"{max_n}/cranelift_fused"]
                    diffs = [b-r for b,r in zip(base,repl,strict=True)]
                    gain = median(diffs)
                    rng = Random(6400+i)
                    boot = sorted(mean(rng.choices(diffs,k=len(diffs))) for _ in range(1000))
                    wall["paired_mean_saving_ns_ci95"] = [boot[25],boot[974]]
                    upfront = search_ns+wall["proof_and_lowering_ns"]+wall["independent_check_ns"]+wall["deployment_ns"]["cranelift_fused"]
                    wall["break_even_calls"] = int(upfront//gain)+1 if gain>0 else None
                    samples.append(wall)
            results.append(dict(family=family,index=i,library=size,keys=keys,costs=costs,budget=budget,
                                runs=runs,certificate=root))
        assert measured and roots
        pending = [r["root"] for r in roots]
        nodes = {}
        while pending:
            key = pending.pop()
            if key not in nodes:
                node = store.load(key)
                nodes[key] = json.loads(node.bytes())
                pending.extend(node.parents)
        controls = proof_controls(store,roots)
        recheck_counts = []
        for mode in ("cold_leaves","shared_leaves"):
            proof_cache = {}
            for row in roots:
                locals_cache = proof_cache if mode=="shared_leaves" else {}
                before = len(locals_cache)
                _,elapsed = timing(lambda: validate(store,row["root"],tuple(row["modules"]),locals_cache))
                phase[f"reuse/{mode}"].append(elapsed)
                recheck_counts.append((mode,len(locals_cache)-before))
        controls["component_checks"] = {mode:sum(n for name,n in recheck_counts if name==mode)
                                       for mode in ("cold_leaves","shared_leaves")}
    paired = {}
    rng = Random(1983)
    for family in ("in_range","deep","wide_shift"):
        for name in nets:
            deltas = [int(row["runs"][name]["solution"] is not None)-int(row["runs"]["entropy"]["solution"] is not None)
                      for row in results if row["family"]==family]
            values = sorted(mean(rng.choices(deltas,k=len(deltas))) for _ in range(1000))
            paired[f"{family}/{name}"] = [values[25],values[974]]
    data = dict(both_certified_engines=measured,roots=len(roots),distinct_components=len(seen_components),
                controls=controls,scale=scaling,summary=summary,results=results,paired_vs_entropy=paired,
                hashes={name:digest(path.read_bytes()) for name,path in fps.items()},
                graph_sha256=digest(graph_path.read_bytes()))
    bundle = dict(nodes=nodes,roots=roots)
    for name,value in (("outcomes.json",data),("proofs.json",bundle)):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes()))
    if args.write:
        (OUT/"timings.json").write_bytes(canonical(dict(search=dict(phase),scale=scale_times,execution=samples))+b"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
