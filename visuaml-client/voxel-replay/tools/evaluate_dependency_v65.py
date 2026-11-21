"""Deterministic search and replay, plus paired timings for dependent fusion."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import ctypes
from itertools import product
import json
from math import prod
from pathlib import Path
from random import Random
import shutil
from statistics import mean, median
import struct
import subprocess
from sys import getsizeof
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime
import z3

from dependent_synthesis_v65 import OUT, Space, TESTS, choices, signed, utility
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from verify_dependency_v65 import PAGE, SCRATCH, certificate, contract, leaf, obligation, open_bundle, publish, verify, wasm


def tree(kind,n):
    return [(i-1 if kind=="chain" else (i-1)//2,i) for i in range(1,n)]


def reply(space,keys,i,j):
    a,b,_ = space.factors[i]
    x,y = TESTS[j]
    return signed(x*keys[a]+y*keys[b])


def search(task,name,net,clock):
    space = Space.build(task["width"],task["library"],tree(task["layout"],task["width"]))
    budget,trace = task["budget"],[]
    memory = torch.zeros(1,32)
    history = []
    while True:
        start = perf_counter_ns()
        state = space.posterior()
        options = choices(space,task["costs"],budget,state)
        clock["symbolic_ns"].append(perf_counter_ns()-start)
        if not options:
            break
        start = perf_counter_ns()
        if net is not None:
            features = [row[3] for row in options]
            if name.startswith("full-history"):
                features.extend(history)
            tensor = torch.tensor([features])
            with torch.inference_mode():
                scores,memory = net(tensor,torch.zeros_like(memory) if name=="reset" else memory)
            selected = options[int(scores[0,:len(options)].argmax())]
        elif name=="entropy":
            selected = max(options,key=utility)
        else:
            selected = max(options,key=lambda row:(row[3][2]/row[2],utility(row)))
        clock["policy_ns"].append(perf_counter_ns()-start)
        edge,test,charge,feature = selected
        assert charge<=budget
        answer = reply(space,task["keys"],edge,test)
        token = [*feature]
        token[4],token[7] = -1,answer/24
        history.append(token)
        space = space.observe(edge,test,answer)
        budget -= charge
        trace.append(dict(edge=edge,test=test,charge=charge,response=answer,models=state[0]))
    count,singles,_ = state
    found = [weights.index(1)+1 for weights in singles] if count==1 else None
    assert not found or found==task["keys"]
    return dict(found=found,remaining=budget,posterior=count,trace=trace)


def scalability():
    rng = Random(265)
    data,wall = [],[]
    for kind,n,k in (("chain",5,5),("tree",7,6),("chain",20,9),("tree",32,12)):
        fresh = Space.build(n,k,tree(kind,n))
        keys = [rng.randint(1,k) for _ in range(n)]
        start = perf_counter_ns()
        flat = list(product(range(1,k+1),repeat=n)) if k**n<=300000 else None
        flat_time = perf_counter_ns()-start
        memory = getsizeof(flat)+sum(getsizeof(row) for row in flat) if flat is not None else None
        refs = {id(pairs):pairs for _,_,pairs in fresh.factors}
        factor_bytes = getsizeof(fresh.factors)+sum(getsizeof(row) for row in fresh.factors)+sum(
            getsizeof(items)+sum(getsizeof(pair) for pair in items) for items in refs.values())
        arrangement = [(edge,0) for edge in range(n-1)]+[(0,5)]
        observations,counts = [],[]
        factor_ns,flat_ns,smt_ns = [],[],[]
        space = fresh
        independent_gap = False
        v = [z3.Int(f"lane{i}") for i in range(n)]
        solver = z3.Solver()
        solver.add(*(z3.And(1<=x,x<=k) for x in v))
        for edge,test in arrangement:
            answer = reply(fresh,keys,edge,test)
            a,b,_ = fresh.factors[edge]
            p,q = TESTS[test]
            start = perf_counter_ns()
            space = space.observe(edge,test,answer)
            exact,singles,_ = space.posterior()
            factor_ns.append(perf_counter_ns()-start)
            projected = prod(sum(c>0 for c in row) for row in singles)
            independent_gap |= projected>exact
            if flat is not None:
                start = perf_counter_ns()
                flat = [row for row in flat if p*row[a]+q*row[b]==answer]
                flat_ns.append(perf_counter_ns()-start)
                assert len(flat)==exact
            start = perf_counter_ns()
            solver.add(p*v[a]+q*v[b]==answer)
            assert solver.check()==z3.sat
            smt_ns.append(perf_counter_ns()-start)
            observations.append([edge,test,answer])
            counts.append(exact)
        solver.add(z3.Or(*(x!=value for x,value in zip(v,keys,strict=True))))
        assert solver.check()==z3.unsat and counts[-1]==1 and independent_gap
        data.append(dict(layout=kind,nodes=n,library=k,treewidth=1,initial=k**n,counts=counts,
                    independent_overcounts=independent_gap,observations=observations,
                    flat_container_bytes=memory,factor_container_bytes=factor_bytes))
        wall.append(dict(layout=kind,nodes=n,flat_setup_ns=flat_time if memory else None,
                    updates_flat_ns=flat_ns,updates_factor_ns=factor_ns,updates_incremental_smt_ns=smt_ns))
    return data,wall


def tasks():
    rng = Random(778652)
    for family,width,library,budget,layout in (
            ("chain",7,6,22,"chain"),("tree",12,8,52,"tree"),("wide_shift",20,10,112,"chain")):
        for case in range(16):
            yield dict(family=family,case=case,width=width,library=library,budget=budget,layout=layout,
                keys=[rng.randint(1,library) for _ in range(width)],
                costs=[[rng.randint(4,11) if family=="wide_shift" else rng.randint(1,5)
                        for _ in TESTS] for _ in range(width-1)])


def expected(keys,n,page):
    result = bytearray(page)
    width = len(keys)
    if not 0<=n<=SCRATCH//(4*width):
        return -1,result
    prev = 0
    for i in range(n*width):
        x = struct.unpack_from("<i",page,4*i)[0]
        struct.pack_into("<i",result,SCRATCH+4*i,x)
        struct.pack_into("<i",result,4*i,signed(keys[i%width]*x+keys[(i-1)%width]*prev))
        prev = x
    return n,result


def native_c(folder,keys):
    source = Path(folder)/"rolling.c"
    data = ",".join(str(k) for k in keys)
    source.write_text(f"""#include <stdint.h>
int run(uint32_t *p,int n) {{
  static const uint32_t a[]={{{data}}};
  if(n<0 || n>{SCRATCH//(4*len(keys))}) return -1;
  uint32_t prev = 0;
  for(int i=0;i<n;++i) for(int j=0;j<{len(keys)};++j) {{
    int k=i*{len(keys)}+j;
    uint32_t x=p[k]; p[k+{SCRATCH//4}]=x;
    p[k]=a[j]*x+a[(j+{len(keys)}-1)%{len(keys)}]*prev; prev=x;
  }}
  return n;
}}
""")
    target = Path(folder)/"roll.so"
    compiler = shutil.which("clang") or shutil.which("cc")
    started = perf_counter_ns()
    subprocess.run([compiler,"-O3","-shared","-fPIC",str(source),"-o",str(target)],check=True)
    elapsed = perf_counter_ns()-started
    c = ctypes.CDLL(str(target)).run
    c.argtypes = [ctypes.POINTER(ctypes.c_uint32),ctypes.c_int]
    c.restype = ctypes.c_int
    return c,elapsed,Path(compiler).name


def benchmark(keys,graph):
    rng = Random(9211)
    page = rng.randbytes(PAGE)
    binaries = [wasm(keys,kind) for kind in ("staged","fused","invalid")]
    checks = []
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for n in (-1,0,1,2,SCRATCH//(4*len(keys))+1):
            wanted = expected(keys,n,page)
            steps = []
            for raw in binaries[:2]:
                output,observed = invoke(graph,raw,n,bytearray(page),fuel=180+len(keys)*160)
                assert (observed["return_i32"],output)==wanted
                assert wasmtime_run(raw,n,bytearray(page))==wanted
                steps.append(observed["source_steps"])
            checks.append(dict(n=n,steps=steps,page=digest(wanted[1])))
    fuel_witness = None
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for fuel in (35,45,55,65,75):
            short = [invoke(graph,raw,1,bytearray(page),fuel=fuel) for raw in binaries[:2]]
            if all(record["status"]==2 for _,record in short) and short[0][0]!=short[1][0]:
                fuel_witness = dict(fuel=fuel,first_byte=next(i for i,(a,b) in enumerate(
                    zip(short[0][0],short[1][0],strict=True)) if a!=b))
                break
    assert fuel_witness is not None
    bad = wasmtime_run(binaries[2],1,bytearray(page))
    ref = expected(keys,1,page)
    first = next(i for i,(a,b) in enumerate(zip(bad[1],ref[1],strict=True)) if a!=b)
    execution,setup = {},{}
    engine = wasmtime.Engine()
    largest = SCRATCH//(4*len(keys))
    expected_page = expected(keys,largest,page)[1]
    for method,binary in zip(("staged","fused"),binaries,strict=False):
        start = perf_counter_ns()
        room = wasmtime.Store(engine)
        exports = wasmtime.Instance(room,wasmtime.Module(engine,binary),[]).exports(room)
        setup[method] = perf_counter_ns()-start
        exports["memory"].write(room,page)
        call = exports["run"]
        assert call(room,largest)==largest and exports["memory"].read(room,0,PAGE)==expected_page
        execution[method] = lambda n,env=room,fn=call: fn(env,n)
    samples = defaultdict(list)
    with TemporaryDirectory() as folder:
        call,setup["native"],compiler = native_c(folder,keys)
        pinned = (ctypes.c_uint32*(PAGE//4)).from_buffer_copy(page)
        assert call(pinned,largest)==largest and bytes(pinned)==expected_page
        execution["native"] = lambda n: call(pinned,n)
        for step in range(24):
            order = sorted(execution)
            rng.shuffle(order)
            for name in order:
                runner = execution[name]
                assert runner(largest)==largest
                started = perf_counter_ns()
                for _ in range(64):
                    runner(largest)
                samples[name].append((perf_counter_ns()-started)/64)
    pair = [a-b for a,b in zip(samples["staged"],samples["fused"],strict=True)]
    bootstrap = sorted(mean(rng.choices(pair,k=len(pair))) for _ in range(1000))
    return dict(checks=checks,incorrect_first_byte=first,fuel_witness=fuel_witness,hashes=[digest(b) for b in binaries]),dict(
        samples=dict(samples),setup_ns=setup,compiler=compiler,records=largest,
        mean_paired_saving_ci95_ns=[bootstrap[25],bootstrap[974]])


def faults(store,root,keys):
    proof = store.load(root)
    semantic,*edges = proof.parents
    rounds = [
        certificate(keys,semantic,edges,variant="invalid"),
        certificate(keys,semantic,edges[::-1]),
    ]
    scope = store.load(semantic)
    aliased = store.add(Node(scope.kind,dict(scope.payload,scratch=0,overlap=True)))
    rounds.append(certificate(keys,aliased,edges))
    fuel = store.add(Node(scope.kind,dict(scope.payload,authority="every-source-fuel-boundary")))
    rounds.append(certificate(keys,fuel,edges))
    rounds.append(Node(proof.kind,dict(proof.payload,upper_bound=65536),proof.parents))
    for forged in rounds:
        try:
            verify(store,store.add(forged),keys)
        except ValueError:
            pass
        else:
            raise AssertionError("rehashed false certificate accepted")
    a,b = keys[0],keys[-1]
    witness = obligation(a,b,invalid=True)
    assert witness is not None
    assert root in store.affected([edges[0]])
    return dict(rejected=["updated-carry","stale-edges","alias","changed-fuel-contract","invalid-loop-bound"],
                induction_counterexample=witness,dependent_roots=sum(
                    store.load(key).kind=="certificate" for key in store.affected([edges[0]])))


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.import_bundle:
        with TemporaryDirectory() as folder:
            print(open_bundle(OUT/"proofs.json",folder))
        return
    paths = {name:OUT/f"{name}.pt" for name in ("full-651","full-652","truncated-651","stateless-651","full-history-651")}
    models = {name:torch.jit.load(str(path)) for name,path in paths.items()}
    methods = dict(models,reset=models["full-651"],entropy=None,coverage=None)
    summary,rows,timings,roots,examples = {},[],defaultdict(list),[],[]
    scale,scale_times = scalability()
    graph_path = EXECUTOR/"machine.pt"
    graph = torch.jit.load(str(graph_path))
    with TemporaryDirectory() as folder:
        store = Store(folder)
        checked = {}
        for task in tasks():
            runs = {}
            for name,net in methods.items():
                phases = defaultdict(list)
                start = perf_counter_ns()
                trial = search(task,name,net,phases)
                phases["total_ns"].append(perf_counter_ns()-start)
                for key,values in phases.items():
                    timings[f"{task['family']}/{name}/{key}"].extend(values)
                runs[name] = trial
                summary.setdefault(task["family"]+"/"+name,Counter()).update(tasks=1,
                    certified=int(trial["found"] is not None),queries=len(trial["trace"]),
                    spent=task["budget"]-trial["remaining"])
            root = None
            if any(r["found"] for r in runs.values()):
                started = perf_counter_ns()
                root = publish(store,task["keys"])
                publish_time = perf_counter_ns()-started
                started = perf_counter_ns()
                assert verify(store,root,task["keys"],checked)
                check_time = perf_counter_ns()-started
                roots.append(dict(root=root,keys=task["keys"]))
                timings["proof/publication_ns"].append(publish_time)
                timings["proof/check_ns"].append(check_time)
                if task["case"] in (0,1):
                    sample,timing = benchmark(task["keys"],graph)
                    saving = median([a-b for a,b in zip(timing["samples"]["staged"],
                                                        timing["samples"]["fused"],strict=True)])
                    search_ns = timings[f"{task['family']}/coverage/total_ns"][-1]
                    upfront = search_ns+publish_time+check_time+timing["setup_ns"]["fused"]
                    timing.update(upfront_ns=upfront,
                        break_even=int(upfront//saving)+1 if saving>0 else None)
                    examples.append(dict(task=task["family"],case=task["case"],keys=task["keys"],**sample))
                    timings["benchmark"].append(timing)
            rows.append(dict(task=task,runs=runs,root=root))
        assert roots and examples
        nodes = {p.name:json.loads(p.read_bytes()) for p in Path(folder).iterdir()}
        control = faults(store,roots[0]["root"],roots[0]["keys"])
        bundle = dict(roots=roots,nodes=nodes)
    paired = {}
    rng = Random(76521)
    for family in ("chain","tree","wide_shift"):
        for name in paths:
            deltas = [int(row["runs"][name]["found"] is not None)-int(row["runs"]["coverage"]["found"] is not None)
                      for row in rows if row["task"]["family"]==family]
            values = sorted(mean(rng.choices(deltas,k=len(deltas))) for _ in range(1000))
            paired[f"{family}/{name}"] = [values[25],values[974]]
    data = dict(summary=summary,rows=rows,scale=scale,faults=control,examples=examples,
                paired_vs_coverage=paired,graph_sha256=digest(graph_path.read_bytes()),
                policy_sha256={k:digest(p.read_bytes()) for k,p in paths.items()})
    for name,value in (("outcomes.json",data),("proofs.json",bundle)):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes()))
    if args.write:
        (OUT/"timings.json").write_bytes(canonical(dict(phases=timings,scale=scale_times))+b"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
