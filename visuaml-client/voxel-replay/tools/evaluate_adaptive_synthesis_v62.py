"""Compare adaptive synthesis policies and independently validate emitted code."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
import json
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime
import z3

from adaptive_synthesis_v62 import OUT, ALL, HALT, MASKS, POINTS, PROGRAMS, WIDTH, encode, optimal, possible
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke


def points(program):
    return list(range(*program)) if program else []


def compile_predicate(values, offset, specialized):
    if not values:
        code = "i32.const 0"
    elif specialized:
        assert values==list(range(values[0],values[-1]+1))
        lo,hi = offset+values[0],offset+values[-1]+1
        code = f"local.get 0 i32.const {lo} i32.lt_s i32.eqz local.get 0 i32.const {hi} i32.lt_s i32.mul"
    else:
        code = " ".join(f"local.get 0 i32.const {offset+x} i32.sub i32.eqz"+
                        (" i32.add" if i else "") for i,x in enumerate(values))
    return bytes(wasmtime.wat2wasm(
        f'(module (memory (export "memory") 1 1) (func (export "run") (param i32) (result i32) {code}))'))


def prove(goal, candidate, offset):
    x = z3.BitVec("x",32)
    expected = z3.Or(*(x==offset+p for p in goal))
    actual = z3.And(x>=offset+candidate[0],x<offset+candidate[-1]+1) if candidate else z3.BoolVal(False)
    solver = z3.SolverFor("QF_BV")
    solver.add(expected!=actual)
    status = solver.check()
    if status==z3.sat:
        v = solver.model().eval(x,model_completion=True).as_long()
        return False,v if v<2**31 else v-2**32
    assert status==z3.unsat
    return True,None


def verify_bundle(bundle):
    def require(ok,message):
        if not ok:
            raise ValueError(message)

    with TemporaryDirectory() as folder:
        store = Store(folder)
        for key,row in bundle["nodes"].items():
            require(store.add(Node(row["kind"],row["payload"],tuple(row["parents"])))==key,
                    "node identity mismatch")
        cert = store.load(bundle["root"])
        require(cert.kind=="certificate" and cert.payload=={"contract":"i32-conditional-completion-v1"},
                "certificate contract differs")
        query_key, = cert.parents
        query = store.load(query_key)
        require(query.kind=="query" and
                query.payload=={"authority":"universal","proof":"signed-interval-partition-v1"},
                "unsupported proof obligation")
        left,right = [store.load(key) for key in query.parents]
        require(left.kind==right.kind=="source" and not left.parents and not right.parents,
                "unexpected source provenance")
        goal,output = left.payload["points"],right.payload["points"]
        offset = left.payload["offset"]
        require(type(offset) is int and right.payload["offset"]==offset and
                -2**31<=offset<=2**31-POINTS-1,"invalid signed domain")
        for values in (goal,output):
            require(all(type(p) is int and 0<=p<POINTS for p in values) and
                    values==sorted(set(values)),"invalid domain points")
        require(not output or output==list(range(output[0],output[-1]+1)),"candidate outside grammar")
        require(goal==output,"different signed partitions")
        require(left.payload["wasm"]==digest(compile_predicate(goal,offset,False)),
                "reference binary differs")
        require(right.payload["wasm"]==digest(compile_predicate(output,offset,True)),
                "candidate binary differs")


def bundle_for(goal,output,offset,binaries):
    with TemporaryDirectory() as folder:
        store = Store(folder)
        keys = [store.add(Node("source",dict(points=value,offset=offset,wasm=digest(binary))))
                for value,binary in zip((goal,output),binaries,strict=True)]
        obligation = store.add(Node("query",dict(authority="universal",proof="signed-interval-partition-v1"),tuple(keys)))
        root = store.add(Node("certificate",dict(contract="i32-conditional-completion-v1"),(obligation,)))
        from pathlib import Path
        nodes = {path.name:json.loads(path.read_bytes()) for path in Path(folder).iterdir()}
    return dict(root=root,nodes=nodes)


def refinement_witness():
    # A narrower fast predicate refines the reference for positive answers.
    # Equality is licensed only over the explicit assumed input range.
    goal, output = [0,1,2], [0,1]
    contract = dict(observes="return",assumed_inputs=[0,1],scope="signed-i32")
    def verify(reference,candidate,assumption):
        covered = set(assumption["assumed_inputs"])
        if not covered or covered & (set(reference)^set(candidate)):
            raise ValueError("refinement cannot authorize equality on the requested scope")
        return True
    assert verify(goal,output,contract)
    try:
        verify(goal,output,dict(contract,assumed_inputs=[0,1,2]))
    except ValueError:
        pass
    else:
        raise AssertionError("widening the assumption licensed invalid equality")
    assert prove(goal,output,0)==(False,2)
    scope = Node("assumptions",contract)
    source = Node("source",dict(goal=goal,candidate=output))
    parents = tuple(digest(node.bytes()) for node in (scope,source))
    certificate = Node("derived-refinement",dict(positive_subset=set(output)<=set(goal)),parents)
    assert certificate.payload["positive_subset"]
    return dict(scope=json.loads(scope.bytes()),source=json.loads(source.bytes()),
                certificate=json.loads(certificate.bytes()),outside_scope_counterexample=2)


def learned(model,observed,costs,budget,proof,state):
    with torch.inference_mode():
        logits,state = model(encode(observed,costs,budget,proof).unsqueeze(0),state)
    return int(logits[0].argmax()),state


def run(task,policy,model,clock,intervention=None):
    costs = tuple(task["costs"])
    budget,proof = task["budget"],task["proof"]
    observed,memory,trace = [0]*POINTS,torch.zeros(1,WIDTH),[]
    approved = None
    for _ in range(10):
        started = perf_counter_ns()
        if policy=="json_tool":
            observed = json.loads(canonical(observed))
        viable = possible(observed)
        if model is not None:
            if policy=="reset":
                memory = torch.zeros_like(memory)
            action,memory = learned(model,observed,costs,budget,proof,memory)
        elif policy=="dp":
            action = optimal(viable,costs,budget-1)[2] if proof else HALT
        elif viable.bit_count()==1:
            action = POINTS+viable.bit_length()-1
        elif viable and proof:
            candidates = [(min((viable&m).bit_count(),(viable&~m).bit_count())/costs[i],i)
                          for i,m in enumerate(MASKS) if not observed[i] and costs[i]<budget]
            action = (max(candidates,key=lambda row:(row[0],-row[1]))[1]
                      if candidates and max(candidates)[0]>0 else HALT)
        else:
            action = HALT
        clock["policy_ns"].append(perf_counter_ns()-started)
        event = dict(raw=action,budget=budget,state=observed.copy())
        trace.append(event)
        if action==HALT:
            event["result"]="abstain"
            break
        charge = costs[action] if action<POINTS else 1
        if charge>budget or action<POINTS and observed[action]:
            event["result"]="denied"
            break
        budget -= charge
        if action<POINTS:
            answer = int(action in task["goal"])
            shown = 1-answer if intervention=="first" and len(trace)==1 else answer
            observed[action] = 1 if shown else -1
            event.update(result="sample",actual=answer,shown=shown,cost=charge)
        elif proof:
            candidate = points(PROGRAMS[action-POINTS])
            started = perf_counter_ns()
            # Exact partition equality is necessary and sufficient in this grammar.
            accepted = candidate==task["goal"]
            clock["verification_ns"].append(perf_counter_ns()-started)
            event.update(result="certified" if accepted else "counterexample",cost=charge,candidate=candidate)
            if accepted:
                approved = candidate
                break
            witness = min(set(candidate)^set(task["goal"]))
            observed[witness] = 1 if witness in task["goal"] else -1
            event["counterexample"] = witness+task["offset"]
        else:
            event["result"]="proof-unavailable"
            break
    return dict(approved=approved,spent=task["budget"]-budget,trace=trace)


def tasks():
    rng = Random(18852)
    for family in ("nominal","skew_cost","non_interval","proof_missing"):
        for case in range(64):
            goal = points(PROGRAMS[rng.randrange(len(PROGRAMS))])
            if family=="non_interval":
                goal = [rng.randrange(3),rng.randrange(5,8)]
            yield dict(family=family,case=case,goal=goal,offset=rng.randint(-700,700),
                       costs=[rng.randint(1,4) for _ in range(8)] if family=="skew_cost" else [1]*8,
                       budget=rng.randint(5,11),proof=family!="proof_missing")


def performance(task,approved,graph):
    offset = task["offset"]
    start = perf_counter_ns()
    source = compile_predicate(task["goal"],offset,False)
    candidate = compile_predicate(approved,offset,True)
    compile_ns = perf_counter_ns()-start
    start = perf_counter_ns()
    assert prove(task["goal"],approved,offset)[0]
    proof_ns = perf_counter_ns()-start
    start = perf_counter_ns()
    portable = bundle_for(task["goal"],approved,offset,(source,candidate))
    publish_ns = perf_counter_ns()-start
    start = perf_counter_ns()
    verify_bundle(portable)
    import_ns = perf_counter_ns()-start
    start = perf_counter_ns()
    lookup = {portable["root"]:portable}
    assert lookup.get(portable["root"]) is portable
    lookup_ns = perf_counter_ns()-start
    repeats = [[],[]]
    results = []
    for index,binary in enumerate((source,candidate)):
        engine = wasmtime.Engine()
        room = wasmtime.Store(engine)
        extern = wasmtime.Instance(room,wasmtime.Module(engine,binary),[]).exports(room)
        for argument in (-2**31,2**31-1,offset-1,offset+8,*range(offset,offset+8)):
            expected = int(argument-offset in task["goal"])
            with patch(PACKED,side_effect=AssertionError("packed fallback")):
                page,record = invoke(graph,binary,argument,bytearray(65536),128)
            assert page==bytes(65536) and record["return_i32"]==expected
            assert wasmtime_run(binary,argument,bytearray(65536))==(expected,page)
            results.append(record["source_steps"])
        for i in range(32):
            at = perf_counter_ns()
            extern["run"](room,offset+i%8)
            repeats[index].append(perf_counter_ns()-at)
    return portable,dict(compilation_ns=compile_ns,verification_ns=proof_ns,native_ns=repeats,
                      persistence_ns=publish_ns,import_ns=import_ns,lookup_ns=lookup_ns,
                      source_steps=[results[0],results[12]])


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--verify-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.verify_bundle:
        for bundle in json.loads((OUT/"proofs.json").read_bytes()):
            verify_bundle(bundle)
        return
    models = {name:torch.jit.load(str(OUT/(name+".pt"))) for name in ("recurrent","attention","mlp")}
    methods = dict(models,reset=models["recurrent"],dp=None,greedy=None,json_tool=models["attention"])
    totals,rows,timings = {},[],defaultdict(list)
    for task in tasks():
        record = {}
        for method,model in methods.items():
            phases = defaultdict(list)
            start = perf_counter_ns()
            result = run(task,method,model,phases)
            phases["total_ns"].append(perf_counter_ns()-start)
            for key,values in phases.items():
                timings[f"{task['family']}/{method}/{key}"].extend(values)
            assert result["approved"] is None or result["approved"]==task["goal"] and task["proof"]
            summary = totals.setdefault(task["family"]+"/"+method,Counter())
            summary.update(total=1,approved=int(result["approved"] is not None),spent=result["spent"],
                           counterexamples=sum(e["result"]=="counterexample" for e in result["trace"]),
                           denied=sum(e["result"]=="denied" for e in result["trace"]))
            record[method] = result
        rows.append(dict(task=task,runs=record))
    frozen = torch.jit.load(str(EXECUTOR/"machine.pt"))
    portable,perf,interventions = [],[],[]
    for row in rows[:64]:
        task = row["task"]
        result = row["runs"]["recurrent"]
        if result["approved"] is not None and len(result["approved"])>=2 and len(portable)<8:
            b,t = performance(task,result["approved"],frozen)
            portable.append(b)
            perf.append(t)
        changed = run(task,"recurrent",models["recurrent"],defaultdict(list),"first")
        assert changed["approved"] is None or changed["approved"]==task["goal"]
        interventions.append(changed)
    assert portable
    malformed = deepcopy(portable[0])
    malformed["nodes"][malformed["root"]]["payload"]["contract"] = "alien"
    try:
        verify_bundle(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid proof accepted")
    forged = bundle_for([0,2],[0,1,2],0,(compile_predicate([0,2],0,False),
                                         compile_predicate([0,1,2],0,True)))
    try:
        verify_bundle(forged)
    except ValueError:
        pass
    else:
        raise AssertionError("rehashing a false premise granted authority")
    equivalent,witness = prove([0,2],[0,1,2],0)
    assert not equivalent and witness==1
    negative = [wasmtime_run(compile_predicate(p,0,specialized),witness,bytearray(65536))[0]
                for p,specialized in (([0,2],False),([0,1,2],True))]
    assert negative==[0,1]
    report = dict(summary=totals,rows=rows,feedback_interventions=interventions,
                  rejected=dict(input=witness,returns=negative),refinement=refinement_witness(),
                  hashes={name:digest((OUT/(name+".pt")).read_bytes()) for name in models})
    for name,value in (("outcomes.json",report),("proofs.json",portable)):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes()))
    if args.write:
        savings = [median(p["native_ns"][0])-median(p["native_ns"][1]) for p in perf]
        rng = Random(927)
        boot = sorted(mean(rng.choices(savings,k=len(savings))) for _ in range(2000))
        for p,delta in zip(perf,savings,strict=True):
            upfront = p["compilation_ns"]+p["verification_ns"]+p["persistence_ns"]+p["import_ns"]
            p["native_saving_ns"] = delta
            p["break_even_uses"] = int(upfront//delta)+1 if delta>0 else None
        (OUT/"timings.json").write_bytes(canonical(dict(synthesis=dict(timings),execution=perf,
            paired_mean_native_saving_bootstrap_95_ns=[boot[50],boot[1949]]))+b"\n")
    print(json.dumps(totals,indent=2))


if __name__=="__main__":
    main()
