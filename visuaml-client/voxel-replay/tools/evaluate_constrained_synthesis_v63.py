"""Action-interface ablations and exact CEGIS replay for v63."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from math import log2
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from adaptive_synthesis_v62 import OUT as V62
from constrained_synthesis_v63 import OUT, STOP, WIDTH, VersionSpace, disagreement, state, teacher
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke
from verify_synthesis_v63 import import_archive, prove, publish, ranges, scope, validate, wasm


def audit_previous():
    prior = json.loads((V62/"outcomes.json").read_bytes())
    errors = Counter()
    for row in prior["rows"]:
        task = row["task"]
        if task["family"]!="skew_cost":
            continue
        for event in row["runs"]["recurrent"]["trace"]:
            if event["result"]=="denied":
                action = event["raw"]
                low = action<8 and event["state"][action]!=0
                charge = task["costs"][action] if action<8 else 1
                expensive = charge>event["budget"]
                reason = "both" if low and expensive else "repeat" if low else "budget"
                assert low or expensive
                errors[reason] += 1
    assert sum(errors.values())==63
    return dict(errors)


def masked_v62():
    from adaptive_synthesis_v62 import HALT, POINTS, PROGRAMS, WIDTH as OLD_WIDTH, encode
    from evaluate_adaptive_synthesis_v62 import points

    model = torch.jit.load(str(V62/"recurrent.pt"))
    rows = []
    archive = json.loads((V62/"outcomes.json").read_bytes())
    for item in archive["rows"]:
        task = item["task"]
        if task["family"]!="skew_cost":
            continue
        remaining = task["budget"]
        observed,events = [0]*POINTS,[]
        memory = torch.zeros(1,OLD_WIDTH)
        certified = False
        for _ in range(10):
            with torch.inference_mode():
                logits,memory = model(encode(observed,task["costs"],remaining,True).unsqueeze(0),memory)
            legal = torch.tensor([not observed[x] and task["costs"][x]<=remaining for x in range(POINTS)]+
                                 [remaining>0]*len(PROGRAMS)+[True])
            raw = int(logits[0].argmax())
            action = int(logits[0].masked_fill(~legal,-float("inf")).argmax())
            events.append(dict(raw=raw,dispatched=action))
            if action==HALT:
                break
            if action<POINTS:
                point = action
                remaining -= task["costs"][point]
            else:
                candidate = points(PROGRAMS[action-POINTS])
                remaining -= 1
                certified = candidate==task["goal"]
                if certified:
                    break
                point = min(set(candidate)^set(task["goal"]))
            observed[point] = 1 if point in task["goal"] else -1
        assert remaining>=0
        rows.append(dict(case=task["case"],certified=certified,events=events))
    return rows


def bounded(space, options, features, remaining):
    """Two-step lookahead values worst-case live version-space reduction."""
    viable = space.viable
    if len(viable)<=1:
        return teacher(options,features)
    best = (-float("inf"),STOP)
    for i,(option,row) in enumerate(zip(options,features,strict=True)):
        if not row[8] or option.kind=="stop":
            continue
        groups = defaultdict(list)
        if option.kind=="probe":
            for word in viable:
                groups[word>>option.operand&1].append(word)
        else:
            for word in viable:
                groups[disagreement(word,option.operand)].append(word)
        # For large groups this considers every second probe, including cheap
        # probes omitted from a greedy first-step shortlist.
        worst = 0
        for key,words in groups.items():
            if option.kind=="prove" and key==-1:
                continue
            residual = len(words)
            for candidate in options:
                if candidate.kind=="probe" and candidate.charge<=remaining-option.charge:
                    left = sum(word>>candidate.operand&1 for word in words)
                    residual = min(residual,max(left,len(words)-left))
            worst = max(worst,residual)
        score = (len(viable)-worst)/option.charge
        if score>best[0]:
            best = score,i
    return best[1]


def dispatch(mode,logits,legal,options):
    raw = int(logits.argmax())
    if mode in ("hard","reserve_proof"):
        return raw,int(logits.masked_fill(~legal,-float("inf")).argmax())
    if mode=="constrained":
        return raw,next(i for i in sorted(range(len(options)),key=lambda i:(-float(logits[i]),i)) if legal[i])
    if mode=="penalty":
        penalized = logits-(~legal)*1.5
        return raw,int(penalized.argmax())
    classes = {kind:[i for i,opt in enumerate(options) if opt.kind==kind and legal[i]]
               for kind in ("probe","prove","stop")}
    viable = [group for group in classes.values() if group]
    group = max(viable,key=lambda group:sum(float(logits[i]) for i in group)/len(group))
    return raw,max(group,key=lambda i:(float(logits[i]),-i))


def run(task,policy,model,mode,store,phases):
    reference = ranges(task["target"],task["width"])
    contract = scope(task)
    hypothesis = VersionSpace(task["width"],task["runs"])
    remaining = task["budget"]
    memory,trace = torch.zeros(1,WIDTH),[]
    parent = store.add(Node("assumptions",dict(task=digest(canonical(task)),scope=contract)))
    root = None
    for _ in range(task["width"]+2):
        started = perf_counter_ns()
        options,features = state(hypothesis,task["costs"],task["proof_cost"],remaining,task["proof"])
        phases["version_space"].append(perf_counter_ns()-started)
        legal = features[:,8].bool()
        if mode=="reserve_proof":
            legal &= torch.tensor([opt.kind!="probe" or task["proof"] and opt.charge+task["proof_cost"]<=remaining
                                   for opt in options])
        started = perf_counter_ns()
        if model is None:
            if policy=="bounded":
                raw = bounded(hypothesis,options,features,remaining)
            elif policy=="enumerative":
                raw = next((i for i,opt in enumerate(options) if opt.kind=="prove" and legal[i]),
                           teacher(options,features))
            else:
                raw = teacher(options,features)
            scheduled = raw
        else:
            with torch.inference_mode():
                logits,memory = model(features.unsqueeze(0),memory)
            raw,scheduled = dispatch(mode,logits[0],legal,options)
        phases["proposal"].append(perf_counter_ns()-started)
        request = options[scheduled]
        before = len(hypothesis.viable)
        event = dict(raw=raw,raw_legal=bool(legal[raw]),scheduled=scheduled,
                     request=vars(request),remaining=remaining,before=before,after=before)
        trace.append(event)
        # One authoritative dispatch gate for every policy, including the
        # soft-penalty ablation. No invalid request reaches a tool.
        if not legal[scheduled] or request.charge>remaining:
            event["result"] = "denied-before-dispatch"
            break
        if request.kind=="stop":
            event["result"] = "no-proof" if not task["proof"] else "abstained"
            break
        remaining -= request.charge
        if request.kind=="probe":
            point = request.operand
        else:
            assert hypothesis.contains(request.operand)
            started = perf_counter_ns()
            candidate = ranges(request.operand,task["width"])
            proposal = store.add(Node("candidate",dict(segments=candidate,scope=contract),(parent,)))
            phases["persistence"].append(perf_counter_ns()-started)
            event["candidate"] = proposal
            if candidate==reference:
                started = perf_counter_ns()
                root = publish(store,contract,reference,candidate)
                if validate(store,root,contract,reference)!=candidate:
                    raise AssertionError("checker disagrees")
                phases["publish_and_recheck"].append(perf_counter_ns()-started)
                event.update(result="certified",root=root,authorized_by=store.add(
                    Node("decision",dict(action="authorize"),(proposal,root))))
                break
            point = disagreement(task["target"],request.operand)
            parent = proposal
        value = task["target"]>>point&1
        event.update(result="example" if request.kind=="probe" else "counterexample",
                     point=point,value=value)
        started = perf_counter_ns()
        parent = store.add(Node("observation",dict(point=point,value=value,result=event["result"]),(parent,)))
        phases["persistence"].append(perf_counter_ns()-started)
        event["artifact"] = parent
        hypothesis = hypothesis.observe(point,value)
        assert hypothesis.contains(task["target"])
        event["after"] = len(hypothesis.viable)
        event["bits_removed"] = log2(before/event["after"])
    assert remaining>=0
    return dict(trace=trace,spent=task["budget"]-remaining,root=root)


def tasks():
    rng = Random(963201)
    for family in ("nominal","shifted_cost","compositional","no_proof"):
        for i in range(32):
            width = rng.randint(11,14) if family=="compositional" else rng.randint(7,9)
            runs = 3 if family=="compositional" else 2
            viable = VersionSpace(width,runs).viable
            if family=="compositional":
                viable = [word for word in viable if len(ranges(word,width))==3]
            patching = i%2==1
            yield dict(id=f"{family}-{i}",family=family,width=width,runs=runs,
                       fragment="patch" if patching else "predicate",delta=(57 if i%4==1 else -71) if patching else 0,
                       offset=(2**31-width-1 if i%8==1 else rng.randint(-300,300)),
                       target=rng.choice(viable),proof=family!="no_proof",
                       costs=[rng.randint(6,10) if family=="shifted_cost" else rng.randint(1,5)
                              for _ in range(width)],proof_cost=rng.randint(3,8),
                       budget=rng.randint(20,40) if family in ("compositional","shifted_cost") else rng.randint(9,22))
    original = json.loads((V62/"outcomes.json").read_bytes())
    for item in original["rows"]:
        task = item["task"]
        if task["family"]=="skew_cost":
            yield dict(id=f"v62-{task['case']}",family="matched_v62",width=8,runs=1,
                       fragment="predicate",delta=0,offset=task["offset"],
                       target=sum(1<<x for x in task["goal"]),proof=task["proof"],
                       costs=task["costs"],proof_cost=1,budget=task["budget"])


def sampled_replay(task,graph):
    contract = scope(task)
    values = ranges(task["target"],task["width"])
    pointwise = [[i,i+1] for i in range(task["width"]) if task["target"]>>i&1]
    proof_start = perf_counter_ns()
    assert prove(contract,pointwise,values)==(True,None)
    smt_ns = perf_counter_ns()-proof_start
    started = perf_counter_ns()
    source,candidate = [wasm(contract,parts) for parts in (pointwise,values)]
    lowering_ns = perf_counter_ns()-started
    samples = [[],[]]
    steps = []
    for index,binary in enumerate((source,candidate)):
        engine = wasmtime.Engine()
        room = wasmtime.Store(engine)
        handle = wasmtime.Instance(room,wasmtime.Module(engine,binary),[]).exports(room)["run"]
        for arg in (-2**31,2**31-1,task["offset"]-1,*range(task["offset"],task["offset"]+task["width"])):
            bit = int(0<=arg-task["offset"]<task["width"] and
                      task["target"]>>(arg-task["offset"])&1)
            expected = ((arg+task["delta"]*bit+2**31)%2**32)-2**31 if task["fragment"]=="patch" else bit
            with patch(PACKED,side_effect=AssertionError("packed fallback")):
                page,result = invoke(graph,binary,arg,bytearray(65536))
            assert page==bytes(65536) and result["return_i32"]==expected and handle(room,arg)==expected
        steps.append(result["source_steps"])
        for _ in range(32):
            start = perf_counter_ns()
            handle(room,task["offset"])
            samples[index].append(perf_counter_ns()-start)
    return dict(id=task["id"],source_steps=steps,binaries=[digest(source),digest(candidate)]),dict(
        samples=samples,lowering_ns=lowering_ns,smt_ns=smt_ns)


def negative(store):
    data = dict(width=8,offset=2147483638,delta=31,fragment="patch")
    ref,bad = [[1,3]],[[1,4]]
    certificate = publish(store,data,ref,bad)
    for root,contract in ((certificate,data),(certificate,dict(data,delta=0)),
                           (certificate,dict(data,width=9))):
        try:
            validate(store,root,contract,ref)
        except ValueError:
            pass
        else:
            raise AssertionError("forged certificate adopted")
    # A matching sample does not cover the differing interval. A rehashed
    # certificate with a different source-fuel contract also lacks authority.
    assert wasmtime_run(wasm(data,ref),0,bytearray(65536))[0]==0
    assert wasmtime_run(wasm(data,bad),0,bytearray(65536))[0]==0
    good = publish(store,data,ref,ref)
    query = store.load(store.load(good).parents[0])
    assumptions = store.load(query.parents[0])
    changed = store.add(Node("assumptions",dict(assumptions.payload,fuel="all-source-boundaries")))
    forged_query = store.add(Node(query.kind,query.payload,(changed,*query.parents[1:])))
    forged_cert = store.add(Node("certificate",dict(checker="finite-signed-partition-v1"),(forged_query,)))
    try:
        validate(store,forged_cert,data,ref)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-fuel contract adoption")
    valid,counterexample = prove(data,ref,bad)
    assert not valid
    left = wasmtime_run(wasm(data,ref),counterexample,bytearray(65536))[0]
    right = wasmtime_run(wasm(data,bad),counterexample,bytearray(65536))[0]
    assert left!=right
    return dict(input=counterexample,returns=[left,right])


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-bundle",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.import_bundle:
        with TemporaryDirectory() as folder:
            print(import_archive(OUT/"dag.json",folder))
        return
    names = ("recurrent-631","recurrent-632","attention-631","mlp-631")
    models = {name:torch.jit.load(str(OUT/(name+".pt"))) for name in names}
    methods = [(name,models[name],"hard") for name in names]+[
               (variant,models["recurrent-631"],variant) for variant in ("hierarchical","penalty","constrained","reserve_proof")]+[
               (name,None,None) for name in ("entropy","bounded","enumerative")]
    rows,archive,roots,summary,measurement = [],[],{}, {},defaultdict(list)
    with TemporaryDirectory() as folder:
        store = Store(folder)
        graph = torch.jit.load(str(EXECUTOR/"machine.pt"))
        for task in tasks():
            record = {}
            for name,model,mode in methods:
                time = defaultdict(list)
                start = perf_counter_ns()
                row = run(task,name,model,mode,store,time)
                time["elapsed"].append(perf_counter_ns()-start)
                for key,measure in time.items():
                    measurement[f"{task['family']}/{name}/{key}"].extend(measure)
                record[name] = row
                if row["root"]:
                    roots[row["root"]] = dict(root=row["root"],scope=scope(task),
                                              reference=ranges(task["target"],task["width"]))
                counts = summary.setdefault(task["family"]+"/"+name,Counter())
                last = row["trace"][-1]["result"]
                counts.update(tasks=1,certified=int(row["root"] is not None),spent=row["spent"],
                    raw_invalid=sum(not e["raw_legal"] for e in row["trace"]),
                    no_dispatch_denials=sum(e["result"]=="denied-before-dispatch" for e in row["trace"]),
                    counterexamples=sum(e["result"]=="counterexample" for e in row["trace"]),
                    abstained=int(last in ("abstained","no-proof")),unsupported_dispatched=0,
                    bits_removed_milli=round(1000*sum(e.get("bits_removed",0) for e in row["trace"])))
            assert record["constrained"]==record["recurrent-631"]
            assert not any(r["root"] for r in record.values()) or task["proof"]
            if (task["family"]=="compositional" and record["recurrent-631"]["root"] and
                    len(archive)<8):
                accepted,timing = sampled_replay(task,graph)
                overhead = sum(measurement[f"{task['family']}/recurrent-631/elapsed"][-1:] or [0])
                timing["synthesis_and_recheck_ns"] = overhead
                saving = median(timing["samples"][0])-median(timing["samples"][1])
                timing["break_even_repetitions"] = int((overhead+timing["lowering_ns"]+timing["smt_ns"])//saving)+1 if saving>0 else None
                archive.append(accepted)
                measurement["execution"].append(timing)
            rows.append(dict(task=task,runs=record))
        with TemporaryDirectory() as negative_folder:
            rejection = negative(Store(negative_folder))
        dag = dict(nodes={p.name:json.loads(p.read_bytes()) for p in Path(folder).iterdir()},
                   roots=list(roots.values()))
    audit = audit_previous()
    differences = {}
    rng = Random(772)
    for family in sorted({row["task"]["family"] for row in rows}):
        for mode in ("recurrent-631","recurrent-632","mlp-631","attention-631"):
            pair = [int(row["runs"][mode]["root"] is not None)-int(row["runs"]["bounded"]["root"] is not None)
                    for row in rows if row["task"]["family"]==family]
            boot = sorted(mean(rng.choices(pair,k=len(pair))) for _ in range(1000))
            differences[family+"/"+mode] = [boot[25],boot[974]]
    execution = measurement["execution"]
    if execution:
        savings = [median(pair["samples"][0])-median(pair["samples"][1]) for pair in execution]
        samples = sorted(mean(rng.choices(savings,k=len(savings))) for _ in range(1000))
        measurement["native_mean_saving_ci95_ns"] = [samples[25],samples[974]]
    report = dict(rows=rows,summary=summary,v62_denial_audit=audit,v62_mask_only=masked_v62(),paired_vs_bounded=differences,
                  replays=archive,negative=rejection,hashes={name:digest((OUT/(name+".pt")).read_bytes())
                                                             for name in names})
    for name,value in (("outcomes.json",report),("dag.json",dag)):
        path = OUT/name
        if args.write:
            path.write_bytes(canonical(value)+b"\n")
        else:
            assert canonical(value)==canonical(json.loads(path.read_bytes())), name
    if args.write:
        (OUT/"timings.json").write_bytes(canonical(dict(measurement))+b"\n")
    print(json.dumps(dict(audit=audit,summary=summary),indent=2))


if __name__=="__main__":
    main()
