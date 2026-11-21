"""Replay held-out assemblies and persist the independently checked modules."""
from argparse import ArgumentParser
from collections import Counter
import json
import subprocess
import sys
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from assembled_loop_v61 import OUT, checkpoint, expression, initial, restore, sample, step
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import MASK, Store, affine, prepare, publish, smt, validate
from replay_frozen_recurrent import PACKED
from replay_guest_rewrite import EXECUTOR, invoke


def run(task,model,external=False):
    state = initial(task)
    while state["status"]=="running":
        if external:
            # Model the request encoding boundary of a stateless external
            # agent without charging for networking or a separate server.
            task,state = json.loads(canonical([task,state]))
        step(task,state,model)
    return state


def signed(value):
    return value if value<1<<31 else value-(1<<32)


def specialize(tokens):
    coefficient,offset = affine(tokens)
    if not coefficient:
        return [("const",signed(offset))]
    result = [("x",0)]
    if coefficient!=1:
        result.extend([("const",signed(coefficient)),("mul",0)])
    if offset:
        result.extend([("const",signed(offset)),("add",0)])
    return result


def verify_compilation(task,state,executor,store):
    reference = expression(task)
    emitted = state["registers"][3]
    compiled = specialize(emitted)
    scope = prepare(reference,compiled)
    root,_ = publish(store,scope)
    assert root and validate(store,root,scope)
    engine = wasmtime.Engine()
    native_times,exact_times,outputs,steps = [],[],[],[]
    for raw in scope[2]:
        room = wasmtime.Store(engine)
        instance = wasmtime.Instance(room,wasmtime.Module(engine,raw),[])
        wasm = instance.exports(room)["run"]
        native,exact = [],[]
        for _ in range(6):
            started = perf_counter_ns()
            result = wasm(room,task["argument"])
            native.append(perf_counter_ns()-started)
            started = perf_counter_ns()
            page,info = invoke(executor,raw,task["argument"],bytearray(65536))
            exact.append(perf_counter_ns()-started)
            assert page==bytes(65536) and info["status"]==1 and info["return_i32"]==result
        native_times.append(native[1:])
        exact_times.append(exact[1:])
        outputs.append(result)
        steps.append(info["source_steps"])
    assert outputs[0]==outputs[1]
    return dict(root=root,reference=reference,emitted=emitted,compiled=compiled,
                binaries=[digest(raw) for raw in scope[2]],result=outputs[0],source_steps=steps),dict(
                    native_ns=native_times,exact_ns=exact_times)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.resume:
        payload = json.load(sys.stdin)
        path = OUT/"recurrent.pt"
        model = torch.jit.load(str(path))
        state = restore(canonical(payload["checkpoint"]),payload["task"],model,digest(path.read_bytes()))
        while state["status"]=="running":
            step(payload["task"],state,model)
        print(canonical(state).decode())
        return
    rng = Random(97821)
    tasks = [sample(rng,holdout=True) for _ in range(100)]
    paths = {name:OUT/(name+".pt") for name in ("recurrent","attention","mlp")}
    models = {name:torch.jit.load(str(path)) for name,path in paths.items()}
    hashes = {name:digest(path.read_bytes()) for name,path in paths.items()}
    methods = {**models,"deterministic":None,"external":models["attention"]}
    summary,rows,timings = {},[],{}
    for task in tasks:
        expected = affine(expression(task))
        record = {}
        for name,model in methods.items():
            started = perf_counter_ns()
            state = run(task,model,external=name=="external")
            timings.setdefault(name,[]).append(perf_counter_ns()-started)
            calls = len(state["journal"])
            assert calls<=task["budget"]
            accepted = state["status"]=="proposed" and affine(state["registers"][3])==expected
            counts = summary.setdefault(name,Counter())
            counts.update(assigned=1,accepted=int(accepted),calls=calls,
                          exact_recipe=int([row["module"] for row in state["journal"]][:-1]==
                              [("add","sub","mul")[op] for op,_,_ in task["recipe"]]))
            record[name] = dict(accepted=accepted,status=state["status"],journal=state["journal"])
        rows.append(dict(task=task,runs=record))
    model = models["recurrent"]
    task = tasks[0]
    state = initial(task)
    for _ in range(2):
        step(task,state,model)
    mid = checkpoint(task,state,hashes["recurrent"])
    resumed = restore(mid,task,model,hashes["recurrent"])
    while resumed["status"]=="running":
        step(task,resumed,model)
    assert canonical(resumed)==canonical(run(task,model))
    restored = subprocess.run([sys.executable,__file__,"--resume"],
                              input=canonical(dict(task=task,checkpoint=json.loads(mid))),
                              capture_output=True,check=True)
    assert canonical(json.loads(restored.stdout))==canonical(resumed)
    for bad_task,bad_hash,bad_raw in (
        (dict(task,argument=task["argument"]^1),hashes["recurrent"],mid),
        (task,hashes["attention"],mid),
        (task,hashes["recurrent"],canonical(dict(json.loads(mid),state=dict(state,remaining=999)))),
    ):
        try:
            restore(bad_raw,bad_task,model,bad_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a stale or modified checkpoint")
    starved = dict(task,budget=1)
    insufficient = run(starved,model)
    assert insufficient["status"]=="fuel" and len(insufficient["journal"])==1
    assert canonical(restore(checkpoint(starved,insufficient,hashes["recurrent"]),starved,model,
                            hashes["recurrent"]))==canonical(insufficient)

    machine = EXECUTOR/"machine.pt"
    executor = torch.jit.load(str(machine))
    artifacts,bench = [],[]
    with TemporaryDirectory() as dirpath,patch(PACKED,side_effect=AssertionError("packed fallback")):
        store = Store(dirpath)
        for task in tasks[:12]:
            state = run(task,model)
            if state["status"]=="proposed" and affine(state["registers"][3])==affine(expression(task)):
                built,timing = verify_compilation(task,state,executor,store)
                artifacts.append(built)
                bench.append(timing)
        assert artifacts, "no recurrent assembly passed the independent compilation checks"
        nodes = {}
        pending = [record["root"] for record in artifacts]
        while pending:
            key = pending.pop()
            if key not in nodes:
                node = store.load(key)
                nodes[key] = json.loads(node.bytes())
                pending.extend(node.parents)
        bundle = dict(nodes=nodes,replays=[dict(root=record["root"],source=record["reference"],
                      candidate=record["compiled"]) for record in artifacts])
        sample_reference = artifacts[0]["reference"]
        invalid = [*artifacts[0]["compiled"],("const",1),("add",0)]
        assert affine(invalid)!=affine(sample_reference)
        rejection,argument = smt(sample_reference,invalid)
        assert not rejection and argument is not None
        # An accepted root cannot authorize a freshly changed candidate.
        rejected_scope = prepare(sample_reference,invalid)
        returns = []
        for raw in rejected_scope[2]:
            page, execution = invoke(executor,raw,signed(argument&MASK),bytearray(65536))
            assert page==bytes(65536) and execution["status"]==1
            returns.append(execution["return_i32"])
        assert returns[0]!=returns[1]
        try:
            validate(store,artifacts[0]["root"],rejected_scope)
        except ValueError:
            pass
        else:
            raise AssertionError("stale evidence authorized a changed candidate")
    result = dict(hashes=hashes,machine=digest(machine.read_bytes()),summary=summary,rows=rows,
                  restored_checkpoint=json.loads(mid),fuel_test=insufficient["status"],
                  specialized=artifacts,rejected=dict(program=invalid,counterexample=argument&MASK,returns=returns))
    if args.write:
        (OUT/"verified-assemblies.json").write_bytes(canonical(bundle)+b"\n")
        (OUT/"outcomes.json").write_bytes(canonical(result)+b"\n")
        (OUT/"timings.json").write_bytes(canonical(dict(controller_ns=timings,binary=bench))+b"\n")
    else:
        assert canonical(bundle)==canonical(json.loads((OUT/"verified-assemblies.json").read_bytes()))
        assert canonical(result)==canonical(json.loads((OUT/"outcomes.json").read_bytes()))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
