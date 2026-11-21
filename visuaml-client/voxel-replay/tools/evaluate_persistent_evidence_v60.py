"""Cross-process import, independent exact execution, and cost accounting."""
from argparse import ArgumentParser
import json
from random import Random
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import wasmtime

from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import (
    Node, Store, identity, prepare, publish, publish_world, query_node, validate, world_scope,
)
from evaluate_evidence_economics import world_case
from pocket_world_inference import OUT as WORLD
from replay_guest_rewrite import EXECUTOR, invoke
from replay_frozen_recurrent import PACKED
from reusable_evidence_v59 import OUT as V59

OUT = V59.parent/"persistent-evidence-v60"
EMPTY = bytes(65536)
X = ("x",0)
ADD = ("add",0)
SUB = ("sub",0)
MUL = ("mul",0)


def programs():
    def const(value):
        return ("const",value)
    # Multiple generated modules with different source structure and proof cost.
    for n in (2,5,12,40):
        yield f"cancel-{n}",[X]+[const(n),ADD,const(n),SUB]*n,[X],True
    for value in (3,9,31):
        yield f"distribute-{value}",[X,const(7),ADD,const(value),MUL],[
            X,const(value),MUL,const(7*value),ADD],True
    yield "cancel-invalid",[X,const(13),ADD,const(12),SUB],[X],False
    yield "distribute-invalid",[X,const(7),ADD,const(9),MUL],[X,const(9),MUL,const(7),ADD],False


def timed(callback):
    start = perf_counter_ns()
    result = callback()
    return result,perf_counter_ns()-start


def native_pair(graph,binaries,argument):
    engine = wasmtime.Engine()
    samples = []
    results = []
    for binary in binaries:
        instance = wasmtime.Store(engine)
        exported = wasmtime.Instance(instance,wasmtime.Module(engine,binary),[]).exports(instance)
        native = exported["run"]
        expected = native(instance,argument)
        frozen = lambda: invoke(graph,binary,argument,bytearray(EMPTY))
        (page,report),_ = timed(frozen)
        assert (report["return_i32"],page,report["status"])==(expected,EMPTY,1)
        native_samples,vm_samples = [],[]
        for _ in range(7):
            result,delta = timed(lambda: native(instance,argument))
            assert result==expected
            native_samples.append(delta)
            (page,check),delta = timed(frozen)
            assert (check["return_i32"],page)==(expected,EMPTY)
            vm_samples.append(delta)
        results.append((expected,report["source_steps"],digest(binary)))
        samples.append(dict(native_ns=native_samples,exact_vm_ns=vm_samples))
    return results,samples


def exercise(graph,directory):
    store = Store(directory)
    worker = Path(__file__).with_name("persistent_evidence_v60.py")
    # Publish a seed certificate, then start the independent reader.
    seed = next(programs())
    prepublished,_ = publish(store,prepare(seed[1],seed[2]))
    started = perf_counter_ns()
    proc = subprocess.Popen([sys.executable,str(worker),directory],stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,text=True)
    assert proc.stdout.readline().strip()=="ready"
    startup = perf_counter_ns()-started

    def request(envelope):
        started = perf_counter_ns()
        data,serialize_ns = timed(lambda: json.dumps(envelope))
        proc.stdin.write(data+"\n")
        proc.stdin.flush()
        response = json.loads(proc.stdout.readline())
        response.update(round_trip_ns=perf_counter_ns()-started,serialize_ns=serialize_ns)
        return response

    def import_cert(source,candidate,root):
        return request(dict(source=source,candidate=candidate,root=root))

    rows,timings = [],[]
    try:
        assert import_cert(seed[1],seed[2],prepublished)["accepted"]
        for name,source,candidate,valid in programs():
            scope,scope_ns = timed(lambda: prepare(source,candidate))
            query = query_node(scope)
            _,hash_ns = timed(lambda: identity(query))
            phases = {}
            (root,witness),publish_ns = timed(lambda: publish(store,scope,phases))
            assert bool(root)==valid
            arg = 2147483647 if valid else witness
            if arg>=1<<31:
                arg-=1<<32
            execution,execution_times = native_pair(graph,scope[2],arg)
            assert (execution[0][0]==execution[1][0])==valid
            log = dict(name=name,scope_ns=scope_ns,hash_ns=hash_ns,cold_publish_ns=publish_ns,
                       execution=execution_times,**phases)
            sample_query = Node("query",dict(obligation="completion-sample",argument=arg,fuel=200,
                                      page_sha256=digest(EMPTY),authority="sample-only"),
                         tuple(identity(node) for node in [scope[0],*scope[1]]))
            observation = Node("witness",dict(executors=["frozen-reference","wasmtime"],
                                              results=execution),(identity(sample_query),))
            row = dict(name=name,source=source,candidate=candidate,valid=valid,root=root,
                       execution=dict(query=json.loads(sample_query.bytes()),witness=json.loads(observation.bytes())))
            if valid:
                # Normal form authority is rederived once per freshly imported root.
                # Repeat through a previously validated in-process root separately.
                assert validate(store,root,scope)
                hot,hot_times,remote = {identity(query):root},[],[]
                for _ in range(12):
                    known,delta = timed(lambda: hot.get(identity(query_node(prepare(source,candidate)))))
                    assert known==root
                    hot_times.append(delta)
                    reply = import_cert(source,candidate,root)
                    assert reply["accepted"]
                    remote.append(reply)
                changed = [*candidate,("const",1),ADD]
                assert not import_cert(source,changed,root)["accepted"]
                changed_scope = prepare(source,changed)
                affected,invalidate_ns = timed(lambda: store.affected([identity(scope[1][1])]))
                assert root in affected
                assert identity(changed_scope[1][1]) not in affected
                if name=="cancel-2":
                    # Rehash forged nodes all the way to the root. Digest checks
                    # succeed, then independent algebra rejects the false claim.
                    forged_query = store.add(query_node(changed_scope))
                    for node in (changed_scope[0],*changed_scope[1]):
                        store.add(node)
                    forged = store.add(Node("witness",dict(affine=[[1,0],[1,0]]),(forged_query,)))
                    forged_cert = store.add(Node("certificate",dict(claim="equivalent"),(forged,)))
                    forged_root = store.add(Node("decision",dict(action="authorize"),(forged_cert,)))
                    assert not import_cert(source,changed,forged_root)["accepted"]
                    bad_root = store.add(Node("decision",dict(action="unrelated"),()))
                    store.path(bad_root).write_text("{}")
                    assert not import_cert(source,candidate,bad_root)["accepted"]
                    store.path(bad_root).unlink()
                row.update(invalidated=len(affected),reused=True)
                log.update(hot_lookup_ns=hot_times,remote=remote,invalidate_ns=invalidate_ns)
            else:
                row["counterexample"]=arg
            rows.append(row)
            timings.append(log)
        args,fixture = world_case("valid",0,Random(301))
        _,_,walls,signal,target,horizon = args
        scene = fixture["scene"]
        simulator = torch.jit.load(str(WORLD/"inference.pt"))
        scope = world_scope(scene,horizon)
        value = simulator.forecast(walls,signal,0,target,0,horizon)[0]
        root = publish_world(store,scope,value)
        original = request(dict(domain="world",scene=scene,horizon=horizon,root=root))
        scene["walls"][0] ^= 1
        related = request(dict(domain="world",scene=scene,horizon=horizon,root=root))
        scene["walls"][target] = 1
        changed = request(dict(domain="world",scene=scene,horizon=horizon,root=root))
        assert original["accepted"] and related["accepted"] and not changed["accepted"]
        assert root in store.affected([identity(scope[1])])
        world = dict(root=root,outside_cone_reused=True,changed_target_rejected=True)
        # Persist the reachable node set from two roots as a portable audit bundle.
        scene["walls"][target] = 0
        nodes = {}
        pending = [prepublished,root]
        while pending:
            key = pending.pop()
            if key not in nodes:
                node = store.load(key)
                nodes[key] = json.loads(node.bytes())
                pending.extend(node.parents)
        bundle = dict(nodes=nodes,replays=[
            dict(root=prepublished,source=seed[1],candidate=seed[2]),
            dict(root=root,domain="world",scene=scene,horizon=horizon)])
        name,source,candidate,_ = list(programs())[3]
        assert name=="cancel-40"
        long_scope = prepare(source,candidate)
        long_root,_ = publish(store,long_scope)
        sweep = {}
        for hits in (0,4,8,12,16):
            samples = []
            for index in range(16):
                if index<hits:
                    reply = import_cert(source,candidate,long_root)
                    assert reply["accepted"]
                    samples.append(reply["round_trip_ns"])
                else:
                    def cold():
                        with TemporaryDirectory() as fresh:
                            return publish(Store(fresh),prepare(source,candidate))[0]
                    fresh_root,delta = timed(cold)
                    assert fresh_root==long_root
                    samples.append(delta)
            sweep[str(hits)] = samples
    finally:
        proc.stdin.close()
        assert proc.wait(timeout=15)==0
    return rows,world,bundle,dict(startup_ns=startup,cases=timings,
                                 world=[original,related,changed],hits_of_16=sweep)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    machine = EXECUTOR/"machine.pt"
    graph = torch.jit.load(str(machine))
    with TemporaryDirectory() as folder,patch(PACKED,side_effect=AssertionError("packed fallback")):
        rows,world,bundle,timings = exercise(graph,folder)
    result = dict(graph=digest(machine.read_bytes()),results=rows,world=world)
    if args.write:
        (OUT/"dag.json").write_bytes(canonical(bundle)+b"\n")
        (OUT/"persistent.json").write_bytes(canonical(result)+b"\n")
        (OUT/"timings.json").write_bytes(canonical(timings)+b"\n")
    else:
        assert canonical(bundle)==canonical(json.loads((OUT/"dag.json").read_bytes()))
        assert canonical(result)==canonical(json.loads((OUT/"persistent.json").read_bytes()))
    print(json.dumps(dict(verified=len(rows),authorized=sum(r["valid"] for r in rows),
                          startup_ms=timings["startup_ns"]/1e6)))


if __name__=="__main__":
    main()
