"""Measure performance identities separately from independently checked code."""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import torch
import wasmtime

from evaluate_dependency_v65 import expected
from frozen_call_checkpoint import canonical, digest
from optimization_histories_v68 import OUT as V68, Authority, admit, binary
from persistent_evidence_v60 import Store
from verify_dependency_v65 import PAGE, SCRATCH

OUT = V68.parent/"uncertain-optimization-v69"
VARIANTS = ("rolling","pointer")
METHOD = "alternating paired warmed batches of 40 calls, single host v1"


@dataclass(frozen=True)
class Environment:
    cpu: str
    machine: str
    wasmtime: str
    optimization: str

    def identity(self):
        return digest(canonical(asdict(self)))


def environment(level):
    info = Path("/proc/cpuinfo")
    cpu = next((line.partition(":")[2].strip() for line in info.read_text().splitlines()
                if line.startswith("model name")),platform.processor()) if info.exists() else platform.processor()
    return Environment(cpu,platform.machine(),version("wasmtime"),level)


def engine(level):
    config = wasmtime.Config()
    config.cranelift_opt_level = level
    return wasmtime.Engine(config)


def deploy(runtime,raw,page,keys,n):
    room = wasmtime.Store(runtime)
    exports = wasmtime.Instance(room,wasmtime.Module(runtime,raw),[]).exports(room)
    memory,fn = exports["memory"],exports["run"]
    memory.write(room,page)
    assert fn(room,n)==n
    assert memory.read(room,0,PAGE)==expected(keys,n,page)[1]
    return lambda: fn(room,n)


def run(option,runtime,keys,n,page,authority):
    results,proofs,stages = {},{},{}
    for name in ("reload",*VARIANTS):
        if name=="reload":
            raw = binary(keys,name)
        else:
            start = perf_counter_ns()
            root,raw,work = authority.prove(keys,name)
            verified_ns = perf_counter_ns()-start
            start = perf_counter_ns()
            assert authority.prove(keys,name)[:2]==(root,raw)
            lookup_ns = perf_counter_ns()-start
            start = perf_counter_ns()
            assert admit(authority.store,root,keys,name,set())==raw
            independent_ns = perf_counter_ns()-start
            proofs[name] = dict(root=root,binary=digest(raw),cold_checks=work["fresh"])
            stages[name] = dict(verification_ns=verified_ns+independent_ns,retrieval_ns=lookup_ns,
                                independently_rechecked_ns=independent_ns)
        start = perf_counter_ns()
        results[name] = deploy(runtime,raw,page,keys,n)
        if name!="reload":
            stages[name]["deployment_ns"] = perf_counter_ns()-start
    rng = Random(127 + option)
    observations = defaultdict(list)
    for i in range(16):
        indices = list(results)
        rng.shuffle(indices)
        for name in indices:
            f = results[name]
            assert f()==n
            start = perf_counter_ns()
            for _ in range(40):
                assert f()==n
            observations[name].append((perf_counter_ns()-start)/40)
    quick = defaultdict(list)
    start = perf_counter_ns()
    for i in range(5):
        for name in ("reload",*VARIANTS) if i%2 else (*reversed(VARIANTS),"reload"):
            f = results[name]
            mark = perf_counter_ns()
            for _ in range(16):
                assert f()==n
            quick[name].append((perf_counter_ns()-mark)/16)
    return dict(samples=dict(observations),quick=dict(quick),quick_ns=perf_counter_ns()-start,stages=stages,
                powers=proofs)


def main():
    parser = ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if not args.measure:
        raise ValueError("profiling changes the evidence identity; explicitly use --measure")
    torch.set_num_threads(1)
    rng = Random(690011)
    walls,roots = [],[]
    with TemporaryDirectory() as folder:
        store = Store(folder)
        engines = {level:engine(level) for level in ("speed","none")}
        for split,repeats in (("train",2),("calibrate",1),("holdout",2)):
            for width in (8,16):
                for index in range(repeats):
                    keys = [rng.randrange(1,13) for _ in range(width)]
                    for mode in VARIANTS:
                        authority = Authority(store)
                        root,_,_ = authority.prove(keys,mode)
                        roots.append(dict(root=root,keys=keys,mode=mode))
                    for level,runtime in engines.items():
                        for profile in ("short","full"):
                            n = 47 if profile=="short" else SCRATCH//(4*width)
                            target = environment(level)
                            if index%2:
                                page = (b"\x00\xff\x03\x80"*16384)
                                distribution = "periodic"
                            else:
                                page = rng.randbytes(PAGE)
                                distribution = "uniform-bytes"
                            result = run(index,runtime,keys,n,page,Authority(store))
                            walls.append(dict(id=len(walls),split=split,width=width,profile=profile,n=n,keys=keys,
                                hardware=asdict(target),hardware_id=target.identity(),distribution=distribution,
                                input_sha256=digest(page),method=METHOD,**result))
        nodes = {path.name:json.loads(path.read_bytes()) for path in store.root.iterdir()}
    (OUT/"performance.json").write_bytes(canonical(walls)+b"\n")
    (OUT/"proofs.json").write_bytes(canonical(dict(roots=roots,nodes=nodes))+b"\n")
    print(f"Measured {len(walls)} performance scopes on one CPU")


if __name__=="__main__":
    main()
