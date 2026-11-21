"""Measure actual deployment payback, cache growth, and migration economics."""
from argparse import ArgumentParser
import json
from pathlib import Path
from random import Random
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from frozen_call_checkpoint import canonical, digest
from measure_uncertainty_v69 import OUT, deploy, engine
from migrate_snapshot_v67 import certificate, packet, resume, step
from optimization_histories_v68 import Authority, Branch, admit, binary
from persistent_evidence_v60 import Node, Store
from verify_dependency_v65 import PAGE


def timed(fn):
    start = perf_counter_ns()
    value = fn()
    return value,perf_counter_ns()-start


def migration():
    rng = Random(6933)
    original = rng.randbytes(PAGE)
    keys = [2,3,5,7]
    rows = []
    with TemporaryDirectory() as folder:
        store = Store(Path(folder)/"evidence")
        file = Path(folder)/"snapshot.json"
        for remaining in (4,32,256,1024):
            n = 4+remaining
            total = 2*n
            root = certificate(store,keys,n)
            state,page = dict(record=0,phase="prepare",carry=0,fuel=total),bytearray(original)
            for _ in range(9):
                assert step(state,page,keys,n,"rolling")
            assert state["phase"]=="snapshot-ready"
            export,load,validation,rolling,reverse = [],[],[],[],[]
            for i in range(20):
                def write():
                    raw = canonical(packet(original,keys,n,root,state,page))
                    file.write_bytes(raw)
                    return len(raw)
                size,ns = timed(write)
                export.append(ns)
                data,ns = timed(lambda: json.loads(file.read_bytes()))
                load.append(ns)
                restored,ns = timed(lambda: resume(data,original,keys,n,root,Store(store.root),total))
                validation.append(ns)
                for kind in (("rolling","reverse") if i%2 else ("reverse","rolling")):
                    memory,live = page.copy(),state.copy()
                    at = perf_counter_ns()
                    while step(live,memory,keys,n,kind):
                        pass
                    (rolling if kind=="rolling" else reverse).append(perf_counter_ns()-at)
                    assert (live,digest(memory))==restored
            net = [before-write-read-resume_all for before,write,read,resume_all in
                   zip(rolling,export,load,validation,strict=True)]
            rows.append(dict(remaining_records=remaining,packet_bytes=size,export_ns=export,load_ns=load,
                authorized_resume_ns=validation,rolling_suffix_ns=rolling,reverse_suffix_ns=reverse,
                net_switch_ns=net,median_net_ns=median(net)))
    return rows


def measured_deployment():
    data = json.loads((OUT/"performance.json").read_bytes())
    row = next(row for row in data if row["split"]=="holdout" and row["width"]==8
               and row["profile"]=="full" and row["hardware"]["optimization"]=="none")
    keys,n = row["keys"],row["n"]
    page = Random(98268).randbytes(PAGE)
    runtime = engine("none")
    start = perf_counter_ns()
    with TemporaryDirectory() as directory:
        store = Store(directory)
        proof_begin = perf_counter_ns()
        root,raw,_ = Authority(store).prove(keys,"pointer")
        assert root==row["powers"]["pointer"]["root"]
        assert admit(Store(directory),root,keys,"pointer",set())==raw
        proof_time = perf_counter_ns()-proof_begin
        reversed_,setup = timed(lambda: deploy(runtime,raw,page,keys,n))
    end_to_end = perf_counter_ns()-start
    baseline = deploy(runtime,binary(keys,"reload"),page,keys,n)
    samples = []
    rng = Random(221)
    for uses in (256,16000):
        timings = {"base":[],"pointer":[]}
        calls = {"base":baseline,"pointer":reversed_}
        for i in range(7):
            for which in calls if i%2 else reversed(calls):
                call = calls[which]
                start = perf_counter_ns()
                for _ in range(uses):
                    assert call()==n
                timings[which].append(perf_counter_ns()-start)
        net = [a-b-end_to_end for a,b in zip(timings["base"],timings["pointer"],strict=True)]
        boot = sorted(mean(rng.choices(net,k=len(net))) for _ in range(1500))
        samples.append(dict(uses=uses,proof_ns=proof_time,deployment_ns=setup,upfront_ns=end_to_end,
            runtime_ns=timings,paired_net_ns=net,paired_mean_ci95_ns=[boot[37],boot[1462]],median_net_ns=median(net)))
    return dict(observation=row["id"],root=root,hardware_id=row["hardware_id"],samples=samples)


def reuse():
    common = [2,5,9,4,11,6,8,3]
    visits = []
    rng = Random(2609)
    for i in range(48):
        layout = common.copy()
        if i%4:
            j = rng.randrange(len(layout))
            layout[j] = (layout[j]+i)%12+1
        visits.append(layout)
    rows = {}
    for setting in ("isolated","shared"):
        with TemporaryDirectory() as folder:
            store = Store(folder)
            shared = Authority(store)
            branch = Branch(store)
            source = branch.edit(common)
            sibling = branch.fork()
            records = []
            for i,layout in enumerate(visits):
                selected = (branch,sibling)[i%2]
                revision,maintenance = timed(lambda: selected.edit(layout))
                active = shared if setting=="shared" else Authority(store)
                mode = "pointer" if i%3==0 else "rolling"
                (root,raw,work),search = timed(lambda: active.prove(layout,mode))
                assert admit(store,root,layout,mode,set())==raw
                records.append(dict(revision=revision,root=root,mode=mode,fresh=work["fresh"],
                    roots=len(active.roots),premises=len(active.checked),maintenance_ns=maintenance,
                    proof_path_ns=search))
            try:
                branch.rollback(sibling.head)
            except ValueError:
                pass
            else:
                raise AssertionError("cross-ancestry rollback accepted")
            branch.rollback(source)
            assert branch.head==source
            rows[setting] = dict(records=records,nodes=len(list(Path(folder).iterdir())),
                                bytes=sum(p.stat().st_size for p in Path(folder).iterdir()))
    assert [r["root"] for r in rows["shared"]["records"]]==[r["root"] for r in rows["isolated"]["records"]]
    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if not args.measure:
        raise ValueError("fresh lifecycle timings require --measure")
    result = dict(migration=migration(),deployment=measured_deployment(),reuse=reuse())
    (OUT/"lifecycle.json").write_bytes(canonical(result)+b"\n")
    print("Measured authorized deployments, four checkpoint sizes and 48 proof requests per cache policy")


if __name__=="__main__":
    main()
