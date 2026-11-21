"""Reviewed snapshot-ready migration, separate from WASM completion proofs."""
from argparse import ArgumentParser
import json
from pathlib import Path
from random import Random
import struct
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import z3

from factorized_synthesis_v64 import signed
from frozen_call_checkpoint import canonical, digest
from incremental_optimization_v67 import OUT
from persistent_evidence_v60 import Node, Store
from verify_dependency_v65 import PAGE, SCRATCH

ABI = "stencil-snapshot-ready-v1"


def scope(keys,n):
    if not 2<=len(keys)<=32 or type(n) is not int or not 0<=n<=SCRATCH//(4*len(keys)):
        raise ValueError("invalid extent")
    if any(type(k) is not int or not 1<=k<=12 for k in keys):
        raise ValueError("invalid coefficient")
    return Node("assumptions",dict(abi=ABI,keys=keys,count=n,scratch=SCRATCH,page=PAGE,
        boundary="snapshot-ready",charge="one preparation and one commit per record",
        checker=digest(Path(__file__).read_bytes())))


def certificate(store,keys,n,root=None):
    binding = scope(keys,n)
    sid = digest(binding.bytes())
    node = Node("certificate",dict(reads=["pending-snapshot","original-left-carry"],
        writes=["current-output"],preserves=["scratch","other-records"],
        schedules=["rolling","reverse"]),(sid,))
    if root is not None and (store.load(sid)!=binding or store.load(root)!=node):
        raise ValueError("checkpoint authority, layout or semantics changed")
    words = [z3.BitVec(f"x{i}",32) for i in range(len(keys))]
    previous = z3.BitVec("carry",32)
    rolling,reverse = [],[None]*len(keys)
    carry = previous
    for i,value in enumerate(words):
        rolling.append(value*keys[i]+carry*keys[i-1])
        carry = value
    for i in reversed(range(len(keys))):
        reverse[i] = words[i]*keys[i]+(words[i-1] if i else previous)*keys[i-1]
    solver = z3.SolverFor("QF_BV")
    solver.add(z3.Or(carry!=words[-1],*(a!=b for a,b in zip(rolling,reverse,strict=True))))
    if solver.check()!=z3.unsat:
        raise ValueError("schedule proof failed")
    if root is None:
        store.add(binding)
        return store.add(node)
    return root


def step(state,page,keys,n,order):
    if order not in ("rolling","reverse") or state["phase"] not in ("prepare","snapshot-ready"):
        raise ValueError("unknown checkpoint schedule")
    if state["fuel"]==0 or state["record"]==n:
        return False
    offset = state["record"]*4*len(keys)
    size = 4*len(keys)
    if state["phase"]=="prepare":
        page[SCRATCH+offset:SCRATCH+offset+size] = page[offset:offset+size]
        state["phase"] = "snapshot-ready"
    else:
        data = struct.unpack_from(f"<{len(keys)}i",page,SCRATCH+offset)
        indices = reversed(range(len(keys))) if order=="reverse" else range(len(keys))
        for i in indices:
            left = data[i-1] if i else state["carry"]
            struct.pack_into("<i",page,offset+4*i,signed(keys[i]*data[i]+keys[i-1]*left))
        state.update(phase="prepare",record=state["record"]+1,carry=data[-1])
    state["fuel"] -= 1
    return True


def packet(initial,keys,n,root,state,page):
    return dict(version=ABI,initial=digest(initial),layout=keys,records=n,root=root,
                state=state.copy(),page=page.hex())


def resume(data,initial,keys,n,root,store,total):
    certificate(store,keys,n,root)
    page = bytearray(initial)
    state = dict(record=0,phase="prepare",carry=0,fuel=total)
    remaining = data["state"]["fuel"]
    if type(remaining) is not int or not 0<=remaining<=total or len(initial)!=PAGE:
        raise ValueError("invalid checkpoint budget")
    while state["fuel"]>remaining:
        if not step(state,page,keys,n,"rolling"):
            raise ValueError("impossible checkpoint boundary")
    if state["phase"]!="snapshot-ready" or packet(initial,keys,n,root,state,page)!=data:
        raise ValueError("checkpoint differs from verified prefix")
    while step(state,page,keys,n,"reverse"):
        pass
    return state,digest(page)


def build():
    keys,n,total,seed = [2,3,5,7],9,18,12067
    initial = Random(seed).randbytes(PAGE)
    with TemporaryDirectory() as folder:
        store = Store(folder)
        root = certificate(store,keys,n)
        page,st = bytearray(initial),dict(record=0,phase="prepare",carry=0,fuel=total)
        for _ in range(7):
            assert step(st,page,keys,n,"rolling")
        assert st["phase"]=="snapshot-ready"
        saved = packet(initial,keys,n,root,st,page)
        while step(st,page,keys,n,"rolling"):
            pass
        result = (st,digest(page))
        assert resume(saved,initial,keys,n,root,store,total)==result
        starved = dict(record=0,phase="prepare",carry=0,fuel=1)
        page = bytearray(initial)
        assert step(starved,page,keys,n,"rolling")
        frozen = bytes(page)
        assert not step(starved,page,keys,n,"reverse") and bytes(page)==frozen
        assert page[:SCRATCH]==initial[:SCRATCH]
        try:
            resume(dict(saved,state=dict(saved["state"],carry=saved["state"]["carry"]^1)),
                   initial,keys,n,root,store,total)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted carry accepted")
        semantics = scope(keys,n)
        aliased = store.add(Node(semantics.kind,dict(semantics.payload,scratch=0)))
        proper = store.load(root)
        wrong_scope = store.add(Node(proper.kind,proper.payload,(aliased,)))
        wrong_authority = store.add(Node(proper.kind,dict(proper.payload,schedules=["completion-only"]),proper.parents))
        for invalid in (wrong_scope,wrong_authority):
            try:
                certificate(store,keys,n,invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong alias or completion scope accepted")
        for changed in (keys[::-1],keys[:-1]):
            try:
                certificate(store,changed,n,root)
            except (ValueError,OSError):
                pass
            else:
                raise AssertionError("stale layout accepted")
        nodes = {p.name:json.loads(p.read_bytes()) for p in Path(folder).iterdir()}
    return dict(seed=seed,total=total,saved=saved,expected=list(result),nodes=nodes,
                rejected=["stale-layout","corrupted-carry","insufficient-commit-budget","aliased-scratch","completion-only"])


def main():
    ap = ArgumentParser()
    ap.add_argument("--write",action="store_true")
    ap.add_argument("--benchmark",action="store_true")
    args = ap.parse_args()
    path = OUT/"migration.json"
    if args.write:
        path.write_bytes(canonical(build())+b"\n")
    else:
        data = json.loads(path.read_bytes())
        with TemporaryDirectory() as folder:
            store = Store(folder)
            for key,r in data["nodes"].items():
                node = Node(r["kind"],r["payload"],tuple(r["parents"]))
                if any(p not in data["nodes"] for p in node.parents) or store.add(node)!=key:
                    raise ValueError("broken archive")
            saved = data["saved"]
            initial = Random(data["seed"]).randbytes(PAGE)
            repeats = []
            for _ in range(24 if args.benchmark else 1):
                started = perf_counter_ns()
                report = resume(saved,initial,saved["layout"],saved["records"],saved["root"],store,data["total"])
                repeats.append(perf_counter_ns()-started)
                assert list(report)==data["expected"]
            if args.benchmark:
                (OUT/"migration-timings.json").write_bytes(canonical(dict(resume_ns=repeats,
                    bytes=len(canonical(saved))))+b"\n")
        assert canonical(build())==canonical(data)
    print("Verified snapshot-ready migration and exact completion")


if __name__=="__main__":
    main()
