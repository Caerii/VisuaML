"""Reviewed separation of successful completion and logical checkpoint authority."""
import json
from pathlib import Path
import struct

import z3

from factorized_synthesis_v64 import signed
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store
from structural_decomposition_v66 import OUT
from verify_dependency_v65 import PAGE, SCRATCH

KINDS = ("fuse","fission","snapshot-reorder")
SEMANTICS = "rolling-record-checkpoint-v1"


def guard(keys,count,scratch):
    if (not 2<=len(keys)<=32 or any(type(k) is not int or not 1<=k<=12 for k in keys)
            or type(count) is not int or not 0<=count<=SCRATCH//(4*len(keys)) or scratch!=SCRATCH):
        raise ValueError("record or memory extent violates checkpoint contract")


def obligation(keys,kind):
    width = len(keys)
    original = [z3.BitVec(f"original_{i}",32) for i in range(width)]
    previous = z3.BitVec("prior_record_carry",32)
    reference = [a*x+keys[i-1]*(original[i-1] if i else previous)
                 for i,(a,x) in enumerate(zip(keys,original,strict=True))]
    scratch = [None]*width
    output = [None]*width
    if kind=="fuse":
        carry = previous
        for i in range(width):
            x = original[i]
            scratch[i] = x
            output[i] = keys[i]*x+keys[i-1]*carry
            carry = x
    else:
        scratch = original.copy()
        indices = reversed(range(width)) if kind=="snapshot-reorder" else range(width)
        for i in indices:
            left = scratch[i-1] if i else previous
            output[i] = keys[i]*scratch[i]+keys[i-1]*left
        carry = scratch[-1]
    arguments = (*reference,*original,original[-1])
    candidate = (*output,*scratch,carry)
    solver = z3.SolverFor("QF_BV")
    solver.add(z3.Or(*(u!=v for u,v in zip(arguments,candidate,strict=True))))
    if solver.check()!=z3.unsat:
        raise ValueError("record boundary induction failed")
    # Entire memory outside the two distinct record footprints is a frame.
    assert 4*width<=SCRATCH


def expected_certificate(keys,count,kind,authority,scratch):
    guard(keys,count,scratch)
    if kind not in KINDS or authority not in ("completion","record-boundary"):
        raise ValueError("unsupported transformation or observation boundary")
    sem = Node("assumptions",dict(version=SEMANTICS,scratch=scratch,
               snapshot_before_outputs=True,generated_from=digest(Path(__file__).read_bytes())))
    parent = digest(sem.bytes())
    children = []
    for i,a in enumerate(keys):
        b = keys[i-1]
        children.append(Node("obligation",dict(current=a,previous=b,reads=["original-current","original-left"],
            writes=["output-current","scratch-current"],carry="original-current"),(parent,)))
    root = Node("certificate",dict(keys=keys,records=count,schedule=kind,authority=authority,
        observation=["return","page","original-carry","record-index"]),(parent,*[digest(child.bytes()) for child in children]))
    return [sem,*children,root]


def publish(store,keys,count,kind,authority="record-boundary"):
    return [store.add(node) for node in expected_certificate(keys,count,kind,authority,SCRATCH)][-1]


def verify(store,root,keys,count,kind,authority,scratch=SCRATCH):
    nodes = expected_certificate(keys,count,kind,authority,scratch)
    for node in nodes:
        if store.load(digest(node.bytes()))!=node:
            raise ValueError("proof node differs from authoritative obligation")
    if root!=digest(nodes[-1].bytes()):
        raise ValueError("certificate does not grant requested authority")
    obligation(keys,kind)
    return True


def advance(page,keys,record,previous,kind):
    """One host-level record transaction. The loop may suspend after return."""
    width = len(keys)
    offset = 4*record*width
    original = tuple(struct.unpack_from("<i",page,offset+4*i)[0] for i in range(width))
    if kind in ("fission","snapshot-reorder"):
        for i,x in enumerate(original):
            struct.pack_into("<i",page,SCRATCH+offset+4*i,x)
        # The captured record makes all original reads stable. Distinct
        # output words can now retire in reverse order as well.
        indices = reversed(range(width)) if kind=="snapshot-reorder" else range(width)
        for i in indices:
            left = previous if i==0 else original[i-1]
            struct.pack_into("<i",page,offset+4*i,signed(keys[i]*original[i]+keys[i-1]*left))
        return original[-1]
    for i,x in enumerate(original):
        struct.pack_into("<i",page,SCRATCH+offset+4*i,x)
        struct.pack_into("<i",page,offset+4*i,signed(keys[i]*x+keys[i-1]*previous))
        previous = x
    return previous


def checkpoint(initial,keys,count,completed,carry,page):
    return canonical(dict(version=SEMANTICS,initial=digest(initial),keys=keys,count=count,
                          completed=completed,carry=carry,page=page.hex()))


def restore(raw,initial,keys,count):
    data = json.loads(raw)
    guard(keys,count,SCRATCH)
    n = data["completed"]
    if type(n) is not int or not 0<=n<=count or len(initial)!=PAGE:
        raise ValueError("bad boundary")
    page,carry = bytearray(initial),0
    for i in range(n):
        carry = advance(page,keys,i,carry,"fission")
    if checkpoint(initial,keys,count,n,carry,page)!=raw:
        raise ValueError("state differs from independent prefix")
    return n,carry,page


def exercise(store):
    import random

    keys,count = [2,5,3,7],14
    initial = random.Random(112).randbytes(PAGE)
    roots = {kind:publish(store,keys,count,kind) for kind in KINDS}
    for kind,root in roots.items():
        verify(store,root,keys,count,kind,"record-boundary")
    states = {kind:(bytearray(initial),0) for kind in KINDS}
    boundaries = []
    saved = None
    for i in range(count):
        for kind,(page,carry) in states.items():
            states[kind] = (page,advance(page,keys,i,carry,kind))
        assert states["fuse"]==states["fission"]==states["snapshot-reorder"]
        page,carry = states["fuse"]
        boundaries.append(dict(record=i+1,page=digest(page),carry=carry))
        if i==5:
            saved = checkpoint(initial,keys,count,i+1,carry,page)
    assert saved
    n,carry,page = restore(saved,initial,keys,count)
    for i in range(n,count):
        carry = advance(page,keys,i,carry,"snapshot-reorder")
        assert dict(record=i+1,page=digest(page),carry=carry)==boundaries[i]
    for root,requested,scratch in ((publish(store,keys,count,"fuse","completion"),"record-boundary",SCRATCH),
                                      (roots["fuse"],"record-boundary",0)):
        try:
            verify(store,root,keys,count,"fuse",requested,scratch)
        except ValueError:
            pass
        else:
            raise AssertionError("excess authority granted")
    corrupted = canonical(dict(json.loads(saved),carry=0))
    try:
        restore(corrupted,initial,keys,count)
    except ValueError:
        pass
    else:
        raise AssertionError("forged checkpoint accepted")
    return dict(keys=keys,count=count,roots=roots,boundaries=boundaries,
                saved_sha256=digest(saved),checkpoint=json.loads(saved),fixture_seed=112,rejected=["completion-only","alias","corrupted-carry"])


def main():
    from argparse import ArgumentParser
    from tempfile import TemporaryDirectory
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    opts = parser.parse_args()
    with TemporaryDirectory() as folder:
        store = Store(folder)
        if opts.write:
            result = exercise(store)
            nodes = {p.name:json.loads(p.read_bytes()) for p in Path(folder).iterdir()}
            (OUT/"checkpoints.json").write_bytes(canonical(dict(result=result,nodes=nodes))+b"\n")
        else:
            bundle = json.loads((OUT/"checkpoints.json").read_bytes())
            for key,raw in bundle["nodes"].items():
                node = Node(raw["kind"],raw["payload"],tuple(raw["parents"]))
                if store.add(node)!=key or any(p not in bundle["nodes"] for p in node.parents):
                    raise ValueError("invalid provenance")
            result = bundle["result"]
            for kind,root in result["roots"].items():
                assert verify(store,root,result["keys"],result["count"],kind,"record-boundary")
            from random import Random
            initial = Random(result["fixture_seed"]).randbytes(PAGE)
            raw = canonical(result["checkpoint"])
            assert digest(raw)==result["saved_sha256"]
            completed,carry,page = restore(raw,initial,result["keys"],result["count"])
            for i in range(completed,result["count"]):
                carry = advance(page,result["keys"],i,carry,"snapshot-reorder")
                assert result["boundaries"][i]==dict(record=i+1,page=digest(page),carry=carry)
            assert exercise(store)==result
    print("Verified three schedules across 14 declared boundaries")


if __name__=="__main__":
    main()
