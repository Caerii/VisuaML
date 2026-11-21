"""Reviewed byte-pointer lowering, branch identity, and campaign value model."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import wasmtime
import z3

from frozen_call_checkpoint import canonical, digest
from incremental_optimization_v67 import Campaign, OUT as V67, compile_stencil, verify
from persistent_evidence_v60 import Node
from verify_dependency_v65 import PAGE, SCRATCH

OUT = V67.parent/"optimization-histories-v68"
MODES = ("rolling","specialized","pointer")


def pointer_binary(keys,wrong_stride=False):
    stride = 4*len(keys)
    cap = SCRATCH//stride
    parts = []
    for i,a in enumerate(keys):
        address = f"local.get 1 i32.const {4*i} i32.add "
        parts.append(address+"i32.load local.set 3 "+
            address+"i32.const 32768 i32.add local.get 3 i32.store "+
            address+f"local.get 3 i32.const {a} i32.mul local.get 4 i32.const {keys[i-1]} "
            "i32.mul i32.add i32.store local.get 3 local.set 4 ")
    wat = f"""(module (memory (export "memory") 1 1)
    (func (export "run") (param i32) (result i32) (local i32 i32 i32 i32)
      local.get 0 i32.const 0 i32.lt_s if i32.const -1 return end
      local.get 0 i32.const {cap} i32.gt_s if i32.const -1 return end
      local.get 0 i32.const {stride} i32.mul local.set 2
      block loop local.get 1 local.get 2 i32.lt_s i32.eqz br_if 1
      {" ".join(parts)}
      local.get 1 i32.const {stride-int(wrong_stride)*4} i32.add local.set 1 br 0 end end
      local.get 0))"""
    return bytes(wasmtime.wat2wasm(wat))


def binary(keys,mode):
    return pointer_binary(keys) if mode=="pointer" else compile_stencil(keys,mode)


def pointer_proof(width,wrong_stride=False):
    n,i,j = z3.Ints("records record lane")
    stride = 4*width
    cap = SCRATCH//stride
    p = i*stride
    solver = z3.Solver()
    solver.add(0<=i,i<n,n<=cap,0<=j,j<width)
    solver.add(z3.Or(p+stride-int(wrong_stride)*4!=(i+1)*stride,
                     p+4*j<0,p+4*j+4>SCRATCH,p+SCRATCH+4*j+4>PAGE,
                     p>=2**31,n*stride>=2**31))
    status = solver.check()
    if status==z3.unsat:
        return None
    if status!=z3.sat:
        raise ValueError("address proof unresolved")
    return [solver.model().eval(v).as_long() for v in (n,i,j)]


def pointer_node(keys,parent,wrong=False):
    return Node("certificate",dict(authority="successful-completion-full-page",
        rule="byte-pointer-induction-v1",width=len(keys),bounds=[0,SCRATCH//(4*len(keys))],
        checker=digest(Path(__file__).read_bytes()),emitted=digest(pointer_binary(keys,wrong))), (parent,))


def admit(store,root,keys,mode,cache):
    if mode not in MODES:
        raise ValueError("unknown schema")
    if mode!="pointer":
        verify(store,root,keys,mode,cache)
    else:
        node = store.load(root)
        parent, = node.parents
        verify(store,parent,keys,"rolling",cache)
        if node!=pointer_node(keys,parent) or pointer_proof(len(keys)) is not None:
            raise ValueError("pointer scope or induction changed")
    return binary(keys,mode)


class Authority:
    """Proof memos are shared across forks; current revision identity is local."""
    def __init__(self,store):
        self.store = store
        self.checked = set()
        self.validation_cache = set()
        self.roots = {}

    def prove(self,keys,mode):
        key = tuple(keys),mode
        if key not in self.roots:
            campaign = Campaign(self.store,True)
            campaign.checked = self.checked
            root,work,_ = campaign.revise(keys,"rolling" if mode=="pointer" else mode)
            if mode=="pointer":
                root = self.store.add(pointer_node(keys,root))
            raw = admit(self.store,root,keys,mode,self.validation_cache)
            self.roots[key] = root,raw
            return root,raw,work
        root,raw = self.roots[key]
        return root,raw,dict(fresh=0,reused=len(keys),recomputed=0,invalidated=[])


@dataclass(frozen=True)
class Revision:
    keys: tuple
    parent: str | None

    def node(self):
        return Node("revision",dict(keys=list(self.keys),abi="one-page-stencil"),
                    () if self.parent is None else (self.parent,))


class Branch:
    def __init__(self,store,head=None):
        self.store,self.head = store,head

    def edit(self,keys):
        self.head = self.store.add(Revision(tuple(keys),self.head).node())
        return self.head

    def fork(self):
        return Branch(self.store,self.head)

    def rollback(self,destination):
        cursor = self.head
        while cursor is not None and cursor!=destination:
            parents = self.store.load(cursor).parents
            cursor = parents[0] if parents else None
        if cursor is None:
            raise ValueError("rollback destination is not an ancestor")
        self.head = destination


def obligations(keys,mode):
    return frozenset(("rolling" if mode=="pointer" else mode,a,keys[i-1])
                     for i,a in enumerate(keys))


def price(keys,mode,known,proven,installed):
    """Microsecond accounting, separately calibrated from execution telemetry."""
    symbol = (tuple(keys),mode)
    missing = obligations(keys,mode)-known
    proof = 0 if symbol in proven else 130+470*len(missing)+(180 if mode=="pointer" else 0)
    deploy = 0 if symbol==installed else 2450
    return proof,deploy


def advance(task,action,mode,known,proven,installed,estimate):
    """Pure accounting transition. The actual proof gate is external to planning."""
    keys = task["keys"]
    if action=="keep":
        return 0,known,proven,None,0
    symbol = (tuple(keys),mode)
    proof,deploy = price(keys,mode,known,proven,installed)
    charge = proof if action=="prove" else proof+deploy
    if charge>task["budget"]:
        return None
    return ((estimate*task["uses"] if action=="apply" else 0)-charge,
            known|obligations(keys,mode),proven|{symbol},symbol if action=="apply" else None,charge)


def decisions(task,known,proven,installed):
    yield "keep",None
    for mode in MODES:
        proof,deploy = price(task["keys"],mode,known,proven,installed)
        if proof and proof<=task["budget"]:
            yield "prove",mode
        if proof+deploy<=task["budget"]:
            yield "apply",mode


def optimize(tasks,estimates,known,proven,installed,beam=None):
    """Finite-horizon exact DP or cost-aware beam when a cap is given."""
    if beam is None:
        @lru_cache(None)
        def value(t,known,proven,installed):
            if t==len(tasks):
                return 0,()
            best = (-float("inf"),())
            for act,mode in decisions(tasks[t],known,proven,installed):
                gain,learned,proved,next_,_ = advance(tasks[t],act,mode,known,proven,installed,
                                                      estimates[t].get(mode,0))
                tail,path = value(t+1,learned,proved,next_)
                if gain+tail>best[0]:
                    best = gain+tail,((act,mode),*path)
            return best
        return value(0,known,proven,installed)
    states = [(0,(),known,proven,installed)]
    for t,task in enumerate(tasks):
        next_states = {}
        for value,path,known,proven,installed in states:
            for act,mode in decisions(task,known,proven,installed):
                gain,learned,proved,next_,_ = advance(task,act,mode,known,proven,installed,
                                                       estimates[t].get(mode,0))
                key = learned,proved,next_
                choice = (value+gain,(*path,(act,mode)),*key)
                if key not in next_states or choice[0]>next_states[key][0]:
                    next_states[key] = choice
        states = sorted(next_states.values(),key=lambda state:-state[0])[:beam]
    return states[0][:2]
