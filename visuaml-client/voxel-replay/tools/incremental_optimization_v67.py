"""Typed revision dependencies and reviewed completion-only stencil rewrites."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import wasmtime
import z3

from frozen_call_checkpoint import digest
from persistent_evidence_v60 import Node
from structural_decomposition_v66 import OUT as V66
from verify_dependency_v65 import PAGE, SCRATCH

OUT = V66.parent/"incremental-optimization-v67"
SCHEMAS = ("rolling","specialized")
AUTHORITY = "successful-completion-full-page"


@dataclass(frozen=True)
class Slot:
    domain: str
    kind: str
    index: int


class RevisionIndex:
    def __init__(self):
        self.values = {}
        self.parents = {}
        self.children = defaultdict(set)

    def get(self,slot,dependencies,build):
        if slot in self.values:
            if self.parents[slot]!=dependencies:
                raise ValueError("stale dependency declaration")
            return self.values[slot]
        if any(parent not in self.values for parent in dependencies):
            raise ValueError("missing dependency")
        value = build()
        self.parents[slot] = dependencies
        self.values[slot] = value
        for parent in dependencies:
            self.children[parent].add(slot)
        return value

    def invalidate(self,roots):
        visited,pending = set(),list(roots)
        while pending:
            slot = pending.pop()
            if slot in visited or slot not in self.values:
                continue
            visited.add(slot)
            pending.extend(self.children.pop(slot,()))
        for slot in visited:
            for parent in self.parents.pop(slot):
                self.children[parent].discard(slot)
            del self.values[slot]
        return visited


def compile_stencil(keys,mode):
    if mode not in ("reload",*SCHEMAS):
        raise ValueError("unknown lowering")
    stride = 4*len(keys)
    operations = []
    for j,a in enumerate(keys):
        b = keys[j-1]
        scale = lambda k: "" if mode=="specialized" and k==1 else f"i32.const {k} i32.mul "
        operations.append(
            f"local.get 1 i32.const {stride} i32.mul i32.const {4*j} i32.add local.set 2 "
            "local.get 2 i32.load local.set 3 "
            "local.get 2 i32.const 32768 i32.add local.get 3 i32.store "
            f"local.get 2 local.get 3 {scale(a)}local.get 4 {scale(b)}i32.add i32.store "+
            ("local.get 2 i32.const 32768 i32.add i32.load " if mode=="reload" else "local.get 3 ")+
            "local.set 4 ")
    cap = SCRATCH//stride
    body = " ".join(operations)
    return bytes(wasmtime.wat2wasm(f"""(module (memory (export "memory") 1 1)
      (func (export "run") (param i32) (result i32) (local i32 i32 i32 i32)
       local.get 0 i32.const 0 i32.lt_s if i32.const -1 return end
       local.get 0 i32.const {cap} i32.gt_s if i32.const -1 return end
       block loop local.get 1 local.get 0 i32.lt_s i32.eqz br_if 1
       {body} local.get 1 i32.const 1 i32.add local.set 1 br 0 end end
       local.get 0))"""))


def semantic_node():
    return Node("assumptions",dict(abi="rolling-reload-v67",page=PAGE,scratch=SCRATCH,
                authority=AUTHORITY,checker=digest(Path(__file__).read_bytes())))


def coefficient_node(key,semantic):
    return Node("source-op",dict(coefficient=key), (semantic,))


def effect_node(a,b,left,right,semantic):
    return Node("dependence",dict(a=a,b=b,read=["original-value","previous-original"],
        write=["scratch-current","output-current"],alias=False),(semantic,left,right))


def obligation(effect,mode):
    return Node("obligation",dict(schema=mode,authority=AUTHORITY),(effect,))


def check_smt(a,b,mode):
    x,p = z3.BitVecs("input previous",32)
    page = z3.Array("before",z3.BitVecSort(16),z3.BitVecSort(32))
    address = z3.BitVecVal(SCRATCH,16)
    scratch = z3.Store(page,address,x)
    original_next = z3.Select(scratch,address)
    a_term = x if mode=="specialized" and a==1 else x*a
    b_term = p if mode=="specialized" and b==1 else p*b
    solver = z3.SolverFor("QF_AUFBV")
    solver.add(z3.Or((a*x+b*p)!=(a_term+b_term),original_next!=x))
    if solver.check()!=z3.unsat:
        raise ValueError("reviewed rewrite obligation failed")


def checked_node(proof):
    return Node("checked",dict(result="unsat",verifier="stencil-z3-v1"),(proof,))


def module_node(keys,mode,semantic,checks):
    return Node("certificate",dict(schema=mode,authority=AUTHORITY,keys=keys,
                bounds=[0,SCRATCH//(4*len(keys))],
                reference=digest(compile_stencil(keys,"reload")),
                emitted=digest(compile_stencil(keys,mode))),
                (semantic,*checks))


class Campaign:
    def __init__(self,store,reuse):
        self.index = RevisionIndex()
        self.store,self.reuse = store,reuse
        self.semantic = store.add(semantic_node())
        self.checked = set()
        self.keys,self.mode = (),None

    def revise(self,keys,mode):
        if mode not in SCHEMAS or not 2<=len(keys)<=32 or any(type(k) is not int or not 1<=k<=12 for k in keys):
            raise ValueError("unsupported rewrite")
        start = perf_counter_ns()
        roots = [Slot("compiler","source",i) for i,k in enumerate(self.keys)
                 if not self.reuse or len(keys)!=len(self.keys) or keys[i]!=k]
        if self.mode!=mode:
            roots.extend(s for s in self.index.values if s.kind=="proof")
        invalidated = self.index.invalidate(roots)
        source,edge,proof = ([Slot("compiler",kind,i) for i in range(len(keys))]
                             for kind in ("source","effect","proof"))
        reused,issued,recomputed,smt_ns = 0,0,0,0
        if not self.reuse:
            self.checked.clear()
        checks = []
        for i,k in enumerate(keys):
            self.index.get(source[i],(),lambda: self.store.add(coefficient_node(k,self.semantic)))
        for i,a in enumerate(keys):
            b = keys[i-1]
            recomputed += edge[i] not in self.index.values
            dep = self.index.get(edge[i],(source[i],source[i-1]),lambda: self.store.add(
                effect_node(a,b,self.index.values[source[i]],self.index.values[source[i-1]],self.semantic)))

            def solve():
                nonlocal reused,issued,smt_ns
                query = self.store.add(obligation(dep,mode))
                if self.reuse and query in self.checked:
                    reused += 1
                else:
                    begun = perf_counter_ns()
                    check_smt(a,b,mode)
                    smt_ns += perf_counter_ns()-begun
                    issued += 1
                    self.checked.add(query)
                return self.store.add(checked_node(query))

            if proof[i] in self.index.values:
                reused += 1
            checks.append(self.index.get(proof[i],(edge[i],),solve))
        analysis = perf_counter_ns()-start
        start = perf_counter_ns()
        root = self.store.add(module_node(keys,mode,self.semantic,checks))
        publication = perf_counter_ns()-start
        self.keys,self.mode = keys,mode
        return root,dict(fresh=issued,reused=reused,recomputed=recomputed,
                         invalidated=sorted((slot.kind,slot.index) for slot in invalidated)),dict(
                            analysis_ns=analysis,smt_ns=smt_ns,publication_ns=publication)


def verify(store,root,keys,mode,cache):
    if not 2<=len(keys)<=32 or any(type(k) is not int or not 1<=k<=12 for k in keys):
        raise ValueError("unsupported claimed layout")
    expected_scope = semantic_node()
    sid = digest(expected_scope.bytes())
    if store.load(sid)!=expected_scope or mode not in SCHEMAS:
        raise ValueError("different proof authority")
    artifact = store.load(root)
    if len(artifact.parents)!=len(keys)+1:
        raise ValueError("wrong layout")
    checked = []
    for i,a in enumerate(keys):
        b = keys[i-1]
        left,right = [digest(coefficient_node(value,sid).bytes()) for value in (a,b)]
        effect = effect_node(a,b,left,right,sid)
        eid = digest(effect.bytes())
        query = obligation(eid,mode)
        qid = digest(query.bytes())
        result = checked_node(qid)
        rid = digest(result.bytes())
        if artifact.parents[i+1]!=rid:
            raise ValueError("reordered or stale dependency")
        if rid not in cache:
            if (store.load(left)!=coefficient_node(a,sid) or store.load(right)!=coefficient_node(b,sid)
                    or store.load(eid)!=effect or store.load(qid)!=query or store.load(rid)!=result):
                raise ValueError("proof DAG differs")
            check_smt(a,b,mode)
            cache.add(rid)
        checked.append(rid)
    if artifact!=module_node(keys,mode,sid,checked):
        raise ValueError("changed code or scope")
    return True
