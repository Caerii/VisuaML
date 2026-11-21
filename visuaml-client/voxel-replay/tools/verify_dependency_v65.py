"""Proof-carrying rolling stencil preserving the whole linear-memory page."""
import json
from pathlib import Path

import wasmtime
import z3

from frozen_call_checkpoint import digest
from persistent_evidence_v60 import Node, Store

PAGE = 65536
SCRATCH = PAGE//2
ABI = "i32-rolling-neighbor-page-v1"


def wasm(coefficients, variant):
    width = len(coefficients)
    stride = width*4
    cap = SCRATCH//stride

    def loop(stage):
        body = []
        for j,a in enumerate(coefficients):
            part = f"local.get 1 i32.const {stride} i32.mul i32.const {4*j} i32.add local.set 2 "
            read_base = "i32.const 32768 i32.add " if stage=="staged" else ""
            part += f"local.get 2 {read_base}i32.load local.set 3 "
            if stage in ("copy","fused","invalid"):
                part += "local.get 2 i32.const 32768 i32.add local.get 3 i32.store "
            if stage!="copy":
                b = coefficients[(j-1)%width]
                part += (f"local.get 2 local.get 3 i32.const {a} i32.mul "
                         f"local.get 4 i32.const {b} i32.mul i32.add i32.store ")
                part += ("local.get 2 i32.load " if stage=="invalid" else "local.get 3 ")+ "local.set 4 "
            body.append(part)
        return ("i32.const 0 local.set 1 block loop local.get 1 local.get 0 "
                "i32.lt_s i32.eqz br_if 1 "+" ".join(body)+
                " local.get 1 i32.const 1 i32.add local.set 1 br 0 end end ")

    code = loop("copy")+loop("staged") if variant=="staged" else loop(variant)
    wat = f"""(module (memory (export "memory") 1 1)
       (func (export "run") (param i32) (result i32) (local i32 i32 i32 i32)
       local.get 0 i32.const 0 i32.lt_s if i32.const -1 return end
       local.get 0 i32.const {cap} i32.gt_s if i32.const -1 return end
       {code} local.get 0))"""
    return bytes(wasmtime.wat2wasm(wat))


def contract():
    return Node("assumptions",dict(abi=ABI,page=PAGE,scratch=SCRATCH,overlap=False,
                authority="completion-with-sufficient-fuel",lowering=digest(Path(__file__).read_bytes())))


def leaf(a,b,semantics):
    return Node("edge",dict(a=a,b=b,reads=["current-original","previous-original"],
                writes=["current-output","scratch-current"],carry="current-original"),(semantics,))


def obligation(a,b,invalid=False):
    x,p = z3.BitVecs("original current_left",32)
    staged = (a*x+b*p,x,x)
    fused = (a*x+b*p,x,a*x+b*p if invalid else x)
    solver = z3.SolverFor("QF_BV")
    solver.add(z3.Or(*(u!=v for u,v in zip(staged,fused,strict=True))))
    status = solver.check()
    if status==z3.unsat:
        return None
    if status!=z3.sat:
        raise ValueError("unresolved induction step")
    model = solver.model()
    return [model.eval(v,model_completion=True).as_long() for v in (x,p)]


def certificate(keys,semantics,children,variant="fused"):
    return Node("certificate",dict(rule="rolling-carry-induction",width=len(keys),
                    upper_bound=SCRATCH//(4*len(keys)),staged=digest(wasm(keys,"staged")),
                    candidate=digest(wasm(keys,variant))),(semantics,*children))


def publish(store,keys):
    semantics = store.add(contract())
    children = [store.add(leaf(a,keys[i-1],semantics)) for i,a in enumerate(keys)]
    return store.add(certificate(keys,semantics,children))


def verify(store,root,expected,cache=None):
    if not 2<=len(expected)<=32 or any(type(k) is not int or not 1<=k<=12 for k in expected):
        raise ValueError("unsupported workload shape")
    artifact = store.load(root)
    semantic,*edges = artifact.parents
    if store.load(semantic)!=contract() or len(edges)!=len(expected):
        raise ValueError("changed scope or aliasing assumptions")
    cache = {} if cache is None else cache
    for i,(a,key) in enumerate(zip(expected,edges,strict=True)):
        b = expected[i-1]
        if key not in cache:
            node = store.load(key)
            if node!=leaf(a,b,semantic) or obligation(a,b) is not None:
                raise ValueError("changed effects, incompatible dependency or failed induction")
            cache[key] = (a,b,semantic)
        if cache[key]!=(a,b,semantic):
            raise ValueError("stale edge assignment")
    if artifact!=certificate(expected,semantic,edges):
        raise ValueError("lowering identity or loop contract differs")
    # Base case: preceding original is zero. The induction step identifies
    # the stored output, the same scratch write, and the next original carry.
    # Index n is bounded before touching either disjoint half-page.
    return True


def open_bundle(path,directory):
    data = json.loads(Path(path).read_bytes())
    store = Store(directory)
    for key,row in data["nodes"].items():
        node = Node(row["kind"],row["payload"],tuple(row["parents"]))
        if any(parent not in data["nodes"] for parent in node.parents) or store.add(node)!=key:
            raise ValueError("broken provenance")
    cache = {}
    for row in data["roots"]:
        verify(store,row["root"],row["keys"],cache)
    return len(data["roots"]),len(cache)
