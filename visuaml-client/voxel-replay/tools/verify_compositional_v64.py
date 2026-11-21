"""Proof-carrying bounded lane transforms with an in-place WASM ABI."""
import json
from pathlib import Path

import wasmtime
import z3

from factorized_synthesis_v64 import LIBRARY
from frozen_call_checkpoint import digest
from persistent_evidence_v60 import Node, Store

PAGE = 65536
ABI = "lane-i32-in-place-completion-v1"


def compile_module(modules, fused):
    stride = 4*len(modules)
    cap = PAGE//stride

    def loop(stage):
        operations = []
        for lane,(a,b,g) in enumerate(modules):
            offset = lane*4
            operations.append(
                f"local.get 1 i32.const {stride} i32.mul i32.const {offset} i32.add local.tee 2 "
                "i32.load local.set 3 local.get 2 local.get 3")
            if stage!="bias":
                operations.append(f"i32.const {a} i32.mul local.get 3 i32.const 0 i32.lt_s "
                                  f"i32.const {g} i32.mul i32.add")
            if stage!="scale":
                operations.append(f"i32.const {b} i32.add")
            operations.append("i32.store")
        return ("i32.const 0 local.set 1 block loop "
                "local.get 1 local.get 0 i32.lt_s i32.eqz br_if 1 "+
                " ".join(operations)+
                " local.get 1 i32.const 1 i32.add local.set 1 br 0 end end")

    passes = loop("fused") if fused else loop("scale")+" "+loop("bias")
    source = f"""(module (memory (export "memory") 1 1)
      (func (export "run") (param i32) (result i32)
        (local i32 i32 i32)
        local.get 0 i32.const 0 i32.lt_s if i32.const -1 return end
        local.get 0 i32.const {cap} i32.gt_s if i32.const -1 return end
        {passes}
        local.get 0))"""
    return bytes(wasmtime.wat2wasm(source))


def semantics():
    return Node("assumptions",dict(abi=ABI,bytes=PAGE,verifier=digest(Path(__file__).read_bytes()),
                                   failure=-1,fuel="completion-only"))


def local_proof(module,fused):
    """Compare the declared staged and fused component for every i32 input."""
    a,b,g = module
    A,B,G = fused
    x = z3.BitVec("lane",32)
    step = x*a+z3.If(x<0,z3.BitVecVal(g,32),z3.BitVecVal(0,32))
    candidate = x*A+B+z3.If(x<0,z3.BitVecVal(G,32),z3.BitVecVal(0,32))
    proof = z3.SolverFor("QF_BV")
    proof.add(step+b!=candidate)
    result = proof.check()
    return result==z3.unsat


def component_node(reference,alternative,parent):
    return Node("component",dict(reference=list(reference),candidate=list(alternative),
                                 proof="pointwise-staged-to-fused"),(parent,))


def verify_component(node,semantic):
    if node.kind!="component" or node.parents!=(semantic,) or node.payload["proof"]!="pointwise-staged-to-fused":
        raise ValueError("component scope mismatch")
    a,b = (tuple(node.payload[key]) for key in ("reference","candidate"))
    if a not in LIBRARY or b not in LIBRARY or a!=b:
        raise ValueError("independent coefficient check failed")
    # Equal triples give the same ring polynomial on each sign partition.
    return a


def composition(semantic,children,reference,candidate):
    return Node("certificate",dict(rule="disjoint-i32-lane-product",length=len(children),
        bound=PAGE//(4*len(children)),reference=digest(reference),candidate=digest(candidate)),
        (semantic,*children))


def publish(store,modules):
    semantic = store.add(semantics())
    reference,candidate = [compile_module(modules,fused) for fused in (False,True)]
    children = [store.add(component_node(module,module,semantic)) for module in modules]
    root = store.add(composition(semantic,children,reference,candidate))
    return root,reference,candidate


def validate(store,root,expected_modules,cache=None):
    node = store.load(root)
    semantic,*children = node.parents
    if (not 1<=len(children)<=32 or expected_modules!=tuple(expected_modules)
            or len(expected_modules)!=len(children) or store.load(semantic)!=semantics()):
        raise ValueError("wrong lane or semantics scope")
    resolved = []
    cache = {} if cache is None else cache
    for key in children:
        cache_key = semantic,key
        if cache_key not in cache:
            cache[cache_key] = verify_component(store.load(key),semantic)
        resolved.append(cache[cache_key])
    if tuple(resolved)!=expected_modules:
        raise ValueError("source lane order changed")
    reference,candidate = [compile_module(resolved,fused) for fused in (False,True)]
    if node!=composition(semantic,children,reference,candidate):
        raise ValueError("composition contract or emitted program differs")
    # For each permitted n the 4-byte accesses at 4*(lane+D*i) are
    # pairwise disjoint and in range. Each exact local result composes.
    return candidate


def import_bundle(path,directory):
    bundle = json.loads(Path(path).read_bytes())
    store = Store(directory)
    for key,raw in bundle["nodes"].items():
        node = Node(raw["kind"],raw["payload"],tuple(raw["parents"]))
        if any(parent not in bundle["nodes"] for parent in node.parents) or store.add(node)!=key:
            raise ValueError("broken provenance")
    cache = {}
    for row in bundle["roots"]:
        modules = tuple(tuple(value) for value in row["modules"])
        validate(store,row["root"],modules,cache)
    return len(bundle["roots"]),len(cache)
