"""Independent universal checker for bounded interval unions and affine patches."""
import json

import wasmtime
import z3

from frozen_call_checkpoint import digest
from persistent_evidence_v60 import Node, Store

CONTRACT = "signed-i32-piecewise-completion-v1"


def ranges(word,width):
    output = []
    point = 0
    while point<width:
        if not word>>point&1:
            point += 1
            continue
        start = point
        while point<width and word>>point&1:
            point += 1
        output.append([start,point])
    return output


def scope(task):
    data = {k:task[k] for k in ("width","offset","delta","fragment")}
    width,offset,delta = data["width"],data["offset"],data["delta"]
    if (type(width) is not int or not 1<=width<=14 or type(offset) is not int or
            not -(1<<31)<=offset<=(1<<31)-1-width or type(delta) is not int or
            not -(1<<31)<=delta<1<<31):
        raise ValueError("invalid signed or modular domain")
    if data["fragment"]=="predicate" and delta!=0 or data["fragment"]=="patch" and delta==0:
        raise ValueError("invalid fragment coefficient")
    if data["fragment"] not in ("predicate","patch"):
        raise ValueError("unknown fragment")
    return data


def wasm(data,segments):
    blocks = []
    for lo,hi in segments:
        lower,upper = data["offset"]+lo,data["offset"]+hi
        if hi-lo==1:
            code = f"local.get 0 i32.const {lower} i32.sub i32.eqz"
        else:
            code = (f"local.get 0 i32.const {lower} i32.lt_s i32.eqz "
                    f"local.get 0 i32.const {upper} i32.lt_s i32.mul")
        blocks.extend([code]+(["i32.add"] if blocks else []))
    body = " ".join(blocks) if blocks else "i32.const 0"
    if data["fragment"]=="patch":
        body = f"local.get 0 {body} i32.const {data['delta']} i32.mul i32.add"
    return bytes(wasmtime.wat2wasm(
        f'(module (memory (export "memory") 1 1) (func (export "run") (param i32) (result i32) {body}))'))


def checked_segments(data,segments):
    end = 0
    members = set()
    for lo,hi in segments:
        if type(lo) is not int or type(hi) is not int or not end<=lo<hi<=data["width"]:
            raise ValueError("intervals must be ordered, disjoint and in scope")
        members.update(range(lo,hi))
        end = hi
    return members


def prove(data,reference,candidate):
    x = z3.BitVec("value",32)
    def meaning(rows):
        condition = z3.Or(*(z3.And(x>=data["offset"]+lo,x<data["offset"]+hi) for lo,hi in rows))
        if data["fragment"]=="patch":
            return x+z3.If(condition,z3.BitVecVal(data["delta"],32),z3.BitVecVal(0,32))
        return z3.If(condition,z3.BitVecVal(1,32),z3.BitVecVal(0,32))
    solver = z3.SolverFor("QF_BV")
    solver.add(meaning(reference)!=meaning(candidate))
    outcome = solver.check()
    if outcome==z3.unsat:
        return True,None
    if outcome==z3.sat:
        value = solver.model().eval(x,model_completion=True).as_long()
        return False,value if value<2**31 else value-2**32
    raise RuntimeError("proof unresolved")


def publish(store,data,reference,candidate):
    sem = store.add(Node("assumptions",dict(scope=data,semantics=CONTRACT,fuel="completion-only")))
    parents = [store.add(Node("source",dict(segments=rows,binary=digest(wasm(data,rows)))))
               for rows in (reference,candidate)]
    query = store.add(Node("query",dict(obligation="universal-i32"),(sem,*parents)))
    return store.add(Node("certificate",dict(checker="finite-signed-partition-v1"),(query,)))


def validate(store,root,expected,reference):
    cert = store.load(root)
    if cert.kind!="certificate" or cert.payload!={"checker":"finite-signed-partition-v1"}:
        raise ValueError("unknown checker")
    query_key, = cert.parents
    query = store.load(query_key)
    if query.kind!="query" or query.payload!={"obligation":"universal-i32"}:
        raise ValueError("incorrect obligation")
    contract,left,right = [store.load(key) for key in query.parents]
    if contract!=Node("assumptions",dict(scope=expected,semantics=CONTRACT,fuel="completion-only")):
        raise ValueError("scope or fuel contract differs")
    if left.payload["segments"]!=reference:
        raise ValueError("reference identity differs")
    sets = []
    for node in (left,right):
        if node.kind!="source" or node.parents:
            raise ValueError("incorrect artifact provenance")
        rows = node.payload["segments"]
        sets.append(checked_segments(expected,rows))
        if digest(wasm(expected,rows))!=node.payload["binary"]:
            raise ValueError("binary identity differs")
    if sets[0]!=sets[1]:
        raise ValueError("signed partitions differ")
    return right.payload["segments"]


def import_archive(path,directory):
    bundle = json.loads(path.read_bytes())
    store = Store(directory)
    nodes = {}
    for key,row in bundle["nodes"].items():
        node = Node(row["kind"],row["payload"],tuple(row["parents"]))
        if any(parent not in bundle["nodes"] for parent in node.parents):
            raise ValueError("missing dependency")
        if store.add(node)!=key:
            raise ValueError("artifact identity differs")
        nodes[key] = node
    approved = {}
    for row in bundle["roots"]:
        expected = scope(row["scope"])
        approved[row["root"]] = dict(scope=expected,
            segments=validate(store,row["root"],expected,row["reference"]))
    for node in nodes.values():
        if node.kind=="decision":
            candidate,root = node.parents
            if (node.payload!={"action":"authorize"} or root not in approved or
                    nodes[candidate].kind!="candidate" or
                    nodes[candidate].payload!=approved[root]):
                raise ValueError("decision exceeds certificate authority")
    return len(approved)
