"""Content-addressed evidence DAG for a reviewed, pure i32 fragment."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

import wasmtime

from frozen_call_checkpoint import canonical, digest

MASK = (1<<32)-1
SEMANTICS = "wasm-i32-affine-completion-v1"
OPS = {"x":"local.get 0","add":"i32.add","sub":"i32.sub","mul":"i32.mul"}


def wasm(tokens):
    body = " ".join(f"i32.const {value}" if op=="const" else OPS[op] for op,value in tokens)
    return bytes(wasmtime.wat2wasm(
        f'(module (memory (export "memory") 1 1) (func (export "run") (param i32) (result i32) {body}))'))


def affine(tokens):
    """Inductively normalize each stack value ax+b in Z/(2**32)."""
    stack = []
    if not 0<len(tokens)<=512:
        raise ValueError("unsupported expression length")
    for op,immediate in tokens:
        if type(immediate) is not int or not -(1<<31)<=immediate<1<<31:
            raise ValueError("invalid immediate")
        if op=="x" and immediate==0:
            stack.append((1,0))
        elif op=="const":
            stack.append((0,immediate&MASK))
        elif op in ("add","sub","mul") and immediate==0 and len(stack)>=2:
            c,d = stack.pop()
            a,b = stack.pop()
            if op=="mul":
                if a and c:
                    raise ValueError("non-affine product")
                pair = (a*d+b*c,b*d)
            else:
                sign = 1 if op=="add" else -1
                pair = (a+sign*c,b+sign*d)
            stack.append(tuple(x&MASK for x in pair))
        else:
            raise ValueError("unsupported instruction")
    result, = stack
    return list(result)


def smt(source,candidate):
    import z3

    x = z3.BitVec("input",32)

    def lower(tokens):
        stack = []
        for op,imm in tokens:
            if op in ("x","const"):
                stack.append(x if op=="x" else z3.BitVecVal(imm,32))
            else:
                b,a = stack.pop(),stack.pop()
                stack.append(a+b if op=="add" else a-b if op=="sub" else a*b)
        return stack[0]

    solver = z3.SolverFor("QF_BV")
    solver.add(lower(source)!=lower(candidate))
    status = solver.check()
    if status==z3.unsat:
        return True,None
    if status==z3.sat:
        return False,solver.model().eval(x,model_completion=True).as_long()
    raise RuntimeError("proof unresolved")


@dataclass(frozen=True)
class Node:
    kind: str
    payload: dict
    parents: tuple = ()

    def bytes(self):
        return canonical(dict(kind=self.kind,payload=self.payload,parents=list(self.parents)))


class Store:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lookup_ns = 0

    def path(self, key):
        if len(key)!=64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid node identity")
        return self.root/key

    def add(self, node):
        data = node.bytes()
        key = digest(data)
        path = self.path(key)
        if not path.exists():
            with NamedTemporaryFile(dir=self.root,delete=False) as temp:
                temp.write(data)
            os.replace(temp.name,path)
        return key

    def load(self, key):
        started = perf_counter_ns()
        data = self.path(key).read_bytes()
        self.lookup_ns += perf_counter_ns()-started
        if digest(data)!=key:
            raise ValueError("node digest mismatch")
        raw = json.loads(data)
        return Node(raw["kind"],raw["payload"],tuple(raw["parents"]))

    def affected(self, roots):
        """Return transitive descendants of changed nodes in this snapshot."""
        affected = set(roots)
        remaining = {p.name:self.load(p.name).parents for p in self.root.iterdir() if len(p.name)==64}
        while True:
            additions = {key for key,parents in remaining.items() if key not in affected and affected.intersection(parents)}
            if not additions:
                return sorted(affected)
            affected.update(additions)


def prepare(source,candidate):
    # The token validator defines the reviewed fragment before assembly.
    for tokens in (source,candidate):
        affine(tokens)
    binaries = [wasm(tokens) for tokens in (source,candidate)]
    assumption = Node("assumptions",dict(semantics=SEMANTICS, checker=digest(Path(__file__).read_bytes()),
                                        observes=["return","unchanged-memory"],fuel="completion-only"))
    inputs = [Node("source",dict(tokens=tokens,wasm=digest(binary)))
              for tokens,binary in zip((source,candidate),binaries,strict=True)]
    return assumption, inputs, binaries


def identity(node):
    return digest(node.bytes())


def query_node(scope):
    assumption,inputs,_ = scope
    return Node("query",dict(obligation="return-equivalence-for-all-i32"),
                tuple(identity(node) for node in [assumption,*inputs]))


def publish(store,scope,metrics=None):
    assumption,inputs,_ = scope
    query = query_node(scope)
    source,candidate = [node.payload["tokens"] for node in inputs]
    started = perf_counter_ns()
    valid,counterexample = smt(source,candidate)
    if metrics is not None:
        metrics["smt_ns"] = perf_counter_ns()-started
    started = perf_counter_ns()
    if not valid:
        return None,counterexample
    for node in (assumption,*inputs,query):
        store.add(node)
    witness = Node("witness",dict(affine=[affine(source),affine(candidate)]),(identity(query),))
    certificate = Node("certificate",dict(claim="equivalent"),(store.add(witness),))
    decision = Node("decision",dict(action="authorize"),(store.add(certificate),))
    root = store.add(decision)
    if metrics is not None:
        metrics["persist_ns"] = perf_counter_ns()-started
    return root,None


def validate(store,root,scope):
    """Only independently checked, in-scope evidence may authorize."""
    def exact(node, kind, payload):
        if node.kind!=kind or node.payload!=payload:
            raise ValueError(f"unexpected {kind}")

    def only(node):
        parent, = node.parents
        return store.load(parent)

    decision = store.load(root)
    exact(decision,"decision",{"action":"authorize"})
    certificate = only(decision)
    exact(certificate,"certificate",{"claim":"equivalent"})
    witness = only(certificate)
    query = only(witness)
    expected_query = query_node(scope)
    if query!=expected_query:
        raise ValueError("query scope changed")
    for key,expected in zip(query.parents,[scope[0],*scope[1]],strict=True):
        if store.load(key).bytes()!=expected.bytes():
            raise ValueError("input scope changed")
    derived = [affine(node.payload["tokens"]) for node in scope[1]]
    exact(witness,"witness",{"affine":derived})
    if derived[0]!=derived[1]:
        raise ValueError("non-equivalent expression")
    return True


def world_scope(scene, horizon):
    from pocket_world_inference import OUT as WORLD
    width,height = scene["width"],scene["height"]
    x,y = scene["target"]
    assert 1<=horizon<=8 and 0<=x<width and 0<=y<height
    cells = [[i,scene["walls"][i],scene["signal"][i]] for i in range(width*height)
             if abs(i%width-x)+abs(i//width-y)<=horizon]
    semantic = Node("assumptions",dict(semantics="cardinal-persistence-v1", graph=digest(
        (WORLD/"inference.pt").read_bytes()),checker=digest(Path(__file__).read_bytes())))
    source = Node("source",dict(width=width,height=height,target=[x,y],cells=cells))
    query = Node("query",dict(obligation="forecast",horizon=horizon),(identity(semantic),identity(source)))
    return semantic,source,query


def publish_world(store, scope, result):
    for node in scope:
        store.add(node)
    witness = Node("witness",dict(value=result),(identity(scope[-1]),))
    certificate = Node("certificate",dict(claim="forecast"),(store.add(witness),))
    decision = Node("decision",dict(action="record-forecast"),(store.add(certificate),))
    return store.add(decision)


def validate_world(store,root,scope):
    from replay_cellular_automata import scalar
    decision = store.load(root)
    cert = store.load(decision.parents[0])
    witness = store.load(cert.parents[0])
    semantic,source,query = scope
    expected = [Node("decision",dict(action="record-forecast"),(identity(cert),)),
                Node("certificate",dict(claim="forecast"),(identity(witness),))]
    if [decision,cert]!=expected or witness.kind!="witness" or witness.parents!=(identity(query),):
        raise ValueError("invalid forecast chain")
    if any(store.load(identity(node)).bytes()!=node.bytes() for node in scope):
        raise ValueError("forecast scope changed")
    data = source.payload
    width,height = data["width"],data["height"]
    walls,signal = [0]*(width*height),[0]*(width*height)
    for i,wall,value in data["cells"]:
        walls[i],signal[i] = wall,value
    for _ in range(query.payload["horizon"]):
        signal = scalar(walls,signal,width,height)
    x,y = data["target"]
    if witness.payload!={"value":signal[y*width+x]}:
        raise ValueError("forecast mismatch")
    return True


def worker(directory):
    import sys

    store = Store(directory)
    print("ready",flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        started = perf_counter_ns()
        try:
            world = request.get("domain")=="world"
            scope = world_scope(request["scene"],request["horizon"]) if world else prepare(
                request["source"],request["candidate"])
            scoped = perf_counter_ns()
            store.lookup_ns = 0
            checker = validate_world if world else validate
            result = checker(store,request["root"],scope)
            row = dict(accepted=result,scope_ns=scoped-started,
                       validate_ns=perf_counter_ns()-scoped,lookup_ns=store.lookup_ns)
        except (ValueError,OSError,KeyError,TypeError,IndexError,AssertionError) as error:
            row = dict(accepted=False,error=str(error))
        print(json.dumps(row),flush=True)


def import_bundle(path):
    """Stage untrusted bytes and independently verify every requested root."""
    from tempfile import TemporaryDirectory

    data = json.loads(Path(path).read_bytes())
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for key,node in data["nodes"].items():
            imported = Node(node["kind"],node["payload"],tuple(node["parents"]))
            if identity(imported)!=key:
                raise ValueError("bundle node identity mismatch")
            store.add(imported)
        if not data["replays"]:
            raise ValueError("no replay roots to validate")
        for request in data["replays"]:
            if request.get("domain")=="world":
                scope = world_scope(request["scene"],request["horizon"])
                validate_world(store,request["root"],scope)
            else:
                scope = prepare(request["source"],request["candidate"])
                validate(store,request["root"],scope)
            print(request["root"])


if __name__=="__main__":
    import sys
    if sys.argv[1]=="--bundle":
        import_bundle(sys.argv[2])
    else:
        worker(sys.argv[1])
