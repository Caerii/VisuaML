"""Exact scoped derivations from independently rechecked v60 roots."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from frozen_call_checkpoint import canonical
from persistent_evidence_v60 import Node, Store, identity, prepare, publish, validate


def derive(store, expressions, left, right):
    a,b,c = expressions
    ab,bc = prepare(a,b),prepare(b,c)
    if identity(ab[0])!=identity(bc[0]) or identity(ab[1][1])!=identity(bc[1][0]):
        raise ValueError("incompatible equivalence scopes")
    if not (validate(store,left,ab) and validate(store,right,bc)):
        raise ValueError("invalid premise")
    ac = prepare(a,c)
    return Node("derived-equivalence",dict(semantics=identity(ac[0]),
                endpoints=[identity(n) for n in ac[1]],bridge=identity(ab[1][1])),(left,right))


def check(store, root, expressions):
    node = store.load(root)
    left,right = node.parents
    if node!=derive(store,expressions,left,right):
        raise ValueError("derived scope differs")
    return True


def bounded_relations(intervals, edges):
    """Edges express x[to] >= x[from] + lag, with nonnegative integer lag."""
    lower = [bounds[0] for bounds in intervals]
    if any(lo>hi for lo,hi in intervals):
        raise ValueError("inconsistent bounds")
    for iteration in range(len(lower)):
        changed = False
        for src,dst,lag in edges:
            if not 0<=src<len(lower) or not 0<=dst<len(lower) or lag<0:
                raise ValueError("invalid certified relation")
            if lower[dst]<lower[src]+lag:
                lower[dst] = lower[src]+lag
                changed = True
        if any(value>bounds[1] for value,bounds in zip(lower,intervals,strict=True)):
            raise ValueError("inconsistent bounds")
        if not changed:
            return lower
    raise ValueError("positive cycle")


def main():
    import argparse
    from recurrent_evidence_v61 import OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--import-bundle",type=Path)
    args = parser.parse_args()
    if args.import_bundle:
        bundle = json.loads(args.import_bundle.read_bytes())
        with TemporaryDirectory() as folder:
            store = Store(folder)
            for key,data in bundle["nodes"].items():
                node = Node(data["kind"],data["payload"],tuple(data["parents"]))
                if identity(node)!=key:
                    raise ValueError("node digest mismatch")
                store.add(node)
            if not check(store,bundle["root"],bundle["inputs"]):
                raise ValueError("derived validation failed")
        print(bundle["root"])
        return
    x,k,add = ["x",0],["const",3],["add",0]
    a,b,c = [x,k,add],[k,x,add],[x,k,add,["const",0],add]
    with TemporaryDirectory() as folder:
        store = Store(folder)
        ab,_ = publish(store,prepare(a,b))
        bc,_ = publish(store,prepare(b,c))
        root = store.add(derive(store,(a,b,c),ab,bc))
        # A new reader uses only disk nodes and its independent validation.
        assert check(Store(folder),root,(a,b,c))
        try:
            check(Store(folder),root,(a,b,[x]))
        except ValueError:
            pass
        else:
            raise AssertionError("stale derived root")
        affected = store.affected([identity(prepare(a,b)[1][1])])
        assert root in affected
        nodes = {p.name:json.loads(p.read_bytes()) for p in Path(folder).iterdir()}
    # B is a later target on the same single-lane corridor. All arrivals
    # cross A, then at least two edges. Verify the exact distance structure.
    import collections
    graph = {0:[1],1:[0,2],2:[1,3],3:[2,4],4:[3,5],5:[4]}
    def distances(start,blocked=-1):
        seen,queue = {start:0},collections.deque([start])
        while queue:
            pc = queue.popleft()
            for nxt in graph[pc]:
                if nxt!=blocked and nxt not in seen:
                    seen[nxt]=seen[pc]+1
                    queue.append(nxt)
        return seen
    assert 5 not in distances(0,3)
    lag = distances(3)[5]
    bounds = [(3,4),(3,7)]
    assert bounded_relations(bounds,[(0,1,lag)])==[3,5]
    result = dict(root=root,nodes=nodes,inputs=[a,b,c],invalidated=affected,
                  relation=dict(graph=graph,intervals=bounds,lower=[3,5]))
    path = OUT/"composition.json"
    if args.write:
        path.write_bytes(canonical(result)+b"\n")
    else:
        assert canonical(result)==canonical(json.loads(path.read_bytes()))
    print(root)


if __name__=="__main__":
    main()
