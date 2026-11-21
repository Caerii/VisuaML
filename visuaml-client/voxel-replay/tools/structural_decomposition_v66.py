"""Exact edit-aware tree inference and width-bounded variable elimination."""
from functools import lru_cache
from itertools import product
from math import prod
from random import Random

import torch
from torch import nn

from dependent_synthesis_v65 import OUT as V65
from frozen_call_checkpoint import canonical

OUT = V65.parent/"structural-decomposition-v66"


class CachedTree:
    def __init__(self,space):
        self.size,self.domain = space.size,range(1,space.library+1)
        self.links = [[] for _ in range(space.size)]
        self.tables = {}
        self.cache = {}
        self.recomputed = 0
        for a,b,pairs in space.factors:
            self.links[a].append(b)
            self.links[b].append(a)
            self.install(a,b,pairs)

    def install(self,a,b,pairs):
        self.tables[a,b] = frozenset(pairs)
        self.tables[b,a] = frozenset((y,x) for x,y in pairs)

    def edit(self,a,b,pairs):
        """Invalidate each directed message whose source side includes the edit."""
        if (a,b) not in self.tables:
            raise ValueError("only existing tree factors may change")
        if frozenset(pairs)==self.tables[a,b]:
            return []
        self.install(a,b,pairs)
        invalidated = []
        pending = [(a,b),(b,a)]
        while pending:
            src,dst = pending.pop()
            # An uncached arrow cannot have a cached dependent; skip that subtree.
            if (src,dst) not in self.cache:
                continue
            del self.cache[src,dst]
            invalidated.append((src,dst))
            pending.extend((dst,nxt) for nxt in self.links[dst] if nxt!=src)
        return sorted(invalidated)

    def message(self,src,dst):
        key = src,dst
        if key not in self.cache:
            incoming = [self.message(other,src) for other in self.links[src] if other!=dst]
            weights = [prod(vector[i] for vector in incoming) for i in range(len(self.domain))]
            factor = self.tables[key]
            self.cache[key] = tuple(sum(w for x,w in zip(self.domain,weights,strict=True)
                                       if (x,y) in factor) for y in self.domain)
            self.recomputed += 1
        return self.cache[key]

    def marginal(self,node):
        incoming = [self.message(src,node) for src in self.links[node]]
        return tuple(prod(m[i] for m in incoming) for i in range(len(self.domain)))


def graph(n,edges):
    if n<=0 or any(a==b or not 0<=a<n or not 0<=b<n for a,b in edges):
        raise ValueError("invalid graph")
    return tuple(tuple(sorted({b if a==v else a for a,b in edges if v in (a,b)})) for v in range(n))


def remove(g,node):
    neighbors = set(g[node])
    return tuple(tuple(sorted(((set(branch)|neighbors)-{node,v}) if v in neighbors else set(branch)-{node}))
                 if v!=node else () for v,branch in enumerate(g))


def features(g,node,active):
    adjacent = g[node]
    d = len(adjacent)
    missing = sum(b not in g[a] for i,a in enumerate(adjacent) for b in adjacent[i+1:])
    return [d/8,missing/24,sum(len(g[v]) for v in adjacent)/40,
            max((len(g[v]) for v in adjacent),default=0)/8,len(active)/32,
            missing/max(1,d*(d-1)/2)]


@lru_cache(maxsize=40000)
def optimal_width(g,active):
    if not active:
        return 0
    return min(max(len(g[v]),optimal_width(remove(g,v),tuple(w for w in active if w!=v)))
               for v in active)


def order(g,kind,model=None):
    active = list(range(len(g)))
    selected = []
    while active:
        if model is not None:
            with torch.inference_mode():
                data = torch.tensor([features(g,v,active) for v in active])
                node = active[int(model(data).argmin())]
        elif kind=="min_degree":
            node = min(active,key=lambda v:len(g[v]))
        elif kind=="optimal":
            node = min(active,key=lambda v:max(len(g[v]),optimal_width(remove(g,v),tuple(
                w for w in active if w!=v))))
        else:
            node = min(active,key=lambda v:(features(g,v,active)[1],len(g[v]),v))
        selected.append(node)
        g = remove(g,node)
        active.remove(node)
    return selected


def decomposition(g,sequence):
    if sorted(sequence)!=list(range(len(g))):
        raise ValueError("order is not a permutation")
    bags = []
    for v in sequence:
        bags.append([v,*g[v]])
        g = remove(g,v)
    positions = {v:i for i,v in enumerate(sequence)}
    parents = [min((positions[w] for w in bag[1:]),default=None) for bag in bags]
    # The elimination bags construct a forest. Independently check the
    # running-intersection relation along the path to every parent.
    for i,bag in enumerate(bags):
        if parents[i] is not None and not set(bag[1:]).issubset(bags[parents[i]]):
            raise ValueError("invalid separator")
    return bags,parents,max(len(b)-1 for b in bags)


def count(n,k,factors,sequence,width_cap=3,unary=None):
    g = graph(n,[(a,b) for a,b,_ in factors])
    _,_,width = decomposition(g,sequence)
    if width>width_cap:
        raise ValueError("induced width exceeds exact inference cap")
    domain = range(1,k+1)
    tables = [((a,b),{pair:1 for pair in pairs}) for a,b,pairs in factors]
    tables.extend(((v,),{(x,):1 for x in domain if unary is None or v!=unary[0] or x==unary[1]})
                  for v in range(n))
    for v in sequence:
        affected = [(scope,data) for scope,data in tables if v in scope]
        tables = [(scope,data) for scope,data in tables if v not in scope]
        scope = tuple(sorted({j for old,_ in affected for j in old}))
        index = {j:i for i,j in enumerate(scope)}
        mapped = [(tuple(index[j] for j in old),data) for old,data in affected]
        target = tuple(j for j in scope if j!=v)
        output = {}
        for values in product(domain,repeat=len(scope)):
            weight = prod(data.get(tuple(values[i] for i in indices),0) for indices,data in mapped)
            if weight:
                key = tuple(values[index[j]] for j in target)
                output[key] = output.get(key,0)+weight
        tables.append((target,output))
    return prod(data.get((),0) for _,data in tables)


def train(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = Random(990261)
    x,y = [],[]
    for _ in range(420):
        n = rng.randrange(5,9)
        edges = [(i,rng.randrange(i)) for i in range(1,n)]
        edges += [(i,j) for i in range(n) for j in range(i) if rng.random()<.13]
        g = graph(n,edges)
        active = tuple(range(n))
        while active:
            values = [max(len(g[v]),optimal_width(remove(g,v),tuple(w for w in active if w!=v))) for v in active]
            for v,value in zip(active,values,strict=True):
                x.append(features(g,v,active))
                y.append(value+.03*len(g[v]))
            v = rng.choice(active)
            g = remove(g,v)
            active = tuple(w for w in active if w!=v)
    x,y = torch.tensor(x),torch.tensor(y).reshape(-1,1)
    net = nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
    optimizer = torch.optim.AdamW(net.parameters(),lr=.002)
    for _ in range(700):
        idx = torch.randint(len(x),(64,))
        loss = nn.functional.mse_loss(net(x[idx]),y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.jit.freeze(torch.jit.trace(net.eval(),torch.zeros(4,6))).save(str(OUT/f"order-{seed}.pt"))
    (OUT/f"order-{seed}.json").write_bytes(canonical(dict(seed=seed,train_nodes=[5,8],
        graphs=420,updates=700,teacher="exact residual induced-width DP"))+b"\n")


if __name__=="__main__":
    import sys
    train(int(sys.argv[1]))
