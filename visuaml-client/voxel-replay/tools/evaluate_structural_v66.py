"""Replay incremental tree inference and bounded cyclic decomposition."""
from argparse import ArgumentParser
from itertools import product
import json
from random import Random
from time import perf_counter_ns

import torch
import z3

from dependent_synthesis_v65 import Space
from frozen_call_checkpoint import canonical, digest
from structural_decomposition_v66 import OUT, CachedTree, count, decomposition, graph, order


def topology(kind,n,rng):
    return [(i-1 if kind=="chain" else (i-1)//2 if kind=="balanced"
             else 0 if kind=="star" else rng.randrange(i),i) for i in range(1,n)]


def oracle(n,k,factors,flat):
    totals = [[0]*k for _ in range(n)]
    if flat:
        possibilities = product(range(1,k+1),repeat=n)
        relations = [(a,b,set(pairs)) for a,b,pairs in factors]
        models = [row for row in possibilities if all((row[a],row[b]) in table for a,b,table in relations)]
    else:
        vars_ = [z3.Int(f"x{i}") for i in range(n)]
        solver = z3.Solver()
        solver.add(*(z3.And(1<=v,v<=k) for v in vars_))
        for a,b,pairs in factors:
            solver.add(z3.Or(*(z3.And(vars_[a]==p,vars_[b]==q) for p,q in pairs)))
        models = []
        while solver.check()==z3.sat:
            m = solver.model()
            row = tuple(m.eval(v).as_long() for v in vars_)
            models.append(row)
            assert len(models)<=k*k
            solver.add(z3.Or(*(v!=x for v,x in zip(vars_,row,strict=True))))
    for row in models:
        for v,x in enumerate(row):
            totals[v][x-1] += 1
    return len(models),[tuple(v) for v in totals]


class IncrementalOracle:
    def __init__(self,n,k,factors):
        self.k,self.n = k,n
        self.vars = [z3.Int(f"t{i}") for i in range(n)]
        self.enabled = [z3.Bool(f"factor{i}") for i in range(len(factors))]
        self.solver = z3.Solver()
        self.solver.add(*(z3.And(1<=x,x<=k) for x in self.vars))
        for signal,(a,b,pairs) in zip(self.enabled,factors,strict=True):
            self.solver.add(z3.Implies(signal,z3.Or(*(z3.And(self.vars[a]==p,self.vars[b]==q)
                                                    for p,q in pairs))))

    def enumerate(self,edge,opened):
        assumptions = [z3.Not(flag) if opened and i==edge else flag
                       for i,flag in enumerate(self.enabled)]
        result = [[0]*self.k for _ in range(self.n)]
        self.solver.push()
        found = 0
        while self.solver.check(*assumptions)==z3.sat:
            model = self.solver.model()
            row = [model.eval(var).as_long() for var in self.vars]
            found += 1
            assert found<=self.k*self.k
            for i,value in enumerate(row):
                result[i][value-1] += 1
            self.solver.add(z3.Or(*(var!=val for var,val in zip(self.vars,row,strict=True))))
        self.solver.pop()
        return found,[tuple(row) for row in result]


def tree_trials():
    rng = Random(75133)
    rows,timings = [],[]
    for kind in ("chain","balanced","star","random"):
        for n,k in ((6,3),(24,6),(72,8)):
            space = Space.build(n,k,topology(kind,n,rng))
            keys = [rng.randint(1,k) for _ in range(n)]
            initial = space
            for edge,(a,b,_) in enumerate(space.factors):
                space = space.observe(edge,0,keys[a]+keys[b])
            base = space.factors
            independent = IncrementalOracle(n,k,base) if n>6 else None
            cached = CachedTree(space)
            full = CachedTree(space)
            for i in range(n):
                cached.marginal(i)
                full.marginal(i)
            events,wall = [],[]
            for index in range(10):
                edge = rng.randrange(n-1)
                a,b,pairs = base[edge]
                # Toggle one factor, then restore it before touching another.
                # This bounds model enumeration for the independent SMT oracle.
                wide = initial.factors[edge][2]
                for replacement in (wide,pairs):
                    factors = base[:edge]+((a,b,replacement),)+base[edge+1:]
                    start = perf_counter_ns()
                    invalidated = cached.edit(a,b,replacement)
                    before = cached.recomputed
                    local = cached.marginal(a)
                    elapsed = perf_counter_ns()-start
                    changed = cached.recomputed-before
                    fresh = Space(n,k,factors)
                    start = perf_counter_ns()
                    rebuilt_local = CachedTree(fresh).marginal(a)
                    local_rebuild = perf_counter_ns()-start
                    assert rebuilt_local==local
                    start = perf_counter_ns()
                    _,all_marginals,_ = fresh.posterior()
                    baseline = perf_counter_ns()-start
                    start = perf_counter_ns()
                    full.edit(a,b,replacement)
                    before = full.recomputed
                    exact = [full.marginal(i) for i in range(n)]
                    full_elapsed = perf_counter_ns()-start
                    all_changes = full.recomputed-before
                    assert exact==all_marginals and local==exact[a]
                    opened = len(replacement)==len(wide)
                    count_,marginals = (oracle(n,k,factors,True) if independent is None
                                        else independent.enumerate(edge,opened))
                    assert exact==marginals and sum(local)==count_
                    events.append(dict(edge=[a,b],queried=a,count=count_,local=list(local),
                        open=len(replacement)==len(wide),cached_work=changed,all_work=all_changes,
                        baseline_work=2*(n-1),invalidated=invalidated))
                    wall.append(dict(local_ns=elapsed,local_rebuild_ns=local_rebuild,
                                     global_cached_ns=full_elapsed,rebuild_ns=baseline))
                    # Audit the lazy cache without warming the timed instance.
                    assert cached.tables[a,b]==frozenset(replacement)
            assert not full.edit(*base[-1])
            rows.append(dict(layout=kind,nodes=n,domain=k,edges=[[a,b] for a,b,_ in base],events=events))
            timings.append(dict(layout=kind,nodes=n,events=wall))
    return rows,timings


def cyclic_tasks(rng):
    for n in (6,8,10,12,16):
        for family in ("cycle","ladder","triangulated","chorded","guard_clique"):
            for case in range(3):
                permutation = list(range(n))
                rng.shuffle(permutation)
                if family=="cycle":
                    raw = [(i,(i+1)%n) for i in range(n)]
                elif family=="ladder":
                    raw = [(i,i+2) for i in range(n-2)]+[(i,i+1) for i in range(0,n,2)]
                elif family=="triangulated":
                    raw = [(i,i+1) for i in range(n-1)]+[(i,i+2) for i in range(n-2)]
                elif family=="guard_clique":
                    raw = [(i,i+1) for i in range(n-1)]+[(j,i) for i in range(5) for j in range(i)]
                else:
                    raw = [(i,(i+1)%n) for i in range(n)]+[(i,i+2) for i in range(0,n-2,3)]
                    raw.extend((i,i+4) for i in range(1,n-4,4))
                edges = sorted({tuple(sorted((permutation[a],permutation[b]))) for a,b in raw})
                k = 3 if case<2 else 4
                keys = [rng.randint(1,k) for _ in range(n)]
                factors = [(a,b,tuple((x,y) for x,y in product(range(1,k+1),repeat=2)
                                       if x+y==keys[a]+keys[b])) for a,b in edges]
                yield n,k,family,case,edges,factors


def cyclic_trials(models):
    rng = Random(25100)
    rows,wall = [],[]
    for n,k,family,case,edges,factors in cyclic_tasks(rng):
        g = graph(n,edges)
        expected,marginals = oracle(n,k,factors,n<=6)
        methods = dict(min_fill=None,min_degree=None,**models)
        if n<=8:
            methods["optimal"] = None
        choices,elapsed = {},{}
        for label,model in methods.items():
            start = perf_counter_ns()
            proposed = order(g,label,model)
            bags,parents,width = decomposition(g,proposed)
            elapsed[label+"_plan_ns"] = perf_counter_ns()-start
            launched = perf_counter_ns()
            if width<=3:
                observed = count(n,k,factors,proposed)
                assert observed==expected
                if label=="min_fill":
                    for v in range(n):
                        assert [count(n,k,factors,proposed,unary=(v,x)) for x in range(1,k+1)]==list(marginals[v])
            else:
                try:
                    count(n,k,factors,proposed)
                except ValueError:
                    observed = None
                else:
                    raise AssertionError("width guard failed")
            elapsed[label+"_inference_ns"] = perf_counter_ns()-launched
            choices[label] = dict(order=proposed,bags=bags,parents=parents,width=width,count=observed)
        rows.append(dict(nodes=n,domain=k,family=family,case=case,edges=edges,count=expected,methods=choices))
        wall.append(dict(nodes=n,domain=k,family=family,case=case,times=elapsed))
    return rows,wall


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {f"learned-{seed}":OUT/f"order-{seed}.pt" for seed in (661,662)}
    models = {name:torch.jit.load(str(p)) for name,p in paths.items()}
    trees,tree_times = tree_trials()
    cycles,cycle_times = cyclic_trials(models)
    result = dict(trees=trees,cycles=cycles,weights={name:digest(p.read_bytes()) for name,p in paths.items()})
    if args.write:
        (OUT/"inference.json").write_bytes(canonical(result)+b"\n")
        (OUT/"inference-timings.json").write_bytes(canonical(dict(trees=tree_times,cycles=cycle_times))+b"\n")
    else:
        assert canonical(result)==canonical(json.loads((OUT/"inference.json").read_bytes()))
    print(f"Verified {sum(len(case['events']) for case in trees)} tree updates and {len(cycles)} cyclic cases")


if __name__=="__main__":
    main()
