"""Benchmark real staged/fused binaries and host checkpoint overhead."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from random import Random
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import torch

from checkpoint_dependence_v66 import advance, checkpoint, restore
from dependent_synthesis_v65 import OUT as V65, Space, choices, utility
from evaluate_dependency_v65 import benchmark, reply, tree
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Store
from replay_guest_rewrite import EXECUTOR
from structural_decomposition_v66 import OUT, CachedTree
from verify_dependency_v65 import PAGE, publish, verify


def search(task):
    space = Space.build(task["width"],task["library"],tree(task["layout"],task["width"]))
    tree_cache = CachedTree(space)
    costs,budget = task["costs"],task["budget"]
    while True:
        singles = [tree_cache.marginal(i) for i in range(space.size)]
        total = sum(singles[0])
        if total==1:
            return [row.index(1)+1 for row in singles]
        pairs = []
        for a,b,allowed in space.factors:
            left,right = ([tree_cache.message(n,node) for n in tree_cache.links[node] if n!=exclude]
                          for node,exclude in ((a,b),(b,a)))
            from math import prod
            pairs.append([(x,y,prod(v[x-1] for v in left)*prod(v[y-1] for v in right))
                          for x,y in allowed if all(v[x-1] for v in left) and all(v[y-1] for v in right)])
        options = choices(space,costs,budget,(total,singles,pairs))
        if not options:
            return None
        edge,probe,charge,_ = max(options,key=utility)
        space = space.observe(edge,probe,reply(space,task["keys"],edge,probe))
        tree_cache.edit(*space.factors[edge])
        budget -= charge


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    opts = parser.parse_args()
    torch.set_num_threads(1)
    prior = json.loads((V65/"outcomes.json").read_bytes())
    graph = torch.jit.load(str(EXECUTOR/"machine.pt"))
    rows,timings = [],[]
    for family in ("chain","wide_shift"):
        task = next(row["task"] for row in prior["rows"] if row["task"]["family"]==family)
        start = perf_counter_ns()
        keys = search(task)
        search_ns = perf_counter_ns()-start
        assert keys==task["keys"]
        with TemporaryDirectory() as folder:
            store = Store(folder)
            start = perf_counter_ns()
            root = publish(store,keys)
            proof_publish_ns = perf_counter_ns()-start
            start = perf_counter_ns()
            assert verify(Store(folder),root,keys)
            independent_ns = perf_counter_ns()-start
        runtime,times = benchmark(keys,graph)
        saving = median(a-b for a,b in zip(times["samples"]["staged"],times["samples"]["fused"],strict=True))
        upfront = search_ns+proof_publish_ns+independent_ns+times["setup_ns"]["fused"]
        times.update(search_ns=search_ns,proof_publish_ns=proof_publish_ns,validation_ns=independent_ns,
                     break_even=int(upfront//saving)+1 if saving>0 else None)
        initial = Random(421).randbytes(PAGE)
        page,carry = bytearray(initial),0
        for i in range(5):
            carry = advance(page,keys,i,carry,"fuse")
        samples = defaultdict(list)
        for _ in range(24):
            start = perf_counter_ns()
            raw = checkpoint(initial,keys,10,5,carry,page)
            samples["export_ns"].append(perf_counter_ns()-start)
            start = perf_counter_ns()
            n,c,restored = restore(raw,initial,keys,10)
            samples["restore_ns"].append(perf_counter_ns()-start)
            assert (n,c,restored)==(5,carry,page)
        times["host_checkpoint"] = dict(samples,bytes=len(raw))
        timings.append(times)
        rows.append(dict(family=family,root=root,checkpoint_sha256=digest(raw),runtime=runtime))
    path = OUT/"performance.json"
    if opts.write:
        path.write_bytes(canonical(rows)+b"\n")
        (OUT/"performance-timings.json").write_bytes(canonical(timings)+b"\n")
    else:
        assert canonical(rows)==canonical(json.loads(path.read_bytes()))
    print("Verified two native layouts and their logical checkpoint samples")


if __name__=="__main__":
    main()
