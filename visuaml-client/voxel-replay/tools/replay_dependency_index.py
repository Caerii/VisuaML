"""Index observation dependencies and replay edits across Pocket guest effects."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest
from replay_cellular_automata import initial
from replay_coupled_signal import full_region
from replay_demand_region import OUT as DEMAND, Relay, cone, execute, recover
from replay_frozen_recurrent import PACKED
from replay_shared_recovery import (CachedAnswer, RecoveryCache, compare, deliver, effect,
                                    history_prefix, project, shortcut)

OUT = DEMAND.parent / "pocket-dependency-index"
PHASES = ("ledger_loading_ns","dependency_lookup_ns","cache_validation_ns",
          "source_loading_ns","recompute_ns")


class DependencyIndex:
    def __init__(self):
        self.queries = {}
        self.by_effect = defaultdict(set)
        self.by_source = defaultdict(set)
        self.guest_effects = defaultdict(list)

    def query(self, coord, tick, dimensions):
        key = (*coord,tick,*dimensions)
        if key not in self.queries:
            slots = {(time,x,y) for time in range(1,tick+1)
                     for x,y in cone(coord,tick-time,*dimensions)}
            cells = cone(coord,tick,*dimensions)
            self.queries[key] = dict(source_cells=sorted(cells), effect_slots=sorted(slots))
            for slot in slots:
                self.by_effect[slot].add(key)
            for cell in cells:
                self.by_source[cell].add(key)
        return key

    def affected(self, tick, coordinate):
        return sorted(self.by_effect.get((tick,*coordinate), ()))

    def record_guest_effect(self, provenance, tick, coordinate, effect_pc, dimensions):
        request = provenance["request"]
        key = self.query(request["coordinate"], request["tick"], dimensions)
        event = dict(tick=tick, coordinate=coordinate, guest_pc=effect_pc, provenance=provenance)
        if event not in self.guest_effects[key]:
            self.guest_effects[key].append(event)

    def project(self, key, ledger):
        return [[row["tick"],edit["coordinate"],edit.get("wall",0)]
                for row in ledger for edit in row["edits"]
                if key in self.by_effect.get((row["tick"],*edit["coordinate"]), ())]


class ProfiledCache(RecoveryCache):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.index = DependencyIndex()
        self.samples = []

    def __call__(self, source, ledger, coord, tick, dimensions):
        start = perf_counter_ns()
        prefix = history_prefix(ledger,tick,dimensions)
        ledger_hash = digest(canonical(prefix))
        ledger_ns = perf_counter_ns()-start
        start = perf_counter_ns()
        if self.strategy == "indexed":
            query = self.index.query(coord,tick,dimensions)
            relevant = self.index.project(query,prefix)
        else:
            relevant = project(prefix,coord,tick,dimensions)
        phase_ns = perf_counter_ns()-start
        start = perf_counter_ns()
        identity = dict(source_sha256=source.shared["source_sha256"], source_tick=source.shared["source_tick"],
                        checkpoint_sha256=source.shared["local"]["checkpoint_sha256"], rule_sha256=source.rule_hash,
                        dimensions=list(dimensions), coordinate=coord, tick=tick)
        key = digest(canonical([identity,"dependencies",relevant]))
        entry = self.answers.get(key)
        hit = entry is not None
        profile = dict(ledger_loading_ns=ledger_ns, dependency_lookup_ns=phase_ns,
                       source_loading_ns=0, recompute_ns=0)
        if hit:
            value = deliver(entry,key)
        profile["cache_validation_ns"] = perf_counter_ns()-start
        if hit:
            work = dict(checkpoint_bytes=0, source_cells=0, cells_recomputed=0, applied_edits=[])
        else:
            value, work = recover(source,prefix,coord,tick,dimensions,profile)
            entry = CachedAnswer(key,value,ledger_hash)
            self.answers[key] = entry
        self.samples.append(profile)
        proof = dict(**identity, ledger_sha256=ledger_hash, relevant_events=relevant,
                     sealed_through=tick, strategy="dependencies")
        return value, dict(**work, hit=hit, cache_key=key, proof=proof,
                           producing_ledger=entry.producing_ledger)


def trace_change(left,right):
    for index,(a,b) in enumerate(zip(left,right),1):
        if (a["pc"],a["next_pc"],a["access"]) != (b["pc"],b["next_pc"],b["access"]):
            return dict(instruction=index, pc=a["pc"], old_access=a["access"], new_access=b["access"],
                        old_next=a["next_pc"], new_next=b["next_pc"])
    raise AssertionError("Expected an observable guest divergence")


def run_case(relay,shared,grid,cache,coord,**flags):
    full,_ = execute(relay,shared,coord,full=grid,**flags)
    observed, raw = execute(relay,shared,coord,resolver=cache,**flags)
    compare(full,observed)
    return observed,raw


def edited_replay(relay, shared, grid):
    cache = ProfiledCache("indexed")
    base,_ = run_case(relay,shared,grid,cache,173)
    far = effect(17,10,1)
    planned = cache.index.affected(1,far["coordinate"])
    assert planned == []
    edited,_ = run_case(relay,shared,grid,cache,173,edits_by_tick={1:[far]})
    assert all(o["recovery"]["hit"] for o in edited["observations"])
    base_plan, producer, _ = shortcut(relay,shared,grid,True,cache)
    control,_ = run_case(relay,shared,grid,cache,173,dig=False,edits_by_tick=base_plan)
    affected_remote = cache.index.affected(1,[2,8])
    assert (2,8,1,20,18) in affected_remote and all(key[:2] != (13,8) for key in affected_remote)
    changed_plan, altered, _ = shortcut(relay,shared,grid,False,cache)
    assert not altered["recovery"]["hit"]
    assert producer["action"] == 0 and altered["action"] == 1
    event = changed_plan[2][0]
    assert 2 not in base_plan
    cache.index.record_guest_effect(event["cause"],2,event["coordinate"],event["pc"],[20,18])
    found = cache.index.affected(2,[9,8])
    assert found == [(13,8,6,20,18)]
    changed,_ = run_case(relay,shared,grid,cache,173,dig=False,edits_by_tick=changed_plan)
    expected, result = control["observations"][-1],changed["observations"][-1]
    assert (expected["value"],result["value"]) == (0,1)
    assert [row["recovery"]["hit"] for row in changed["observations"]] == [True]*5+[False]
    first = trace_change(producer["trace"],altered["trace"])
    later = trace_change(expected["trace"],result["trace"])
    assert first["pc"] == later["pc"] == 38
    stopped,_ = execute(relay,shared,173,resolver=cache,events_available=False)
    assert not stopped["observations"] and stopped["stopped"]["reason"] == "event_history_unavailable"
    return dict(far=dict(edit=far, invalidated=planned, baseline=base, edited=edited),
                shortcut=dict(remote_invalidated=affected_remote, newly_affected=found, before=control, after=changed,
                              old_remote=producer,new_remote=altered, first_guest_change=first,
                              first_consumer_change=later, materialized_effect=event),
                incomplete=stopped["stopped"]["pending_read"],
                index=[dict(query=list(key), **value) for key,value in cache.index.queries.items()],
                guest_effect_index=[dict(query=list(key), effects=value)
                                    for key,value in cache.index.guest_effects.items()])


def measure(relay, shared, grid, strategy):
    cache = ProfiledCache(strategy)
    # Reuse across adjacent horizons and three request coordinates. Several
    # unrelated histories reuse the same path; some remove the real gate edit.
    work = []
    for i in range(18):
        request = (172,173,175)[i%3]
        flags = dict(dig=i%5 != 4, edits_by_tick={1:[effect(18+i%2,10,i//3%2)]})
        _,samples = run_case(relay,shared,grid,cache,request,**flags)
        work.extend(samples)
    assert len(work) == len(cache.samples)
    rows = []
    for item,phase in zip(work,cache.samples):
        row = {key:item[key] for key in ("hit","checkpoint_bytes","cells_recomputed","suspended_ns","guest_execution_ns")}
        row.update(phase)
        row["guest_restore_and_delivery_ns"] = item["suspended_ns"]-item["recovery_ns"]
        rows.append(row)
    return rows


def replay(n):
    relay = Relay(DEMAND/"signal-relay.wasm")
    shared = json.loads((DEMAND/"shared-checkpoint.json").read_text())
    grid = full_region(initial())
    result = edited_replay(relay,shared,grid)
    samples = {"direct-key": [], "indexed": []}
    for i in range(n):
        order = ("direct-key","indexed") if i%2 == 0 else ("indexed","direct-key")
        for strategy in order:
            samples[strategy].append(measure(relay,shared,grid,strategy))
    for a,b in zip(samples["direct-key"],samples["indexed"]):
        assert [(r["hit"],r["checkpoint_bytes"],r["cells_recomputed"]) for r in a] == [
                (r["hit"],r["checkpoint_bytes"],r["cells_recomputed"]) for r in b]
    keys = ("hit","checkpoint_bytes","cells_recomputed","suspended_ns","guest_restore_and_delivery_ns","guest_execution_ns")+PHASES
    summary = {name:{key:median(sum(r[key] for r in batch) for batch in rows) for key in keys}
               for name,rows in samples.items()}
    return dict(profile="visuaml-pocket-dependency-index-v1", edits=result), dict(samples=samples, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--open",type=Path)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--effect", type=int, nargs=3, metavar=("TICK","X","Y"))
    query.add_argument("--read", type=int, nargs=3, metavar=("X","Y","TICK"))
    args = parser.parse_args()
    if args.effect or args.read:
        archive = json.loads((args.open or OUT/"replay.json").read_text())["edits"]
        if args.effect:
            result = [entry["query"] for entry in archive["index"] if args.effect in entry["effect_slots"]]
        else:
            result = dict(dependencies=[entry for entry in archive["index"] if entry["query"][:3] == args.read],
                          guest_effects=[entry for entry in archive["guest_effect_index"]
                                         if entry["query"][:3] == args.read])
        print(json.dumps(result,indent=2))
        raise SystemExit
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        bundle,timings = replay(4 if args.write else 1)
    if args.write:
        for name,data in (("replay",bundle),("timings",timings)):
            (OUT/(name+".json")).write_text(json.dumps(data,separators=(",",":"))+"\n")
    else:
        assert json.loads(json.dumps(bundle)) == json.loads((args.open or OUT/"replay.json").read_text())
    print("Verified reverse lookup, edited guest effects, and exact full-world counterparts.")
