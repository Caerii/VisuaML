"""Share certified observations across divergent Pocket world forks."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest
from replay_cellular_automata import initial
from replay_coupled_signal import full_region
from replay_demand_region import OUT as DEMAND, Relay, Unavailable, cone, execute, recover
from replay_frozen_recurrent import PACKED

OUT = DEMAND.parent / "pocket-shared-recovery"


def history_prefix(ledger, tick, dimensions):
    """Validate the scheduler's sealed, ordered ledger through the read."""
    if ledger is None or [row["tick"] for row in ledger[:tick]] != list(range(1,tick+1)):
        raise Unavailable("event_history_unavailable")
    width,height = dimensions
    for row in ledger[:tick]:
        if set(row) != {"tick","edits"}:
            raise Unavailable("unsealed_event_row")
        for edit in row["edits"]:
            if (not {"coordinate","pc"} <= edit.keys() or
                set(edit) - {"coordinate","pc","wall","cause"} or
                not isinstance(edit["pc"], int) or edit.get("wall",0) not in (0,1)):
                raise Unavailable("uncertified_event_kind")
            x,y = edit["coordinate"]
            if not (0 <= x < width and 0 <= y < height):
                raise Unavailable("invalid_event_coordinate")
    return ledger[:tick]


def project(history, coord, tick, dimensions, *, spatial_only=False):
    edits = []
    for row in history:
        visible = cone(coord,tick-row["tick"],*dimensions)
        for edit in row["edits"]:
            if tuple(edit["coordinate"]) in visible:
                if spatial_only and "cause" in edit:
                    continue
                # The CA observes only materialized effects. Keep order within a tick.
                edits.append([row["tick"],edit["coordinate"],edit.get("wall",0)])
    return edits


def proof_for(source, ledger, coord, tick, dimensions, strategy):
    prefix = history_prefix(ledger,tick,dimensions)
    relevant = project(prefix,coord,tick,dimensions)
    data = dict(source_sha256=source.shared["source_sha256"], source_tick=source.shared["source_tick"],
                checkpoint_sha256=source.shared["local"]["checkpoint_sha256"],
                rule_sha256=source.rule_hash, dimensions=list(dimensions), coordinate=coord, tick=tick)
    selector = prefix if strategy == "full-history" else relevant
    key = digest(canonical([data,strategy,selector]))
    return key, dict(**data, ledger_sha256=digest(canonical(prefix)), relevant_events=relevant,
                     sealed_through=tick, strategy=strategy)


@dataclass(frozen=True)
class CachedAnswer:
    key: str
    value: int
    producing_ledger: str


def deliver(entry, expected_key):
    if entry.key != expected_key:
        raise Unavailable("observation_binding_mismatch")
    return entry.value


class RecoveryCache:
    def __init__(self, strategy):
        self.strategy = strategy
        self.answers = {}

    def __call__(self, source, ledger, coord, tick, dimensions):
        key, proof = proof_for(source,ledger,coord,tick,dimensions,self.strategy)
        if key in self.answers:
            entry = self.answers[key]
            value = deliver(entry,key)
            work = dict(checkpoint_bytes=0, source_cells=0, cells_recomputed=0, applied_edits=[])
            hit = True
        else:
            value, work = recover(source,ledger,coord,tick,dimensions)
            entry = CachedAnswer(key,value,proof["ledger_sha256"])
            if self.strategy != "uncached":
                self.answers[key] = entry
            hit = False
        return value, dict(**work, hit=hit, cache_key=key, proof=proof,
                           producing_ledger=entry.producing_ledger)


def effect(x,y,wall,pc=-1,**extra):
    return dict(coordinate=[x,y],wall=wall,pc=pc,**extra)


def shortcut(relay, shared, grid, close, resolver):
    edits = {1:[effect(2,8,1)]} if close else {}
    flags = dict(dig=False, edits_by_tick=edits, horizon=1)
    full, _ = execute(relay,shared,162,full=grid,**flags)
    region, metrics = execute(relay,shared,162,resolver=resolver,**flags)
    compare(full, region)
    obs = region["observations"][0]
    if obs["action"]:
        store = next(r for r in obs["trace"] if r["access"][0] == 2)
        provenance = dict(request=obs["request"], value=obs["value"], action_pc=store["pc"],
                          action_page_sha256=obs["page_sha256"])
        edits[2] = [effect(9,8,0,store["pc"],cause=provenance)]
    return edits, obs, metrics


def compare(full, regional):
    assert full["helper"] == regional["helper"]
    assert full["ledger"] == regional["ledger"] and full["clock"] == regional["clock"]
    assert full["stopped"] == regional["stopped"]
    assert len(full["observations"]) == len(regional["observations"])
    for a,b in zip(full["observations"],regional["observations"]):
        assert {k:v for k,v in a.items() if k != "recovery"} == {k:v for k,v in b.items() if k != "recovery"}


def one_trial(relay, shared, grid, strategy):
    cache = RecoveryCache(strategy)
    rows, times = {}, {}
    for name in ("open","identical","far_edit","gate_closed","shortcut_on","shortcut_off"):
        flags = {}
        extra_metrics = []
        producer = None
        if name == "far_edit":
            flags["edits_by_tick"] = {1:[effect(17,10,1)]}
        elif name == "gate_closed":
            flags["dig"] = False
        elif name.startswith("shortcut"):
            edits, producer, extra_metrics = shortcut(relay,shared,grid,name=="shortcut_off",cache)
            flags.update(dig=False,edits_by_tick=edits)
        full,_ = execute(relay,shared,173,full=grid,**flags)
        result,timing = execute(relay,shared,173,resolver=cache,**flags)
        compare(full,result)
        rows[name] = dict(consumer=result, producer=producer)
        counters = ("tick","hit","checkpoint_bytes","source_cells","cells_recomputed",
                    "suspended_ns","recovery_ns")
        times[name] = [{key:row[key] for key in counters} for row in extra_metrics+timing]
    return rows,times


def assertions(trials, relay, shared):
    dep = trials["dependencies"]
    full = trials["full-history"]
    def records(data,name):
        return data[name]["consumer"]["observations"]
    assert all(o["recovery"]["hit"] for o in records(full,"identical"))
    assert not any(o["recovery"]["hit"] for o in records(full,"far_edit"))
    assert all(o["recovery"]["hit"] for o in records(dep,"far_edit"))
    assert [o["recovery"]["hit"] for o in records(dep,"gate_closed")] == [True]*4+[False]*2
    assert [o["value"] for o in records(dep,"gate_closed")] == [0]*6
    origin = records(dep,"open")[4]
    alternate = records(dep,"gate_closed")[4]
    wrong_tick = records(dep,"open")[3]
    rejected = {}
    for label, row in (("wrong_tick",wrong_tick),("wrong_branch",alternate)):
        entry = CachedAnswer(row["recovery"]["cache_key"],row["value"],row["recovery"]["producing_ledger"])
        cache = RecoveryCache("dependencies")

        def inject(source, ledger, coord, tick, dimensions):
            if tick == 5:
                expected, _ = proof_for(source,ledger,coord,tick,dimensions,"dependencies")
                assert expected == origin["recovery"]["cache_key"]
                deliver(entry,expected)
                raise AssertionError("Stale value passed its binding check")
            return cache(source,ledger,coord,tick,dimensions)

        halted,_ = execute(relay,shared,173,resolver=inject)
        assert halted["clock"] == 5 and len(halted["observations"]) == 4
        assert halted["stopped"]["reason"] == "observation_binding_mismatch"
        assert halted["stopped"]["pending_read"] == origin["request"]
        rejected[label] = halted["stopped"]["pending_read"]
    # The only spatially nearby edit that distinguishes the two futures
    # was materialized by the earlier remote Rust observation and action.
    on = dep["shortcut_on"]["consumer"]
    off = dep["shortcut_off"]["consumer"]
    pos, tick = [13,8],6
    assert dep["shortcut_on"]["producer"]["action"] == 1
    assert dep["shortcut_off"]["producer"]["action"] == 0
    naive = [project(world["ledger"],pos,tick,[20,18],spatial_only=True) for world in (on,off)]
    assert naive[0] == naive[1]
    good,bad = on["observations"][-1],off["observations"][-1]
    assert good["value"] == 1 and bad["value"] == 0
    assert good["recovery"]["cache_key"] != bad["recovery"]["cache_key"]
    assert any(e["tick"] == 2 and e["edits"][0]["cause"]["request"]["coordinate"] == [2,8]
               for e in on["ledger"] if e["edits"] and "cause" in e["edits"][0])
    return dict(rejected=rejected, shortcut=dict(naive_projection=naive[0],
                correct_values=[good["value"],bad["value"]], tick=tick, coordinate=pos,
                first_distinguishing_effect=on["ledger"][1]["edits"][0]))


def summarize(samples):
    per_trial = []
    for trial in samples:
        values = [item for events in trial.values() for item in events]
        per_trial.append(dict(reads=len(values), hits=sum(row["hit"] for row in values),
                              checkpoint_bytes=sum(row["checkpoint_bytes"] for row in values),
                              cells_recomputed=sum(row["cells_recomputed"] for row in values),
                              suspension_ns=sum(row["suspended_ns"] for row in values)))
    return {key:statistics.median(row[key] for row in per_trial) for key in per_trial[0]}


def replay(count):
    relay = Relay(DEMAND/"signal-relay.wasm")
    shared = json.loads((DEMAND/"shared-checkpoint.json").read_text())
    grid = full_region(initial())
    order = ["uncached","full-history","dependencies"]
    saved, samples = {}, {name:[] for name in order}
    for n in range(count):
        for name in order[::1 if n%2 == 0 else -1]:
            data, times = one_trial(relay,shared,grid,name)
            assert saved.get(name,data) == data
            saved[name] = data
            samples[name].append(times)
    controls = assertions(saved,relay,shared)
    # An unsealed ledger cannot justify reusing even an existing answer.
    cache = RecoveryCache("dependencies")
    _, absent = execute(relay,shared,173,resolver=cache)
    assert absent
    stopped,_ = execute(relay,shared,173,resolver=cache,events_available=False)
    assert stopped["stopped"]["reason"] == "event_history_unavailable"
    assert not stopped["observations"]
    controls["incomplete_ledger_stop"] = stopped["stopped"]["pending_read"]
    return dict(profile="visuaml-shared-recovery-v1", shared_sha256=digest(canonical(shared)),
                strategies=saved, controls=controls), dict(samples=samples, summary={
                    name:summarize(items) for name,items in samples.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--open",type=Path)
    args = parser.parse_args()
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        bundle,timing = replay(4 if args.write else 1)
    if args.write:
        (OUT/"replay.json").write_bytes(canonical(bundle)+b"\n")
        (OUT/"timings.json").write_text(json.dumps(timing,indent=2)+"\n")
    else:
        assert bundle == json.loads((args.open or OUT/"replay.json").read_text())
    print("Verified full histories, dependency hits, guest-mediated shortcut, and stale-answer rejection.")
