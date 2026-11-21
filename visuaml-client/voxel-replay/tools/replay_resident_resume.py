"""Compare portable and resident continuations at Pocket's pending load."""
import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest, encode, restore
from replay_cellular_automata import initial
from replay_coupled_signal import full_region
from replay_demand_region import (OUT as DEMAND, LOAD_PC, MAILBOX, Relay, Unavailable,
                                  execute, finish_read, page, supply)
from replay_dependency_index import ProfiledCache
from replay_frozen_recurrent import PACKED
from replay_shared_recovery import compare, effect, history_prefix

OUT = DEMAND.parent / "pocket-resident-resume"


@dataclass(frozen=True)
class Binding:
    branch: str
    sequence: int
    pc: int
    coordinate: tuple
    tick: int
    history: str | None
    checkpoint: str


@dataclass(frozen=True)
class Response:
    binding: Binding
    value: int


@dataclass
class Waiting:
    binding: Binding
    resident: object
    verified: Response | None = None


class Broker:
    """In-process ownership for a private waiting VM and a trusted resolver."""

    def __init__(self, branch, world_size):
        self.branch = branch
        self.world_size = world_size
        self.sequence = 0
        self.pending = {}

    def wait(self, state, request, ledger):
        assert int(state.vm.pc) == request["source_pc"] == LOAD_PC
        self.sequence += 1
        try:
            history = digest(canonical(history_prefix(ledger,request["tick"],self.world_size)))
        except Unavailable:
            history = None
        bound = Binding(self.branch,self.sequence,LOAD_PC,tuple(request["coordinate"]),
                        request["tick"],history,request["checkpoint_sha256"])
        waiting = Waiting(bound,state)
        self.pending[self.sequence] = waiting
        return waiting

    def certify(self, waiting, value):
        # Called by the authoritative resolver after successful recovery or a
        # checked cache hit. Object identity is the in-process authorization.
        if (waiting.resident is None or waiting.binding.history is None
                or self.pending.get(waiting.binding.sequence) is not waiting):
            raise Unavailable("no_certifiable_pending_read")
        if type(value) is not int or value not in (0,1) or waiting.verified is not None:
            raise Unavailable("invalid_observation")
        waiting.verified = Response(waiting.binding,value)
        return waiting.verified

    def resume(self, relay, waiting, response, prefix, complete, began, profile=None):
        if waiting.resident is None:
            raise Unavailable("pending_read_already_consumed")
        if (self.pending.get(waiting.binding.sequence) is not waiting
            or response.binding.branch != self.branch or response.binding != waiting.binding
            or response is not waiting.verified):
            raise Unavailable("unverified_observation_binding")
        state = waiting.resident
        assert int(state.vm.pc) == LOAD_PC
        self.evict(waiting)
        return finish_read(relay,state,prefix,complete,response.value,began,profile)


    def evict(self, waiting):
        if self.pending.get(waiting.binding.sequence) is not waiting:
            raise Unavailable("pending_read_not_owned")
        del self.pending[waiting.binding.sequence]
        waiting.resident = None


def checked(relay, shared, grid, cache, coord, branch, resident, **flags):
    reference,_ = execute(relay,shared,coord,full=grid,**flags)
    broker = Broker(branch,shared["local"]["world_size"]) if resident else None
    got,samples = execute(relay,shared,coord,resolver=cache,broker=broker,**flags)
    compare(reference,got)
    return got,samples


def edits(relay, shared, grid):
    cache = ProfiledCache("indexed")
    original,_ = checked(relay,shared,grid,cache,173,"original",True)
    distant,_ = checked(relay,shared,grid,cache,173,"far-edit",True,
                        edits_by_tick={1:[effect(17,10,1)]})
    assert all(o["recovery"]["hit"] for o in distant["observations"])
    cases = []
    for wall in (True,False):
        planned = {1:[effect(2,8,1)]} if wall else {}
        remote,_ = checked(relay,shared,grid,cache,162,f"producer-{wall}",True,
                           horizon=1,dig=False,edits_by_tick=planned)
        observed = remote["observations"][0]
        assert observed["action"] == int(not wall)
        if observed["action"]:
            pc = next(r["pc"] for r in observed["trace"] if r["access"][0] == 2)
            provenance = dict(request=observed["request"], value=observed["value"], action_pc=pc,
                              action_page_sha256=observed["page_sha256"])
            planned[2] = [effect(9,8,0,pc,cause=provenance)]
            assert cache.index.affected(2,[9,8]) == [(13,8,6,20,18)]
        consumer,_ = checked(relay,shared,grid,cache,173,f"consumer-{wall}",True,
                             dig=False,edits_by_tick=planned)
        cases.append(dict(remote=remote,consumer=consumer))
    changed = cases[1]["consumer"]["observations"]
    assert [o["recovery"]["hit"] for o in changed] == [True]*5+[False]
    assert cases[0]["consumer"]["observations"][-1]["value"] == 0 and changed[-1]["value"] == 1
    return dict(original=original,distant=distant,shortcut=cases)


def controls(relay, shared, grid):
    # Capture two pending calls at the same tick and PC, but fork-specific
    # histories and private pages. The real broker must handle reverse delivery.
    snapshots, tokens, retained = {}, {}, []
    for name, wall in (("open",False),("blocked",True)):
        flags = dict(dig=False,horizon=1,edits_by_tick={1:[effect(2,8,1)]} if wall else {})
        unavailable,_ = execute(relay,shared,162,events_available=False,**flags)
        pending = unavailable["stopped"]
        expected,_ = execute(relay,shared,162,full=grid,**flags)
        state = restore(pending["vm"],relay.identity,relay.image)
        broker = Broker(name,shared["local"]["world_size"])
        waiting = broker.wait(state,pending["pending_read"],expected["ledger"])
        tokens[name] = broker.certify(waiting,expected["observations"][0]["value"])
        snapshots[name] = (state,broker,waiting,pending,expected)
        tensors = [*state.vm,state.depth,state.frames]
        retained.append(sum(t.numel()*t.element_size() for t in tensors))
    assert snapshots["open"][3]["vm"] == snapshots["blocked"][3]["vm"]
    assert snapshots["open"][0].vm.page.data_ptr() != snapshots["blocked"][0].vm.page.data_ptr()
    state,broker,waiting,pending,_ = snapshots["open"]
    correct = tokens["open"]
    rejected = []
    wrong = {
        "wrong_branch":tokens["blocked"],
        "wrong_tick":replace(correct,binding=replace(correct.binding,tick=2)),
        "wrong_pc":replace(correct,binding=replace(correct.binding,pc=37)),
        "wrong_coordinate":replace(correct,binding=replace(correct.binding,coordinate=(3,8))),
        "wrong_history":replace(correct,binding=replace(correct.binding,history="0"*64)),
        "unverified":replace(correct),
    }
    for label,value in wrong.items():
        try:
            broker.resume(relay,waiting,value,[],page(state),perf_counter_ns())
        except Unavailable:
            assert encode(state,relay.identity) == pending["vm"]
            rejected.append(label)
        else:
            raise AssertionError(label)
    offline = Broker("missing-ledger",shared["local"]["world_size"])
    stopped,_ = execute(relay,shared,162,events_available=False,broker=offline,horizon=1,dig=False)
    incomplete, = offline.pending.values()
    assert not stopped["observations"] and stopped["stopped"]["reason"] == "event_history_unavailable"
    assert encode(incomplete.resident,relay.identity) == pending["vm"]
    try:
        offline.certify(incomplete,1)
    except Unavailable:
        assert encode(state,relay.identity) == pending["vm"]
        rejected.append("ledger_incomplete")
    else:
        raise AssertionError("Certified an incomplete history")
    actual = []
    portable = None
    costs = []
    for label in ("blocked","open"):
        state,broker,waiting,pending,expected = snapshots[label]
        before = page(state)
        began = perf_counter_ns()
        # Explicit export and eviction are separately measured, using a
        # second private waiting object so the real request stays live.
        exported = encode(state,relay.identity)
        costs.append(dict(export_ns=perf_counter_ns()-began))
        assert exported == pending["vm"]
        evictor = Broker("evict",shared["local"]["world_size"])
        scratch = evictor.wait(restore(exported,relay.identity,relay.image),pending["pending_read"],
                               expected["ledger"])
        began = perf_counter_ns()
        evictor.evict(scratch)
        costs[-1]["eviction_ns"] = perf_counter_ns()-began
        completed,trace,*_ = broker.resume(relay,waiting,tokens[label],pending["prefix"],before,perf_counter_ns())
        row = expected["observations"][0]
        assert trace == row["trace"] and digest(completed) == row["page_sha256"]
        assert sum(r["access"][:2] == [1,MAILBOX+16] for r in trace) == 1
        try:
            broker.resume(relay,waiting,tokens[label],pending["prefix"],before,perf_counter_ns())
        except Unavailable:
            rejected.append(f"duplicate_{label}")
        else:
            raise AssertionError("Duplicate committed")
        if label == "blocked":
            open_state,_,open_wait,open_snapshot,_ = snapshots["open"]
            assert open_wait.resident is open_state
            assert encode(open_state,relay.identity) == open_snapshot["vm"]
        actual.append(dict(branch=label,value=row["value"],trace=trace,page_sha256=digest(completed)))
        if label == "open":
            portable = dict(vm=exported,prefix=pending["prefix"],value=row["value"])
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/"pending.json"
        path.write_bytes(canonical(portable))
        result = json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),
                                                     "--resume",str(path)]))
    assert result == {key:actual[-1][key] for key in ("trace","page_sha256")}
    return dict(rejected=rejected,reverse_delivery=actual,portable=portable,
                retained_tensor_bytes=retained,serialized_bytes=len(canonical(portable["vm"]))),costs


FIELDS = ("restore_ns","integrity_ns","page_decode_ns","page_initialize_ns","registers_ns","delivery_ns",
          "guest_execution_ns","suspended_ns","export_ns","hit","checkpoint_bytes","cells_recomputed")


def bench(relay,shared,grid,forks,resident):
    cache = ProfiledCache("indexed")
    rows = []
    for i in range(forks):
        flags = dict(dig=i%5 != 4,edits_by_tick={1:[effect(18+i%2,10,i//3%2)]})
        _,samples = checked(relay,shared,grid,cache,(172,173,175)[i%3],str(i),resident,**flags)
        rows.extend({name:r[name] for name in FIELDS} for r in samples)
    return rows


def replay(pairs):
    relay = Relay(DEMAND/"signal-relay.wasm")
    shared = json.loads((DEMAND/"shared-checkpoint.json").read_text())
    grid = full_region(initial())
    edited = edits(relay,shared,grid)
    guard,costs = controls(relay,shared,grid)
    samples = {}
    for size in (18,36):
        rows = {"portable":[],"resident":[]}
        for i in range(pairs):
            pair = {}
            for resident in (False,True)[::1 if i%2 == 0 else -1]:
                key = "resident" if resident else "portable"
                pair[key] = bench(relay,shared,grid,size,resident)
                rows[key].append(pair[key])
            a,b = pair["portable"],pair["resident"]
            fields = ("hit","checkpoint_bytes","cells_recomputed")
            assert [tuple(r[k] for k in fields) for r in a] == [tuple(r[k] for k in fields) for r in b]
        samples[str(size)] = rows
    summary = {size:{mode:{key:median(sum(item[key] for item in row) for row in raw)
                           for key in FIELDS} for mode,raw in rows.items()} for size,rows in samples.items()}
    return dict(profile="visuaml-resident-resume-v1", edits=edited, controls=guard),dict(
                raw=samples,summary=summary, export_and_eviction=costs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--resume",type=Path)
    args = parser.parse_args()
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        if args.resume:
            relay = Relay(DEMAND/"signal-relay.wasm")
            packet = json.loads(args.resume.read_text())
            previous = page(restore(packet["vm"],relay.identity,relay.image))
            complete, trace,*_ = supply(relay,packet["vm"],packet["prefix"],previous,packet["value"],perf_counter_ns())
            print(json.dumps(dict(trace=trace,page_sha256=digest(complete))))
            raise SystemExit
        bundle, raw = replay(4 if args.write else 1)
    if args.write:
        for name,data in (("replay",bundle),("timings",raw),("pending-portable",bundle["controls"]["portable"])):
            (OUT/(name+".json")).write_text(json.dumps(data,separators=(",",":"))+"\n")
    else:
        assert bundle == json.loads((OUT/"replay.json").read_text())
    print("Verified resident/portable boundaries, reverse delivery, rejected responses and edited forks.")
