"""Recover certified observations from tick-zero row shards at a pending Rust load."""
import argparse
import json
from pathlib import Path
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest, encode, restore
from replay_cellular_automata import initial, tensor
from replay_coupled_signal import HORIZON, MAILBOX, Relay, full_region, mailbox, page
from replay_frozen_recurrent import PACKED, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image

OUT = Path(__file__).resolve().parents[1] / "evidence/pocket-demand-region"
LOAD_PC = 38
OFFSETS = ((0,0),(1,0),(-1,0),(0,1),(0,-1))


class Unavailable(Exception):
    pass


def publish(scene, relay):
    width,height = scene["width"],scene["height"]
    raw = bytes(item for pair in zip(scene["walls"],scene["signal"]) for item in pair)
    hashes = [digest(raw[y*2*width:(y+1)*2*width]) for y in range(height)]
    manifest = dict(tick=0, width=width, height=height, row_sha256=hashes,
                    rule_sha256=relay.rule_hash)
    (OUT/"world.bin").write_bytes(raw)
    (OUT/"world-index.json").write_bytes(canonical(manifest))
    _, _, _, local = relay.capture(scene, 1)
    binding = dict(local=local, source_sha256=digest(canonical(manifest)), source_tick=0)
    (OUT/"shared-checkpoint.json").write_bytes(canonical(binding)+b"\n")


def cone(center, radius, w, h):
    x,y = center
    return {(a,b) for b in range(max(0,y-radius),min(h,y+radius+1))
            for a in range(max(0,x-radius),min(w,x+radius+1))
            if abs(x-a)+abs(y-b) <= radius}


class RowSource:
    def __init__(self, shared, relay, available):
        self.shared = shared
        self.rule_hash = relay.rule_hash
        self.available = available
        self.cache = {}
        self.index = None
        self.bytes = 0

    def read(self, cells):
        if not self.available:
            raise Unavailable("checkpoint_data_unavailable")
        try:
            if self.index is None:
                raw = (OUT/"world-index.json").read_bytes()
                self.bytes += len(raw)
                if digest(raw) != self.shared["source_sha256"]:
                    raise Unavailable("checkpoint_identity_mismatch")
                self.index = json.loads(raw)
                assert self.index["tick"] == self.shared["source_tick"] == 0
                assert self.index["rule_sha256"] == self.rule_hash
            width = self.index["width"]
            missing = sorted({y for _,y in cells} - self.cache.keys())
            if missing:
                with (OUT/"world.bin").open("rb") as handle:
                    for y in missing:
                        handle.seek(2*width*y)
                        raw = handle.read(2*width)
                        self.bytes += len(raw)
                        if digest(raw) != self.index["row_sha256"][y]:
                            raise Unavailable("checkpoint_row_missing_or_corrupt")
                        self.cache[y] = raw
        except OSError as exc:
            raise Unavailable("checkpoint_data_unavailable") from exc
        return {cell: self.cache[cell[1]][2*cell[0]:2*cell[0]+2] for cell in cells}


def recover(source, ledger, coord, tick, dimensions, profile=None):
    if ledger is None or len(ledger) < tick or [e["tick"] for e in ledger[:tick]] != list(range(1,tick+1)):
        raise Unavailable("event_history_unavailable")
    before = source.bytes
    needed = cone(coord,tick,*dimensions)
    began = perf_counter_ns()
    loaded = source.read(needed)
    load_ns = perf_counter_ns()-began
    began = perf_counter_ns()
    walls = {p:data[0] for p,data in loaded.items()}
    signal = {p:data[1] for p,data in loaded.items()}
    recomputed, edits = 0, []
    for now,row in enumerate(ledger[:tick],1):
        for change in row["edits"]:
            cell = tuple(change["coordinate"])
            if cell in walls:
                walls[cell] = change.get("wall", 0)
                edits.append(dict(tick=now, **change))
        next_cells = cone(coord,tick-now,*dimensions)
        signal = {p:int(not walls[p] and any(signal.get((p[0]+dx,p[1]+dy),0)
                                              for dx,dy in OFFSETS)) for p in next_cells}
        recomputed += len(signal)
    if profile is not None:
        profile.update(source_loading_ns=load_ns, recompute_ns=perf_counter_ns()-began)
    return signal[tuple(coord)], dict(checkpoint_bytes=source.bytes-before, source_cells=len(needed),
                                      cells_recomputed=recomputed, applied_edits=edits)


def pause_live(relay, complete):
    state = relay.start(complete)
    prefix = []
    while int(state.vm.pc) != LOAD_PC:
        assert int(state.vm.status) == 0
        state, entry = relay.step(state)
        assert entry["access"][:2] != [1,MAILBOX+16]
        assert entry["access"][0] != 2
        prefix.append(entry)
    return state, prefix


def supply(relay, pending, prefix, complete, value, requested_at, profile=None):
    began = perf_counter_ns()
    state = restore(pending,relay.identity,relay.image,profile)
    if profile is not None:
        profile["restore_ns"] = perf_counter_ns()-began
    return finish_read(relay,state,prefix,complete,value,requested_at,profile)


def finish_read(relay, state, prefix, complete, value, requested_at, profile=None):
    began = perf_counter_ns()
    assert int(state.vm.pc) == LOAD_PC
    state.vm.page.numpy()[MAILBOX+16:MAILBOX+20] = list(struct.pack("<i",value))
    resumed_at = perf_counter_ns()
    if profile is not None:
        profile["delivery_ns"] = resumed_at-began
    suspended = resumed_at-requested_at
    state, suffix = relay.finish(state)
    resumed_execution = perf_counter_ns()-resumed_at
    trace = prefix+suffix
    completed = page(state)
    reference_page = bytearray(complete)
    struct.pack_into("<i",reference_page,MAILBOX+16,value)
    assert wasmtime_run(relay.raw,MAILBOX,bytes(reference_page)) == (mailbox(completed)[5],completed)
    assert suffix[0]["access"][:2] == [1,MAILBOX+16]
    return completed, trace, suspended, resumed_execution


def execute(relay, shared, choice, *, full=None, source_available=True, events_available=True, stale_tick_control=False, dig=True, edits_by_tick=None,
            resolver=recover, horizon=HORIZON, broker=None, journal=None):
    small = shared["local"]
    body = {key:value for key,value in small.items() if key != "checkpoint_sha256"}
    assert digest(canonical(body)) == small["checkpoint_sha256"]
    assert small["rule_sha256"] == relay.rule_hash and small["cursor"] == 0
    state = restore(small["vm"],relay.identity,relay.image)
    if not dig:
        state.vm.page.numpy()[MAILBOX+4:MAILBOX+8] = 0
    state.vm.page.numpy()[MAILBOX+28:MAILBOX+32] = list(struct.pack("<i",choice))
    state, helper = relay.finish(state)
    complete = page(state)
    region = full or small["region"]
    width,height = region["width"],region["height"]
    ox,oy = region["origin"]
    walls = tensor(region["walls"],height,width)
    signal = tensor(region["signal"],height,width)
    gate = region["gate"]
    store = next((e for e in helper if e["access"][:2] == [2,MAILBOX]), None)
    assert (store is not None) == dig
    if store is not None:
        assert store["access"] == [2,MAILBOX,7,0]
    dimensions = small["world_size"]
    flat = mailbox(complete)[3]
    assert 0 <= flat < dimensions[0]*dimensions[1]
    coord = [flat%dimensions[0],flat//dimensions[0]]
    source = RowSource(shared, relay, source_available)
    ledger, observations, metrics = [], [], []
    stopped = None
    for tick in range(1,horizon+1):
        edits = [dict(coordinate=gate, pc=store["pc"])] if tick == 1 and store else []
        edits.extend((edits_by_tick or {}).get(tick, ()))
        ledger.append(dict(tick=tick,edits=edits))
        for edit in edits:
            x,y = edit["coordinate"]
            if ox <= x < ox+width and oy <= y < oy+height:
                walls[0,0,y-oy,x-ox] = edit.get("wall", 0)
        signal = relay.rule(walls,signal)
        resident, prefix = pause_live(relay,complete)
        pending, export_ns = None, 0
        if broker is None or "state_sha256" not in prefix[-1] or journal is not None:
            export_start = perf_counter_ns()
            pending = encode(resident,relay.identity)
            export_ns = perf_counter_ns()-export_start
        begin = perf_counter_ns()
        read = dict(coordinate=coord,tick=tick,source_pc=LOAD_PC, checkpoint_sha256=(pending["checkpoint_sha256"] if pending else prefix[-1]["state_sha256"]))
        waiting = (broker.wait(resident,read,ledger if events_available else None)
                   if broker is not None else None)
        try:
            if full is not None or coord == region["target"]:
                value = int(signal[0,0,coord[1]-oy,coord[0]-ox])
                detail = None
            else:
                replay_tick = 4 if stale_tick_control and tick == 5 else tick
                value, detail = resolver(source,ledger if events_available else None,coord,replay_tick,dimensions)
        except Unavailable as exc:
            if pending is None:
                pending = encode(resident,relay.identity)
            stopped = dict(pending_read=read, reason=str(exc), vm=pending, prefix=prefix)
            break
        elapsed = perf_counter_ns()-begin
        phases = dict(restore_ns=0, integrity_ns=0, page_decode_ns=0, page_initialize_ns=0,
                      registers_ns=0, delivery_ns=0, export_ns=export_ns)
        if broker is None:
            outcome = supply(relay,pending,prefix,complete,value,begin,phases)
        else:
            response = broker.certify(waiting,value)
            if journal is not None:
                journal.observe(pending, waiting.binding, response, prefix)
            outcome = broker.resume(relay,waiting,response,prefix,complete,begin,phases)
        complete, trace, suspended, resumed_execution = outcome
        branch = next(row for row in trace[len(prefix)+1:] if row["opcode"] == "br_if")
        row = dict(request=read, value=value, action=mailbox(complete)[5],
                   trace=trace, page_sha256=digest(complete), branch=branch, recovery=detail)
        observations.append(row if journal is None else journal.record(row))
        metrics.append(dict(tick=tick,suspended_ns=suspended,recovery_ns=elapsed, guest_execution_ns=resumed_execution, **phases, **(detail or {})))
    return dict(helper=helper, gate_store=store, observations=observations, stopped=stopped,
                source_bytes=source.bytes, clock=len(ledger), ledger=ledger), metrics


def verify():
    relay = Relay(OUT/"signal-relay.wasm")
    shared = json.loads((OUT/"shared-checkpoint.json").read_text())
    scene = initial()
    results, timings = {}, {}
    for name, choice in (("local",0),("remote",162),("delayed",173)):
        direct,_ = execute(relay,shared,choice,full=full_region(scene))
        regional, samples = execute(relay,shared,choice)
        assert direct["helper"] == regional["helper"]
        assert sum(e["access"][:2] == [2,MAILBOX] for e in regional["helper"]) == 1
        assert len(regional["observations"]) == HORIZON and regional["clock"] == HORIZON
        for a,b in zip(direct["observations"],regional["observations"]):
            assert {k:v for k,v in a.items() if k != "recovery"} == {k:v for k,v in b.items() if k != "recovery"}
        results[name] = regional
        timings[name] = samples
    assert results["local"]["source_bytes"] == 0
    assert results["remote"]["observations"][0]["action"] == 1
    delayed = results["delayed"]["observations"]
    assert [r["value"] for r in delayed] == [0,0,0,0,1,1]
    assert delayed[4]["recovery"]["applied_edits"] == [dict(tick=1,coordinate=[9,8],pc=10)]
    negatives = {}
    for key, options in (("missing_checkpoint",dict(source_available=False)),
                         ("missing_events",dict(events_available=False))):
        stopped,_ = execute(relay,shared,162,**options)
        assert stopped["observations"] == [] and stopped["clock"] == 1
        assert stopped["stopped"]["pending_read"] == results["remote"]["observations"][0]["request"]
        negatives[key] = stopped["stopped"]
    stale, _ = execute(relay, shared, 173, stale_tick_control=True)
    wrong = stale["observations"][4]
    assert wrong["request"] == delayed[4]["request"]
    assert wrong["value"] == delayed[3]["value"] == 0 and wrong["action"] == 0
    assert wrong["branch"]["next_pc"] != delayed[4]["branch"]["next_pc"]
    return dict(profile="visuaml-demand-region-v1", shared_sha256=digest(canonical(shared)),
                cases=results, unavailable=negatives,
                wrong_tick_control=dict(coordinate=[13,8],requested_tick=5,correct=delayed[4]["value"],
                                         tick_used=4,wrong=wrong["value"], resumed=wrong)), timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--fork",choices=("local","remote","delayed"))
    parser.add_argument("--checkpoint",type=Path,default=OUT/"shared-checkpoint.json")
    args = parser.parse_args()
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        if args.fork:
            relay = Relay(OUT/"signal-relay.wasm")
            shared = json.loads(args.checkpoint.read_text())
            fork, _ = execute(relay, shared, {"local":0,"remote":162,"delayed":173}[args.fork])
            print(json.dumps(fork))
            raise SystemExit
        if args.write:
            relay = Relay(OUT/"signal-relay.wasm")
            publish(initial(),relay)
        replay, samples = verify()
    if args.write:
        (OUT/"replay.json").write_bytes(canonical(replay)+b"\n")
        (OUT/"timings.json").write_text(json.dumps(samples,indent=2)+"\n")
    else:
        assert replay == json.loads((OUT/"replay.json").read_text())
    print("All three forks matched full-world guest boundaries; missing-data stops preserved.")
