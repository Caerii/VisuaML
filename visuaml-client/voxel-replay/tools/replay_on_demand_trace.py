"""Explore with a compact journal and reconstruct selected source intervals."""
import argparse
import base64
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch
import zlib

import torch

from frozen_call_checkpoint import canonical, digest, encode, restore
from replay_cellular_automata import initial
from replay_coupled_signal import full_region
from replay_demand_region import OUT as DEMAND, Relay, execute, page, pause_live, supply
from replay_dependency_index import ProfiledCache
from replay_frozen_recurrent import PACKED
from replay_resident_resume import Broker
from replay_shared_recovery import effect

OUT = DEMAND.parent / "pocket-on-demand-trace"


def thin(row):
    return {key:row[key] for key in ("pc","next_pc","opcode","access")}


class CompactRelay(Relay):
    """Use the same frozen transition and retain only small pulse metadata."""
    def __init__(self, wasm):
        super().__init__(wasm)
        self.execution_ns = 0
        self.capture_ns = 0

    def step(self, state):
        start = perf_counter_ns()
        pc = int(state.vm.pc)
        function = int(pc >= self.image.starts[1])
        opcode = self.image.functions[function][pc-self.image.starts[function]]
        pulse = self.machine(self.image.rows,state)
        now = perf_counter_ns()
        self.execution_ns += now-start
        row = dict(pc=pc,opcode=opcode,access=pulse.access.tolist(),next_pc=int(pulse.state.vm.pc))
        self.capture_ns += perf_counter_ns()-now
        return pulse.state,row


class ProfiledRelay(CompactRelay):
    def step(self,state):
        start = perf_counter_ns()
        pc = int(state.vm.pc)
        function = int(pc >= self.image.starts[1])
        opcode = self.image.functions[function][pc-self.image.starts[function]]
        pulse = self.machine(self.image.rows,state)
        now = perf_counter_ns()
        self.execution_ns += now-start
        state = pulse.state
        boundary = encode(state,self.identity)
        row = dict(pc=pc,function=function,opcode=opcode,access=pulse.access.tolist(),
                   next_pc=int(state.vm.pc),fuel=int(state.vm.fuel),depth=int(state.depth),
                   locals=state.vm.locals.tolist(),stack=state.vm.stack[:int(state.vm.height)].tolist(),
                   state_sha256=boundary["checkpoint_sha256"])
        self.capture_ns += perf_counter_ns()-now
        return state,row


def boundary(row):
    if "trace" not in row:
        return {k:v for k,v in row.items() if k not in ("reply","ledger_sha256")}
    trace = row["trace"]
    return dict(request=row["request"], value=row["value"], action=row["action"],
                retired=len(trace), branch=thin(row["branch"]),
                effects=[thin(entry) for entry in trace if entry["access"][0] == 2],
                page_sha256=row["page_sha256"], recovery=row["recovery"])


class Journal:
    def __init__(self, cadence):
        self.cadence = cadence
        self.anchors = {}
        self.capture_ns = 0
        self.capture_count = 0
        self.reply = None

    def observe(self, pending, binding, response, prefix):
        started = perf_counter_ns()
        self.reply = dict(binding=json.loads(json.dumps(asdict(binding))), value=response.value)
        if (binding.tick-1) % self.cadence == 0:
            packet = dict(vm=pending, reply=self.reply, retired_before=len(prefix))
            self.anchors[str(binding.tick)] = base64.b64encode(zlib.compress(canonical(packet))).decode()
            self.capture_count += 1
        self.capture_ns += perf_counter_ns()-started

    def record(self,row):
        start = perf_counter_ns()
        result = dict(**boundary(row), reply=self.reply)
        self.capture_ns += perf_counter_ns()-start
        return result


def reconstruct(packet, relay=None):
    """Run a private copy of a single observation suffix. No world sink is passed."""
    relay = relay or Relay(DEMAND/"signal-relay.wasm")
    doc = packet["vm"]
    binding = packet["reply"]["binding"]
    if (binding["checkpoint"] != doc["checkpoint_sha256"] or not binding["history"]
            or binding["pc"] != doc["registers"]["pc"]):
        raise ValueError("Checkpoint and reply disagree")
    previous = page(restore(doc,relay.identity,relay.image))
    begin = perf_counter_ns()
    complete, trace,*_ = supply(relay,doc,[],previous,packet["reply"]["value"],begin)
    assert trace[0]["pc"] == binding["pc"]
    return dict(label="reconstructed", branch=binding["branch"], tick=binding["tick"], trace=trace,
                final_page_sha256=digest(complete), world_effects_delivered=0)


def reconstruct_later(packet, observations, target, relay):
    """Advance earlier journaled requests privately to reach the selected one."""
    working = packet
    for recorded in observations:
        tick = recorded["request"]["tick"]
        assert working["reply"] == recorded["reply"]
        previous = page(restore(working["vm"],relay.identity,relay.image))
        complete, trace,*_ = supply(relay,working["vm"],[],previous,working["reply"]["value"],perf_counter_ns())
        assert digest(complete) == recorded["page_sha256"]
        if tick == target:
            return dict(label="reconstructed",tick=tick,trace=trace,final_page_sha256=digest(complete),
                        world_effects_delivered=0)
        if tick >= len(observations):
            break
        state,prefix = pause_live(relay,complete)
        doc = encode(state,relay.identity)
        reply = observations[tick]["reply"]
        assert reply["binding"]["checkpoint"] == doc["checkpoint_sha256"]
        working = dict(vm=doc,reply=reply,retired_before=len(prefix))
    raise ValueError("Requested tick is outside the recorded journal")


def check_world(expected, candidate):
    assert list(map(thin,expected["helper"])) == list(map(thin,candidate["helper"]))
    assert expected["ledger"] == candidate["ledger"]
    def visible(rows):
        return [{k:v for k,v in boundary(row).items() if k != "recovery"} for row in rows]
    assert visible(expected["observations"]) == visible(candidate["observations"])


def run_one(relay, shared, cache, coord, name, journal=None, **flags):
    return execute(relay,shared,coord,resolver=cache,broker=Broker(name,shared["local"]["world_size"]),
                   journal=journal,**flags)


def explanation(shared, grid):
    wasm = DEMAND/"signal-relay.wasm"
    exact, compact = Relay(wasm), CompactRelay(wasm)
    journal = Journal(6)
    closed,_ = run_one(exact,shared,ProfiledCache("indexed"),162,"wall-present",dig=False,horizon=1,
                       edits_by_tick={1:[effect(2,8,1)]})
    before = closed["observations"][0]
    full,_ = run_one(exact,shared,ProfiledCache("indexed"),162,"wall-removed",dig=False,horizon=1)
    small,_ = run_one(compact,shared,ProfiledCache("indexed"),162,"wall-removed",journal,dig=False,horizon=1)
    world,_ = execute(exact,shared,162,full=grid,dig=False,horizon=1)
    check_world(world,small)
    check_world(full,small)
    assert before["value"] == 0 and small["observations"][0]["value"] == 1
    packet = json.loads(zlib.decompress(base64.b64decode(journal.anchors["1"])))
    replay = reconstruct(packet,exact)
    observation = full["observations"][0]
    assert replay["trace"] == observation["trace"][packet["retired_before"]:]
    assert replay["final_page_sha256"] == observation["page_sha256"]
    # Replay runs on another private page. Scheduler only publishes the
    # action from the original path and accepts one idempotency key.
    effect_store, = small["observations"][0]["effects"]
    assert effect_store in [thin(row) for row in replay["trace"]]
    gate = effect(9,8,0,effect_store["pc"],cause=dict(
        request=observation["request"],value=observation["value"],
        action_pc=effect_store["pc"],action_page_sha256=observation["page_sha256"]))
    effects = {digest(canonical([packet["reply"]["binding"],gate])):gate}
    assert reconstruct(packet,exact) == replay and len(effects) == 1
    cache = ProfiledCache("indexed")
    before_gate,_ = run_one(exact,shared,cache,173,"before-gate",dig=False)
    consumer_log = Journal(6)
    after_gate,_ = run_one(compact,shared,cache,173,"after-gate",consumer_log,
                           dig=False,edits_by_tick={2:list(effects.values())})
    world,_ = execute(exact,shared,173,full=grid,dig=False,edits_by_tick={2:list(effects.values())})
    check_world(world,after_gate)
    assert [r["recovery"]["hit"] for r in after_gate["observations"]] == [True]*5+[False]
    assert before_gate["observations"][-1]["value"] == 0 and after_gate["observations"][-1]["value"] == 1
    consumer_packet = json.loads(zlib.decompress(base64.b64decode(consumer_log.anchors["1"])))
    later = reconstruct_later(consumer_packet,after_gate["observations"],6,exact)
    final_trace = world["observations"][-1]["trace"]
    start = next(i for i,row in enumerate(final_trace) if row["pc"] == 38)
    assert later["trace"] == final_trace[start:]
    return dict(producer=small,reconstructed=replay,gate=gate,consumer=after_gate,consumer_reconstructed=later,
                delivered_effect_count=len(effects),compact_anchor=journal.anchors["1"]),packet


def workload(relay, shared, forks, cadence):
    relay.execution_ns = relay.capture_ns = 0
    cache = ProfiledCache("indexed")
    retained = []
    stats = dict(hits=0, bytes_loaded=0, cell_ticks=0, checkpoints=0, journal_capture_ns=0)
    began = perf_counter_ns()
    for i in range(forks):
        journal = Journal(cadence) if cadence else None
        result,samples = run_one(relay,shared,cache,(172,173,175)[i%3],str(i),journal,
                                dig=i%5!=4,edits_by_tick={1:[effect(18+i%2,10,i//3%2)]})
        retained.append(dict(result=result, anchors=journal.anchors if journal else {}))
        stats["hits"] += sum(row["hit"] for row in samples)
        stats["bytes_loaded"] += sum(row["checkpoint_bytes"] for row in samples)
        stats["cell_ticks"] += sum(row["cells_recomputed"] for row in samples)
        if journal:
            stats["checkpoints"] += journal.capture_count
            stats["journal_capture_ns"] += journal.capture_ns
    stats["total_ns"] = perf_counter_ns()-began
    stats["guest_step_ns"] = relay.execution_ns
    stats["instruction_capture_ns"] = relay.capture_ns
    payload = canonical(retained)
    stats["retained_json_bytes"] = len(payload)
    stats["compressed_trial_bytes"] = len(zlib.compress(payload))
    stats["anchor_bytes"] = sum(sum(len(value) for value in fork["anchors"].values()) for fork in retained)
    return retained,stats


def replay(pairs):
    wasm = DEMAND/"signal-relay.wasm"
    shared = json.loads((DEMAND/"shared-checkpoint.json").read_text())
    grid = full_region(initial())
    original = ProfiledRelay(wasm)
    compact = CompactRelay(wasm)
    detail,packet = explanation(shared,grid)
    modes = [("full",original,0),("compact_first",compact,6),("compact_every",compact,1)]
    measurements = {}
    for size in (18,36):
        stats = {name:[] for name,_,_ in modes}
        for pair in range(pairs):
            runs = {}
            for name,relay,cadence in modes[::1 if pair%2==0 else -1]:
                runs[name], row = workload(relay,shared,size,cadence)
                stats[name].append(row)
            for i in range(size):
                for name in ("compact_first","compact_every"):
                    check_world(runs["full"][i]["result"],runs[name][i]["result"])
            for key in ("hits","bytes_loaded","cell_ticks"):
                assert len({stats[name][-1][key] for name,_,_ in modes}) == 1
        # Independent full-world comparison outside all timed sections.
        oracle = Relay(wasm)
        for i,bundle in enumerate(runs["compact_first"]):
            full,_ = execute(oracle,shared,(172,173,175)[i%3],full=grid,dig=i%5!=4,
                            edits_by_tick={1:[effect(18+i%2,10,i//3%2)]})
            check_world(full,bundle["result"])
        measurements[str(size)] = dict(samples=stats,summary={
            name:{key:median(row[key] for row in rows) for key in rows[0]} for name,rows in stats.items()})
    relay = Relay(wasm)
    open_samples = []
    for _ in range(12):
        start = perf_counter_ns()
        reconstructed = reconstruct(packet,relay)
        open_samples.append(perf_counter_ns()-start)
        assert reconstructed == detail["reconstructed"]
    return dict(profile="visuaml-on-demand-trace-v1", **detail),packet,dict(
        workloads=measurements, warm_open_ns=open_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--reconstruct",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        if args.reconstruct:
            print(json.dumps(reconstruct(json.loads(args.reconstruct.read_text()))))
            raise SystemExit
        bundle,packet,timings = replay(3 if args.write else 1)
    if args.write:
        for name,data in (("replay",bundle),("pending",packet),("timings",timings)):
            (OUT/(name+".json")).write_bytes(canonical(data)+b"\n")
    else:
        assert bundle == json.loads((OUT/"replay.json").read_text())
        assert packet == json.loads((OUT/"pending.json").read_text())
    print("Verified compact/full boundaries, checkpoint cadence and reconstructed private trace.")
