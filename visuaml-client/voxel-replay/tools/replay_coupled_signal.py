"""Resume a suspended Rust call and replay its explicit spatial dependencies."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest, encode, identities, restore
from replay_cellular_automata import OUT as CA, export_region, initial, scalar, tensor
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_voxel_call_checkpoint import CALLS, FUEL, Witness, page
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = CA.parent / "pocket-coupled-checkpoint"
MAILBOX = 65472
TARGET = [14, 8]
HORIZON = 6


def mailbox(raw):
    return list(struct.unpack_from("<8i", raw, MAILBOX))


class Relay(Witness):
    def __init__(self):
        torch.set_num_threads(1)
        self.raw = (OUT / "signal-relay.wasm").read_bytes()
        self.image = load_call_image(self.raw)
        graph = CALLS / "machine.pt"
        self.machine = torch.jit.load(str(graph))
        self.identity = identities(graph.read_bytes(), self.image)
        rule = CA / "neighborhood.pt"
        self.rule = torch.jit.load(str(rule))
        self.rule_hash = digest(rule.read_bytes())

    def start(self, raw):
        return initialize(self.image, MAILBOX, raw, FUEL)

    def invoke(self, raw):
        state, trace = self.finish(self.start(raw))
        expected = wasmtime_run(self.raw, MAILBOX, raw)
        assert expected == (mailbox(page(state))[5], page(state))
        return state, trace

    def capture(self, scene, dig, outside=False, halo=6):
        raw = bytearray(65536)
        struct.pack_into("<8i", raw, MAILBOX, 7, dig, -1, -1, 0, 0, 0, outside)
        state = self.start(raw)
        prefix = []
        while int(state.vm.status) == 0:
            state, record = self.step(state)
            prefix.append(record)
            if record["access"][:2] == [1, MAILBOX] and record["depth"] == 1:
                break
        else:
            raise AssertionError("Missing suspended helper read")
        region = export_region(scene, TARGET, halo, HORIZON)
        payload = dict(profile="visuaml-coupled-signal-v1", vm=encode(state, self.identity),
                       region=region, halo=halo, rule_sha256=self.rule_hash, cursor=0,
                       stage="finish_helper", world_size=[scene["width"],scene["height"]])
        return bytes(raw), prefix, state, dict(payload, checkpoint_sha256=digest(canonical(payload)))

    def resume(self, bundle, diagnostic=False):
        body = {key:value for key,value in bundle.items() if key != "checkpoint_sha256"}
        if (digest(canonical(body)) != bundle["checkpoint_sha256"]
                or body["profile"] != "visuaml-coupled-signal-v1" or body["rule_sha256"] != self.rule_hash
                or body["cursor"] != 0 or body["stage"] != "finish_helper"):
            raise ValueError("Checkpoint integrity, rule or cursor mismatch")
        region = body["region"]
        width,height = body["world_size"]
        ox,oy = region["origin"]
        if (region["format"] != "signal-relay-region-v1" or region["horizon"] != HORIZON
                or region["target"] != TARGET or region["tick"] != 0
                or region["gate"] != [9,8]
                or not (0 <= ox < ox+region["width"] <= width and 0 <= oy < oy+region["height"] <= height)):
            raise ValueError("Invalid regional checkpoint")
        x,y = TARGET
        needed = (max(0,x-HORIZON),max(0,y-HORIZON),min(width,x+HORIZON+1),min(height,y+HORIZON+1))
        covers = ox <= needed[0] and oy <= needed[1] and (
            ox+region["width"] >= needed[2] and oy+region["height"] >= needed[3])
        if not covers and not diagnostic:
            raise ValueError("Checkpoint halo does not cover the declared horizon")
        return self.world(restore(body["vm"], self.identity, self.image), body["region"], body["world_size"])

    def world(self, state, region, world_size):
        width,height = region["width"],region["height"]
        ox,oy = region["origin"]
        walls = tensor(region["walls"],height,width)
        signal = tensor(region["signal"],height,width)

        def index(coord):
            x,y = coord
            if not (ox <= x < ox+width and oy <= y < oy+height):
                return None
            return (y-oy,x-ox)

        state, helper_suffix = self.finish(state)
        complete = page(state)
        words = mailbox(complete)
        assert words[2] == 7 and words[6] == 1
        gate_store = next((row for row in helper_suffix if row["access"][:2] == [2,MAILBOX]), None)
        assert (gate_store is not None) == (words[0] == 0)
        if gate_store:
            assert gate_store["access"] == [2,MAILBOX,7,0]
        request_store = [row for row in helper_suffix if row["access"][:2] == [2,MAILBOX+12]][-1]
        gate_y,gate_x = index(region["gate"])
        if gate_store:
            walls[0,0,gate_y,gate_x] = 0
        flat = words[3]
        assert 0 <= flat < world_size[0]*world_size[1]
        request = [flat%world_size[0],flat//world_size[0]]
        region_index = index(request)
        # The six-tick contract covers only the target cone. A different guest
        # read must be re-certified even if the coordinate happens to be retained.
        covered = region_index is not None and (request == region["target"] or
                                              [width,height] == world_size)
        frames, observations = [], []
        scalar_walls = walls.flatten().int().tolist()
        scalar_signal = region["signal"]
        for tick in range(1,HORIZON+1):
            signal = self.rule(walls,signal)
            scalar_signal = scalar(scalar_walls, scalar_signal, width, height)
            frames.append(signal.flatten().int().tolist())
            assert frames[-1] == scalar_signal
            if not covered:
                return dict(helper_suffix=helper_suffix, gate_store=gate_store, request_store=request_store, observation_request=request,
                            observations=observations, frames=frames,
                            violation=dict(tick=tick, coordinate=request, reason="outside_retained_region"
                                           if region_index is None else "uncertified_observation"))
            y,x = region_index
            observed = int(signal[0,0,y,x].item())
            resumed = bytearray(complete)
            struct.pack_into("<i", resumed, MAILBOX+16, observed)
            state, trace = self.invoke(bytes(resumed))
            complete = page(state)
            load_at = next(i for i,row in enumerate(trace) if row["access"][:2] == [1,MAILBOX+16])
            branch = next(row for row in trace[load_at+1:] if row["opcode"] == "br_if")
            action = next((row for row in trace if row["access"][:2] == [2,MAILBOX+20]), None)
            assert (action is not None) == bool(observed)
            observations.append(dict(tick=tick, coordinate=request, value=observed,
                                     load=trace[load_at], branch=branch, action_store=action,
                                     action=mailbox(complete)[5], trace=trace, page_sha256=digest(complete)))
        return dict(helper_suffix=helper_suffix, gate_store=gate_store, request_store=request_store, observation_request=request,
                    observations=observations, frames=frames, violation=None)


def full_region(scene):
    return dict(origin=[0,0], width=scene["width"],height=scene["height"],
                walls=scene["walls"],signal=scene["signal"],target=TARGET,
                gate=[scene["gate"]%scene["width"],scene["gate"]//scene["width"]])


def compare(full, regional, region, width):
    for key in ("helper_suffix","gate_store","request_store","observation_request","observations","violation"):
        assert full[key] == regional[key], key
    ox,oy = region["origin"]
    # A finite halo guarantees the target cone. Other crop boundary cells can diverge.
    for tick,(a,b) in enumerate(zip(full["frames"],regional["frames"]),1):
        for i,value in enumerate(b):
            x,y = ox+i%region["width"],oy+i//region["width"]
            if abs(x-TARGET[0])+abs(y-TARGET[1]) <= HORIZON-tick:
                assert value == a[y*width+x]


def fresh(bundle, diagnostic=False):
    with TemporaryDirectory() as temp:
        path = Path(temp)/"coupled.json"
        path.write_bytes(canonical(bundle))
        cmd = [sys.executable,str(Path(__file__).resolve()),"--resume",str(path)]
        if diagnostic:
            cmd.append("--diagnostic")
        return json.loads(subprocess.check_output(cmd))


def replay(relay):
    scene = initial()
    full = full_region(scene)
    world_size = [scene["width"],scene["height"]]
    saved, result = {}, {}
    for name,dig,outside,halo in (("dig",1,0,6),("omitted",0,0,6),
                                  ("short_halo",1,0,5),("outside_read",1,1,6)):
        raw,prefix,state,snapshot = relay.capture(scene,dig,outside,halo)
        direct = relay.world(state,full,world_size)
        complete, whole = relay.invoke(raw)
        assert prefix + direct["helper_suffix"] == whole
        assert mailbox(page(complete))[0] == (0 if dig else 7)
        regional = fresh(snapshot, diagnostic=halo==5)
        if name in ("dig","omitted"):
            compare(direct,regional,snapshot["region"],scene["width"])
        saved[name] = snapshot
        result[name] = dict(prefix=prefix, region=snapshot["region"], full=direct, regional=regional)
    dig = result["dig"]["regional"]
    closed = result["omitted"]["regional"]
    short = result["short_halo"]["regional"]
    assert [o["value"] for o in dig["observations"]] == [0]*5+[1]
    assert [o["action"] for o in closed["observations"]] == [0]*6
    first = next((a,b) for a,b in zip(dig["observations"],short["observations"]) if a["value"] != b["value"])
    assert first[0]["tick"] == 6 and first[0]["action"] == 1 and first[1]["action"] == 0
    assert first[0]["branch"]["next_pc"] != first[1]["branch"]["next_pc"]
    outside = result["outside_read"]
    assert outside["regional"]["violation"] == dict(tick=1,coordinate=[2,8],reason="outside_retained_region")
    assert outside["regional"]["observations"] == [] and outside["full"]["observations"][0]["action"] == 1
    try:
        relay.resume(saved["short_halo"])
    except ValueError as error:
        assert "halo" in str(error)
    else:
        raise AssertionError("Uncertified checkpoint accepted")
    path = []
    width = saved["dig"]["region"]["width"]
    ox,oy = saved["dig"]["region"]["origin"]
    for tick in range(1,7):
        x,y = 8+tick,8
        assert dig["frames"][tick-1][(y-oy)*width+x-ox] == 1
        path.append(dict(tick=tick,cell=[x,y]))
    return saved, dict(profile="visuaml-coupled-signal-v1", packed_fallback=False, identities=relay.identity,
                       rule_sha256=relay.rule_hash, checkpoints={key:value["checkpoint_sha256"]
                                                               for key,value in saved.items()},
                       cases=result, first_short_observation_difference=dict(tick=6, coordinate=TARGET,
                           full=first[0]["value"], short=first[1]["value"]),
                       signal_path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--diagnostic",action="store_true")
    cli = parser.parse_args()
    relay = Relay()
    with torch.inference_mode(), patch(PACKED,side_effect=AssertionError("Unexpected packed fallback")):
        if cli.resume:
            print(json.dumps(relay.resume(json.loads(cli.resume.read_text()),cli.diagnostic)))
            raise SystemExit
        saved, bundle = replay(relay)
    if cli.write:
        for name,snapshot in saved.items():
            (OUT/f"checkpoint-{name}.json").write_bytes(canonical(snapshot)+b"\n")
        (OUT/"replay.json").write_text(json.dumps(bundle,separators=(",",":"))+"\n")
    else:
        for name,snapshot in saved.items():
            assert snapshot == json.loads((OUT/f"checkpoint-{name}.json").read_text())
        assert bundle == json.loads((OUT/"replay.json").read_text())
    print("Fresh-process VM plus region resumes match full-world runs; short and outside-read controls verified.")
