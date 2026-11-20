"""Pixel-equivalent redraw comparison and exact typed-program replay."""
import argparse
from base64 import b64encode
import json
from pathlib import Path
import statistics
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch

from build_compiled_ordinary import OUT as COMPILED, register
from replay_cartridges import keys, transcript
from replay_compiled_ordinary import native_rows
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha
from replay_ordinary_effects import no_regions
from replay_pocket import BASE, FRAME, SIZE, oracle, player
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/monitor-program"
FUEL = 500_000


def runlog(graph, raw, buttons, compiler=None, retain=False):
    plain = load_call_image(raw)
    image = native_rows(no_regions(plain))
    memory, phase = player(graph, image, FUEL)
    ref = oracle(raw)
    unlowered = oracle(compiler) if compiler else None
    records, frames, captures = [], [], []
    for i, code in enumerate(buttons):
        phases = []
        for label, mode in (("simulation", 0), ("drawing", 1)):
            start = perf_counter_ns()
            steps, calls, got, _ = phase(code, mode)
            elapsed = perf_counter_ns() - start
            expected, page = ref(code, mode)
            assert memory.tobytes() == page and got == expected
            if unlowered:
                assert unlowered(code, mode) == (expected, page)
            if mode == 0:
                first, last = struct.unpack_from("<2i", page, BASE+20)
                captures.append(page)
            phases.append({"stage": label, "source_steps": steps, "host_calls": calls,
                           "page_sha256": sha(page), "elapsed_ns": elapsed})
        frame = page[FRAME:FRAME+SIZE]
        kind = ("boot" if i == 0 else "scroll" if code == 13 and first == 0 else
                "enter" if code == 13 else "backspace" if code == 8 else "printable")
        record = {"tick": i, "button": code, "kind": kind, "first_row": first, "last_row": last,
                  "page_sha256": sha(page), "frame_sha256": sha(frame), "phases": phases,
                  "transcript": transcript(page), "events": {}, "acc": struct.unpack_from("<i", page, BASE+92)[0],
                  "pc": struct.unpack_from("<i", page, BASE+96)[0]}
        if retain:
            record["frame_base64"] = b64encode(frame).decode()
        records.append(record)
        frames.append(frame)
    return records, frames, captures, plain


def exact_store_count(single, image, page):
    state = initialize(image, BASE, page, FUEL)
    stores = 0
    struct.pack_into("<i", state.vm.page.numpy(), BASE+4, 1)
    while int(state.vm.status) == 0:
        result = single(image.rows, state)
        state = result.state
        access = result.access
        if int(access[0]) == 2 and FRAME <= int(access[1]) < FRAME+SIZE:
            stores += 1
    assert int(state.vm.status) == 1
    return stores, FUEL-int(state.vm.fuel)


def stats(values):
    values = sorted(values)
    return {"samples": len(values), "median": statistics.median(values),
            "p95": values[(95*len(values)+99)//100-1]}


def compare_renderers(graph, one):
    buttons = [0, *keys("STX"), 8, *keys("ATUS\nDIG\nSTEP\nMAP\n"),
               *keys("MAP\n")*5, *keys("H?"), 8, *keys("ELP\n")]
    versions = {}
    base = (OUT / "baseline.wasm").read_bytes()
    dirty = (OUT / "dirty.wasm").read_bytes()
    for name, raw in (("baseline", base), ("dirty", dirty)):
        versions[name] = runlog(graph, raw, buttons)
    old, new = versions["baseline"], versions["dirty"]
    assert old[1] == new[1]
    assert [tick["transcript"] for tick in old[0]] == [tick["transcript"] for tick in new[0]]
    lookup = {}
    for kind in ("printable", "backspace", "enter", "scroll"):
        selected = next(i for i, tick in enumerate(old[0]) if tick["kind"] == kind)
        note = {}
        for name, (logs, _, pages, plain) in versions.items():
            actual, steps = exact_store_count(one, plain, pages[selected])
            assert steps == logs[selected]["phases"][1]["source_steps"]
            note[name] = {"framebuffer_stores": actual, "source_steps": steps}
        lookup[kind] = {"witness_tick": selected, **note}
    # Count scheduled framebuffer stores for each frame from the renderer
    # contract: full pairs in the old span, changed character pairs in the new.
    previous = (0,) * 360
    for a, b, page in zip(old[0], new[0], new[2], strict=True):
        lo, hi = a["first_row"], a["last_row"]
        chars = struct.unpack_from("<360i", page, 512)
        changed = sum(chars[i:i+2] != previous[i:i+2] for i in range(20*lo, 20*hi, 2))
        a["framebuffer_stores"] = (hi-lo)*80
        b["framebuffer_stores"] = changed*8
        assert all(previous[i] == chars[i] for i in range(360) if not 20*lo <= i < 20*hi)
        previous = chars
    for note in lookup.values():
        for name in versions:
            assert versions[name][0][note["witness_tick"]]["framebuffer_stores"] == note[name]["framebuffer_stores"]
    summary = {kind: {name: {
        mode: stats([tick["phases"][1]["elapsed_ns"] if mode == "drawing_ns" else tick[mode]
                     for tick in versions[name][0] if tick["kind"] == kind])
        for mode in ("drawing_ns", "framebuffer_stores")}
        for name in versions} for kind in lookup}
    return {"buttons": buttons, "both_framebuffers_exact": True, "both_transcripts_exact": True,
            "audited_store_counts": lookup, "summary": summary,
            "wasm_sha256": {"baseline": sha(base), "dirty": sha(dirty)},
            "runs": {name: values[0] for name, values in versions.items()}}


def program_events(single, raw, buttons, records):
    image = load_call_image(raw)
    reference = oracle(raw)
    previous = bytes(65536)
    enters = 0
    for i, key in enumerate(buttons):
        if key == 13:
            enters += 1
            if enters in (5, 6, 7):
                state = initialize(image, BASE, previous, FUEL)
                struct.pack_into("<2i", state.vm.page.numpy(), BASE, key, 0)
                writes = []
                while int(state.vm.status) == 0:
                    pc = int(state.vm.pc)
                    pulse = single(image.rows, state)
                    state = pulse.state
                    kind, address, before, after = map(int, pulse.access)
                    if kind == 2 and address in (BASE + 92, BASE + 96) and before != after:
                        writes.append({"pc": pc, "address": address, "before": before, "after": after})
                assert int(state.vm.status) == 1
                assert sha(state.vm.page.numpy().tobytes()) == records[i]["phases"][0]["page_sha256"]
                records[i]["events"]["simulation"] = writes
        reference(key, 0)
        _, previous = reference(key, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--checks", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    graph_path = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_path))
    one = torch.jit.load(str(CALLS / "machine.pt"))
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.profile:
            print(json.dumps(compare_renderers(graph, one), indent=2))
            return
        raw = (ROOT / "guest/pocket_program.wasm").read_bytes()
        compiler = (OUT / "program.compiler.wasm").read_bytes()
        if args.checks:
            text = "10 SET 999\n10 SET 1000\n20 ADD 5\n30 HALT\n40 PRINT\nRUN\n"
            buttons = [0, *keys(text)]
            records, _, _, _ = runlog(graph, raw, buttons, compiler)
            lines = [line for line in records[-1]["transcript"] if line]
            assert "BAD LINE" in lines
            assert lines[-3:] == [">RUN", "OK", ">_"]
            assert (records[-1]["acc"], records[-1]["pc"]) == (4, 5)
            print(json.dumps({"buttons": buttons, "transcript": lines, "acc": 4, "pc": 5,
                              "ticks": len(records), "wasm_sha256": sha(raw)}, indent=2))
            return
        for value, result in ((5, 12), (6, 13)):
            text = f"10 SET 7\n20 ADD {value}\n30 PRINT\n40 HALT\nNEXT\nNEXT\nRUN\n"
            session = {"version": 1, "cartridge": "program-monitor",
                       "binary_sha256": sha(raw), "buttons": [0, *keys(text)]}
            ticks, _, _, _ = runlog(graph, raw, session["buttons"], compiler, retain=True)
            program_events(one, raw, session["buttons"], ticks)
            ends = [tick for tick in ticks if tick["button"] == 13]
            assert ends[4]["acc"] == 7 and ends[4]["pc"] == 2
            assert ends[5]["acc"] == result and ends[5]["pc"] == 3
            assert any(write["address"] == BASE + 92 and write["before"] == 7
                       and write["after"] == result for write in ends[5]["events"]["simulation"])
            assert ends[6]["acc"] == result and ends[6]["pc"] == 5
            assert str(result) in ticks[-1]["transcript"]
            for tick in ticks:
                for phase in tick["phases"]:
                    phase.pop("elapsed_ns")
            proof = {"session": session, "wasm_sha256": sha(raw), "graph_sha256": sha(graph_path.read_bytes()),
                     "packed_fallback": False, "ticks": ticks}
            path = ROOT / "fixtures" / f"monitor-program-{value}.json"
            if args.write:
                path.write_text(json.dumps(session, indent=2) + "\n")
                path.with_suffix(".proof.json").write_text(json.dumps(proof, indent=2) + "\n")
            else:
                assert proof == json.loads(path.with_suffix(".proof.json").read_text())
            print(f"Verified ADD {value}, output {result}")


if __name__ == "__main__":
    main()
