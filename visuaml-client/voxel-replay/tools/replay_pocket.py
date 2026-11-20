"""Verify Pocket across engines and measure the paced compiled rendering pipeline."""
import argparse
from base64 import b64encode
import json
import math
from pathlib import Path
import statistics
import struct
from time import perf_counter_ns, sleep
from unittest.mock import patch

import numpy as np
import torch
import wasmtime

from build_compiled_ordinary import OUT as COMPILED, register
from replay_compiled_ordinary import native_rows
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha
from replay_ordinary_effects import no_regions
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/pocket"
BASE = 256
FRAME = 4096
SIZE = 5760
PERIOD = 50_000_000
PALETTE = np.array([[220, 235, 179, 255], [147, 171, 112, 255],
                    [65, 98, 78, 255], [21, 45, 40, 255]], dtype=np.uint8)
SHIFTS = np.array([6, 4, 2, 0], dtype=np.uint8)


def drive(graph, rows, state, audit=False):
    events, count = [], 0
    while int(state.vm.status) == 0:
        pc = int(state.vm.pc) if audit else 0
        tick = graph(rows, state) if audit else graph(rows, state, 64)
        state = tick.state
        if audit:
            kind, address, before, after = map(int, tick.access)
            if kind == 2 and before != after and (address in (BASE + 12, BASE + 16)
                                                    or FRAME <= address < FRAME + SIZE):
                events.append({"pc": pc, "address": address, "before": before, "after": after})
        count += 1
    assert int(state.vm.status) == 1
    return 5000 - int(state.vm.fuel), count, int(state.vm.stack[0]), events


def oracle(raw):
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    memory = exports["memory"]

    def phase(value, mode):
        memory.write(store, struct.pack("<2i", value, mode), BASE)
        result = exports["run"](store, BASE)
        return result, bytes(memory.read(store))

    return phase


def pixel(frame, x, y):
    return (frame[40*y+x//4] >> (6-2*(x%4))) & 3


def check_art(frame, block, crossed):
    assert pixel(frame, 0, 0) == 0 and pixel(frame, 34, 70) == 1
    assert pixel(frame, 81, 59) == 1 and pixel(frame, 81, 82) == 1
    assert pixel(frame, 82, 70) == (3 if block else 2 if crossed else 0)
    assert pixel(frame, 114, 70) == (0 if crossed else 2)


def player(graph, image):
    state = initialize(image, BASE, bytes(65536), 5000)
    page = state.vm.page.numpy()

    def phase(value, mode, audit=False, write_inputs=True):
        if write_inputs:
            struct.pack_into("<2i", page, BASE, value, mode)
        return drive(graph, image.rows, state, audit)

    return page, phase


def verify(session, graph, image, single, original, raw, published=False):
    assert session["version"] == 1 and session["binary_sha256"] == sha(raw)
    assert session["buttons"] and session["buttons"][0] == 0 and all(type(x) is int and x in (-1, 0, 1) for x in session["buttons"])
    page, phase = player(graph, image)
    audit_page, audit_phase = player(single, original)
    compare = oracle(raw)
    evidence = []
    retained = []
    block, crossed, ready = 0, 0, False
    for index, button in enumerate(session["buttons"]):
        if not ready or button == -1:
            block, crossed, ready = 1, 0, True
        else:
            if not block:
                crossed = 1
            if button == 1:
                block = 0
        timing = []
        event_log = {}
        for stage, mode in (("simulation", 0), ("drawing", 1)):
            steps, calls, out, _ = phase(button, mode)
            reference_steps, _, reference_out, events = audit_phase(button, mode, audit=True)
            expected, reference_page = compare(button, mode)
            assert (steps, out) == (reference_steps, reference_out) == (steps, expected)
            assert page.tobytes() == audit_page.tobytes() == reference_page
            timing.append({"stage": stage, "source_steps": steps, "host_calls": calls,
                           "page_sha256": sha(reference_page)})
            event_log[stage] = events
        assert struct.unpack_from("<2i", page, BASE + 12) == (block, crossed)
        frame = page[FRAME:FRAME+SIZE].tobytes()
        check_art(frame, block, crossed)
        retained.append((frame, sha(frame)))
        record = {"tick": index, "button": button, "block": block, "crossed": crossed,
                  "page_sha256": timing[-1]["page_sha256"], "frame_sha256": sha(frame),
                  "phases": timing, "events": event_log}
        if published:
            record["frame_base64"] = b64encode(frame).decode()
        evidence.append(record)
    assert all(sha(frame) == digest for frame, digest in retained)
    if published and session["buttons"] == [0, 1, 0]:
        for state, stage, address in ((1, "simulation", BASE + 12), (1, "drawing", FRAME + 40*66+20),
                                      (2, "simulation", BASE + 16), (2, "drawing", FRAME+40*66+28)):
            assert any(event["address"] == address for event in evidence[state]["events"][stage])
    return evidence


def describe(samples):
    phases = {}
    for key in samples[0]:
        values = sorted(row[key] for row in samples)
        phases[key] = {"median_ns": statistics.median(values), "p95_ns": values[math.ceil(.95*len(values))-1]}
    return {"samples": len(samples), "misses": sum(row["scheduled_ns"] > PERIOD for row in samples),
            "phases": phases, "raw": samples}


def paced(graph, image, raw, buttons):
    page, phase = player(graph, image)
    phase(0, 0)
    phase(0, 1)
    frames, samples = [], []
    epoch = perf_counter_ns()
    for i, button in enumerate(buttons):
        target = epoch + i * PERIOD
        slack = target - perf_counter_ns()
        if slack > 0:
            sleep(slack / 1e9)
        began = perf_counter_ns()
        struct.pack_into("<2i", page, BASE, button, 0)
        mark = perf_counter_ns()
        phase(button, 0, write_inputs=False)
        updated = perf_counter_ns()
        struct.pack_into("<i", page, BASE + 4, 1)
        draw_start = perf_counter_ns()
        phase(button, 1, write_inputs=False)
        drawn = perf_counter_ns()
        frame = page[FRAME:FRAME+SIZE].tobytes()
        published = perf_counter_ns()
        rgba = PALETTE[((np.frombuffer(frame, dtype=np.uint8)[:, None] >> SHIFTS) & 3).ravel()]
        assert rgba.shape == (23040, 4)
        done = perf_counter_ns()
        frames.append(frame)
        samples.append({"input_ns": mark-began+draw_start-updated, "simulation_ns": updated-mark,
                        "drawing_ns": drawn-draw_start, "publication_ns": published-drawn,
                        "palette_expand_ns": done-published, "scheduled_ns": done-target})
    compare = oracle(raw)
    compare(0, 0)
    compare(0, 1)
    for button, frame in zip(buttons, frames, strict=True):
        compare(button, 0)
        _, reference = compare(button, 1)
        assert frame == reference[FRAME:FRAME+SIZE]
    assert page.tobytes() == reference
    return describe(samples)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write-proofs", action="store_true")
    modes.add_argument("--paced", action="store_true")
    modes.add_argument("--session", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    registration = register()
    graph_path = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_path))
    single = torch.jit.load(str(CALLS / "machine.pt"))
    raw = (ROOT / "guest/pocket.wasm").read_bytes()
    original = load_call_image(raw)
    image = native_rows(no_regions(original))
    metadata = {"graph_sha256": sha(graph_path.read_bytes()), "wasm_sha256": sha(raw),
                "image_sha256": image.image_sha256, "packed_fallback": False,
                "cpu_operator_sha256": registration["library_sha256"]}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.paced:
            patterns = {"passage_repeat": ([-1, 0, 1, 0] + [0]*16)*12,
                        "frequent_changes": [-1, 1, 0]*80}
            print(json.dumps({**metadata, **{label: paced(graph, image, raw, stream)
                                            for label, stream in patterns.items()}}, indent=2))
            return
        files = [args.session] if args.session else [ROOT / "fixtures" / f"pocket-{name}.json"
                                                     for name in ("edit", "omitted")]
        for path in files:
            session = json.loads(path.read_text())
            record = {**metadata, "session": session,
                      "ticks": verify(session, graph, image, single, original, raw, published=True)}
            if args.write_proofs:
                path.with_suffix(".proof.json").write_text(json.dumps(record, indent=2) + "\n")
            elif args.session:
                print(json.dumps(record, indent=2))
            else:
                saved = json.loads(path.with_suffix(".proof.json").read_text())
                # Host-specific machine code can be rebuilt on another CPU/toolchain.
                saved.pop("cpu_operator_sha256")
                assert {k: v for k, v in record.items() if k != "cpu_operator_sha256"} == saved
                print(f"Verified {path.stem} across Wasmtime and both frozen paths")


if __name__ == "__main__":
    main()
