"""Verify cartridge authoring fixtures against the same frozen Pocket runner."""
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
from replay_compiled_ordinary import native_rows
from replay_frozen_recurrent import PACKED, sha
from replay_ordinary_effects import no_regions
from replay_pocket import FRAME, SIZE, oracle, pixel, player
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image

ROOT = Path(__file__).resolve().parents[1]
def keys(text):
    return [ord(c) if c != "\n" else 13 for c in text]


def inputs():
    return {
        "rail-crate-play": ("rail-crate", [0, 1, 1, 1]),
        "rail-crate-omitted": ("rail-crate", [0, 2, 1, 1]),
        "monitor-play": ("monitor", [0, *keys("STATUS\nDIG\nSTEP\nMAP\n")]),
        "monitor-omitted": ("monitor", [0, *keys("STATUS\nSTEP\nMAP\n")]),
    }


def transcript(page):
    chars = struct.unpack_from("<360i", page, 512)
    return ["".join(chr(c) if c else " " for c in chars[i:i+20]).rstrip()
            for i in range(0, len(chars), 20)]


def render_check(frame, player_slot, crate_slot):
    for slot in range(5):
        got = pixel(frame, 34 + slot*16, 70)
        assert got == (2 if slot == player_slot else 3 if slot == crate_slot else 1 if slot == 4 else 0)
        assert pixel(frame, 34 + slot*16, 61) == 1


def run(manifest, session, graph, raw, compiler, retain=True):
    assert session["binary_sha256"] == sha(raw) and session["buttons"][0] == 0
    assert all(type(button) is int for button in session["buttons"])
    commands = {-1, 0, *(control["code"] for control in manifest["controls"])}
    assert all(button in commands or manifest.get("keyboard") == "ascii" and
               (button in (8, 13) or 32 <= button <= 126) for button in session["buttons"])
    image = native_rows(no_regions(load_call_image(raw)))
    page, phase = player(graph, image, 500_000)
    reference, unmodified = oracle(raw), oracle(compiler)
    result, raw_timing = [], []
    player_slot, crate_slot = 1, 2
    for index, button in enumerate(session["buttons"]):
        phases, timings = [], {}
        for stage, mode in (("simulation", 0), ("drawing", 1)):
            start = perf_counter_ns()
            steps, calls, value, _ = phase(button, mode)
            timings[f"{stage}_ns"] = perf_counter_ns() - start
            expected, full = reference(button, mode)
            assert (expected, full) == unmodified(button, mode)
            assert value == expected and page.tobytes() == full
            phases.append({"stage": stage, "source_steps": steps, "host_calls": calls,
                           "page_sha256": sha(full)})
        frame = full[FRAME:FRAME+SIZE]
        record = {"tick": index, "button": button, "page_sha256": phases[-1]["page_sha256"],
                  "frame_sha256": sha(frame), "phases": phases, "events": {}}
        if retain:
            record["frame_base64"] = b64encode(frame).decode()
        if manifest["id"] == "rail-crate":
            if button == -1:
                player_slot, crate_slot = 1, 2
            elif index and button in (1, 2):
                delta = 1 if button == 1 else -1
                dest = player_slot + delta
                if 0 <= dest <= 4 and (dest != crate_slot or 0 <= crate_slot + delta <= 4):
                    if dest == crate_slot:
                        crate_slot += delta
                    player_slot = dest
            assert struct.unpack_from("<2i", full, 268) == (4*player_slot, 4*crate_slot)
            render_check(frame, player_slot, crate_slot)
        elif manifest["id"] == "monitor":
            record["transcript"] = transcript(full)
            if index == 0:
                # Upper-left stem and top bar of P, and the prompt cursor.
                assert [pixel(frame, x, y) for x, y in ((1, 1), (2, 1), (1, 6), (7, 6),
                                                        (9, 15), (15, 15))] == [3, 3, 3, 0, 3, 0]
        result.append(record)
        raw_timing.append(timings)
    return result, raw_timing


def summarize(rows):
    values = {}
    for label in ("simulation_ns", "drawing_ns"):
        samples = sorted(row[label] for row in rows)
        values[label] = {"median_ns": statistics.median(samples),
                         "p95_ns": samples[(95 * len(samples) + 99)//100 - 1]}
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-proofs", action="store_true")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--manifest", type=Path, help="Local manifest for an exported session")
    parser.add_argument("--wasm", type=Path, help="Local WASM for an exported session")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.wasm) or (args.manifest and not args.session):
        parser.error("Use --manifest and --wasm together with --session")
    torch.set_num_threads(1)
    register()
    graph_file = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_file))
    timing = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.session:
            session = json.loads(args.session.read_text())
            if session.get("cartridge") == "passage":
                parser.error("Use replay_pocket.py --session for Passage Relay")
            sessions = {"exported": (session["cartridge"], session["buttons"])}
        else:
            sessions = inputs()
        for name, (cartridge, stream) in sessions.items():
            manifest = json.loads((args.manifest or ROOT / "cartridges" / f"{cartridge}.json").read_text())
            path = args.wasm or ROOT / "guest" / manifest["binary"]
            raw = path.read_bytes()
            assert manifest["binary_sha256"] == sha(raw)
            original = ROOT / "evidence/cartridges" / f"{path.stem}.compiler.wasm"
            compiler = original.read_bytes() if not args.wasm and original.exists() else raw
            if not args.session:
                session = {"version": 1, "cartridge": cartridge,
                           "binary_sha256": manifest["binary_sha256"], "buttons": stream}
            ticks, phases = run(manifest, session, graph, raw, compiler)
            if name == "monitor-play":
                assert ticks[-1]["transcript"][:10] == [
                    "POCKET MONITOR", ">STATUS", "T:000 A:0 B:2", ">DIG", "PASSAGE OPEN",
                    ">STEP", "STEP COMPLETE", ">MAP", "MAP AB.", ">_",
                ]
            if name == "monitor-omitted":
                assert "MAP A#B" in ticks[-1]["transcript"] and "PASSAGE OPEN" not in ticks[-1]["transcript"]
            output = {"graph_sha256": sha(graph_file.read_bytes()), "wasm_sha256": sha(raw),
                      "compiler_wasm_sha256": sha(compiler), "packed_fallback": False,
                      "session": session, "ticks": ticks}
            path = ROOT / "fixtures" / f"{name}.json"
            if args.write_proofs:
                path.write_text(json.dumps(session, indent=2) + "\n")
                path.with_suffix(".proof.json").write_text(json.dumps(output, indent=2) + "\n")
            elif args.session:
                print(json.dumps(output, indent=2))
            else:
                assert output == json.loads(path.with_suffix(".proof.json").read_text())
                print(f"Verified {name}")
            timing[name] = {"warm_samples": len(phases)-1, "phases": summarize(phases[1:]), "raw": phases}
        if not args.session:
            manifest = json.loads((ROOT / "cartridges/monitor.json").read_text())
            raw = (ROOT / "guest/pocket_monitor.wasm").read_bytes()
            compiler = (ROOT / "evidence/cartridges/pocket_monitor.compiler.wasm").read_bytes()
            regression = [0, *keys("HX"), 8, *keys("ELP\nnope\nRESET\n"),
                          *keys("MAP\n")*6, *keys("STEP\nSTATUS\n")]
            log, measurements = run(manifest, {"buttons": regression, "binary_sha256": sha(raw)},
                                    graph, raw, compiler, retain=False)
            assert "HELP STATUS MAP" in log[7]["transcript"]
            assert any("UNKNOWN COMMAND" in row["transcript"] for row in log)
            assert "POCKET MONITOR" not in log[-1]["transcript"]
            assert "T:001 A:0 B:2" in log[-1]["transcript"]
            assert log[-1]["transcript"][-1] == ">_"
            timing["monitor_edit_error_scroll"] = {"buttons": regression, "ticks": log,
                                                  "timing": measurements}
            if args.write_proofs:
                (ROOT / "evidence/cartridges/visuaml.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
