"""Replay the compiler cartridge and run its exact emitted bytes."""
import argparse
import json
from pathlib import Path
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch

from replay_monitor_program import COMPILED, FUEL, ROOT, keys, load_call_image, native_rows, no_regions, register, runlog
from replay_frozen_recurrent import PACKED, sha, wasmtime_run
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize

OUT = ROOT / "evidence/pocket-compile"


def words(page, offset, count=1):
    return struct.unpack_from(f"<{count}i", page, offset)


def emitted(graph, page, expected=None, states=None):
    length, count = words(page, 400, 2)
    assert 54 <= length <= 174 and 0 <= count <= 4
    if states is not None:
        assert count == len(states)
    binary = page[24576:24576 + length]
    image = native_rows(no_regions(load_call_image(binary)))
    state = initialize(image, 0, bytes(65536), FUEL)
    steps, calls, result, _ = drive(graph, image.rows, state, fuel=FUEL)
    output = state.vm.page.numpy().tobytes()
    assert (result, output) == wasmtime_run(binary, 0, bytes(65536))
    assert 0 <= result <= 4
    printed = list(words(output, 16, result))
    if expected is not None:
        assert printed == expected
    rows, before = [], 0
    for index in range(count):
        line, start, end = words(page, 24000 + 12 * index, 3)
        assert line in (10, 20, 30, 40) and start == (rows[-1]["end"] if rows else 51) and start < end <= length - 3
        assert not rows or line > rows[-1]["line"]
        slot = 32 + 4 * (line // 10 - 1)
        after, = words(output, slot)
        if states is not None:
            assert (line, before, after) == states[index]
        rows.append(dict(line=line, start=start, end=end, before=before, after=after, accumulator_slot=slot))
        before = after
    assert (rows[-1]["end"] if rows else 51) == length - 3
    return binary, dict(binary_sha256=sha(binary), page_sha256=sha(output), printed=printed,
                        source_steps=steps, host_calls=calls, source_map=rows)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--session", type=Path, help="Verify an input log exported by Pocket")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    graph_path = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_path))
    raw = (ROOT / "guest/pocket_compile.wasm").read_bytes()
    compiler = (OUT / "compiler.raw.wasm").read_bytes()
    if args.session:
        session = json.loads(args.session.read_text())
        assert session["version"] == 1 and session["cartridge"] == "compile-monitor"
        assert session["binary_sha256"] == sha(raw)
        buttons = session["buttons"]
        assert buttons and buttons[0] == 0 and all(
            type(key) is int and (key in (-1, 0, 8, 13) or 32 <= key <= 126) for key in buttons)
        with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
            _, _, pages, _ = runlog(graph, raw, buttons, compiler)
            module = emitted(graph, pages[-1])[1] if words(pages[-1], 400)[0] else None
        print(json.dumps(dict(verified_ticks=len(buttons), packed_fallback=False, module=module), indent=2))
        return
    cases = [
        (f"add-{n}", json.loads((ROOT / f"fixtures/monitor-program-{n}.json").read_text())["buttons"] + keys("COMPILE\n"),
         [7+n], [(10, 0, 7), (20, 7, 7+n), (30, 7+n, 7+n), (40, 7+n, 7+n)])
        for n in (5, 6)
    ]
    cases += [
        ("overflow", [0] + keys("10 SET 999\n20 ADD 5\n30 PRINT\nRUN\nCOMPILE\n"), [4],
         [(10, 0, 999), (20, 999, 4), (30, 4, 4)]),
        ("malformed", [0] + keys("10 SET 7\n20 PRINT\nYTDBFLV\nKBHTKOK\nCOMPILE\n10 SET 1000\nCOMPILE\n"), None, []),
        ("max-size", [0] + keys("40 ADD 999\n10 ADD 999\n30 ADD 999\n20 ADD 999\nCOMPILE\n"), [],
         [(10, 0, 999), (20, 999, 998), (30, 998, 997), (40, 997, 996)]),
    ]
    def save(path, value):
        if not isinstance(value, bytes):
            value = (json.dumps(value, indent=2) + "\n").encode()
        if args.write:
            path.write_bytes(value)
        else:
            assert path.read_bytes() == value, path

    timings = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, buttons, expected, states in cases:
            records, _, pages, _ = runlog(graph, raw, buttons, compiler, retain=True)
            if name in ("add-5", "add-6"):
                original = ROOT / f"fixtures/monitor-program-{name[-1]}.proof.json"
                prefix = json.loads(original.read_text())["ticks"]
                assert len(prefix) == len(buttons) - len(keys("COMPILE\n"))
                for reference, current in zip(prefix, records):
                    for field in ("frame_base64", "transcript", "acc", "pc"):
                        assert reference[field] == current[field], (name, current["tick"], field)
            start = perf_counter_ns()
            module = None
            if expected is None:
                assert words(pages[-1], 400, 2) == (0, 0)
                assert "COMPILE ERROR" in records[-1]["transcript"]
                rejected = [tick for tick in records if tick["button"] == 13
                            and any(command in tick["transcript"] for command in (">YTDBFLV", ">KBHTKOK"))]
                assert "UNKNOWN COMMAND" in rejected[0]["transcript"]
                assert "UNKNOWN COMMAND" in rejected[1]["transcript"]
                assert all(words(pages[tick["tick"]], 400)[0] == 0 for tick in rejected[:2])
                assert any(words(page, 400)[0] for page in pages[:-1])
            else:
                assert "WASM READY" in records[-1]["transcript"]
                if expected:
                    run = next(tick for tick in reversed(records) if ">RUN" in tick["transcript"])
                    assert " ".join(map(str, expected)) in run["transcript"]
                binary, module = emitted(graph, pages[-1], expected, states)
                save(OUT / f"{name}.wasm", binary)
                save(OUT / f"{name}.map.json", module)
            timings[name] = dict(emitted_check_ns=perf_counter_ns()-start,
                                 compile_enter_ns=records[-1]["phases"][0]["elapsed_ns"],
                                 ticks=[{p["stage"]: p.pop("elapsed_ns") for p in t["phases"]} for t in records])
            session = dict(version=1, cartridge="compile-monitor", binary_sha256=sha(raw), buttons=buttons)
            proof = dict(session=session, wasm_sha256=sha(raw), graph_sha256=sha(graph_path.read_bytes()),
                         packed_fallback=False, ticks=records, module=module)
            path = ROOT / "fixtures" / f"pocket-compile-{name}.json"
            save(path, session)
            save(path.with_suffix(".proof.json"), proof)
            print(f"{name}: {len(buttons)} exact ticks; emitted prints {expected}", flush=True)
    add5, add6 = ((OUT / f"add-{value}.wasm").read_bytes() for value in (5, 6))
    assert len(add5) == len(add6)
    assert [(i, first, second) for i, (first, second) in enumerate(zip(add5, add6))
            if first != second] == [(65, 5, 6)]
    if args.write:
        save(OUT / "timings.json", timings)


if __name__ == "__main__":
    main()
