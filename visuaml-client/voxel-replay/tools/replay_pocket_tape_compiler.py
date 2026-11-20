"""Verify the guest compiler, linked helper, and source-attributed tape stores."""
import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from replay_frozen_calls import run
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha, wasmtime_run
from replay_monitor_program import COMPILED, ROOT, keys, register, runlog
from replay_pocket_compile import words
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from replay_compiled_ordinary import native_rows
from replay_ordinary_effects import no_regions

OUT = ROOT / "evidence/pocket-tape-compiler"
CASES = {
    "erase": ("112", "112", "202", [0, 0, 0, 1]),
    "keep": ("112", "112", "222", [0, 0, 0, 1]),
    "boundary": ("001", "010", "112", [2]),
    "duplicate": ("112", "112", "122", []),
}
TABLE = 256
CELL = TABLE + 17 * 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    def save(name, data):
        path = OUT / name
        if not isinstance(data, bytes):
            data = (json.dumps(data, indent=2) + "\n").encode()
        if args.write:
            path.write_bytes(data)
        else:
            assert path.read_bytes() == data, path

    torch.set_num_threads(1)
    register()
    artifact = COMPILED / "machine.pt"
    graph = torch.jit.load(str(artifact))
    single = torch.jit.load(str(CALLS / "machine.pt"))
    raw = (ROOT / "guest/pocket_tape_compiler.wasm").read_bytes()
    helper = ROOT / "guest/pocket_tape.wasm"
    cases = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name, (tape, first, second, statuses) in CASES.items():
            program = [f"10 TAPE {tape}", f"20 RULE {first}", f"30 RULE {second}", "40 STEP", "COMPILE"]
            buttons = [0] + keys("\n".join(program) + "\n")
            records, _, pages, _ = runlog(graph, raw, buttons, (OUT / "monitor.raw.wasm").read_bytes())
            page = pages[-1]
            size, = words(page, 400)
            entry = dict(program=program, buttons=buttons, monitor_pages=[r["page_sha256"] for r in records],
                         framebuffer_hashes=[r["frame_sha256"] for r in records])
            cases[name] = entry
            if not statuses:
                assert size == 0 and "COMPILE ERROR" in records[-1]["transcript"]
                continue
            assert "WASM READY" in records[-1]["transcript"]
            code = page[24576:24576 + size]
            data = page[25000:25084]
            rule_lines = words(page, 25100, 3)
            expected_header = (1, 3, 4, 0, 0, 0, 0, 0)
            rule_words = [-1, 0, 0] * 3
            for digits in (first, second):
                symbol, write, move = map(int, digits)
                rule_words[symbol * 3:symbol * 3 + 3] = [0, write, move - 1]
            assert words(data, 0, 21) == (*expected_header, *rule_words, *map(int, tape), 0)
            module = OUT / f"{name}.emitted.wasm"
            save(module.name, code)
            save(f"{name}.table.bin", data)
            with TemporaryDirectory() as temp:
                linked = Path(temp) / "linked.wasm"
                link = json.loads(subprocess.check_output(
                    ["node", ROOT / "tools/link_pocket_rust_call.mjs", module, helper, linked, "tapeStep"], text=True))
                merged = linked.read_bytes()
            save(f"{name}.linked.wasm", merged)
            assert code[-7:] == bytes.fromhex("0041800210000b") and merged[-7:] == code[-7:]
            image = load_call_image(merged)
            native = native_rows(no_regions(image))
            entry.update(linked_sha256=sha(merged), module_sha256=sha(code), table_sha256=sha(data),
                         call_byte=73 + link["delta"], rule_lines=rule_lines, ticks=[])

            def step(memory, status, line):
                prior_head, = words(memory, TABLE + 16)
                state = initialize(native, 0, memory, 2000)
                steps, _, result, _ = drive(graph, native.rows, state, fuel=2000)
                after = state.vm.page.numpy().tobytes()
                assert (result, after) == wasmtime_run(merged, 0, memory)
                reference, trace = run(single, image, 0, memory, trace=True)
                assert steps == len(trace) and reference.vm.page.numpy().tobytes() == after and result == status
                call = next(row for row in trace if row["opcode"] == "call")
                assert call["depth"] == 1 and call["locals"][0] == TABLE
                returned = next(row for row in trace if row["function"] == 0 and row["depth"] == 0)
                assert returned["stack"] == [status]
                symbol = words(memory, CELL + 4 * prior_head)[0]
                selected = TABLE + 4 * (8 + 3 * symbol)
                lookup = next(row for row in trace if row["function"] == 0 and row["access"][:2] == [1, selected])
                stores = [dict(pc=row["pc"], address=row["access"][1], before=row["access"][2], after=row["access"][3])
                          for row in trace if row["access"][0] == 2 and CELL <= row["access"][1] < CELL + 16]
                assert len(stores) == int(status == 0)
                if stores:
                    assert stores[0]["address"] == CELL + 4 * prior_head
                return after, dict(status=status, line=line, head=words(after, TABLE + 16)[0],
                                   tape=words(after, CELL, 4), stores=stores, page_sha256=sha(after),
                                   call_pc=call["pc"], return_pc=returned["pc"],
                                   rule_address=selected, lookup_pc=lookup["pc"], source_steps=steps)

            memory = bytearray(65536)
            memory[TABLE:TABLE + len(data)] = data
            for status in statuses:
                head, = words(memory, TABLE + 16)
                line = rule_lines[words(memory, CELL + 4 * head)[0]]
                memory, tick = step(memory, status, line)
                entry["ticks"].append(tick)
            if name == "boundary":
                assert entry["ticks"][0]["tape"] == (0, 0, 1, 0)
            if name == "erase":
                malformed = bytearray(65536)
                malformed[TABLE:TABLE + len(data)] = data
                # Corrupt the active row's write symbol after compiler admission.
                malformed[TABLE + 12 * 4] = 3
                _, entry["malformed_table"] = step(malformed, 3, 20)
                assert entry["malformed_table"]["tape"] == (1, 1, 2, 0)
            print(name, size, entry["ticks"][-1]["tape"], flush=True)
    a, b = (cases[name]["ticks"] for name in ("erase", "keep"))
    assert [(x["head"], x["tape"], x["stores"]) for x in a[:2]] == [
        (x["head"], x["tape"], x["stores"]) for x in b[:2]]
    assert a[2]["stores"][0]["after"] == 0 and b[2]["stores"][0]["after"] == 2
    assert a[2]["line"] == b[2]["line"] == 30 and a[2]["stores"][0]["address"] == CELL + 8
    assert cases["erase"]["linked_sha256"] == cases["keep"]["linked_sha256"]
    source, changed = ((OUT / f"{name}.table.bin").read_bytes() for name in ("erase", "keep"))
    assert [(i, x, y) for i, (x, y) in enumerate(zip(source, changed)) if x != y] == [(60, 0, 2)]
    save("replay.json", dict(graph_sha256=sha(artifact.read_bytes()), cartridge_sha256=sha(raw),
                            helper_sha256=sha(helper.read_bytes()), packed_fallback=False, cases=cases))


if __name__ == "__main__":
    main()
