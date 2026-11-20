"""Replay an uncommitted right-edge move from a wider tape checkpoint."""
import argparse
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from replay_pocket_universal import BASE, FUEL, LIMIT, OUT as UNIVERSAL
from replay_frozen_recurrent import PACKED, sha
from replay_monitor_program import COMPILED, ROOT, native_rows, no_regions, register
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = ROOT / "evidence/pocket-tape-resume"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    graph_file = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_file))
    wasm = (ROOT / "evidence/pocket-tape-compiler/erase.linked.wasm").read_bytes()
    image = native_rows(no_regions(load_call_image(wasm)))
    table_file = UNIVERSAL / "table.json"
    table = json.loads(table_file.read_text())
    fixture = json.loads((ROOT / "fixtures/pocket-tape-resume.json").read_text())
    states, symbols = table["states"], table["symbols"]
    rows = [-1, 0, 0] * (len(states) * len(symbols))
    for rule in table["rules"].splitlines():
        q, read, dest, write, move = rule.split()
        i = 3 * (int(q) * len(symbols) + int(read))
        rows[i:i+3] = [int(dest), int(write), {"L": -1, "S": 0, "R": 1}[move]]
    program = fixture["program"].splitlines()
    tape = "#" + "".join((";*" if i == 0 else ";-") + op for i, op in enumerate(program)) + "|b"
    tape_base = BASE + (8 + len(rows)) * 4
    short_length = len(tape) + fixture["shortBlanks"]
    long_length = len(tape) + fixture["longBlanks"]

    def initial(length):
        data = [len(states), len(symbols), length, 0, 0, 0, 0, 0, *rows,
                *map(symbols.index, tape.ljust(length, "_"))]
        page = bytearray(65536)
        struct.pack_into(f"<{len(data)}i", page, BASE, *data)
        return bytes(page)

    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, wasm), []).exports(store)

    def tick(page):
        live = initialize(image, 0, page, FUEL)
        steps, _, status, _ = drive(graph, image.rows, live, fuel=FUEL)
        after = live.vm.page.numpy().tobytes()
        exports["memory"].write(store, page)
        assert status == exports["run"](store, 0) and after == bytes(exports["memory"].read(store))
        return after, steps, status

    short, long = initial(short_length), initial(long_length)
    transitions, boundaries = [], []
    growth = None
    simulated = head = pc = 0
    reference_tape = [1] + [0] * fixture["longBlanks"]
    committed_steps = 0
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for _ in range(LIMIT):
            new_long, expected_steps, long_status = tick(long)
            new_short, steps, short_status = tick(short)
            assert long_status == 0
            if short_status:
                assert short_status == 2 and growth is None
                assert struct.unpack_from("<i", new_short, BASE+28)[0] == short_length
                failed = bytearray(short)
                struct.pack_into("<i", failed, BASE+20, 2)
                struct.pack_into("<i", failed, BASE+28, short_length)
                assert new_short == failed
                # Restore pre-attempt checkpoint. Only the logical length changes.
                checkpoint = short
                expanded = bytearray(checkpoint)
                struct.pack_into("<i", expanded, BASE+8, long_length)
                struct.pack_into(f"<{long_length-short_length}i", expanded,
                                 tape_base + 4*short_length, *[symbols.index("_")] * (long_length-short_length))
                assert expanded == long
                growth = dict(transition=len(transitions), simulated_steps=simulated,
                              failed_wasm_steps=steps, checkpoint_sha256=sha(checkpoint),
                              stopped_sha256=sha(new_short), extended_sha256=sha(expanded))
                new_short, steps, short_status = tick(expanded)
            assert short_status == 0 and steps == expected_steps
            q, physical, _, count = struct.unpack_from("<4i", new_short, BASE+12)
            assert count == len(transitions) + 1
            if growth is not None:
                assert new_short == new_long
            else:
                # Normalize the sole header difference while added cells remain blank.
                assert new_short[:BASE+8] == new_long[:BASE+8] and new_short[BASE+12:] == new_long[BASE+12:]
            short, long = new_short, new_long
            committed_steps += steps
            transitions.append(dict(short_sha256=sha(short), wide_sha256=sha(long), head=physical, wasm_steps=steps))
            phase = states[q]
            if phase in ("boundary", "halt"):
                op = program[pc]
                if op == "R":
                    head += 1
                elif op == "M":
                    reference_tape[head] = 1
                if op != "H":
                    pc += 1
                simulated += 1
                decoded = [symbols[i] for i in struct.unpack_from(f"<{long_length}i", long, tape_base)]
                body = decoded[decoded.index("|")+1:]
                assert [int(x in "1b") for x in body] == reference_tape
                assert [i for i, x in enumerate(body) if x in "ab"] == [head]
                assert decoded[:decoded.index("*")].count(";") - 1 == pc
                boundaries.append(dict(simulated=simulated, pc=pc, head=head, tape=reference_tape.copy(),
                                       transitions=count, wasm_steps=committed_steps))
            if phase == "halt":
                break
        else:
            raise AssertionError("Transition allowance exhausted")
    assert growth and growth["simulated_steps"] == 1 and reference_tape == [1, 0, 1, 0, 0]
    result = dict(graph_sha256=sha(graph_file.read_bytes()), wasm_sha256=sha(wasm), table_sha256=sha(table_file.read_bytes()),
                  packed_fallback=False, checkpoint=growth, boundaries=boundaries, transitions=transitions)
    path = OUT / "replay.json"
    output = json.dumps(result, indent=2) + "\n"
    if args.write:
        path.write_text(output)
        (OUT / "checkpoint.bin").write_bytes(checkpoint)
    else:
        assert path.read_text() == output and (OUT / "checkpoint.bin").read_bytes() == checkpoint
    print(growth, boundaries[-1])


if __name__ == "__main__":
    main()
