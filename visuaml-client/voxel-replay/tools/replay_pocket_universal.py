"""Verify every universal transition and independently decode simulated boundaries."""
import argparse
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from build_universal_table import assemble
from replay_frozen_calls import run as trace_run
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha
from replay_monitor_program import COMPILED, ROOT, native_rows, no_regions, register
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = ROOT / "evidence/pocket-universal"
BASE = 256
FUEL = 2000
LIMIT = 4096


def replay(graph, image, single, original, wasm, table, case):
    instructions = [line.split() for line in case["program"].splitlines()]
    program = "".join(";" + ("*" if i == 0 else "-") + op[0] +
                      ("u" * int(op[1]) if len(op) > 1 else "")
                      for i, op in enumerate(instructions))
    encoded = "#" + program + "|" + ("b" if case["input"][0] == "1" else "a") + case["input"][1:] + "_" * 24
    states, symbols = table["states"], table["symbols"]
    data = [word for _ in range(len(states) * len(symbols)) for word in (-1, 0, 0)]
    for line in table["rules"].splitlines():
        q, read, dest, write, move = line.split()
        index = 3 * (int(q) * len(symbols) + int(read))
        data[index:index+3] = [int(dest), int(write), {"L": -1, "R": 1, "S": 0}[move]]
    words = [len(states), len(symbols), len(encoded), 0, 0, 0, 0, 0, *data,
             *map(symbols.index, encoded)]
    page = bytearray(65536)
    struct.pack_into(f"<{len(words)}i", page, BASE, *words)
    tape_offset = BASE + 4 * (8 + len(data))
    reference_tape = list(map(int, case["input"])) + [0] * 24
    head = pc = total = attempted = 0
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, wasm), []).exports(store)
    ref = exports["memory"]
    ref.write(store, page)
    boundaries, ticks, stores = [], [], []
    for _ in range(LIMIT):
        vm = initialize(image, 0, page, FUEL)
        steps, _, result, _ = drive(graph, image.rows, vm, fuel=FUEL)
        after = vm.vm.page.numpy().tobytes()
        assert exports["run"](store, 0) == result == 0
        assert bytes(ref.read(store)) == after
        q, physical, status, count = struct.unpack_from("<4i", after, BASE + 12)
        assert status == 0 and count == len(ticks) + 1
        changes = [(tape_offset + 4*i, x, y) for i, (x, y) in enumerate(zip(
            struct.unpack_from(f"<{len(encoded)}i", page, tape_offset),
            struct.unpack_from(f"<{len(encoded)}i", after, tape_offset))) if x != y]
        ticks.append(dict(phase=states[q], head=physical, tape_changes=changes,
                          wasm_steps=steps, page_sha256=sha(after)))
        prior_state, = struct.unpack_from("<i", page, BASE + 12)
        if states[prior_state] in ("execute_M", "execute_E"):
            audited, trace = trace_run(single, original, 0, page, fuel=FUEL, trace=True)
            assert audited.vm.page.numpy().tobytes() == after and len(trace) == steps
            for instruction, row in enumerate(trace, 1):
                kind, address, before, written = row["access"]
                if kind == 2 and address >= tape_offset:
                    stores.append(dict(address=address, before=symbols[before], after=symbols[written],
                                       universal_transition=count, wasm_step=total + instruction,
                                       rust_pc=row["pc"], simulated_step=attempted + 1))
        total += steps
        page = after
        if q != 0 and states[q] != "halt":
            continue
        attempted += 1
        op, *arg = instructions[pc]
        halted = op == "H"
        if op in "ME":
            reference_tape[head] = int(op == "M")
        if op in "LR":
            head += -1 if op == "L" else 1
        if not halted:
            pc = int(arg[0]) if op == "G" or op == "J" and reference_tape[head] else pc + 1
        observed = [symbols[i] for i in struct.unpack_from(f"<{len(encoded)}i", after, tape_offset)]
        apparent = observed[observed.index("|") + 1:]
        assert [int(s in "1b") for s in apparent] == reference_tape
        assert [i for i, symbol in enumerate(apparent) if symbol in "ab"] == [head]
        cursor = observed.index("*")
        assert observed[:cursor].count(";") - 1 == pc
        assert ("@" not in observed and "!" not in observed and "v" not in observed)
        assert (states[q] == "halt") == halted
        boundaries.append(dict(step=attempted, pc=pc, head=head, tape=reference_tape.copy(),
                               universal_transitions=count, wasm_steps=total, halted=halted))
        if halted:
            return dict(initial_tape=encoded, boundaries=boundaries, transitions=ticks, audited_stores=stores)
    raise AssertionError("Universal transition allowance exhausted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    graph_path = COMPILED / "machine.pt"
    graph = torch.jit.load(str(graph_path))
    wasm = (ROOT / "evidence/pocket-tape-compiler/erase.linked.wasm").read_bytes()
    original = load_call_image(wasm)
    image = native_rows(no_regions(original))
    single = torch.jit.load(str(CALLS / "machine.pt"))
    table_file = OUT / "table.json"
    table = json.loads(table_file.read_text())
    assert table == assemble()
    cases = json.loads((ROOT / "fixtures/pocket-universal.json").read_text())
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        runs = {case["name"]: replay(graph, image, single, original, wasm, table, case) for case in cases}
    left, right = runs.values()
    assert sum(a != b for a, b in zip(left["initial_tape"], right["initial_tape"], strict=True)) == 1
    assert left["boundaries"][:2] == right["boundaries"][:2]
    assert left["boundaries"][2]["tape"][1] == 1 and right["boundaries"][2]["tape"][1] == 0
    assert [run["audited_stores"][0]["after"] for run in runs.values()] == ["b", "a"]
    evidence = dict(graph_sha256=sha(graph_path.read_bytes()), linked_wasm_sha256=sha(wasm),
                    table_sha256=sha(table_file.read_bytes()), packed_fallback=False,
                    universal_transition_limit=LIMIT, wasm_fuel_per_transition=FUEL, runs=runs)
    output = json.dumps(evidence, indent=2) + "\n"
    target = OUT / "replay.json"
    if args.write:
        target.write_text(output)
    else:
        assert target.read_text() == output
    for name, run in runs.items():
        print(name, run["boundaries"][2], flush=True)


if __name__ == "__main__":
    main()
