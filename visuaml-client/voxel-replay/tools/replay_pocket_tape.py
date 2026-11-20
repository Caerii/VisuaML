"""Check transition-level tape state and record the frozen graph's exact stores."""
import argparse
import json
import struct
from unittest.mock import patch

import torch

from replay_frozen_calls import run
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED, sha, wasmtime_run
from replay_monitor_program import COMPILED, ROOT, load_call_image, native_rows, no_regions, register
from replay_pocket import drive
from visuaml.compiler_lab.frozen_calls_v58 import initialize

BASE = 256
OUT = ROOT / "evidence/pocket-tape"


def replay(compiled, single, image, native, wasm, raw, case):
    states, symbols = case["states"], case["symbols"]
    tape = case["tape"].copy()
    head, state, count, status = case["head"], 0, 0, 0
    rules = {}
    for rule in case["rules"].splitlines():
        q, symbol, next_state, write, move = rule.split()
        rules[int(q), int(symbol)] = int(next_state), int(write), {"L": -1, "S": 0, "R": 1}[move]
    table = [word for q in range(states) for symbol in range(symbols)
             for word in rules.get((q, symbol), (-1, 0, 0))]
    initial = [states, symbols, len(tape), state, head, status, count, 0, *table, *tape]
    page = bytearray(65536)
    struct.pack_into(f"<{len(initial)}i", page, BASE, *initial)
    tape_base = BASE + 4 * (8 + len(table))
    ticks = []
    for _ in range(case["limit"]):
        before = [state, head, tape[head]]
        old_head = head
        rule = rules.get((state, tape[head]))
        if rule is None:
            status = 1
        else:
            next_state, symbol, move = rule
            if not 0 <= head + move < len(tape):
                status = 2
            else:
                state, tape[head], head, count = next_state, symbol, head + move, count + 1
        expected, next_page = wasmtime_run(wasm, BASE, page)
        assert wasmtime_run(raw, BASE, page) == (expected, next_page)
        live = initialize(native, BASE, page, 2000)
        steps, calls, result, _ = drive(compiled, native.rows, live, fuel=2000)
        assert result == status == expected
        assert live.vm.page.numpy().tobytes() == next_page
        audited, trace = run(single, image, BASE, page, trace=True)
        assert len(trace) == steps and audited.vm.page.numpy().tobytes() == next_page
        assert struct.unpack_from("<4i", next_page, BASE+12) == (state, head, status, count)
        assert list(struct.unpack_from(f"<{len(tape)}i", next_page, tape_base)) == tape
        accesses = [{"pc": row["pc"], "kind": row["access"][0], "address": row["access"][1],
                     "before": row["access"][2], "after": row["access"][3]}
                    for row in trace if row["access"][0] in (1, 2) and row["access"][1] >= tape_base]
        assert accesses and all(a["address"] == tape_base + 4 * old_head for a in accesses)
        assert sum(a["kind"] == 2 for a in accesses) == int(status == 0)
        ticks.append(dict(before=before, state=state, head=head, tape=tape.copy(),
                          status=status, transitions=count, accesses=accesses, source_steps=steps,
                          host_calls=calls, page_sha256=sha(next_page)))
        page = next_page
        if status:
            break
    assert status in (1, 2)
    return ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    machine = COMPILED / "machine.pt"
    compiled = torch.jit.load(str(machine))
    single = torch.jit.load(str(CALLS / "machine.pt"))
    wasm = (ROOT / "guest/pocket_tape.wasm").read_bytes()
    raw = (OUT / "tape.raw.wasm").read_bytes()
    image = load_call_image(wasm)
    native = native_rows(no_regions(image))
    cases = json.loads((ROOT / "fixtures/pocket-tape.json").read_text())
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        runs = {case["name"]: replay(compiled, single, image, native, wasm, raw, case) for case in cases}
    data = dict(graph_sha256=sha(machine.read_bytes()), wasm_sha256=sha(wasm), packed_fallback=False, runs=runs)
    output = json.dumps(data, indent=2) + "\n"
    path = OUT / "replay.json"
    if args.write:
        path.write_text(output)
    else:
        assert path.read_text() == output
    for name, ticks in runs.items():
        print(name, len(ticks), ticks[-1]["tape"], ticks[-1]["status"])


if __name__ == "__main__":
    main()
