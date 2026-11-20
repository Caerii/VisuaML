"""Verify the frozen call graph against Wasmtime and record source-level traces."""
import json
import struct

import torch
import wasmtime

from replay_frozen_recurrent import EVIDENCE, sha, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page

OUT = EVIDENCE.parent / "frozen-calls"
NESTED = """(module
 (memory (export "memory") 1 1)
 (func (param i32) (result i32) local.get 0 i32.load return)
 (func (param i32) (result i32) local.get 0 call 0)
 (func (export "run") (param i32) (result i32) (local i32)
   i32.const 17 local.set 1
   i32.const 9 local.get 0 call 1 i32.add local.get 1 i32.add))"""


def run(machine, image, argument, page, fuel=2000, trace=False):
    state = initialize(image, argument, page, fuel)
    live = state.vm.page.data_ptr()
    log = []
    while int(state.vm.status) == 0:
        previous = state
        tick = machine(image.rows, state)
        state = tick.state
        assert state.vm.page.data_ptr() == live
        if trace:
            pc = int(previous.vm.pc)
            function = max(i for i, start in enumerate(image.starts) if start <= pc)
            source = pc - image.starts[function]
            log.append({"function": function, "pc": source,
                        "opcode": image.functions[function][source],
                        "next_image_pc": int(state.vm.pc), "depth": int(state.depth),
                        "fuel": int(state.vm.fuel), "status": int(state.vm.status),
                        "stack": state.vm.stack[:int(state.vm.height)].tolist(),
                        "locals": state.vm.locals.tolist(), "access": tick.access.tolist()})
    return state, log


def main():
    torch.set_num_threads(1)
    path = OUT / "machine.pt"
    machine = torch.jit.load(str(path))
    raw = (OUT / "passage.wasm").read_bytes()
    image = load_call_image(raw, memory_pages=256)
    size = 16 * 1024 * 1024
    cell = size - 12
    scenes = {}
    for edit in (0, 1):
        page = bytearray(size)
        struct.pack_into("<iii", page, cell, 7, edit, 0)
        ticks = []
        for expected in ((7, 0), (7, 0)) if not edit else ((0, 0), (0, 1)):
            state, log = run(machine, image, cell, page, trace=True)
            assert int(state.vm.status) == 1 and int(state.depth) == 0
            expected_page = page.copy()
            struct.pack_into("<iii", expected_page, cell, expected[0], page[cell + 4], expected[1])
            actual = snapshot_page(state.vm)
            assert actual == expected_page
            assert wasmtime_run(raw, cell, page) == (expected[1], actual)
            assert int(state.vm.stack[0]) == expected[1]
            calls = [row for row in log if row["opcode"] == "call"]
            assert len(calls) == 1 and calls[0]["depth"] == 1
            helper_return = [row for row in log if row["function"] == 0 and row["depth"] == 0]
            assert len(helper_return) == 1
            assert helper_return[0]["next_image_pc"] == image.starts[1] + calls[0]["pc"] + 1
            ticks.append({"record": list(struct.unpack_from("<iii", actual, cell)),
                          "sha256": sha(actual), "steps": len(log), "trace": log})
            page = bytearray(actual)
            page[cell + 4] = 0
        edit_stores = [row for row in ticks[0]["trace"]
                       if row["access"][0] == 2 and row["access"][1] == cell]
        assert len(edit_stores) == edit
        assert all(row["function"] == 0 and row["depth"] == 1 for row in edit_stores)
        scenes["edit" if edit else "omitted"] = ticks

    raw_nested = bytes(wasmtime.wat2wasm(NESTED))
    nested = load_call_image(raw_nested)
    page = bytearray(65536)
    page[0] = 42
    full, trace = run(machine, nested, 0, page, trace=True)
    assert int(full.vm.stack[0]) == 68
    assert wasmtime_run(raw_nested, 0, page) == (68, snapshot_page(full.vm))
    assert max(record["depth"] for record in trace) == 2
    for fuel in range(len(trace) + 1):
        state, prefix = run(machine, nested, 0, page, fuel, trace=True)
        assert len(prefix) == fuel + int(fuel < len(trace))
        assert int(state.vm.status) == (1 if fuel == len(trace) else 2)
        if fuel < len(trace):
            assert prefix[-1]["access"][0] == 0
            assert int(state.vm.pc) == (nested.entry if fuel == 0 else trace[fuel - 1]["next_image_pc"])
            assert int(state.depth) == (0 if fuel == 0 else trace[fuel - 1]["depth"])
        assert snapshot_page(state.vm) == page
    trapped, trap = run(machine, nested, 65533, page, trace=True)
    assert int(trapped.vm.status) == 3 and int(trapped.depth) == 2 and trap[-1]["access"][0] == 3
    assert wasmtime_run(raw_nested, 65533, page, trap_expected=True) == (None, bytes(page))
    print(json.dumps({"machine_sha256": sha(path.read_bytes()),
                      "operators": sorted(torch.jit.export_opnames(machine)),
                      "binary_sha256": image.binary_sha256, "image_sha256": image.image_sha256,
                      "memory_bytes": size, "record_address": cell, "scenes": scenes,
                      "nested_wasm_sha256": sha(raw_nested), "fuel_boundaries": len(trace) + 1,
                      "nested_trace": trace, "nested_trap_trace": trap}, indent=2))


if __name__ == "__main__":
    main()
