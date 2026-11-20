"""Check live ownership, retained snapshots and the exact store boundaries."""
import json

import torch
import wasmtime

from replay_frozen_recurrent import ARCHIVE, EVIDENCE, sha, wasmtime_run
from visuaml.compiler_lab.frozen_recurrent_v58 import initial_state, load_program, snapshot_page
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlAdmissionError, parse_section_envelope

OUT = EVIDENCE.parent / "bounded-recurrent-store"

OVERLAP = """(module
 (memory (export "memory") 1 1)
 (func (export "run") (param i32) (result i32)
   i32.const 1 i32.const 0x11223344 i32.store align=1
   i32.const 3 i32.const 0x55667788 i32.store align=1
   local.get 0 i32.const 0x12345678 i32.store align=1
   i32.const 1 i32.load align=1))"""


def compare(reference, bounded, raw, arg, initial, fuel, status, expected):
    image = load_program(raw)
    old = initial_state(arg, initial, fuel)
    new = initial_state(arg, initial, fuel)
    independent = initial_state(arg, initial, fuel)
    assert len({state.page.data_ptr() for state in (old, new, independent)}) == 3
    live_pointer = new.page.data_ptr()
    snapshots = [(snapshot_page(new), old.page)]
    log = []
    while not int(new.status):
        a = reference(image.rows, old)
        b = bounded(image.rows, new)
        old, new = a.state, b.state
        assert all(torch.equal(x, y) for x, y in zip(old, new, strict=True))
        assert torch.equal(a.access, b.access)
        assert new.page.data_ptr() == live_pointer
        snapshots.append((snapshot_page(new), old.page))
        log.append([int(new.pc), int(new.fuel), int(new.status), *map(int, b.access)])
    assert snapshot_page(independent) == bytes(initial) == snapshots[0][0]
    for retained, past_reference in snapshots:
        assert retained == past_reference.numpy().tobytes()
    final = snapshot_page(new)
    assert int(new.status) == status
    assert final == expected
    if status != 2:
        result, oracle = wasmtime_run(raw, arg, initial, trap_expected=status == 3)
        assert oracle == final
        if status == 1:
            assert result == int(new.stack[0])
    stopped = bounded(image.rows, new)
    assert all(torch.equal(x, y) for x, y in zip(new, stopped.state, strict=True))
    return {"image_sha256": image.image_sha256, "binary_sha256": image.binary_sha256,
            "fuel": fuel, "status": status, "page_sha256": sha(final), "trace": log}


def check_helper_call():
    raw = (OUT / "passage-with-agent.wasm").read_bytes()
    sections = {section.section_id: section.payload for section in parse_section_envelope(raw)}
    assert sections[3] == bytes.fromhex("020000")
    assert bytes.fromhex("20001080808080000d00") in sections[10]
    try:
        load_program(raw, memory_pages=256)
    except ControlAdmissionError as error:
        admission = str(error)
    else:
        raise AssertionError("Unexpected function admission")
    page = bytearray(16777216)
    cell = len(page) - 12
    page[cell], page[cell + 4] = 7, 1
    first, edited = wasmtime_run(raw, cell, page)
    next_page = bytearray(edited)
    next_page[cell + 4] = 0
    second, moved = wasmtime_run(raw, cell, next_page)
    records = [edited[cell:].hex(), moved[cell:].hex()]
    assert [first, second] == [0, 1]
    assert records == ["000000000100000000000000", "000000000000000001000000"]
    return {"binary_sha256": sha(raw), "function_section_hex": sections[3].hex(),
            "code_section_hex": sections[10].hex(), "direct_call": {"caller": 1, "target": 0},
            "admission_error": admission, "wasmtime_returns": [first, second],
            "records_after_ticks": records}


def main():
    torch.set_num_threads(1)
    reference = torch.jit.load(str(EVIDENCE / "machine.pt"))
    bounded = torch.jit.load(str(OUT / "machine.pt"))
    ops = sorted(torch.jit.export_opnames(bounded))
    ir = str(bounded.inlined_graph)
    assert ir.count("aten::scatter_(") == 1
    assert "aten::scatter(" not in ir and "aten::clone" not in ir
    raw = bytes(wasmtime.wat2wasm(OVERLAP))
    assert load_program(raw, memory_pages=1).memory_pages == 1
    try:
        load_program(raw, memory_pages=16)
    except ControlAdmissionError:
        pass
    else:
        raise AssertionError("Guest memory declaration disagrees with invocation capacity")
    initial = bytearray(65536)
    partial = initial.copy()
    partial[1:7] = bytes.fromhex("443388776655")
    success = partial.copy()
    success[-4:] = bytes.fromhex("78563412")
    cases = {
        "last-four-bytes": compare(reference, bounded, raw, 65532, initial, 100, 1, success),
        "past-end-is-atomic": compare(reference, bounded, raw, 65533, initial, 100, 3, partial),
        "unsigned-negative-address": compare(reference, bounded, raw, -1, initial, 100, 3, partial),
    }
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    initial[1:5] = initial[7:11] = bytes.fromhex("01000080")
    edited = initial.copy()
    edited[7:11] = bytes.fromhex("78563412")
    cases["fuel-before-store"] = compare(reference, bounded, rollback, 0, initial, 8, 2, initial)
    cases["fuel-after-store"] = compare(reference, bounded, rollback, 0, initial, 9, 2, edited)
    executed_ir = str(torch.jit.last_executed_optimized_graph())
    assert "aten::scatter_(" in executed_ir
    assert all(op not in executed_ir for op in ("aten::scatter(", "aten::clone", "aten::copy_"))
    print(json.dumps({"machine_sha256": sha((OUT / "machine.pt").read_bytes()),
                      "operators": ops, "cases": cases, "next_capability": check_helper_call()}, indent=2))


if __name__ == "__main__":
    main()
