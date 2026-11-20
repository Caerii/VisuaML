"""Verify frozen instruction and state reuse, including interleaved invocations."""
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from bench_frozen_state_reuse import CALLS, OUT
from replay_frozen_calls import NESTED
from replay_frozen_multistep import state_equal, terrain_histories, witness
from replay_frozen_recurrent import ARCHIVE, PACKED, sha
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page


def interleave(one, graph, raw, other_raw):
    cases = []
    page = bytes(65536)
    for binary, arg in ((raw, 5), (other_raw, 65533), (raw, 0)):
        image = load_call_image(binary)
        a = initialize(image, arg, page, 100)
        b = initialize(image, arg, page, 100)
        cases.append((image, a, b))
    snapshots = [snapshot_page(b.vm) for _, _, b in cases]
    origins = [state.vm.page.data_ptr() for _, _, state in cases]
    assert len(set(origins)) == len(cases)
    while any(int(state.vm.status) == 0 for _, _, state in cases):
        for i, (image, a, b) in enumerate(cases):
            if int(a.vm.status):
                continue
            x, y = one(image.rows, a), graph(image.rows, b, 1)
            state_equal(x.state, y.state)
            assert torch.equal(x.access, y.access)
            assert y.state.vm.page.data_ptr() == origins[i]
            cases[i] = image, x.state, y.state
    left, _, right = (b for _, _, b in cases)
    assert int(left.vm.stack[0]) == 97 and int(right.vm.stack[0]) == 13
    assert snapshot_page(left.vm)[5:9] == bytes.fromhex("61000000")
    assert snapshot_page(right.vm) == page and all(old == page for old in snapshots)
    return [{"status": int(b.vm.status), "steps": 100 - int(b.vm.fuel),
             "page_sha256": sha(snapshot_page(b.vm))} for _, _, b in cases]


def check_arithmetic(one, selected):
    """Exercise each binary i32 result circuit on live boundary operands."""
    pairs = ((0, 0), (0, 1), (2147483647, 1), (-2147483648, -1),
             (-2147483648, -2147483648), (-1, 1), (65536, 65536))
    coverage = {}
    for op in ("lt_s", "gt_s", "add", "sub", "mul"):
        raw = bytes(wasmtime.wat2wasm(f"""(module
            (memory (export "memory") 1 1)
            (func (export "run") (param i32) (result i32)
               local.get 0 i32.load local.get 0 i32.load offset=4 i32.{op}))"""))
        records = []
        for left, right in pairs:
            page = bytearray(65536)
            struct.pack_into("<ii", page, 0, left, right)
            evidence = witness(one, selected, raw, 0, page, 1)
            records.append({"operands": [left, right], "output_i32": evidence["trace"][-1]["stack"][0]})
        coverage[op] = {"binary_sha256": sha(raw), "cases": records}
    return coverage


def main():
    torch.set_num_threads(1)
    one = torch.jit.load(str(CALLS / "machine.pt"))
    later = (OUT / "after_freeze.wasm").read_bytes()
    assert later == bytes(wasmtime.wat2wasm((OUT / "after_freeze.wat").read_text()))
    nested = bytes(wasmtime.wat2wasm(NESTED))
    empty = bytes(65536)
    undo = bytearray(empty)
    undo[1:5] = undo[7:11] = bytes.fromhex("01000080")
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    output = {}
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for name in ("row", "state", "selected"):
            path = OUT / f"{name}.pt"
            graph = torch.jit.load(str(path))
            ir = str(graph.inlined_graph)
            assert ir.count("aten::index_select(") == 2
            assert all(op not in ir for op in ("prim::PythonOp", "aten::clone", "aten::copy_", "aten::scatter("))
            assert "aten::scatter_(" in ir
            cases = {
                "new-helper-return": witness(one, graph, later, 5, empty, 1),
                "new-branch-bypasses-call": witness(one, graph, later, 0, empty, 1),
                "new-helper-store-trap": witness(one, graph, later, 65533, empty, 3),
                "nested-return": witness(one, graph, nested, 0, empty, 1),
                "nested-trap": witness(one, graph, nested, 65533, empty, 3),
                "rollback": witness(one, graph, rollback, 0, undo, 1),
                "before-write": witness(one, graph, rollback, 0, undo, 2, fuel=8),
                "after-write": witness(one, graph, rollback, 0, undo, 2, fuel=9),
            }
            output[name] = {"sha256": sha(path.read_bytes()), "operators": sorted(torch.jit.export_opnames(graph)),
                            "witnesses": cases, "terrain_histories": terrain_histories(one, graph),
                            "interleaved": interleave(one, graph, later, nested)}
        output["selected"]["arithmetic"] = check_arithmetic(one, graph)
    assert output["row"]["witnesses"] == output["state"]["witnesses"] == output["selected"]["witnesses"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
