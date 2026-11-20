"""Replay complete histories and compare every small witness prefix."""
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from bench_frozen_multistep import CALLS, OUT, drive
from replay_frozen_calls import NESTED, run
from replay_frozen_recurrent import ARCHIVE, PACKED, sha, signed, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image
from visuaml.compiler_lab.frozen_recurrent_v58 import snapshot_page


def state_equal(a, b):
    assert torch.equal(a.depth, b.depth) and torch.equal(a.frames, b.frames)
    assert all(torch.equal(x, y) for x, y in zip(a.vm, b.vm, strict=True))


def witness(one, multi, raw, argument, page, expected_status, fuel=100, pages=1, canonical_no_access=False):
    image = load_call_image(raw, memory_pages=pages)
    final, trace = run(one, image, argument, page, fuel, trace=True)
    assert int(final.vm.status) == expected_status
    if expected_status != 2:
        result, expected_page = wasmtime_run(raw, argument, page, trap_expected=expected_status == 3)
        assert expected_page == snapshot_page(final.vm)
        if expected_status == 1:
            assert result == signed(final.vm.stack[0])
    stopped = multi(image.rows, final, 64)
    state_equal(final, stopped.state)
    assert not torch.any(stopped.access)
    reference = initialize(image, argument, page, fuel)
    for count in range(len(trace) + 1):
        candidate = multi(image.rows, initialize(image, argument, page, fuel), count)
        state_equal(reference, candidate.state)
        if count:
            expected_access = trace[count - 1]["access"]
            actual_access = candidate.access.tolist()
            if canonical_no_access and expected_access[0] == 0:
                assert actual_access[0] == 0
            else:
                assert actual_access == expected_access
        if count < len(trace):
            reference = one(image.rows, reference).state
    for budget in (4, 16, 64):
        reference = initialize(image, argument, page, fuel)
        candidate = initialize(image, argument, page, fuel)
        for start in range(0, len(trace), budget):
            candidate = multi(image.rows, candidate, budget).state
            for _ in trace[start:start + budget]:
                reference = one(image.rows, reference).state
            state_equal(reference, candidate)
        state_equal(final, candidate)
    return {"binary_sha256": image.binary_sha256, "image_sha256": image.image_sha256,
            "argument": argument, "fuel": fuel, "status": expected_status,
            "input_page_sha256": sha(page), "full_page_sha256": sha(snapshot_page(final.vm)),
            "trace": trace, "source_steps": fuel - int(final.vm.fuel)}


def terrain_histories(one, multi):
    raw = terrain.GUEST.read_bytes()
    image = load_call_image(raw)
    results = {}
    for name in ("terrain", "terrain-omitted"):
        proof = json.loads((OUT.parents[1] / "fixtures" / f"{name}.proof.json").read_text())
        session = proof["session"]
        page = terrain.initial_page(session["initial"])
        ticks = []
        for inputs, expected in zip(session["inputs"], proof["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * d for d in inputs["moves"]])
            result, oracle = wasmtime_run(raw, terrain.BASE, page)
            reference, _ = drive(one, multi, image, initialize(image, terrain.BASE, page, 5000), 1)
            for budget in (4, 16, 64):
                candidate, calls = drive(one, multi, image, initialize(image, terrain.BASE, page, 5000), budget)
                state_equal(reference, candidate)
                assert int(candidate.vm.status) == 1 and int(candidate.vm.stack[0]) == result
                assert snapshot_page(candidate.vm) == oracle
            assert sha(oracle) == expected["page_sha256"]
            assert 5000 - int(reference.vm.fuel) == expected["steps"]
            ticks.append({"page_sha256": sha(oracle), "source_steps": expected["steps"],
                          "result": result, "budget_64_host_calls": calls})
            page = bytearray(oracle)
        results[name] = ticks
    return results


def main():
    torch.set_num_threads(1)
    one = torch.jit.load(str(CALLS / "machine.pt"))
    path = OUT / "machine.pt"
    multi = torch.jit.load(str(path))
    ir = str(multi.inlined_graph)
    assert "prim::Loop" in ir and "aten::scatter_" in ir and "prim::PythonOp" not in ir
    page = bytearray(65536)
    nested = bytes(wasmtime.wat2wasm(NESTED))
    later = (OUT / "after_freeze.wasm").read_bytes()
    assert later == bytes(wasmtime.wat2wasm((OUT / "after_freeze.wat").read_text()))
    new_image = load_call_image(later)
    initial = initialize(new_image, 4, page, 100)
    for invalid in (-1, 65):
        try:
            multi(new_image.rows, initial, invalid)
        except torch.jit.Error as error:
            assert "Expected a transition budget" in str(error)
        else:
            raise AssertionError("Unexpected admission of a transition budget")
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    undo = page.copy()
    undo[1:5] = undo[7:11] = bytes.fromhex("01000080")
    passage = (CALLS / "passage.wasm").read_bytes()
    world = bytearray(16 << 20)
    struct.pack_into("<iii", world, len(world) - 12, 7, 1, 0)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        cases = {
            "nested-return": witness(one, multi, nested, 0, page, 1),
            "nested-trap": witness(one, multi, nested, 65533, page, 3),
            "new-call-return": witness(one, multi, later, 4, page, 1),
            "new-call-store-then-trap": witness(one, multi, later, 65533, page, 3),
            "rollback": witness(one, multi, rollback, 0, undo, 1),
            "before-write": witness(one, multi, rollback, 0, undo, 2, fuel=8),
            "after-write-before-reversal": witness(one, multi, rollback, 0, undo, 2, fuel=9),
            "rust-edit-and-return": witness(one, multi, passage, len(world) - 12, world, 1, pages=256),
        }
        trap = cases["new-call-store-then-trap"]
        assert trap["trace"][-1]["access"][0] == 3
        assert [2, 4, 0, 77] in [step["access"] for step in trap["trace"]]
        retained = page.copy()
        retained[4] = 77
        assert trap["full_page_sha256"] == sha(retained)
        rust = cases["rust-edit-and-return"]
        assert rust["source_steps"] == 21
        assert [entry["depth"] for entry in rust["trace"]][2:17] == [1] * 14 + [0]
        assert rust["trace"][13]["access"] == [2, len(world) - 12, 7, 0]
        assert cases["before-write"]["full_page_sha256"] == sha(undo)
        undo[7:11] = bytes.fromhex("78563412")
        assert cases["after-write-before-reversal"]["full_page_sha256"] == sha(undo)
        histories = terrain_histories(one, multi)
    print(json.dumps({"frozen_sha256": sha(path.read_bytes()),
                      "operators": sorted(torch.jit.export_opnames(multi)),
                      "witnesses": cases, "terrain_histories": histories}, indent=2))


if __name__ == "__main__":
    main()
