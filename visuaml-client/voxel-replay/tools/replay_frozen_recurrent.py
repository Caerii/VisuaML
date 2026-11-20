"""Replay unrelated program images through the same frozen TorchScript artifact."""
import hashlib
import json
import struct
from pathlib import Path
from unittest.mock import patch

import torch
import wasmtime

import terrain_verify as terrain
from scripts.stateful_scale_workloads_v58 import PUZZLES, graph_case, sudoku_page, sudoku_reference
from visuaml.compiler_lab import neural_microop_executor_v57 as existing
from visuaml.compiler_lab.frozen_recurrent_v58 import initial_state, load_program
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap, decode_control

HERE = Path(__file__).resolve().parents[1]
EVIDENCE = HERE / "evidence/frozen-recurrent"
ARCHIVE = HERE.parents[1] / "benchmarks/results"
PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signed(word):
    return (int(word) + 2**31) % 2**32 - 2**31


def wasmtime_run(raw, argument, initial, trap_expected=False):
    store = wasmtime.Store()
    exports = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    exports["memory"].write(store, initial)
    try:
        result = exports["run"](store, argument)
    except wasmtime.Trap as trap:
        assert trap_expected and trap.trap_code == wasmtime.TrapCode.MEMORY_OUT_OF_BOUNDS
        result = None
    else:
        assert not trap_expected
    return result, bytes(exports["memory"].read(store))


def execute(machine, raw, argument, initial, fuel, traced=True):
    image = load_program(raw)
    state = initial_state(argument, initial, fuel)
    log = []
    while int(state.status) == 0:
        previous = state
        tick = machine(image.rows, state)
        state = tick.state
        if traced and int(previous.fuel) != int(state.fuel):
            kind, address, before, after = map(int, tick.access)
            access = None
            if kind:
                access = {"kind": "i32.load" if kind == 1 else "i32.store", "address": address,
                          "before_hex": before.to_bytes(4, "little").hex(),
                          "after_hex": after.to_bytes(4, "little").hex()}
            log.append({
                "pc": int(previous.pc), "next_pc": int(state.pc),
                "stack_i32": [signed(x) for x in state.stack[:int(state.height)]],
                "locals_i32": [signed(x) for x in state.locals[:image.local_count + 1]],
                "memory_access": access, "page_sha256": sha(state.page.numpy().tobytes()),
                "fuel_after": int(state.fuel), "status_after": int(state.status),
            })
    return image, state, log


def witness(machine, name, raw, argument, initial, fuel, expected_status):
    image, state, log = execute(machine, raw, argument, initial, fuel)
    assert int(state.status) == expected_status
    stopped = machine(image.rows, state)
    assert all(torch.equal(a, b) for a, b in zip(state, stopped.state, strict=True))
    assert int(stopped.access[0]) == 0
    reference = initial.copy()
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        try:
            result = existing.execute_microop_routed_control(
                decode_control(raw, scaled=True), argument, array_memory=reference, fuel=fuel,
            )
        except (ControlFuelExhausted, LinearMemoryTrap) as error:
            assert expected_status == (2 if isinstance(error, ControlFuelExhausted) else 3)
            records = error.trace
            assert error.trap_site["pc"] == int(state.pc)
        else:
            assert expected_status == 1 and result.result_i32 == signed(state.stack[0])
            records = result.trace
    assert len(log) == len(records) == fuel - int(state.fuel)
    replay_page = initial.copy()
    for actual, checked in zip(log, records, strict=True):
        for key in ("pc", "next_pc", "stack_i32", "locals_i32", "memory_access"):
            assert actual[key] == checked[key], (name, actual, checked)
        access = checked["memory_access"]
        if access and access["kind"] == "i32.store":
            start = access["address"]
            replay_page[start:start + 4] = bytes.fromhex(access["after_hex"])
        assert actual["page_sha256"] == sha(replay_page)
    page = state.page.numpy().tobytes()
    assert page == reference == replay_page
    if expected_status != 2:
        value, oracle = wasmtime_run(raw, argument, initial, trap_expected=expected_status == 3)
        assert oracle == page
        if expected_status == 1:
            assert value == signed(state.stack[0])
    return {"name": name, "binary_sha256": image.binary_sha256, "image_sha256": image.image_sha256,
            "argument": argument, "fuel": fuel, "status": expected_status, "steps": len(log),
            "input_page_sha256": sha(initial), "output_page_sha256": sha(page), "trace": log}


def main(image_file=EVIDENCE / "machine.pt"):
    torch.set_num_threads(1)
    machine = torch.jit.load(image_file, map_location="cpu")
    assert not tuple(machine.parameters()) and not tuple(machine.buffers())
    graph = str(machine.inlined_graph)
    assert "prim::PythonOp" not in graph and "aten::item" not in graph
    rollback = (ARCHIVE / "v58-array-memory/rollback.wasm").read_bytes()
    newer = (EVIDENCE / "after_freeze.wasm").read_bytes()
    assert newer == bytes(wasmtime.wat2wasm((EVIDENCE / "after_freeze.wat").read_text()))
    voxel = (ARCHIVE / "v58-voxel-tick/voxel_tick.wasm").read_bytes()
    live = bytearray(65536)
    live[1:5] = live[7:11] = bytes.fromhex("01000080")
    trap = bytearray(65536)
    trap[65527:65531] = bytes.fromhex("01000080")
    trap[65533:] = bytes.fromhex("aabbcc")
    empty = bytearray(65536)

    def world(position):
        page = empty.copy()
        struct.pack_into("<8i", page, 256, position, 4, -1, 0, 0, 0, 7, 0)
        return page

    cases = [
        ("rollback", rollback, 0, live, 100, 1),
        ("between-write-and-reversal", rollback, 0, live, 9, 2),
        ("branch-untaken", rollback, 0, empty, 100, 1),
        ("straddling-store-trap", rollback, 65526, trap, 100, 3),
        ("new-branch-true", newer, 1, empty, 100, 1),
        ("new-branch-false", newer, 0, empty, 100, 1),
        ("voxel-open", voxel, 256, world(0), 100, 1),
        ("voxel-blocked", voxel, 256, world(4), 100, 1),
    ]
    witnesses = [witness(machine, *case) for case in cases]
    classic = tuple(map(int, PUZZLES["classic"]))
    solution = sudoku_reference(classic)[0]
    sudoku_initial, sudoku_argument = sudoku_page((0, *solution[1:]))
    bfs_argument, bfs_initial, *_ = graph_case(0)
    transfers = []
    for name, initial, argument, steps in (
        ("bfs", bfs_initial, bfs_argument, 41_591),
        ("sudoku", sudoku_initial, sudoku_argument, 893),
    ):
        raw = (ARCHIVE / f"v58-stateful-scale-7125937/{name}.wasm").read_bytes()
        image, state, _ = execute(machine, raw, argument, initial, steps, traced=False)
        expected, oracle = wasmtime_run(raw, argument, initial)
        assert int(state.status) == 1 and int(state.fuel) == 0
        assert signed(state.stack[0]) == expected and state.page.numpy().tobytes() == oracle
        transfers.append({"name": name, "binary_sha256": image.binary_sha256,
                          "image_sha256": image.image_sha256, "source_steps": steps,
                          "page_sha256": sha(oracle), "result": expected})
    terrain_raw = terrain.GUEST.read_bytes()
    for name in ("terrain", "terrain-omitted"):
        fixture = HERE / "fixtures" / f"{name}.proof.json"
        proof = json.loads(fixture.read_text())
        session = proof["session"]
        assert sha(terrain_raw) == session["binary_sha256"]
        page = terrain.initial_page(session["initial"])
        ticks = []
        for inputs, expected in zip(session["inputs"], proof["ticks"], strict=True):
            terrain.write(page, 4, [int(inputs["edit"])])
            terrain.write(page, 320, [4 * move for move in inputs["moves"]])
            image, state, _ = execute(machine, terrain_raw, terrain.BASE, page, expected["steps"], traced=False)
            value, oracle = wasmtime_run(terrain_raw, terrain.BASE, page)
            assert int(state.status) == 1 and int(state.fuel) == 0
            assert signed(state.stack[0]) == value and state.page.numpy().tobytes() == oracle
            assert sha(oracle) == expected["page_sha256"]
            page = bytearray(oracle)
            ticks.append({"source_steps": expected["steps"], "page_sha256": sha(oracle),
                          "result": value, "positions": expected["positions"], "terrain": expected["terrain"]})
        transfers.append({"name": name, "binary_sha256": image.binary_sha256,
                          "image_sha256": image.image_sha256, "ticks": ticks})
    print(json.dumps({"frozen_sha256": sha(image_file.read_bytes()), "torch": torch.__version__,
                      "operators": sorted(torch.jit.export_opnames(machine)), "traces": witnesses,
                      "additional_transfer": transfers}, indent=2))


if __name__ == "__main__":
    import sys

    main(*map(Path, sys.argv[1:]))
