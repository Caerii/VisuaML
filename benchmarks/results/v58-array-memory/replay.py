"""Replay the bounded NumPy memory witness and the unchanged scale modules."""
import argparse
import cProfile
import gzip
import hashlib
import json
import pstats
import platform
import struct
import subprocess
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import wasmtime

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.stateful_scale_workloads_v58 import (
    PUZZLES, contradictory_sudoku, graph_case, sudoku_page, sudoku_reference,
)
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab import neural_microop_executor_v57 as executor
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted, LinearMemoryTrap, decode_control, encode_control,
)

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "v58-stateful-scale-7125937"
MAX_FUEL = 50_000_000
TRACE_FIELDS = ("step", "pc", "opcode", "next_pc", "stack_i32", "locals_i32",
                "control_labels", "halted", "neural_route", "memory_access")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(raw):
    program = decode_control(raw, scaled=True)
    assert encode_control(program) == raw
    return program


def run_array(program, argument, page, **options):
    fail = AssertionError("Native fallback")
    with (
        patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
              side_effect=fail) as native,
        patch("visuaml.compiler_lab.constructed_alu_v58.compiled_alu", side_effect=fail),
    ):
        result = executor.execute_microop_routed_control(
            program, argument, array_memory=page, **options,
        )
        native.assert_not_called()
    assert result.backend == "v58-structured-numpy-memory"
    return result


def reference(raw, argument, initial, *, trap=False):
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
    exports["memory"].write(store, initial)
    try:
        result = exports["run"](store, argument)
    except wasmtime.Trap as error:
        assert trap and error.trap_code == wasmtime.TrapCode.MEMORY_OUT_OF_BOUNDS
        result = None
    else:
        assert not trap
    return result, exports["memory"].read(store)


def trace_rows(trace):
    return [{key: row[key] for key in TRACE_FIELDS} for row in trace]


def compare_traced(program, packed, argument, initial, fuel, expected_error=None):
    records = []
    for run in (
        lambda page: run_array(program, argument, page, fuel=fuel),
        lambda page: packed.run_memory(argument, page, fuel=fuel),
    ):
        page = initial.copy()
        try:
            result = run(page)
        except (ControlFuelExhausted, LinearMemoryTrap) as error:
            assert type(error) is expected_error
            records.append((None, page, trace_rows(error.trace), error.trap_site))
        else:
            assert expected_error is None
            records.append((result.result_i32, page, trace_rows(result.trace), None))
    assert records[0] == records[1]
    return records[0]


def witness():
    raw = (HERE / "rollback.wasm").read_bytes()
    assert raw == bytes(wasmtime.wat2wasm((HERE / "rollback.wat").read_text()))
    program = source(raw)
    packed = prepare(program)
    evidence = []
    for name, argument, fuel, error in (
        ("zero", 0, 100, None),
        ("rollback", 0, 100, None),
        ("cutoff_after_store", 0, 9, ControlFuelExhausted),
        ("trapping_store", 65526, 100, LinearMemoryTrap),
        ("widened_load", -1, 100, LinearMemoryTrap),
    ):
        initial = bytearray(65_536)
        if name in ("rollback", "cutoff_after_store"):
            initial[1:5] = initial[7:11] = bytes.fromhex("01000080")
        if name == "trapping_store":
            initial[65527:65531] = bytes.fromhex("01000080")
            initial[65533:] = bytes.fromhex("aabbcc")
        result, page, trace, trap = compare_traced(program, packed, argument, initial, fuel, error)
        writes = [row["memory_access"] for row in trace if row["opcode"] == "i32.store"]
        if name == "cutoff_after_store":
            assert page[7:11].hex() == "78563412" and len(writes) == 1
        else:
            assert (result, page) == reference(raw, argument, initial, trap=error is LinearMemoryTrap)
            assert page == initial
        if name == "rollback":
            assert result == -2147483647
            assert [write["after_hex"] for write in writes] == ["78563412", "01000080"]
        if name == "zero":
            assert result == 0 and not writes
        evidence.append({"name": name, "argument": argument, "fuel": fuel, "result": result,
                         "input_sha256": digest(initial), "output_sha256": digest(page),
                         "trap": trap, "trace": trace})
    return {"wasm_sha256": digest(raw), "cases": evidence}


def sudoku(name, givens, output, rust_checker):
    raw = (ARCHIVE / "sudoku.wasm").read_bytes()
    assert digest(raw) == json.loads((ARCHIVE / "manifest.json").read_text())["artifacts"]["sudoku.wasm"]["sha256"]
    program = source(raw)
    packed = prepare(program)
    initial, argument = sudoku_page(givens)
    packed_page = initial.copy()
    expected = packed.run_memory(argument, packed_page, fuel=MAX_FUEL, capture_trace=False)
    prefix_steps = min(expected.steps, 2000)
    compare_traced(program, packed, argument, initial, prefix_steps,
                   ControlFuelExhausted if expected.steps > prefix_steps else None)
    placement = {}
    rollback = []

    def observe(step, pc, access):
        cell = access["address"]
        if cell >= 324 or rollback:
            return
        event = {"step": step, "pc": pc, **access}
        if access["before_hex"] == "00000000":
            placement[cell] = event
        elif access["after_hex"] == "00000000" and cell in placement:
            rollback.extend((placement[cell], event))

    page = initial.copy()
    started = perf_counter()
    result = run_array(program, argument, page, fuel=expected.steps,
                       capture_trace=False, observe_store=observe)
    elapsed = perf_counter() - started
    assert (result.result_i32, result.steps, page) == (expected.result_i32, expected.steps, packed_page)
    assert (result.result_i32, page) == reference(raw, argument, initial)
    solutions = sudoku_reference(givens)
    assert len(solutions) <= 1
    assert result.result_i32 == bool(solutions)
    assert struct.unpack_from("<81i", page) == (solutions[0] if solutions else givens)
    path = output / f"{name}.page"
    path.write_bytes(page)
    if rust_checker:
        subprocess.run([rust_checker, "".join(map(str, givens)), path, str(result.result_i32)], check=True)
    if name != "nearly_solved":
        with gzip.open(ARCHIVE / "runtime-1.json.gz", "rt") as stream:
            archived = next(row for row in json.load(stream)["cases"] if row["name"] == name)
        assert (result.result_i32, result.steps, digest(page)) == (
            archived["result"], archived["steps"], archived["output_sha256"],
        )
    if name == "classic_unsatisfiable":
        assert rollback and rollback[0]["after_hex"] == rollback[1]["before_hex"]
        for event in rollback:
            address = event["address"]
            for delta, field in ((0, "before_hex"), (1, "after_hex")):
                prefix_page = initial.copy()
                try:
                    packed.run_memory(argument, prefix_page, fuel=event["step"] + delta,
                                      capture_trace=False)
                except ControlFuelExhausted:
                    pass
                else:
                    raise AssertionError("Expected cutoff around Sudoku board write")
                assert prefix_page[address:address + 4].hex() == event[field]
    return {"name": name, "givens": "".join(map(str, givens)), "argument": argument,
            "result": result.result_i32, "steps": result.steps, "trace_rows_matched": prefix_steps,
            "backend": result.backend, "wasm_sha256": result.source_wasm_sha256,
            "input_sha256": digest(initial), "output_sha256": digest(page),
            "first_rollback": rollback, "execution_with_preparation_seconds": elapsed,
            "rust_checker_passed": bool(rust_checker)}


def bfs():
    raw = (ARCHIVE / "bfs.wasm").read_bytes()
    program = source(raw)
    argument, initial, expected_result, expected_page = graph_case(0)
    packed = prepare(program)
    compare_traced(program, packed, argument, initial, 2000, ControlFuelExhausted)
    packed_page = initial.copy()
    expected = packed.run_memory(argument, packed_page, fuel=MAX_FUEL, capture_trace=False)
    page = initial.copy()
    result = run_array(program, argument, page, fuel=expected.steps, capture_trace=False)
    assert (result.result_i32, result.steps, page) == (expected.result_i32, expected.steps, packed_page)
    assert (result.result_i32, page) == (expected_result, expected_page) == reference(raw, argument, initial)
    return {"seed": 0, "wasm_sha256": digest(raw), "steps": result.steps,
            "result": result.result_i32, "output_sha256": digest(page), "trace_rows_matched": 2000}


def profile(raw, argument, initial, fuel):
    program = source(raw)
    first_tick = None
    route = executor._array_route

    def first_route(*args):
        nonlocal first_tick
        first_tick = perf_counter()
        executor._array_route = route
        return route(*args)

    started = perf_counter()
    with patch.object(executor, "_array_route", first_route):
        run_array(program, argument, initial.copy(), fuel=fuel, capture_trace=False)
    elapsed = perf_counter() - started
    profiler = cProfile.Profile()
    profiler.runcall(run_array, program, argument, initial.copy(), fuel=fuel, capture_trace=False)
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
    return {"through_first_route_seconds": first_tick - started,
            "remaining_execution_seconds": elapsed - (first_tick - started)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("nearly_solved", "classic_unsatisfiable", "classic", "bfs"),
                        action="append", default=[])
    parser.add_argument("--rust-checker", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    small = witness()
    evidence = {
        "source": git_identity(HERE.parents[2]),
        "python": platform.python_version(),
        "witness": small,
        "sudoku": [],
    }
    classic = tuple(map(int, PUZZLES["classic"]))
    solution = sudoku_reference(classic)[0]
    near = (0, *solution[1:])
    boards = {"nearly_solved": near, "classic": classic,
              "classic_unsatisfiable": contradictory_sudoku(classic)}
    for name in args.case:
        print(f"Replaying {name}", flush=True)
        if name == "bfs":
            evidence["bfs"] = bfs()
        else:
            evidence["sudoku"].append(sudoku(name, boards[name], args.output, args.rust_checker))
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if args.profile:
        page = bytearray(65_536)
        page[1:5] = page[7:11] = bytes.fromhex("01000080")
        initial, argument = sudoku_page(near)
        evidence["profile"] = {
            "witness": profile((HERE / "rollback.wasm").read_bytes(), 0, page, 100),
            "nearly_solved": profile((ARCHIVE / "sudoku.wasm").read_bytes(), argument, initial, 2000),
        }
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
