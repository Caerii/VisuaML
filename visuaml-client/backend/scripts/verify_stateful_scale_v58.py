"""Run full Sudoku, graph BFS and matrix multiplication as original WASM."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity
from scripts.stateful_scale_workloads_v58 import (
    PUZZLES, bfs_wat, contradictory_sudoku, graph_case, matrix_case, matrix_wat,
    sudoku_page, sudoku_reference, sudoku_variant, sudoku_wat,
)
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    SCALED_MAX_FUEL, SCALED_MAX_OPERATIONS, ControlFuelExhausted, decode_control, encode_control,
)

BUILDERS = {"sudoku": sudoku_wat, "bfs": bfs_wat, "matrix": matrix_wat,
            "matrix_unrolled": lambda: matrix_wat(32)}


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def collect(root, bulk):
    identity = git_identity(root)
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    programs = {}
    prepared = {}
    for name, builder in BUILDERS.items():
        wat = builder()
        raw = bytes(wasmtime.wat2wasm(wat))
        source = decode_control(raw, scaled=True)
        assert encode_control(source) == raw
        started = perf_counter_ns()
        driver = prepare(source)
        prepare_ns = perf_counter_ns() - started
        store = wasmtime.Store(engine)
        kernel = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
        prepared[name] = driver, store, kernel
        programs[name] = {"instructions": len(source.operations), "sha256": _digest(raw),
                          "wasm_hex": raw.hex(), "preparation_ns": prepare_ns}
    rows = []

    def run(family, name, argument, initial, verify, inputs):
        driver, store, kernel = prepared[family]
        page = bytearray(initial)
        started = perf_counter_ns()
        result = driver.run_memory(argument, page, fuel=SCALED_MAX_FUEL, capture_trace=False)
        packed_ns = perf_counter_ns() - started
        started = perf_counter_ns()
        store.set_fuel(SCALED_MAX_FUEL)
        kernel["memory"].write(store, initial, 0)
        expected = kernel["run"](store, argument)
        oracle = kernel["memory"].read(store)
        wasm_ns = perf_counter_ns() - started
        assert result.result_i32 == expected and page == oracle, (family, name)
        extra = verify(result.result_i32, page)
        rows.append({"family": family, "name": name, "argument": argument, "result": result.result_i32,
                     "steps": result.steps, "output_sha256": _digest(page), "input_sha256": _digest(initial),
                     "input": inputs, "packed_wall_ns": packed_ns, "wasmtime_wall_ns": wasm_ns, **extra})

    boards = {name: tuple(map(int, puzzle)) for name, puzzle in PUZZLES.items()}
    boards["classic_unsatisfiable"] = contradictory_sudoku(boards["classic"])

    def solve(name, board, metadata):
        initial, count = sudoku_page(board)
        solutions = sudoku_reference(board)
        assert len(solutions) <= 1

        def verify(result, page):
            actual = struct.unpack_from("<81i", page)
            assert result == bool(solutions) and actual == (solutions[0] if solutions else board), name
            return {"grid": "".join(map(str, actual)), "solution_count": len(solutions)}

        run("sudoku", name, count, initial, verify, {"givens": "".join(map(str, board)), **metadata})

    for name, board in boards.items():
        solve(name, board, {})
    for index in range(bulk // 8):
        base = ("classic", "euler_grid")[index % 2]
        solve(f"variant_{index}", sudoku_variant(boards[base], index), {"base": base, "seed": index})

    def numeric(family, name, case, inputs):
        argument, initial, expected_result, expected_page = case

        def verify(result, page):
            assert result == expected_result and page == expected_page, name
            return {}

        run(family, name, argument, initial, verify, inputs)

    for seed in range(6 + bulk * 3 // 4):
        numeric("bfs", f"graph_{seed}", graph_case(seed), {"seed": seed})
    for seed in range(8 + bulk // 8):
        size = (4, 8, 16, 32)[seed % 4]
        numeric("matrix", f"matrix_{seed}", matrix_case(size, seed), {"size": size, "seed": seed})
    numeric("matrix_unrolled", "matrix_unrolled_32", matrix_case(32, 3), {"size": 32, "seed": 3})

    initial, argument = sudoku_page(boards["escargot"])
    try:
        prepared["sudoku"][0].run_memory(argument, initial, fuel=2000)
    except ControlFuelExhausted as error:
        assert len(error.trace) == 2000 and error.trap_site["step"] == 2000
        trap = error.trap_site
    else:
        raise AssertionError("Expected an early budget trap on hard Sudoku")
    steps = Counter()
    counts = Counter()
    for row in rows:
        steps[row["family"]] += row["steps"]
        counts[row["family"]] += 1
    return {
        "source": identity, "environment": environment(), "programs": programs,
        "limits": {"instructions": SCALED_MAX_OPERATIONS, "untraced_fuel": SCALED_MAX_FUEL},
        "requested_bulk": bulk, "cases": rows, "steps_by_family": dict(steps),
        "cases_by_family": dict(counts), "steps_total": sum(steps.values()), "sudoku_fuel_trap": trap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk", type=int, default=0, help="additional seeded workloads, multiple of eight")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if not 0 <= args.bulk <= 8192 or args.bulk % 8:
        parser.error("--bulk must be a multiple of eight between 0 and 8192")
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    result = collect(root, args.bulk)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
