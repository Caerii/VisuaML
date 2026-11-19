"""Compare the original and current NumPy multiply column contraction."""
import argparse
import cProfile
import json
import pstats
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from scripts.stateful_scale_workloads_v58 import PUZZLES, graph_case, sudoku_page, sudoku_reference
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab import neural_alu_v56 as alu, neural_microop_executor_v57 as executor
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, decode_control

ARCHIVE = Path(__file__).resolve().parents[1] / "v58-stateful-scale-7125937"
BASELINE = "c0a3d3264c658afd5dae24f2bcaf239171190046"


def original_multiply():
    path = "visuaml-client/backend/visuaml/compiler_lab/bit_neural_vm_v49.py"
    name = "visuaml.compiler_lab._baseline_bit_vm"
    module = types.ModuleType(name)
    module.__package__ = "visuaml.compiler_lab"
    sys.modules[name] = module
    raw = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"])
    exec(compile(raw, path, "exec"), module.__dict__)
    return module._mul


def compare(name, program, initial, argument, baseline, repeats):
    driver = prepare(program)
    expected_page = initial.copy()
    expected = driver.run_memory(argument, expected_page, fuel=50_000_000, capture_trace=False)
    choices = {"original": baseline, "grouped_columns": alu._mul}
    timings = {label: [] for label in choices}

    def run():
        page = initial.copy()
        start = perf_counter()
        result = executor.execute_microop_routed_control(
            program, argument, array_memory=page, capture_trace=False, fuel=expected.steps,
        )
        duration = perf_counter() - start
        assert (page, result.result_i32, result.steps) == (
            expected_page, expected.result_i32, expected.steps,
        )
        return duration

    print(f"Benchmarking {name}: {expected.steps} steps", file=sys.stderr, flush=True)
    for index in range(repeats):
        order = tuple(choices) if index % 2 == 0 else tuple(reversed(choices))
        for label in order:
            with patch.object(alu, "_mul", choices[label]):
                timings[label].append(run())

    counts = Counter()
    execute_alu = executor.execute_alu

    def counted(selector, *args, **kwargs):
        counts[alu.ALU_OPS[int(selector.argmax())]] += 1
        return execute_alu(selector, *args, **kwargs)

    with patch.object(executor, "execute_alu", counted):
        profiler = cProfile.Profile()
        profiler.runcall(run)
    pstats.Stats(profiler, stream=sys.stderr).strip_dirs().sort_stats("cumulative").print_stats(
        "neural_alu|_mul|_signed_lt|_unsigned_lt|_and|_array_route|step_frames|bincount"
    )
    centers = {label: median(samples) for label, samples in timings.items()}
    return {"name": name, "steps": expected.steps, "selected_alu_opcodes": dict(counts),
            "seconds": timings, "median_seconds": centers,
            "paired_median_ratio": centers["original"] / centers["grouped_columns"]}


def escargot_prefix():
    program = decode_control((ARCHIVE / "sudoku.wasm").read_bytes(), scaled=True)
    initial, argument = sudoku_page(tuple(map(int, PUZZLES["escargot"])))
    driver = prepare(program)
    limits = (2000, 20_000)
    seconds = []
    for limit in limits:
        traced = limit == 2000
        outcomes = []
        for run in (
            lambda page: executor.execute_microop_routed_control(
                program, argument, fuel=limit, capture_trace=traced, array_memory=page,
            ),
            lambda page: driver.run_memory(argument, page, fuel=limit, capture_trace=traced),
        ):
            page = initial.copy()
            start = perf_counter()
            try:
                run(page)
            except ControlFuelExhausted as error:
                seconds.append(perf_counter() - start)
                fields = ("pc", "next_pc", "stack_i32", "locals_i32", "memory_access")
                trace = [tuple(row[field] for field in fields) for row in error.trace or ()]
                outcomes.append((page, error.trap_site, trace))
            else:
                raise AssertionError("Expected fuel exhaustion")
        assert outcomes[0] == outcomes[1]
    return {"full_archived_steps": 2_080_993, "matching_trace_steps": limits[0],
            "matching_page_cutoff_steps": limits[1], "array_untraced_seconds": seconds[2],
            "linear_full_run_estimate_seconds": seconds[2] * 2_080_993 / limits[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--case", choices=("sudoku", "bfs", "escargot"), action="append", required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("Use at least one pair")
    cases = {}
    result = {"environment": environment(), "baseline": BASELINE, "cases": cases}
    for name in args.case:
        if name == "escargot":
            cases[name] = escargot_prefix()
            continue
        if name == "sudoku":
            solved = sudoku_reference(tuple(map(int, PUZZLES["classic"])))[0]
            initial, argument = sudoku_page((0, *solved[1:]))
        else:
            argument, initial, *_ = graph_case(0)
        program = decode_control((ARCHIVE / f"{name}.wasm").read_bytes(), scaled=True)
        cases[name] = compare(name, program, initial, argument, original_multiply(), args.pairs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
