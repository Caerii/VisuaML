"""Paired replay and execution-only timings for the operand contraction."""
import argparse
import json
import linecache
import math
import subprocess
import sys
import types
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment
from scripts.stateful_scale_workloads_v58 import PUZZLES, graph_case, sudoku_page, sudoku_reference
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab import neural_operand_adapter_v55 as adapter
from visuaml.compiler_lab import neural_operand_storage_v55 as storage
from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control

import terrain_verify as terrain
from scale_terrain import ARCHIVE

BASELINE = "24dd138e13b0ab1ccc24d8eeb1bb0db7bdb041ea"
STORAGE = "visuaml-client/backend/visuaml/compiler_lab/neural_operand_storage_v55.py"


def reference_transition():
    name = "visuaml.compiler_lab._dense_operand_reference"
    module = types.ModuleType(name)
    module.__package__ = "visuaml.compiler_lab"
    sys.modules[name] = module
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{STORAGE}"], text=True)
    filename = f"<{BASELINE}:{STORAGE}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    for name in ("OperandCircuitError", "OperandState", "OperandWeights"):
        setattr(module, name, getattr(storage, name))
    return module.step_operands


def measured(call, samples):
    first = None
    route = vm._array_route

    def first_route(*args):
        nonlocal first
        first = perf_counter_ns()
        vm._array_route = route
        return route(*args)

    with patch.object(vm, "_array_route", first_route):
        start = perf_counter_ns()
        result = call()
        finish = perf_counter_ns()
    samples.append({"pre_execution_ns": first - start, "execution_ns": finish - first,
                    "steps": result.steps})
    return result


def workload(name):
    if name in ("sparse", "crowded"):
        return terrain.scale_session(8, name == "crowded", 4)
    if name == "bfs":
        argument, initial, expected_return, expected_page = graph_case(0)
        expected_steps = 41_591
    else:
        classic = tuple(map(int, PUZZLES["classic"]))
        solution = sudoku_reference(classic)[0]
        initial, argument = sudoku_page((0, *solution[1:]))
        expected_return = 1
        expected_page = None
        expected_steps = 893
    raw = (ARCHIVE / f"{name}.wasm").read_bytes()
    start = perf_counter_ns()
    program = decode_control(raw, scaled=True)
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    exports = wasmtime.Instance(store, wasmtime.Module(engine, raw), []).exports(store)
    exports["memory"].write(store, initial)
    setup_ns = perf_counter_ns() - start
    assert exports["run"](store, argument) == expected_return
    reference = exports["memory"].read(store)
    if expected_page is not None:
        assert reference == expected_page
    return program, initial, argument, expected_return, reference, expected_steps, setup_ns


def run(case, samples, expected, *, prepare_operand_pop=False, prepare_alu=False, prepare_mul=False, prepare_regions=False, prepare_local_region=False):
    if isinstance(case, dict):
        def execute(*args, **kwargs):
            return measured(lambda: vm.execute_microop_routed_control(
                *args, **kwargs, prepare_operand_pop=prepare_operand_pop, prepare_alu=prepare_alu, prepare_mul=prepare_mul, prepare_regions=prepare_regions,
                prepare_local_region=prepare_local_region,
            ), samples)

        with patch.object(terrain, "execute_microop_routed_control", execute):
            proof = terrain.verify(case, expected)
        for sample in samples:
            sample["module_preparation_ns"] = proof["module_preparation_ns"]
        return proof
    program, initial, argument, expected_return, reference, expected_steps, setup_ns = case
    page = initial.copy()
    result = measured(lambda: vm.execute_microop_routed_control(
        program, argument, array_memory=page, fuel=50_000, capture_trace=False,
        prepare_operand_pop=prepare_operand_pop, prepare_alu=prepare_alu, prepare_mul=prepare_mul, prepare_regions=prepare_regions,
        prepare_local_region=prepare_local_region,
    ), samples)
    assert result.backend == "v58-structured-numpy-memory"
    assert (page, result.result_i32, result.steps) == (reference, expected_return, expected_steps)
    samples[-1]["module_preparation_ns"] = setup_ns
    samples[-1]["page_sha256"] = terrain.digest(page)
    return None


def summarize(samples, agents):
    timings = sorted(s["execution_ns"] for s in samples)
    return {"samples": len(samples), "median_ms": median(timings) / 1e6,
            "p95_ms": timings[math.ceil(len(timings) * .95) - 1] / 1e6,
            "pre_execution_median_ms": median(s["pre_execution_ns"] for s in samples) / 1e6,
            "agent_ticks_per_second": len(timings) * agents * 1e9 / sum(timings) if agents else None,
            "steps_per_sample": sorted({s["steps"] for s in samples})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", type=int, required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("Use at least one pair")
    inputs = workload(args.case)
    paths = {"dense": reference_transition(), "unit_push_pop": storage.step_operands}
    records = {name: [] for name in paths}
    proofs = {name: [] for name in paths}
    expected = None
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            order = tuple(paths) if pair % 2 == 0 else tuple(reversed(paths))
            for name in order:
                samples = []
                with patch.object(adapter, "step_operands", paths[name]):
                    proof = run(inputs, samples, expected)
                if proof is not None:
                    expected = proof
                if pair:
                    records[name].extend(samples)
                    if proof is not None:
                        proofs[name].append([r["page_sha256"] for r in proof["ticks"]])
    binary = terrain.GUEST if isinstance(inputs, dict) else ARCHIVE / f"{args.case}.wasm"
    print(json.dumps({"baseline": BASELINE, "environment": environment(), "workload": args.case,
                      "binary_sha256": terrain.digest(binary.read_bytes()),
                      "pairs_after_warmup": args.pairs, "samples": records, "world_hashes": proofs,
                      "summary": {name: summarize(samples, 8 if isinstance(inputs, dict) else None)
                                  for name, samples in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
