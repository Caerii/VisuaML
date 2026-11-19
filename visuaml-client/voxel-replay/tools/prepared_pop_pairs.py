"""Paired prepared-pop timings plus a separately instrumented counter pass."""
import argparse
import json
import sys
from collections import Counter
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_operand_adapter_v55 as adapter
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted
from visuaml.compiler_lab import neural_microop_executor_v57 as vm

import terrain_verify as terrain
from operand_pairs import run, summarize, workload
from scale_terrain import ARCHIVE


def count_run(case, enabled, expected):
    counts = Counter()
    checked = adapter.TensorOperandStack.pop.__code__
    prepared = adapter._PreparedPopStack.pop.__code__
    validation = adapter.validate_operands.__code__

    def count(frame, event, _arg):
        if event == "call":
            if frame.f_code is checked:
                counts["checked_pops"] += 1
            elif frame.f_code is prepared:
                counts["prepared_pops"] += 1
            elif frame.f_code is validation:
                counts["all_operand_state_scans"] += 1

    sys.setprofile(count)
    try:
        if isinstance(case, dict):
            run(case, [], expected, prepare_operand_pop=enabled)
            invocations = len(case["inputs"])
            source_steps = sum(row["steps"] for row in expected["ticks"])
        else:
            program, page, argument, *_ = case
            source_steps = min(4000, case[5])
            try:
                result = vm.execute_microop_routed_control(
                    program, argument, array_memory=page.copy(), capture_trace=False,
                    fuel=source_steps, prepare_operand_pop=enabled, prepare_alu=False,
                )
                assert result.steps == source_steps
            except ControlFuelExhausted:
                assert source_steps < case[5]
            invocations = 1
    finally:
        sys.setprofile(None)
    return {"source_steps": source_steps, "invocations": invocations, **{
        key: counts[key] for key in ("checked_pops", "prepared_pops", "all_operand_state_scans")
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", required=True, type=int)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("Use at least one pair")
    case = workload(args.case)
    world = isinstance(case, dict)
    binary = terrain.GUEST if world else ARCHIVE / f"{args.case}.wasm"
    paths = {"checked": False, "prepared": True}
    samples = {name: [] for name in paths}
    expected = None
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            order = tuple(paths) if pair % 2 == 0 else tuple(reversed(paths))
            for name in order:
                records = []
                proof = run(case, records, expected, prepare_operand_pop=paths[name])
                if world:
                    expected = proof
                    for record, tick in zip(records, proof["ticks"], strict=True):
                        record["page_sha256"] = tick["page_sha256"]
                if pair:
                    samples[name].extend(records)
        counters = {name: count_run(case, enabled, expected) for name, enabled in paths.items()}
    old, new = counters["checked"], counters["prepared"]
    assert old["checked_pops"] == new["prepared_pops"] and new["checked_pops"] == 0
    scans_avoided = old["all_operand_state_scans"] - new["all_operand_state_scans"]
    assert scans_avoided == 2 * new["prepared_pops"] - new["invocations"]
    print(json.dumps({
        "baseline": "c562449dd6d2d6dd5062007629795176673bce0c",
        "environment": environment(), "workload": args.case,
        "binary_sha256": terrain.digest(binary.read_bytes()),
        "pairs_after_warmup": args.pairs, "samples": samples,
        "counter_pass": {**counters, "state_scans_avoided": scans_avoided},
        "world_session": case if world else None,
        "summary": {name: summarize(rows, 8 if world else None) for name, rows in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
