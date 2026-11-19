"""Separate prepared-multiply and const/add region measurements."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.neural_operand_storage_v55 import _unit_pop, step_operands
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted

import terrain_verify as terrain
from operand_pairs import run, summarize, workload
from prepared_alu_pairs import mirror
from scale_terrain import ARCHIVE

VARIANTS = {
    "baseline": {"prepare_mul": False, "prepare_regions": False},
    "multiply": {"prepare_mul": True, "prepare_regions": False},
    "region": {"prepare_mul": True, "prepare_regions": True, "prepare_local_region": False},
}


def physical_work(case, expected, options):
    functions = {
        vm._run_const_add_region.__code__: "regions",
        vm._run_local_const_add_region.__code__: "local_regions",
        vm._array_route.__code__: "route_calls",
        vm.select_frame_action.__code__: "frame_event_calls",
        step_operands.__code__: "checked_operand_transitions",
        _unit_pop.__code__: "unit_pop_copies",
        alu.execute_alu.__code__: "generic_alu_calls",
        alu._run_prepared_alu.__code__: "prepared_alu_calls",
    }
    counts = Counter()

    def count(frame, event, _arg):
        if event == "call" and frame.f_code in functions:
            counts[functions[frame.f_code]] += 1

    sys.setprofile(count)
    try:
        if isinstance(case, dict):
            proof = run(case, [], expected, prepare_operand_pop=True, prepare_alu=True, **options)
            steps = sum(row["steps"] for row in proof["ticks"])
        else:
            program, page, argument, _, _, full_steps, _ = case
            steps = min(4000, full_steps)
            try:
                result = vm.execute_microop_routed_control(
                    program, argument, array_memory=page.copy(), fuel=steps, capture_trace=False,
                    **options,
                )
                assert result.steps == steps
            except ControlFuelExhausted:
                assert steps < full_steps
    finally:
        sys.setprofile(None)
    return {"steps": steps, **{label: counts[label] for label in functions.values()},
            "alu_candidates": 6 * counts["generic_alu_calls"] + counts["prepared_alu_calls"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", required=True, type=int)
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variants = VARIANTS if not args.extend else {
        "region": VARIANTS["region"],
        "local_region": {"prepare_mul": True, "prepare_regions": True, "prepare_local_region": True},
    }
    if args.pairs < 2:
        parser.error("Use at least two measured sets")
    case = workload(args.case)
    world = isinstance(case, dict)
    logs = [case, mirror(case)] if world else [case]
    expected = [None] * len(logs)
    samples = {name: [] for name in variants}
    binary = terrain.GUEST if world else ARCHIVE / f"{args.case}.wasm"
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            log_id = ((pair - 1) // 2) % len(logs) if pair else 0
            order = tuple(variants) if pair % 2 == 0 else tuple(reversed(variants))
            for variant in order:
                rows = []
                proof = run(logs[log_id], rows, expected[log_id],
                            prepare_operand_pop=True, prepare_alu=True, **variants[variant])
                if world:
                    expected[log_id] = proof
                    for record, tick in zip(rows, proof["ticks"], strict=True):
                        record.update(log=log_id, page_sha256=tick["page_sha256"])
                if pair:
                    for record in rows:
                        record["set"] = pair
                    samples[variant].extend(rows)
        work = {name: physical_work(case, expected[0], flags) for name, flags in variants.items()}
    if args.extend:
        base, extended = work["region"], work["local_region"]
        assert base["steps"] == extended["steps"]
        assert base["alu_candidates"] == extended["alu_candidates"]
        saved = extended["regions"] * 2 + extended["local_regions"] * 3 - base["regions"] * 2
        assert base["route_calls"] - extended["route_calls"] == saved
        assert base["frame_event_calls"] - extended["frame_event_calls"] == saved
    else:
        multiply, region = work["multiply"], work["region"]
        assert region["steps"] == multiply["steps"]
        assert multiply["route_calls"] - region["route_calls"] == 2 * region["regions"]
        assert multiply["frame_event_calls"] - region["frame_event_calls"] == 2 * region["regions"]
        assert multiply["checked_operand_transitions"] - region["checked_operand_transitions"] == 2 * region["regions"]
        assert multiply["unit_pop_copies"] - region["unit_pop_copies"] == 2 * region["regions"]
        assert region["alu_candidates"] == multiply["alu_candidates"]
    report = json.dumps({
        "baseline": "d1ff1df5432fc299c4fab0054ac310226e95c186", "workload": args.case,
        "environment": environment(), "binary_sha256": terrain.digest(binary.read_bytes()),
        "measured_sets_after_warmup": args.pairs, "samples": samples,
        "world_logs": logs if world else None, "counter_pass": work,
        "summary": {name: summarize(rows, 8 if world else None) for name, rows in samples.items()},
    }, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    else:
        print(report)


if __name__ == "__main__":
    main()
