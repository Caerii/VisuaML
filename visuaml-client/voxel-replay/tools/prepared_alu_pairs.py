"""Paired full replays with a separate fixed-circuit work counter."""
import argparse
import json
import sys
from collections import Counter
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted

import terrain_verify as terrain
from operand_pairs import run, summarize, workload
from scale_terrain import ARCHIVE


def mirror(case):
    cells = len(case["initial"]["materials"])
    return {**case, "initial": {
        "positions": [cells - 1 - p for p in case["initial"]["positions"]],
        "materials": list(reversed(case["initial"]["materials"])),
    }, "inputs": [{"moves": [-m for m in tick["moves"]], "edit": False} for tick in case["inputs"]]}


def counters(case, enabled, expected):
    calls = Counter()
    generic = alu.execute_alu.__code__
    prepared = alu._run_prepared_alu.__code__

    def count(frame, event, _arg):
        if event == "call":
            if frame.f_code is generic:
                calls["generic"] += 1
            elif frame.f_code is prepared:
                circuit = frame.f_locals["circuit"].__name__
                calls[circuit] += 1

    sys.setprofile(count)
    try:
        if isinstance(case, dict):
            proof = run(case, [], expected, prepare_operand_pop=True, prepare_alu=enabled)
            steps = sum(tick["steps"] for tick in proof["ticks"])
        else:
            program, page, argument, *_ = case
            steps = min(4000, case[5])
            try:
                result = vm.execute_microop_routed_control(
                    program, argument, array_memory=page.copy(), capture_trace=False,
                    fuel=steps, prepare_alu=enabled, prepare_mul=False, prepare_regions=False,
                )
                assert result.steps == steps
            except ControlFuelExhausted:
                assert steps < case[5]
    finally:
        sys.setprofile(None)
    return {"steps": steps, "generic_ticks": calls["generic"], "prepared_add": calls["_add"],
            "prepared_lt_s": calls["_less_than_word"],
            "candidates_evaluated": 6 * calls["generic"] + calls["_add"] + calls["_less_than_word"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", required=True, type=int)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("Use at least two pairs")
    case = workload(args.case)
    world = isinstance(case, dict)
    logs = [case, mirror(case)] if world else [case]
    binary = terrain.GUEST if world else ARCHIVE / f"{args.case}.wasm"
    chosen = {"six_candidates": False, "prepared": True}
    samples = {name: [] for name in chosen}
    expected = [None] * len(logs)
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            # Alternate order overall and within each deterministic log.
            log = pair % len(logs)
            if world and pair:
                log = ((pair - 1) // 2) % len(logs)
            order = tuple(chosen) if pair % 2 == 0 else tuple(reversed(chosen))
            for name in order:
                rows = []
                proof = run(logs[log], rows, expected[log], prepare_operand_pop=True,
                            prepare_alu=chosen[name])
                if world:
                    expected[log] = proof
                    for record, tick in zip(rows, proof["ticks"], strict=True):
                        record.update(log=log, page_sha256=tick["page_sha256"])
                if pair:
                    for row in rows:
                        row["pair"] = pair
                    samples[name].extend(rows)
        work = {name: [counters(log, enabled, expected[i]) for i, log in enumerate(logs)]
                for name, enabled in chosen.items()}
    for old, new in zip(work["six_candidates"], work["prepared"], strict=True):
        selected = new["prepared_add"] + new["prepared_lt_s"]
        assert old["steps"] == new["steps"] and old["generic_ticks"] == new["generic_ticks"] + selected
        assert old["candidates_evaluated"] - new["candidates_evaluated"] == selected * 5
    print(json.dumps({
        "baseline": "acf7abfa52da9ca8fff4c8eae3c7d0d7b854e669",
        "workload": args.case, "binary_sha256": terrain.digest(binary.read_bytes()),
        "environment": environment(), "pairs_after_warmup": args.pairs,
        "world_logs": logs if world else None, "samples": samples, "counter_pass": work,
        "summary": {name: summarize(rows, 8 if world else None) for name, rows in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
