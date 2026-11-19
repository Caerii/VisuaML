"""Pair the compiled v54 gate against NumPy with exact replay oracles."""
import argparse
import json
import sys
from collections import Counter
from time import perf_counter_ns
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.compiled_frame_event_v58 import CompiledFrameEvents
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu

import terrain_verify as terrain
from operand_pairs import run, summarize, workload
from prepared_alu_pairs import mirror
from scale_terrain import ARCHIVE

SETTINGS = dict(prepare_operand_pop=True, prepare_alu=True, prepare_mul=True,
                prepare_regions=True, prepare_local_region=True)


def work(case, expected, enabled):
    counts = Counter()
    codes = {
        vm.select_frame_action.__code__: "numpy_event_calls",
        CompiledFrameEvents.select.__code__: "kernel_calls",
        vm._run_const_add_region.__code__: "const_add_calls",
        vm._run_local_const_add_region.__code__: "local_const_add_calls",
    }

    def count(frame, event, _arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    records = []
    sys.setprofile(count)
    try:
        run(case, records, expected, **SETTINGS, use_compiled_frame_event=enabled)
    finally:
        sys.setprofile(None)
    source = sum(record["steps"] for record in records)
    assert source == (counts["numpy_event_calls"] + counts["kernel_calls"]
                      + counts["const_add_calls"] * 2 + counts["local_const_add_calls"] * 3)
    return {"source_steps": source, **{name: counts[name] for name in codes.values()},
            # Four np.array scalar guards and two explicit np.array
            # constructions in the reference selector itself.
            "explicit_per_event_array_constructions": 6 * counts["numpy_event_calls"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", type=int, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("Use at least two pairs")
    case = workload(args.case)
    world = isinstance(case, dict)
    logs = [case, mirror(case)] if world else [case]
    binary = terrain.GUEST if world else ARCHIVE / f"{args.case}.wasm"
    started = perf_counter_ns()
    kernel = compiled_alu()
    compilation_ns = perf_counter_ns() - started
    variants = {"numpy": False, "compiled": True}
    samples = {name: [] for name in variants}
    expected = [None] * len(logs)
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            index = ((pair - 1) // 2) % len(logs) if pair else 0
            order = tuple(variants) if pair % 2 == 0 else tuple(reversed(variants))
            for name in order:
                rows = []
                proof = run(logs[index], rows, expected[index], **SETTINGS,
                            use_compiled_frame_event=variants[name])
                if world:
                    expected[index] = proof
                    for record, tick in zip(rows, proof["ticks"], strict=True):
                        record.update(log=index, page_sha256=tick["page_sha256"])
                if pair:
                    for row in rows:
                        row["pair"] = pair
                    samples[name].extend(rows)
        counts = {name: work(case, expected[0], enabled) for name, enabled in variants.items()}
    assert counts["numpy"]["numpy_event_calls"] == counts["compiled"]["kernel_calls"] > 0
    assert counts["compiled"]["numpy_event_calls"] == 0
    print(json.dumps({
        "baseline": "574e6c55c9c7e137d8df6aad3dd783b40b2cc6bb", "case": args.case,
        "binary_sha256": terrain.digest(binary.read_bytes()), "environment": environment(),
        "compile_and_load_ns": compilation_ns, "compiler": kernel.compiler, "flags": kernel.flags,
        "kernel_source_sha256": kernel.source_sha256, "kernel_binary_sha256": kernel.binary_sha256,
        "pairs_after_warmup": args.pairs, "world_logs": logs if world else None,
        "samples": samples, "counter_pass": counts,
        "summary": {name: summarize(rows, 8 if world else None) for name, rows in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
