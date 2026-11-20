"""Paired fused-copy, prepared and compact frame-input replays."""
import argparse
import json
import sys
from collections import Counter
from time import perf_counter_ns
from unittest.mock import patch

from scripts.benchmark_neural_vm_v58 import environment
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.compiled_frame_event_v58 import _CompiledFrameRetain
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu

import terrain_verify as terrain
from compiled_frame_event_pairs import SETTINGS
from operand_pairs import run, summarize, workload
from prepared_alu_pairs import mirror
from scale_terrain import ARCHIVE

MODES = {"copy": (False, False), "prepared": (True, False), "compact": (True, True)}


def execute(case, samples, expected, mode):
    prepared, compact = MODES[mode]
    return run(case, samples, expected, **SETTINGS, use_compiled_frame_event=True,
               compiled_frame_retain="copy", prepare_frame_inputs=prepared, compact_frame_inputs=compact)


def counters(case, expected, mode):
    counts = Counter()
    functions = {
        _CompiledFrameRetain.select.__code__: "four_input_calls",
        _CompiledFrameRetain.select_live.__code__: "three_input_calls",
        vm.FrameTokens.__init__.__code__: "token_objects",
        vm.step_frames.__code__: "checked_entry_calls",
        vm._step_frame_gates.__code__: "frame_gate_calls",
        vm._frame_optional.__code__: "optional_token_constructors",
    }

    def record(frame, event, _arg):
        if event == "call" and frame.f_code in functions:
            counts[functions[frame.f_code]] += 1

    samples = []
    sys.setprofile(record)
    try:
        execute(case, samples, expected, mode)
    finally:
        sys.setprofile(None)
    return {"steps": sum(item["steps"] for item in samples), **{
        name: counts[name] for name in functions.values()
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sparse", "crowded", "bfs", "sudoku"))
    parser.add_argument("--pairs", type=int, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("At least two measured sets required")
    case = workload(args.case)
    world = isinstance(case, dict)
    logs = [case, mirror(case)] if world else [case]
    binary = terrain.GUEST if world else ARCHIVE / f"{args.case}.wasm"
    started = perf_counter_ns()
    kernel = compiled_alu()
    compilation_ns = perf_counter_ns() - started
    samples = {mode: [] for mode in MODES}
    expected = [None] * len(logs)
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for pair in range(args.pairs + 1):
            index = ((pair - 1) // 2) % len(logs) if pair else 0
            order = tuple(MODES) if pair % 2 == 0 else tuple(reversed(MODES))
            for mode in order:
                rows = []
                proof = execute(logs[index], rows, expected[index], mode)
                if world:
                    expected[index] = proof
                    for record, tick in zip(rows, proof["ticks"], strict=True):
                        record.update(log=index, page_sha256=tick["page_sha256"])
                if pair:
                    for row in rows:
                        row["pair"] = pair
                    samples[mode].extend(rows)
        # Separate instrumentation samples the four-tick log, full BFS,
        # and the same one-blank Sudoku invocation.
        counts = {mode: counters(case, expected[0], mode) for mode in MODES}
    assert counts["copy"]["four_input_calls"] == counts["prepared"]["three_input_calls"] == counts["compact"]["three_input_calls"]
    assert len({row["frame_gate_calls"] for row in counts.values()}) == 1
    print(json.dumps({
        "baseline": "0d8da4fc46edb7262ee5fb96512637c2e06c4a72", "case": args.case,
        "environment": environment(), "binary_sha256": terrain.digest(binary.read_bytes()),
        "compile_and_load_ns": compilation_ns, "compiler": kernel.compiler,
        "kernel_binary_sha256": kernel.binary_sha256, "kernel_source_sha256": kernel.source_sha256,
        "pairs_after_warmup": args.pairs, "samples": samples, "counter_pass": counts,
        "world_logs": logs if world else None,
        "summary": {mode: summarize(rows, 8 if world else None) for mode, rows in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
