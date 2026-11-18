"""Measure the combined gain from both native execution changes in one session."""
import json
import statistics
import sys
from functools import partial
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure, workloads
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

source = git_identity(Path.cwd())
kernels = compiled_alu()
rows = []
ratios = {"trace": [], "no_trace": []}
for work in workloads(scaling=True):
    if any(op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
           for op in work.program.operations[:-1]):
        continue
    python = PreparedMicrostepProgram(work.program)
    loop = PreparedMicrostepProgram(work.program, use_native_loop=True)
    for argument in work.arguments:
        expected = python.run(argument)
        assert loop.run(argument) == expected
        action = {"python": python.run, "loop": loop.run}
        order = tuple(action) if len(rows) % 2 == 0 else tuple(reversed(action))
        timings = {}
        for trace, mode in ((True, "trace"), (False, "no_trace")):
            pair = {name: measure(partial(action[name], argument, capture_trace=trace),
                                  repeats=5, warmups=1, expected=expected.result_i32)
                    for name in order}
            ratios[mode].append(pair["python"]["wall"]["median_ns"] / pair["loop"]["wall"]["median_ns"])
            timings[mode] = pair
        rows.append({"program": work.name, "input": argument, "timings": timings})

with Path(sys.argv[1]).open("x") as stream:
    json.dump({
        "source": source, "environment": environment(), "kernel_sha256": kernels.binary_sha256,
        "cases": rows, "paired_speedup": {name: statistics.geometric_mean(values)
                                         for name, values in ratios.items()},
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
