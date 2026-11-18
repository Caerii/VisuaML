"""Recheck the largest observed slowdown in each mode."""
import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from scripts.benchmark_neural_vm_v58 import git_identity, summarize_ns, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    PreparedMicrostepProgram, execute_microstep_control,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import compile_microop_tokens

results = Path(sys.argv[1])
benchmarks = [json.loads((results / f"runtime-{trial}.json").read_bytes()) for trial in (1, 2)]
programs = {work.name: work.program for work in workloads(scaling=True)}
rows = []
for mode in ("trace", "no_trace"):
    factor, trial, name, argument = max(
        (case["timing"][mode]["wall"]["median_ns"] /
         case["timing"][f"indexed_{mode}"]["wall"]["median_ns"],
         trial, row["program"], case["input"])
        for trial, data in enumerate(benchmarks, 1)
        for row in data["rows"] for case in row["cases"]
    )
    program = programs[name]
    options = dict(use_fused_state=True, use_compiled_alu=True,
                   program_tokens=compile_microop_tokens(program))
    prepared = PreparedMicrostepProgram(program, **options)
    run = dict(fuel=1500, capture_trace=(mode == "trace"))
    actions = {
        "indexed": partial(execute_microstep_control, program, argument,
                           use_compact_routing=True, use_indexed_pc=True, **options, **run),
        "prepared": partial(prepared.run, argument, **run),
    }
    expected = actions["indexed"]()
    assert actions["prepared"]() == expected
    samples = {key: [] for key in actions}
    for sample in range(15):
        order = tuple(actions) if sample % 2 == 0 else tuple(reversed(actions))
        for key in order:
            started = perf_counter_ns()
            actual = actions[key]()
            samples[key].append(perf_counter_ns() - started)
            assert actual == expected
    rows.append({
        "mode": mode, "program": name, "input": argument, "benchmark_run": trial,
        "original_regression_factor": factor,
        "paired_recheck": {key: summarize_ns(values) for key, values in samples.items()},
    })
record = {"source": git_identity(Path.cwd()), "repeats": 15, "rows": rows}
with (results / "outliers.json").open("x") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
