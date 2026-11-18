"""Recheck the largest paired slowdown in each measurement mode."""
import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from scripts.benchmark_neural_vm_v58 import git_identity, summarize_ns, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.typed_microop_ir_v57 import compile_microop_tokens

results = Path(sys.argv[1])
benchmarks = [json.loads((results/f"runtime-{trial}.json").read_bytes()) for trial in (1, 2)]
programs = {work.name: work.program for work in workloads(scaling=True)}
rows = []
for mode in ("trace", "no_trace"):
    worst = max((c["timing"][mode]["wall"]["median_ns"] /
                 c["timing"]["compact_"+mode]["wall"]["median_ns"],
                 trial, row["program"], c["input"])
                for trial, d in enumerate(benchmarks, 1)
                for row in d["rows"] for c in row["cases"])
    factor, trial, name, argument = worst
    program = programs[name]
    common = dict(fuel=1500, capture_trace=(mode == "trace"), use_fused_state=True,
                  use_compiled_alu=True, use_compact_routing=True,
                  program_tokens=compile_microop_tokens(program))
    actions = {key: partial(execute_microstep_control, program, argument,
                            use_indexed_pc=(key == "indexed"), **common)
               for key in ("compact", "indexed")}
    expected = actions["compact"]()
    assert actions["indexed"]() == expected
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
with (results/"outliers.json").open("x") as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write("\n")
