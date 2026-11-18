"""Replay timing regressions with interleaved prepared calls."""
import json
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns

from scripts.benchmark_neural_vm_v58 import git_identity, summarize_ns, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

output = Path(sys.argv[1])
trials = [json.loads((output / f"runtime-{trial}.json").read_bytes()) for trial in (1, 2)]
programs = {work.name: work.program for work in workloads(scaling=True)}
rows = []
for trial, run in enumerate(trials, 1):
    for work in run["rows"]:
        for case in work["cases"]:
            for mode in ("trace", "no_trace"):
                before = case["timing"][f"prepared_{mode}"]["wall"]["median_ns"]
                after = case["timing"][mode]["wall"]["median_ns"]
                if after <= before:
                    continue
                program = programs[work["program"]]
                actions = {
                    name: partial(
                        PreparedMicrostepProgram(program, use_native_microstep=native).run,
                        case["input"], fuel=1500, capture_trace=mode == "trace",
                    ) for name, native in (("python", False), ("native", True))
                }
                expected = actions["python"]()
                assert actions["native"]() == expected
                samples = {key: [] for key in actions}
                for repeat in range(15):
                    order = tuple(actions) if repeat % 2 == 0 else tuple(reversed(actions))
                    for key in order:
                        started = perf_counter_ns()
                        result = actions[key]()
                        samples[key].append(perf_counter_ns() - started)
                        assert result == expected
                rows.append({
                    "run": trial, "program": work["program"], "input": case["input"], "mode": mode,
                    "original_slowdown": after / before,
                    "replay": {key: summarize_ns(values) for key, values in samples.items()},
                })
with (output / "outliers.json").open("x") as stream:
    json.dump({"source": git_identity(Path.cwd()), "repeats": 15, "rows": rows},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
