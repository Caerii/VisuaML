"""Compare both structured engines with preparation charged on every invocation."""
import json
import statistics
import sys
from functools import partial
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure
from scripts.verify_packed_control_v58 import prepare, witnesses
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control

source = git_identity(Path.cwd())
rows = []


def packed(program, argument):
    return prepare(program).run(argument)


for name, (program, arguments) in witnesses().items():
    for argument in arguments:
        actions = {"packed": partial(packed, program, argument),
                   "v57": partial(execute_microop_routed_control, program, argument)}
        assert actions["packed"]().result_i32 == actions["v57"]().result_i32
        order = tuple(actions) if len(rows) % 2 == 0 else tuple(reversed(actions))
        timing = {arm: measure(actions[arm], warmups=1, repeats=3) for arm in order}
        rows.append({"fixture": name, "argument": argument, "timings": timing})


def median(row, mode):
    return row["timings"][mode]["wall"]["median_ns"]


with Path(sys.argv[1]).open("x") as stream:
    json.dump({
        "source": source, "environment": environment(), "rows": rows,
        "paired_geomean": statistics.geometric_mean(
            median(row, "v57") / median(row, "packed") for row in rows
        ),
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
