"""Profile only repeated prepared calls on the straight-line scaling corpus."""
import cProfile
import json
import sys
from pathlib import Path

from scripts.benchmark_constructed_runtime_v58 import _profile_rows
from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

path = Path(sys.argv[1])
profile = cProfile.Profile()
items = []
for work in workloads(scaling=True):
    if any(op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
           for op in work.program.operations[:-1]):
        continue
    prepared = PreparedMicrostepProgram(work.program)
    items.extend((prepared, argument) for argument in work.arguments)
records = {}
for trace in (False, True):
    for prepared, argument in items:
        profile.runcall(prepared.run, argument, fuel=1500, capture_trace=trace)
    records["trace" if trace else "no_trace"] = _profile_rows(profile)
    profile.clear()
with path.open("x") as stream:
    json.dump({"source": git_identity(Path.cwd()), "calls_per_mode": len(items),
               "profile": records}, stream, indent=2, sort_keys=True)
    stream.write("\n")
