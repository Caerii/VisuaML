"""Profile the warm native loop separately from latency measurements."""
import cProfile
import json
import sys
from pathlib import Path

from scripts.benchmark_constructed_runtime_v58 import _profile_rows
from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

source = git_identity(Path.cwd())
calls = []
for work in workloads(scaling=True):
    if any(op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
           for op in work.program.operations[:-1]):
        continue
    prepared = PreparedMicrostepProgram(work.program, use_native_loop=True)
    calls.extend((prepared, argument) for argument in work.arguments)
records = {}
for trace in (False, True):
    profile = cProfile.Profile()
    for prepared, argument in calls:
        profile.runcall(prepared.run, argument, fuel=1500, capture_trace=trace)
    records["trace" if trace else "no_trace"] = _profile_rows(profile)
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "calls_per_mode": len(calls), "profile": records},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
