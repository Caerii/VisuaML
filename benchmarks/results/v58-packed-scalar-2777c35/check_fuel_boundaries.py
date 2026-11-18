"""Compare fuel and final-state boundaries on the admitted straight-line corpus."""
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import digest, git_identity, workloads
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted


def outcome(session, value, fuel, capture):
    try:
        result = session.run(value, fuel=fuel, capture_trace=capture)
    except ControlFuelExhausted as error:
        return {"exception": str(error)}
    return {"result": result.result_i32, "steps": result.steps, "trace_sha256": digest(result.trace)}


source = git_identity(Path.cwd())
counts = Counter()
rows = []
for work in workloads(scaling=True):
    if any(op.opcode in ("block", "loop", "if", "else", "br", "br_if", "return")
           for op in work.program.operations[:-1]):
        continue
    microstep = PreparedMicrostepProgram(work.program, use_native_loop=True)
    native = PreparedMicrostepProgram(work.program, use_native_loop=True, use_packed_scalar=True)
    limit = len(work.program.operations)
    for value in work.arguments:
        for fuel in sorted({1, limit - 1, limit, limit + 1}):
            for capture in (False, True):
                observed = outcome(native, value, fuel, capture)
                expected = outcome(microstep, value, fuel, capture)
                assert observed == expected, (work.name, value, fuel, capture, expected, observed)
                counts["exhausted" if "exception" in observed else "completed"] += 1
                rows.append({"program": work.name, "input": value, "fuel": fuel,
                             "traced": capture, "outcome": observed})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "counts": counts, "cases": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
