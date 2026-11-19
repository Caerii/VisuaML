"""Exercise one terminating source invocation immediately below the raised fuel cap."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import wasmtime

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.wasm_control_flow_v51 import SCALED_MAX_FUEL, SCALED_SCHEMA, ControlFuelExhausted, encode_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import countdown

source = git_identity(Path.cwd())
program = replace(countdown(), schema=SCALED_SCHEMA)
raw = encode_control(program)
packed = prepare(program)
argument = (SCALED_MAX_FUEL - 7) // 12
fuel = argument * 12 + 7
start = perf_counter_ns()
result = packed.run(argument, fuel=fuel, capture_trace=False)
packed_ns = perf_counter_ns() - start
assert result.steps == fuel
try:
    packed.run(argument, fuel=fuel - 1, capture_trace=False)
except ControlFuelExhausted:
    pass
else:
    raise AssertionError("Expected one-short fuel to trap")
store = wasmtime.Store()
reference = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)["run"]
arithmetic = (argument * (argument + 1) // 2 + (1 << 31)) % (1 << 32) - (1 << 31)
assert result.result_i32 == reference(store, argument) == arithmetic
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "wasm_sha256": hashlib.sha256(raw).hexdigest(), "argument": argument,
               "result": arithmetic, "packed_steps": fuel, "one_short_trapped": True, "packed_wall_ns": packed_ns},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
