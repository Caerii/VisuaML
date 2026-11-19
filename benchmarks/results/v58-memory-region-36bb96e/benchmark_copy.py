"""Time guarded copy fusion with a fresh page and complete readback in each arm."""
import json
import sys
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure
from scripts.verify_memory_region_v58 import Case
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program

source = git_identity(Path.cwd())
program = copy_program()
packed = prepare(program)
lowered = PreparedCopyRegion(program)
store = wasmtime.Store()
wasm = wasmtime.Instance(store, wasmtime.Module(store.engine, encode_control(program)), []).exports(store)
rows = []
for words in (1, 4, 16, 64):
    for src, dst in ((8, 12), (12, 8)):
        start = Case(words, src, dst).memory()

        def run(session):
            page = bytearray(start)
            result = session.run_memory(words, page, fuel=2000, capture_trace=False)
            return result.result_i32, page

        def oracle():
            wasm["memory"].write(store, start, 0)
            result = wasm["run"](store, words)
            return result, wasm["memory"].read(store)

        assert run(packed) == run(lowered) == oracle()
        assert lowered.run_memory(words, bytearray(start), fuel=2000, capture_trace=False).fused
        actions = {"packed": lambda: run(packed), "fused": lambda: run(lowered), "wasmtime": oracle}
        order = tuple(actions) if len(rows) % 2 == 0 else tuple(reversed(actions))
        timing = {name: measure(actions[name], warmups=3, repeats=21) for name in order}
        rows.append({"words": words, "src": src, "dst": dst, "timings": timing})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "environment": environment(), "rows": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
