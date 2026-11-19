"""Measure warm API latency with explicit page reset and full-page readback."""
import json
import sys
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import environment, git_identity, measure
from scripts.verify_memory_recovery_v58 import initial_page, prefix_program, symbolic
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

source = git_identity(Path.cwd())
program = prefix_program()
packed = prepare(program)
kernel = compiled_alu()
store = wasmtime.Store()
wasm = wasmtime.Instance(store, wasmtime.Module(store.engine, encode_control(program)), []).exports(store)
rows = []
for size in (0, 1, 4, 16, 64):
    values = tuple(((i * 45466329 + 1117) % (1 << 32)) - (1 << 31) for i in range(size))
    start = initial_page(values)

    def native():
        page = bytearray(start)
        result = packed.run_memory(size, page, capture_trace=False, fuel=2000)
        return result.result_i32, page

    def oracle():
        wasm["memory"].write(store, start, 0)
        result = wasm["run"](store, size)
        return result, wasm["memory"].read(store)

    assert native() == oracle() == symbolic("sum", values)
    actions = {"packed": native, "wasmtime": oracle}
    order = tuple(actions) if len(rows) % 2 == 0 else tuple(reversed(actions))
    timing = {name: measure(actions[name], warmups=3, repeats=21) for name in order}
    rows.append({"length": size, "timings": timing})

with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "environment": environment(), "kernel_sha256": kernel.binary_sha256,
               "rows": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
