"""Replay both branch operations over every legal depth in a 12-label nest."""
import json
import sys
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_packed_control_v58 import first_divergence, frame_reference, prepare
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import MAX_NESTING, ControlProgram, encode_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import op


def program(branch, depth):
    setup = (op("i32.const", 31), *(op("block"),) * MAX_NESTING, op("i32.const", 97))
    injected = (op("local.get", 0),) if branch == "br_if" else ()
    tail = (op("drop"),) if branch == "br_if" else ()
    return ControlProgram(0, (*setup, *injected, op(branch, depth), *tail,
                              *(op("end"),) * (MAX_NESTING + 1)))


source = git_identity(Path.cwd())
engine = wasmtime.Engine()
rows = []
for opcode in ("br", "br_if"):
    for depth in range(MAX_NESTING):
        fixture = program(opcode, depth)
        wasm_bytes = encode_control(fixture)
        store = wasmtime.Store(engine)
        wasm = wasmtime.Instance(store, wasmtime.Module(engine, wasm_bytes), []).exports(store)["run"]
        packed = prepare(fixture)
        for value in ((1,) if opcode == "br" else (-1, 0)):
            actual = packed.run(value)
            reference = execute_microop_routed_control(fixture, value)
            assert actual.result_i32 == reference.result_i32 == wasm(store, value) == 31
            expected = tuple(frame_reference(fixture, reference.trace))
            difference = first_divergence(expected, actual.trace)
            assert difference is None, (opcode, depth, value, difference)
            branch = next(record for record in actual.trace if record["opcode"] == opcode)
            taken = opcode == "br" or value != 0
            assert branch["stack_i32"] == ([31] if taken else [31, 97])
            assert len(branch["control_frames"]) == (MAX_NESTING - depth - 1 if taken else MAX_NESTING)
            rows.append({"opcode": opcode, "depth": depth, "argument": value, "taken": taken,
                         "wasm_hex": wasm_bytes.hex(), "branch": branch, "steps": actual.steps})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "witnesses": len(rows), "rows": rows}, stream,
              indent=2, sort_keys=True)
    stream.write("\n")
