"""Replay all nonzero-local role assignments with source-position padding."""
import json
import sys
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from random import Random

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_copy_recognition_v58 import Case, wasm_oracle
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.memory_region_contract_v58 import first_difference, observe
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program

source = git_identity(Path.cwd())
base = copy_program()
random = Random(58)
rows = []
for assignments in permutations(range(1, 5), 3):
    roles = {0: 0, **dict(zip(range(1, 4), assignments, strict=True))}
    padding = sorted(random.sample(range(len(base.operations)), 6))
    ops = []
    for pc, op in enumerate(base.operations):
        if pc in padding:
            ops.append(ControlOperation("nop"))
        if op.opcode in ("local.get", "local.set"):
            op = replace(op, immediate=roles[op.immediate])
        ops.append(op)
    variant = ControlProgram(4, tuple(ops))
    lowered = PreparedCopyRegion(variant)
    packed = prepare(variant)
    wasm = wasm_oracle(variant)
    assert lowered.plan is not None
    for case in (Case(0, 8, 12), Case(3, 8, 9), Case(3, 12, 8), Case(64, 8, 12)):
        initial = case.memory()
        source_result = observe(packed.run_memory, case.count, initial, fuel=2000)
        assert first_difference(wasm(case.count, initial), source_result) is None
        out = bytearray(initial)
        result = lowered.run_memory(case.count, out, fuel=source_result.source_steps, capture_trace=False)
        assert result.fused and result.steps == source_result.source_steps and out == source_result.page
        rows.append({"locals": assignments, "padding_positions": padding, "count": case.count,
                     "src": case.src, "dst": case.dst, "source_steps": result.steps})
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "variants": len(rows) // 4, "rows": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")
