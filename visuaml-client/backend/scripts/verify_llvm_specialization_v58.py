"""Check emitted LLVM scalar expressions against the v43 typed WASM encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import z3

from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.llvm_specialization_v58 import _llvm, _module
from visuaml.compiler_lab.native_specialization_v58 import _lower_ssa
from visuaml.compiler_lab.neural_microstep_v58 import DATA_OPS
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import _encode_wasm_in_z3
from visuaml.compiler_lab.wasm_scale_corpus_v58 import arithmetic_chain
from visuaml.compiler_lab.wasm_state_vm_v40 import MAX_INSTRUCTIONS, I32Program, VMOperation

_TOKEN = r"(%(?:input|v\d+)|-?\d+)"
_BINARY = rf"(%v\d+|%flag\d+) = (add|sub|mul|icmp eq|icmp slt|icmp sgt) i32 {_TOKEN}, {_TOKEN}"
_EXTEND = r"(%v\d+) = zext i1 (%flag\d+) to i32"


def _result(ir, x):
    def value(token):
        return bindings[token] if token.startswith("%") else z3.BitVecVal(int(token), 32)

    scalar, = re.findall(r"^define i32 @visuaml_run\(i32 %input\) \{\n(.*?)\n\}$",
                         ir, re.MULTILINE | re.DOTALL)
    lines = scalar.strip().splitlines()
    bindings = {"%input": x}
    for line in lines[:-1]:
        if match := re.fullmatch(_BINARY, line.strip()):
            name, op, left, right = match.groups()
            a, b = value(left), value(right)
            match op:
                case "add":
                    result = a + b
                case "sub":
                    result = a - b
                case "mul":
                    result = a * b
                case "icmp eq":
                    result = a == b
                case "icmp slt":
                    result = a < b
                case "icmp sgt":
                    result = a > b
        elif match := re.fullmatch(_EXTEND, line.strip()):
            name, flag = match.groups()
            result = z3.If(bindings[flag], z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
        else:
            raise ValueError("Unrecognized LLVM scalar instruction")
        if name in bindings:
            raise ValueError("Duplicate LLVM definition")
        bindings[name] = result
    returned = re.fullmatch(rf"ret i32 {_TOKEN}", lines[-1].strip())
    if returned is None:
        raise ValueError("Unrecognized LLVM scalar return")
    return value(returned[1])


def proof(program, ir):
    source = I32Program(program.local_count, tuple(
        VMOperation(op.opcode, op.immediate)
        for op in program.operations if op.opcode != "nop"
    ))
    if len(source.operations) > MAX_INSTRUCTIONS:
        return "unsupported_source_solver_length"
    x = z3.BitVec("input_i32", 32)
    solver = z3.Solver()
    solver.set(timeout=5000)
    solver.add(_encode_wasm_in_z3(source, x, z3) != _result(ir, x))
    status = solver.check()
    if status == z3.sat:
        raise AssertionError(f"LLVM expression disagrees at {solver.model().eval(x, model_completion=True)}")
    return str(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    source = git_identity(root)
    llvm = _llvm()
    target = llvm.Target.from_default_triple()
    machine = target.create_target_machine(opt=2)
    layout = str(machine.target_data)

    def emit(program):
        definitions, states = _lower_ssa(program)
        return _module(definitions, states[-1][0][0], target.triple, layout)

    rows = []
    for work in workloads(scaling=True):
        if any(op.opcode not in DATA_OPS for op in work.program.operations[:-1]):
            continue
        ir = emit(work.program)
        rows.append({"name": work.name, "llvm_ir_sha256": hashlib.sha256(ir.encode()).hexdigest(),
                     "status": proof(work.program, ir)})

    program = arithmetic_chain(2)
    bad = re.sub(r"(?m)^  ret i32 %v\d+$", "  ret i32 %input", emit(program), count=1)
    try:
        proof(program, bad)
    except AssertionError as exc:
        negative = str(exc)
    else:
        raise AssertionError("Corrupted LLVM result was not detected")

    summary = {status: sum(row["status"] == status for row in rows)
               for status in ("unsat", "unknown", "unsupported_source_solver_length")}
    data = {"source": source, "solver": z3.get_full_version(), "programs": rows,
            "summary": summary, "negative_control": negative,
            "scope": "Pre-optimization LLVM scalar i32 result versus typed WASM source."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
