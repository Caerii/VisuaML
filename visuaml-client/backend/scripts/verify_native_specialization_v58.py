"""Translation validation for v58's emitted straight-line integer expressions.

Z3 checks the lowered expressions against the independent typed stack
encoder. Executing emitted machine code is a separate benchmark gate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import z3

from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.native_specialization_v58 import _emit, _lower
from visuaml.compiler_lab.neural_microstep_v58 import DATA_OPS
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import _encode_wasm_in_z3
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlProgram, encode_control
from visuaml.compiler_lab.wasm_scale_corpus_v58 import arithmetic_chain
from visuaml.compiler_lab.wasm_state_vm_v40 import MAX_INSTRUCTIONS, I32Program, VMOperation


def _expression(node, bindings):
    if isinstance(node, ast.Name):
        return bindings[node.id]
    if (isinstance(node, ast.Constant) and type(node.value) is int
            and 0 <= node.value < (1 << 32)):
        return z3.BitVecVal(node.value, 32)
    if isinstance(node, ast.BinOp):
        left = _expression(node.left, bindings)
        right = _expression(node.right, bindings)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.BitXor):
            return left ^ right
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _expression(node.left, bindings)
        right = _expression(node.comparators[0], bindings)
        match node.ops[0]:
            case ast.Eq():
                test = left == right
            case ast.Lt():
                test = z3.ULT(left, right)
            case ast.Gt():
                test = z3.UGT(left, right)
            case _:
                raise ValueError("Unexpected emitted comparison")
        return z3.If(test, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
    raise ValueError("Unexpected emitted i32 expression")


def proof(program: ControlProgram, cpp: str, *, timeout_ms: int):
    """Compare the warm entry's emitted integer expression against source i32 semantics."""
    original = I32Program(program.local_count, tuple(
        VMOperation(op.opcode, op.immediate)
        for op in program.operations if op.opcode != "nop"
    ))
    if len(original.operations) > MAX_INSTRUCTIONS:
        return {"status": "unsupported_source_solver_length"}
    x = z3.BitVec("input_i32", 32)
    bindings = {"input": x, "sign": z3.BitVecVal(1 << 31, 32)}
    lines = cpp.splitlines()
    if lines[:4] != [
        "#include <cstdint>",
        "using Word = std::uint32_t;",
        "constexpr Word sign = Word{1} << 31;",
        "template <bool Trace> Word run(Word input, Word* out) {",
    ] or lines[-3:] != [
        "}",
        'extern "C" Word visuaml_run(Word x) { return run<false>(x, nullptr); }',
        'extern "C" Word visuaml_trace(Word x, Word* out) { return run<true>(x, out); }',
    ]:
        raise ValueError("Unrecognized native wrapper")
    for line in lines[4:-4]:
        if re.fullmatch(r"  if constexpr \(Trace\) out\[\d+\] = v\d+;", line):
            continue
        assignment = re.fullmatch(r"  const Word (v\d+) = (.+);", line)
        if assignment is None or assignment[1] in bindings:
            raise ValueError("Unrecognized native function body")
        if not re.fullmatch(
            r"input|[0-9]+u|v\d+ [*+\-] v\d+|"
            r"\(v\d+ \^ sign\) [<>] \(v\d+ \^ sign\)|v\d+ == 0u",
            assignment[2],
        ):
            raise ValueError("Unrecognized unsigned native expression")
        normalized = re.sub(r"(\d+)u\b", r"\1", assignment[2])
        bindings[assignment[1]] = _expression(ast.parse(normalized, mode="eval").body, bindings)
    returned = re.fullmatch(r"  return (v\d+);", lines[-4])
    if returned is None:
        raise ValueError("Unrecognized native return")
    source = _encode_wasm_in_z3(original, x, z3)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(source != bindings[returned[1]])
    outcome = solver.check()
    if outcome == z3.sat:
        counterexample = solver.model().eval(x, model_completion=True).as_signed_long()
        raise AssertionError(f"Specialization changes result at input {counterexample}")
    if outcome == z3.unknown:
        return {"status": "unknown", "reason": solver.reason_unknown()}
    return {"status": "unsat"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if (not 1 <= args.timeout_ms <= 60000 or output.exists()
            or output.is_relative_to(root)):
        parser.error("Use a bounded solver timeout and a fresh output outside the checkout")
    source = git_identity(root)
    rows = []
    for work in workloads(scaling=True):
        if any(op.opcode not in DATA_OPS for op in work.program.operations[:-1]):
            continue
        cpp = _emit(*_lower(work.program))
        rows.append({
            "name": work.name,
            "wasm_sha256": hashlib.sha256(encode_control(work.program)).hexdigest(),
            "cpp_sha256": hashlib.sha256(cpp.encode()).hexdigest(),
            **proof(work.program, cpp, timeout_ms=args.timeout_ms),
        })
    summary = {status: sum(row["status"] == status for row in rows)
               for status in ("unsat", "unknown", "unsupported_source_solver_length")}
    negative_program = arithmetic_chain(2)
    bad_cpp, replacements = re.subn(
        r"^  return v\d+;$", "  return v0;", _emit(*_lower(negative_program)),
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise AssertionError("Could not corrupt the native result for the negative control")
    try:
        proof(negative_program, bad_cpp, timeout_ms=args.timeout_ms)
    except AssertionError as exc:
        negative_control = str(exc)
    else:
        raise AssertionError("Incorrect return value was not detected")

    record = {
        "negative_control": negative_control,
        "schema": "visuaml.native-specialization-translation-validation.v58",
        "source": source, "solver": z3.get_full_version(),
        "timeout_ms": args.timeout_ms, "summary": summary, "programs": rows,
        "scope": "Emitted i32 result expression versus typed source over the entire i32 domain. "
                 "No claim about compiled binary, trace states or constructed neural gates.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
