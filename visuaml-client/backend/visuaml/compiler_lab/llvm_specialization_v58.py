"""In-process LLVM specialization of the bounded v58 scalar/batch contract."""
from __future__ import annotations

import ctypes
import hashlib
from functools import lru_cache
from threading import Lock

from .native_specialization_v58 import _WORD_POINTER, _BatchSpecialization, _lower_ssa
from .wasm_control_flow_v51 import decode_control

_OPERATORS = {
    "i32.add": "add",
    "i32.sub": "sub",
    "i32.mul": "mul",
    "i32.lt_s": "icmp slt",
    "i32.gt_s": "icmp sgt",
}
_COMPILE_LOCK = Lock()


@lru_cache(maxsize=1)
def _llvm():
    from llvmlite import binding

    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    return binding


def _body(definitions, result):
    values = []
    lines = []
    for index, (opcode, args) in enumerate(definitions):
        if opcode == "input":
            values.append("%input")
            continue
        if opcode == "i32.const":
            values.append(str(args[0]))
            continue
        value = f"%v{index}"
        values.append(value)
        operands = [values[arg] for arg in args]
        if opcode == "i32.eqz":
            lines.append(f"%flag{index} = icmp eq i32 {operands[0]}, 0")
            lines.append(f"{value} = zext i1 %flag{index} to i32")
        else:
            op = _OPERATORS[opcode]
            dest = f"%flag{index}" if opcode in ("i32.lt_s", "i32.gt_s") else value
            lines.append(f"{dest} = {op} i32 {operands[0]}, {operands[1]}")
            if dest != value:
                lines.append(f"{value} = zext i1 {dest} to i32")
    return "\n  ".join(lines), values[result]


def _module(definitions, result, target, layout):
    body, value = _body(definitions, result)
    return f'''target triple = "{target}"
target datalayout = "{layout}"

define i32 @visuaml_run(i32 %input) {{
  {body}
  ret i32 {value}
}}

define void @visuaml_batch(ptr %buffer, i32 %count) {{
entry:
  %length = zext i32 %count to i64
  br label %test
test:
  %index = phi i64 [ 0, %entry ], [ %next, %loop ]
  %done = icmp eq i64 %index, %length
  br i1 %done, label %end, label %loop
loop:
  %element = getelementptr i32, ptr %buffer, i64 %index
  %input = load i32, ptr %element, align 4
  {body}
  store i32 {value}, ptr %element, align 4
  %next = add i64 %index, 1
  br label %test
end:
  ret void
}}
'''


class LlvmSpecialization(_BatchSpecialization):
    """Process-local JIT for scalar and batch result execution via llvmlite."""

    def __init__(self, source: bytes, *, optimization_level: int = 2):
        if type(optimization_level) is not int or optimization_level not in (0, 1, 2):
            raise ValueError("LLVM optimization level must be 0, 1, or 2")
        self.optimization_level = optimization_level
        program = decode_control(source)
        definitions, states = _lower_ssa(program)
        self._opcodes = tuple(op.opcode for op in program.operations)
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        with _COMPILE_LOCK:
            llvm = _llvm()
            target = llvm.Target.from_default_triple()
            machine = target.create_target_machine(opt=optimization_level)
            ir = _module(definitions, states[-1][0][0], target.triple, str(machine.target_data))
            self.ir_sha256 = hashlib.sha256(ir.encode()).hexdigest()
            self.llvm_version = llvm.llvm_version_info
            module = llvm.parse_assembly(ir)
            module.verify()
            with llvm.PipelineTuningOptions(speed_level=optimization_level) as options:
                with llvm.create_pass_builder(machine, options) as passes:
                    with passes.getModulePassManager() as manager:
                        manager.run(module, passes)
            module.verify()
            self._engine = llvm.create_mcjit_compiler(module, machine)
            self._engine.finalize_object()
            run = self._engine.get_function_address("visuaml_run")
            batch = self._engine.get_function_address("visuaml_batch")
            if not run or not batch:
                raise RuntimeError("LLVM specialization is missing a compiled entry point")
        self._run = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32)(run)
        self._batch = ctypes.CFUNCTYPE(None, _WORD_POINTER, ctypes.c_uint32)(batch)
