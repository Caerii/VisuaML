"""Compile canonical straight-line v58 WASM to a program-specific CPU function.

The source decoder and resource contract are shared with the v58 reference.
The native function executes specialized i32 semantics. It does not execute
or certify constructed neural gates. A diagnostic entry point exposes the
same SSA values at every original source instruction boundary.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from .bit_neural_vm_v49 import BitVMAdmissionError
from .neural_microstep_executor_v58 import MicrostepAdmissionError
from .neural_microstep_v58 import DATA_OPS
from .wasm_control_flow_v51 import (
    ControlProgram,
    MAX_FUEL,
    MAX_STACK,
    ControlExecutionTrap,
    ControlFuelExhausted,
    decode_control,
)

_MASK = (1 << 32) - 1
_BINARY_EXPRESSIONS = {
    "i32.add": "v{left} + v{right}",
    "i32.sub": "v{left} - v{right}",
    "i32.mul": "v{left} * v{right}",
    # Toggling the sign bit converts signed order into unsigned order.
    "i32.lt_s": "(v{left} ^ sign) < (v{right} ^ sign)",
    "i32.gt_s": "(v{left} ^ sign) > (v{right} ^ sign)",
}


def _lower(program: ControlProgram):
    """Resolve operand stack and local-register flow into SSA references."""
    definitions = ["input", "0u"]
    locals_ = [0] + [1] * program.local_count
    stack = []
    states = []
    for op in program.operations:
        kind = op.opcode
        if kind != "end" and kind not in DATA_OPS:
            raise MicrostepAdmissionError("Native specialization requires straight-line WASM")
        required = 2 if kind in _BINARY_EXPRESSIONS else int(
            kind in ("local.set", "local.tee", "drop", "i32.eqz")
        )
        if len(stack) < required:
            raise ControlExecutionTrap("Native specialization found stack underflow")
        if kind == "local.get":
            stack.append(locals_[op.immediate])
        elif kind == "local.set":
            locals_[op.immediate] = stack.pop()
        elif kind == "local.tee":
            locals_[op.immediate] = stack[-1]
        elif kind == "drop":
            stack.pop()
        elif kind == "i32.const":
            stack.append(len(definitions))
            definitions.append(f"{op.immediate & _MASK}u")
        elif kind == "i32.eqz":
            operand = stack.pop()
            stack.append(len(definitions))
            definitions.append(f"v{operand} == 0u")
        elif kind in _BINARY_EXPRESSIONS:
            right, left = stack.pop(), stack.pop()
            stack.append(len(definitions))
            definitions.append(_BINARY_EXPRESSIONS[kind].format(left=left, right=right))
        if len(stack) > MAX_STACK:
            raise ControlExecutionTrap("Native specialization found stack overflow")
        states.append((tuple(stack), tuple(locals_)))
    if len(stack) != 1:
        raise ControlExecutionTrap("Final source WASM end needs exactly one i32")
    return definitions, states


def _emit(definitions, states):
    lines = [
        "#include <cstdint>",
        "using Word = std::uint32_t;",
        "constexpr Word sign = Word{1} << 31;",
        "template <bool Trace> Word run(Word input, Word* out) {",
    ]
    emitted = 0
    offset = 0
    for stack, locals_ in states:
        live = (*stack, *locals_)
        last = max(live)
        for index in range(emitted, last + 1):
            lines.append(f"  const Word v{index} = {definitions[index]};")
        emitted = max(emitted, last + 1)
        for index in live:
            lines.append(f"  if constexpr (Trace) out[{offset}] = v{index};")
            offset += 1
    lines.extend((
        f"  return v{states[-1][0][0]};",
        "}",
        'extern "C" Word visuaml_run(Word x) { return run<false>(x, nullptr); }',
        'extern "C" Word visuaml_trace(Word x, Word* out) { return run<true>(x, out); }',
    ))
    return "\n".join(lines) + "\n"


def _signed(word):
    return ctypes.c_int32(word).value


class NativeSpecialization:
    """Process-local compiled program. Cold construction requires a C++17 compiler."""

    def __init__(self, source: bytes):
        program = decode_control(source)
        definitions, states = _lower(program)
        cpp = _emit(definitions, states).encode()
        compiler = next(
            (path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))),
            None,
        )
        if compiler is None:
            raise RuntimeError("Native specialization requires a C++17 compiler")
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.cpp_sha256 = hashlib.sha256(cpp).hexdigest()
        self.compiler = subprocess.check_output(
            [compiler, "--version"], text=True, timeout=10,
        ).splitlines()[0]
        with tempfile.TemporaryDirectory(prefix="visuaml-v58-native-") as directory:
            cpp_path = Path(directory) / "program.cpp"
            binary = Path(directory) / "program.so"
            cpp_path.write_bytes(cpp)
            result = subprocess.run(
                [compiler, "-std=c++17", "-O3", "-fPIC", "-shared", str(cpp_path),
                 "-o", str(binary)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode:
                raise RuntimeError(f"Native specialization compilation failed: {result.stderr}")
            self.binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
            self._library = ctypes.CDLL(str(binary))
        self._run = self._library.visuaml_run
        self._run.argtypes = (ctypes.c_uint32,)
        self._run.restype = ctypes.c_uint32
        self._trace = self._library.visuaml_trace
        self._trace.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
        self._trace.restype = ctypes.c_uint32
        self._opcodes = tuple(op.opcode for op in program.operations)
        self._stack_sizes = tuple(len(stack) for stack, _ in states)
        self._local_size = program.local_count + 1
        self._buffer_type = ctypes.c_uint32 * (
            sum(self._stack_sizes) + len(self._opcodes) * self._local_size
        )

    def _validate_call(self, argument, fuel):
        if type(fuel) is not int or not 1 <= fuel <= MAX_FUEL:
            raise MicrostepAdmissionError("Invalid v58 execution fuel")
        if type(argument) is not int or not -(1 << 31) <= argument < (1 << 31):
            raise BitVMAdmissionError("Expected signed i32 external value")
        if fuel < len(self._opcodes):
            raise ControlFuelExhausted("Insufficient fuel for the straight-line program")

    def run(self, argument: int, *, fuel: int = 400) -> int:
        self._validate_call(argument, fuel)
        return _signed(self._run(argument & _MASK))

    def trace(self, argument: int, *, fuel: int = 400) -> tuple[dict, ...]:
        self._validate_call(argument, fuel)
        buffer = self._buffer_type()
        self._trace(argument & _MASK, buffer)
        records = []
        offset = 0
        for pc, (op, depth) in enumerate(zip(self._opcodes, self._stack_sizes, strict=True)):
            local_offset = offset + depth
            end = local_offset + self._local_size
            records.append({
                "step": pc,
                "pc": pc,
                "opcode": op,
                "next_pc": pc + 1,
                "stack_i32": [_signed(word) for word in buffer[offset:local_offset]],
                "locals_i32": [_signed(word) for word in buffer[local_offset:end]],
                "control_labels": [],
                "halted": op == "end",
            })
            offset = end
        return tuple(records)
