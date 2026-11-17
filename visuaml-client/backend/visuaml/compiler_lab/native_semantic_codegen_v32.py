"""v32 closed-world pure SemanticProgram -> generated C++17 CPU execution.

This is the first *graph-generated* native path, unlike the fixed Llama decoder.
Admits bounded static float32 SSA graphs with add/relu/matmul/conv2d only.
Weights are explicit runtime inputs. No state effects, dynamic shapes, input
alias mutation, user-supplied C++, arbitrary Python, or portable signed AOT.
The native library is process-owned and is NOT an isolation sandbox.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .semantic_ir import (
    SemanticProgram,
    ValueType,
    validate_semantic_program,
)

SCHEMA = "visuaml.semantic-native-cpu.v32"
_MAX_INPUTS = 16
_MAX_OPS = 64
_MAX_ELEMENTS = 65_536
_MAX_BYTES = 8 * 1024 * 1024
_MAX_MULTIPLIES = 50_000_000
_ALLOWED = frozenset({"matmul", "add", "relu", "conv2d"})
_PTRS = ctypes.POINTER(ctypes.c_void_p)


class NativeIRCodegenError(ValueError):
    """Unadmitted graph, unsafe execution argument or compiler failure."""


@dataclass(frozen=True)
class _Instruction:
    code: str
    sources: tuple[int, ...]
    result_shape: tuple[int, ...]
    parameters: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Plan:
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    instructions: tuple[_Instruction, ...]
    output_index: int
    output_shape: tuple[int, ...]
    retained_bytes_bound: int
    multiply_bound: int


def _concrete(spec: ValueType) -> tuple[int, ...]:
    if not isinstance(spec, ValueType) or spec.dtype != "float32":
        raise NativeIRCodegenError("Native v32 requires float32 ValueType")
    shape = spec.shape
    if not 1 <= len(shape) <= 4 or any(
        type(d) is not int or not 1 <= d <= 128 for d in shape
    ):
        raise NativeIRCodegenError("Native v32 requires bounded concrete dimensions")
    elements = math.prod(shape)
    if elements > _MAX_ELEMENTS:
        raise NativeIRCodegenError("Tensor element budget exceeded")
    if spec.bytes is not None and (
        type(spec.bytes) is not int or spec.bytes != elements * 4
    ):
        raise NativeIRCodegenError("Declared tensor byte count mismatch")
    return shape


def _admit(program: SemanticProgram) -> _Plan:
    if not isinstance(program, SemanticProgram):
        raise NativeIRCodegenError("Expected SemanticProgram")
    try:
        validate_semantic_program(program)
    except ValueError as exc:
        raise NativeIRCodegenError("Invalid semantic envelope") from exc
    if (
        program.origin != "torch-fx"
        or program.state_types
        or len(program.regions) != 1
        or program.regions[0].repetitions != 1
        or not 1 <= len(program.input_types) <= _MAX_INPUTS
        or not 1 <= len(program.regions[0].operations) <= _MAX_OPS
        or len(program.outputs) != 1
    ):
        raise NativeIRCodegenError(
            "v32 admits one pure, single-output, static tensor region"
        )
    names: dict[str, int] = {}
    shapes: list[tuple[int, ...]] = []
    op_names: set[str] = set()
    retained = 0
    for name, spec in program.input_types:
        if name in names:
            raise NativeIRCodegenError("Duplicate native input")
        names[name] = len(shapes)
        shape = _concrete(spec)
        shapes.append(shape)
        retained += 4 * math.prod(shape)
    instructions: list[_Instruction] = []
    multiplied = 0
    for op in program.regions[0].operations:
        if (
            op.name in op_names
            or op.dialect != "tensor"
            or op.effect != "pure"
            or op.state_cell is not None
            or op.revision is not None
            or op.token_in is not None
            or op.token_out is not None
            or op.op not in _ALLOWED
            or op.result is None
            or op.result in names
        ):
            raise NativeIRCodegenError("Unsupported native operator/effect/SSA")
        op_names.add(op.name)
        if not isinstance(op.operands, tuple):
            raise NativeIRCodegenError("Malformed native operands")
        try:
            source_indices = tuple(names[name] for name in op.operands)
        except KeyError as exc:
            raise NativeIRCodegenError("Unknown native SSA operand") from exc
        input_shapes = tuple(shapes[i] for i in source_indices)
        out_shape = _concrete(op.result_type)
        if op.op == "matmul":
            if (
                len(input_shapes) != 2
                or len(input_shapes[0]) != 2
                or len(input_shapes[1]) != 2
                or input_shapes[0][1] != input_shapes[1][0]
                or out_shape != (input_shapes[0][0], input_shapes[1][1])
                or op.parameters
            ):
                raise NativeIRCodegenError("Invalid rank-2 matmul contract")
            m, k = input_shapes[0]
            n = input_shapes[1][1]
            multiplied += m * k * n
            params: tuple[int, ...] = ()
        elif op.op in ("add", "relu"):
            expected_arity = 2 if op.op == "add" else 1
            if (
                len(input_shapes) != expected_arity
                or any(shape != out_shape for shape in input_shapes)
                or op.parameters
            ):
                raise NativeIRCodegenError("Unsupported elementwise shape/broadcast")
            params = ()
        else:
            if (
                len(input_shapes) != 2
                or len(input_shapes[0]) != 4
                or len(input_shapes[1]) != 4
            ):
                raise NativeIRCodegenError("Convolution requires NCHW and OIHW")
            if (
                not isinstance(op.parameters, tuple)
                or any(
                    not isinstance(p, tuple) or len(p) != 2
                    or not isinstance(p[0], str) or type(p[1]) is not int
                    for p in op.parameters
                )
            ):
                raise NativeIRCodegenError("Invalid convolution parameters")
            opts = dict(op.parameters)
            if (
                len(op.parameters) != 2
                or set(opts) != {"stride", "padding"}
                or not 1 <= opts["stride"] <= 4
                or not 0 <= opts["padding"] <= 3
            ):
                raise NativeIRCodegenError("Unsupported convolution geometry")
            batch, in_channels, height, width = input_shapes[0]
            out_channels, weight_channels, kh, kw = input_shapes[1]
            stride, padding = opts["stride"], opts["padding"]
            oh = (height + 2 * padding - kh) // stride + 1
            ow = (width + 2 * padding - kw) // stride + 1
            if (
                in_channels != weight_channels
                or oh <= 0 or ow <= 0
                or out_shape != (batch, out_channels, oh, ow)
            ):
                raise NativeIRCodegenError("Convolution result geometry mismatch")
            multiplied += batch * out_channels * oh * ow * in_channels * kh * kw
            params = (stride, padding)
        retained += 4 * math.prod(out_shape)
        if retained > _MAX_BYTES or multiplied > _MAX_MULTIPLIES:
            raise NativeIRCodegenError("Native retained memory/work budget exceeded")
        instructions.append(_Instruction(op.op, source_indices, out_shape, params))
        names[op.result] = len(shapes)
        shapes.append(out_shape)
    if program.outputs[0] not in names:
        raise NativeIRCodegenError("Missing native output producer")
    result_index = names[program.outputs[0]]
    if result_index < len(program.input_types):
        raise NativeIRCodegenError("Native output must be computed")
    return _Plan(
        tuple(name for name, _ in program.input_types),
        tuple(shapes[:len(program.input_types)]),
        tuple(instructions),
        result_index,
        shapes[result_index],
        retained,
        multiplied,
    )


def _source(plan: _Plan) -> str:
    """Only validated shapes/opcodes appear in code; user names are not emitted."""
    lines = [
        "// Deterministically generated v32 Semantic IR CPU source.",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <vector>",
        'extern "C" int visuaml_semantic_run(const float* const* inputs, float* output) noexcept {',
        "  try {",
        "    if (!inputs || !output) return -1;",
    ]
    for index in range(len(plan.input_names)):
        lines += [
            f"    if (!inputs[{index}]) return -1;",
            f"    const float* v{index} = inputs[{index}];",
        ]
    base = len(plan.input_names)
    for op_index, op in enumerate(plan.instructions):
        slot = base + op_index
        elements = math.prod(op.result_shape)
        lines += [
            f"    std::vector<float> storage_{slot}({elements});",
            f"    float* v{slot} = storage_{slot}.data();",
        ]
        a = f"v{op.sources[0]}"
        if op.code == "relu":
            lines.append(
                f"    for (size_t i=0; i<{elements}; ++i) v{slot}[i] = std::max(0.0f, {a}[i]);"
            )
        elif op.code == "add":
            b = f"v{op.sources[1]}"
            lines.append(
                f"    for (size_t i=0; i<{elements}; ++i) v{slot}[i] = {a}[i] + {b}[i];"
            )
        elif op.code == "matmul":
            m, k = _shape_for(plan, op.sources[0])
            _, n = _shape_for(plan, op.sources[1])
            b = f"v{op.sources[1]}"
            lines += [
                f"    for (int i=0; i<{m}; ++i) for (int j=0; j<{n}; ++j) {{",
                "      float sum = 0.0f;",
                f"      for (int k=0; k<{k}; ++k) sum += {a}[i*{k}+k] * {b}[k*{n}+j];",
                f"      v{slot}[i*{n}+j] = sum;",
                "    }",
            ]
        else:
            batch, channels, height, width = _shape_for(plan, op.sources[0])
            out_channels, _, kh, kw = _shape_for(plan, op.sources[1])
            _, _, oh, ow = op.result_shape
            stride, pad = op.parameters
            b = f"v{op.sources[1]}"
            lines += [
                f"    for (int n=0; n<{batch}; ++n)",
                f"      for (int o=0; o<{out_channels}; ++o)",
                f"        for (int y=0; y<{oh}; ++y)",
                f"          for (int x=0; x<{ow}; ++x) {{",
                "            float sum = 0.0f;",
                f"            for (int c=0; c<{channels}; ++c)",
                f"              for (int ky=0; ky<{kh}; ++ky)",
                f"                for (int kx=0; kx<{kw}; ++kx) {{",
                f"                  const int iy = y*{stride} + ky - {pad};",
                f"                  const int ix = x*{stride} + kx - {pad};",
                f"                  if (iy>=0 && iy<{height} && ix>=0 && ix<{width})",
                f"                    sum += {a}[((n*{channels}+c)*{height}+iy)*{width}+ix]",
                f"                           * {b}[((o*{channels}+c)*{kh}+ky)*{kw}+kx];",
                "                }",
                f"            v{slot}[((n*{out_channels}+o)*{oh}+y)*{ow}+x] = sum;",
                "          }",
            ]
        lines.append(
            f"    for (size_t i=0; i<{elements}; ++i) if (!std::isfinite(v{slot}[i])) return -3;"
        )
    lines += [
        f"    std::copy(v{plan.output_index}, v{plan.output_index}+{math.prod(plan.output_shape)}, output);",
        "    return 0;",
        "  } catch (...) { return -99; }",
        "}",
        "",
    ]
    return "\n".join(lines)


def _shape_for(plan: _Plan, index: int) -> tuple[int, ...]:
    if index < len(plan.input_shapes):
        return plan.input_shapes[index]
    return plan.instructions[index - len(plan.input_shapes)].result_shape


class NativeSemanticCompiler:
    """Compile admitted pure Semantic IR into a process-owned native library."""

    def __init__(self, program: SemanticProgram):
        self.plan = _admit(program)
        self.semantic_sha256 = program.digest()
        self.source = _source(self.plan)
        source_bytes = self.source.encode("utf-8")
        self.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        system = platform.system()
        if system not in ("Linux", "Darwin"):
            raise NativeIRCodegenError("Initial native codegen supports Linux/macOS")
        compiler = next(
            (shutil.which(name) for name in ("c++", "g++", "clang++")
             if shutil.which(name)), None
        )
        if compiler is None:
            raise NativeIRCodegenError("No trusted installed C++17 compiler")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="visuaml-semantic-native-v32-"
        )
        root = Path(self._temporary.name)
        self.source_path = root / "program.cpp"
        binary = root / (
            "libprogram.dylib" if system == "Darwin" else "libprogram.so"
        )
        self.source_path.write_bytes(source_bytes)
        self.flags = ("-std=c++17", "-O2", "-fno-fast-math", "-fPIC", "-shared")
        start_ns = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                [compiler, *self.flags, str(self.source_path), "-o", str(binary)],
                capture_output=True, text=True, timeout=90, check=False,
            )
            identity = subprocess.run(
                [compiler, "--version"], capture_output=True, text=True,
                timeout=10, check=True,
            ).stdout.splitlines()[0][:160]
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            raise NativeIRCodegenError("Native C++17 toolchain invocation failed") from exc
        if completed.returncode != 0 or not binary.is_file():
            raise NativeIRCodegenError(
                "Generated native source failed compilation: "
                + completed.stderr[:900]
            )
        self.compilation_ms = round((time.perf_counter_ns()-start_ns)/1e6, 6)
        self.binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        self.compiler_identity = identity
        try:
            self._library = ctypes.CDLL(str(binary))
        except OSError as exc:
            raise NativeIRCodegenError("Generated library could not be loaded") from exc
        self._execute = self._library.visuaml_semantic_run
        self._execute.restype = ctypes.c_int
        self._execute.argtypes = [_PTRS, ctypes.POINTER(ctypes.c_float)]
        self.source_program_sha256 = program.source_sha256

    def report(self) -> dict:
        return {
            "schema": SCHEMA,
            "semantic_sha256": self.semantic_sha256,
            "source_program_sha256": self.source_program_sha256,
            "generated_cpp_sha256": self.source_sha256,
            "native_binary_sha256": self.binary_sha256,
            "compiler_identity": self.compiler_identity,
            "compiler_flags": list(self.flags),
            "compilation_ms": self.compilation_ms,
            "target": platform.machine(),
            "inputs": [
                {"name": name, "shape": list(shape), "dtype": "float32"}
                for name, shape in zip(
                    self.plan.input_names, self.plan.input_shapes, strict=True
                )
            ],
            "output_shape": list(self.plan.output_shape),
            "retained_bytes_upper_bound": self.plan.retained_bytes_bound,
            "scalar_multiply_upper_bound": self.plan.multiply_bound,
            "op_sequence": [op.code for op in self.plan.instructions],
            "limitations": (
                "pure, static float32 graphs only; no external native sandbox, "
                "dynamic shape, state effect, GPU or portable AOT package"
            ),
        }

    def _inputs(self, arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
        if set(arrays) != set(self.plan.input_names):
            raise NativeIRCodegenError("Exact named native input set required")
        accepted = []
        for name, shape in zip(
            self.plan.input_names, self.plan.input_shapes, strict=True
        ):
            arr = arrays[name]
            if (
                not isinstance(arr, np.ndarray)
                or arr.dtype != np.float32
                or arr.shape != shape
                or not np.isfinite(arr).all()
            ):
                raise NativeIRCodegenError("Invalid native input dtype/shape/values")
            accepted.append(np.ascontiguousarray(arr))
        return accepted

    def run(self, **arrays: np.ndarray) -> np.ndarray:
        accepted = self._inputs(arrays)
        ptrs = (ctypes.c_void_p * len(accepted))(
            *(value.ctypes.data for value in accepted)
        )
        output = np.empty(self.plan.output_shape, dtype=np.float32)
        status = self._execute(
            ptrs, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if status != 0 or not np.isfinite(output).all():
            raise NativeIRCodegenError(
                f"Native semantic kernel failed without publishing output: {status}"
            )
        return output

    def run_with_receipt(
        self, **arrays: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        accepted = self._inputs(arrays)
        result = self.run(**dict(zip(self.plan.input_names, accepted, strict=True)))
        receipt = {
            **self.report(),
            "input_sha256": {
                name: hashlib.sha256(array.tobytes(order="C")).hexdigest()
                for name, array in zip(
                    self.plan.input_names, accepted, strict=True
                )
            },
            "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        }
        return result, receipt
