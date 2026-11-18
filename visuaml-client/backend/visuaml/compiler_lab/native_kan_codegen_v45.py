"""v45 closed-world KAN graph -> independent native C++17 double-precision backend.

This emitter consumes the SAME admitted typed KANProgram as NumPy/PyTorch and
contains no model-specific switch. No dynamic code strings, user-provided C++
identifiers, symbol imports, model execution or callbacks are allowed.
Compilation is local C++17, subprocess-bounded and uses a private temporary
directory. Native input/output buffers are copied and SHA-256 receipts emitted.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .typed_kan_v45 import (
    MAX_ROWS,
    KANAdmissionError,
    KANProgram,
    validate_kan,
)

HELPERS = r"""
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
static constexpr double PI=3.14159265358979323846264338327950288;
static inline double polynomial(double x, const double *c, int n) {
    double y=0.0;
    for (int i=n-1; i>=0; --i) y=y*x+c[i];
    return y;
}
static inline double chebyshev(double x, double lo, double hi,
                               const double *c, int n) {
    const double u=(2.0*x-(lo+hi))/(hi-lo);
    double t0=1.0, t1=u, result=c[0];
    if(n>1) result+=c[1]*t1;
    for(int i=2;i<n;++i) {
        const double next=2.0*u*t1-t0;
        t0=t1; t1=next;
        result+=c[i]*t1;
    }
    return result;
}
static inline double fourier(double x, double lo, double scale,
                             const double *c, int n) {
    const double theta=2.0*PI*(x-lo)/scale;
    double result=c[0];
    for(int k=1;k<=(n-1)/2;++k) {
        result+=c[2*k-1]*std::cos(k*theta) + c[2*k]*std::sin(k*theta);
    }
    return result;
}
static inline double gaussian(double x, double scale,
                              const double *c, const double *knots, int n) {
    double result=0.0;
    for(int i=0;i<n;++i) {
        const double u=(x-knots[i])/scale;
        result+=c[i]*std::exp(-u*u);
    }
    return result;
}
static inline double linear_spline(double x, const double *c,
                                    const double *knots, int n) {
    int k=0;
    while(k<n-2 && x>=knots[k+1]) ++k;
    const double t=(x-knots[k])/(knots[k+1]-knots[k]);
    return (1.0-t)*c[k]+t*c[k+1];
}
"""


def _literal(value: float) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise KANAdmissionError("Nonfinite KAN literal")
    encoded = repr(float(value))
    if encoded in {"nan", "inf", "-inf"}:
        raise KANAdmissionError("Unsafe KAN literal")
    return encoded


def emit_kan_cpp17(program: KANProgram) -> str:
    report = validate_kan(program)
    lines = [HELPERS, f"// Semantic SHA-256: {report['program_sha256']}",
             "extern \"C\" int visuaml_kan_eval(const double* input, std::size_t count, double* output) {"]
    lines += [
        "    if(input==nullptr || output==nullptr || count<1 || count>4096) return -1;",
        "    for(std::size_t row=0; row<count; ++row) {",
        "        double prev[8]={0.0};",
        "        double next[8]={0.0};",
    ]
    for i in range(program.input_dim):
        lines.append(f"        prev[{i}] = input[row*{program.input_dim}+{i}];")
        lines.append(f"        if(!std::isfinite(prev[{i}])) return -2;")
    edge_number = 0
    for stage_index, stage in enumerate(program.stages):
        for node_index, node in enumerate(stage):
            prefix = f"s{stage_index}n{node_index}"
            for index, edge in enumerate(node.edges):
                key = f"{prefix}e{index}"
                coeffs = ",".join(_literal(v) for v in edge.coefficients)
                lines.append(
                    f"        const double {key}_c[{len(edge.coefficients)}]={{ {coeffs} }};"
                )
                if edge.knots:
                    knots = ",".join(_literal(v) for v in edge.knots)
                    lines.append(
                        f"        const double {key}_k[{len(edge.knots)}]={{ {knots} }};"
                    )
                src = f"prev[{edge.source}]"
                lines.append(
                    f"        if(!std::isfinite({src}) || {src} < {_literal(edge.domain[0])}"
                    f" || {src} > {_literal(edge.domain[1])}) return -2;"
                )
                count = len(edge.coefficients)
                if edge.family == "power":
                    call = f"polynomial({src},{key}_c,{count})"
                elif edge.family == "chebyshev":
                    call = (f"chebyshev({src},{_literal(edge.domain[0])},"
                            f"{_literal(edge.domain[1])},{key}_c,{count})")
                elif edge.family == "fourier":
                    call = (f"fourier({src},{_literal(edge.domain[0])},"
                            f"{_literal(edge.scale)},{key}_c,{count})")
                elif edge.family == "gaussian":
                    call = (f"gaussian({src},{_literal(edge.scale)},"
                            f"{key}_c,{key}_k,{count})")
                elif edge.family == "linear_spline":
                    call = f"linear_spline({src},{key}_c,{key}_k,{count})"
                else:
                    raise KANAdmissionError("Unadmitted C++ KAN edge family")
                lines.append(f"        const double {key}={call};")
                lines.append(
                    f"        if(!std::isfinite({key}) || std::abs({key})>1e12) return -3;"
                )
                edge_number += 1
            if node.aggregation == "sum":
                expression = " + ".join(
                    [f"{prefix}e{k}" for k in range(len(node.edges))]
                )
                lines.append(
                    f"        next[{node_index}]={_literal(node.bias)}+({expression});"
                )
            else:
                expression = " * ".join(
                    [f"{prefix}e{k}" for k in range(len(node.edges))]
                )
                lines.append(
                    f"        next[{node_index}]={_literal(node.bias)}+({expression});"
                )
            lines.append(
                f"        if(!std::isfinite(next[{node_index}])"
                f" || std::abs(next[{node_index}])>1e12) return -3;"
            )
        for i in range(len(stage)):
            lines.append(f"        prev[{i}]=next[{i}];")
    output_dim = len(program.stages[-1])
    for k in range(output_dim):
        lines.append(f"        output[row*{output_dim}+{k}]=prev[{k}];")
    lines += ["    }", "    return 0;", "}"]
    result = "\n".join(lines) + "\n"
    if len(result) > 256000 or edge_number != report["edges"]:
        raise KANAdmissionError("Generated native KAN source budget exceeded")
    return result


def run_kan_cpp17(program: KANProgram, inputs: np.ndarray) -> tuple[np.ndarray, dict]:
    report = validate_kan(program)
    if (not isinstance(inputs, np.ndarray) or inputs.dtype != np.float64
            or inputs.ndim != 2 or inputs.shape[1] != program.input_dim
            or not 1 <= inputs.shape[0] <= MAX_ROWS or not np.isfinite(inputs).all()):
        raise KANAdmissionError("Invalid native KAN float64 ABI")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise KANAdmissionError("Native C++17 compiler unavailable")
    source = emit_kan_cpp17(program)
    with tempfile.TemporaryDirectory(prefix="visuaml-kan-v45-") as tmp:
        path = Path(tmp)
        cpp, library = path / "kan.cpp", path / "kan.so"
        cpp.write_text(source, encoding="utf-8")
        command = [
            compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
            "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
            str(cpp), "-o", str(library),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        binary_sha = hashlib.sha256(library.read_bytes()).hexdigest()
        native = ctypes.CDLL(str(library)).visuaml_kan_eval
        native.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        native.restype = ctypes.c_int
        copied = np.ascontiguousarray(inputs).copy()
        original_sha = hashlib.sha256(copied.tobytes()).hexdigest()
        result = np.empty(
            (len(copied), report["output_dim"]), dtype=np.float64
        )
        status = native(
            copied.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(len(copied)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise KANAdmissionError(f"Native KAN violated typed ABI, status={status}")
        if hashlib.sha256(copied.tobytes()).hexdigest() != original_sha:
            raise KANAdmissionError("Native KAN modified source input buffer")
        if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e12:
            raise KANAdmissionError("Native KAN returned unsafe output")
        info = {
            "schema": "visuaml.native-kan-cpp17-receipt.v45",
            "program_sha256": report["program_sha256"],
            "cpp_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "native_binary_sha256": binary_sha,
            "input_sha256": original_sha,
            "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            "output_shape": list(result.shape),
            "native_execution": True,
            "compiler": subprocess.run(
                [compiler, "--version"], check=True, capture_output=True,
                text=True, timeout=10,
            ).stdout.splitlines()[0],
            "numerical_parity_verified": False,
            "performance_speedup_proved": False,
        }
    return result, info
