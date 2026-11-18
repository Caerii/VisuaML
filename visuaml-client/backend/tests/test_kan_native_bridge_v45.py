"""v45 exact modular KAN -> typed WASM/C++17 native backend bridge."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANAdmissionError,
    KANNode,
    KANProgram,
    execute_kan_scalar,
)
from visuaml.compiler_lab.kan_native_bridge_v45 import lower_exact_polynomial_kan
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module, execute_program


def graph():
    return KANProgram(
        ("x",),
        (
            KANNode("product", "product", (
                EdgeFunction("left", "x", "polynomial", (0., 1.)),
                EdgeFunction("right", "x", "polynomial", (0., 1.)),
            )),
            KANNode("result", "sum", (
                EdgeFunction("quad", "product", "polynomial", (0., 1.)),
                EdgeFunction("linear", "x", "polynomial", (0., -3.)),
                EdgeFunction("constant", "x", "polynomial", (7.,)),
            )),
        ),
        ("result",), (-128., 127.),
    )


def test_kan_lowering_binds_original_graph_identity_to_real_wasm_and_cpp():
    source = graph()
    compiled = lower_exact_polynomial_kan(source)
    assert (compiled.symbolic.a, compiled.symbolic.b, compiled.symbolic.c) == (1, -3, 7)
    assert compiled.receipt["source_kan_graph_sha256"] == source.digest()
    assert compiled.receipt["checked_in_domain_integer_witnesses"] == 256
    assert not compiled.receipt["general_kan_compiler"]
    assert not compiled.receipt["all_i32_original_kan_equivalent"]
    assert not compiled.receipt["native_code_performance_measured"]
    assert decode_module(compiled.wasm_bytes).program == compiled.typed_wasm
    for x in range(-128, 128):
        assert execute_kan_scalar(source, {"x": x})["result"] == (
            execute_program(compiled.typed_wasm, x).value
        )


def test_independent_wasmtime_executes_graph_lowered_binary():
    wasmtime = pytest.importorskip("wasmtime")
    source = graph()
    compiler = lower_exact_polynomial_kan(source)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, compiler.wasm_bytes)
    store = wasmtime.Store(engine)
    binary = wasmtime.Module(engine, compiler.wasm_bytes)
    instance = wasmtime.Instance(store, binary, [])
    run = instance.exports(store)["run"]
    for x in (*range(-128, 128), -(1 << 31), (1 << 31) - 1, 65535):
        assert run(store, x) == execute_program(compiler.typed_wasm, x).value
        if -128 <= x <= 127:
            assert run(store, x) == execute_kan_scalar(source, {"x": x})["result"]


def test_compiled_cpp17_bridge_matches_wasm_without_signed_overflow():
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("No C++17 compiler available")
    program = lower_exact_polynomial_kan(graph())
    with tempfile.TemporaryDirectory(prefix="visuaml-kan-v45-") as directory:
        root = Path(directory)
        source = root / "kan_generated.cpp"
        binary = root / "kan_generated.so"
        source.write_text(program.cpp17_source, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall",
             "-Wextra", "-Werror", str(source), "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=45,
        )
        fn = ctypes.CDLL(str(binary)).visuaml_recovered_i32
        fn.argtypes = [ctypes.c_int32]
        fn.restype = ctypes.c_int32
        sample = np.random.default_rng(45).integers(
            -(1 << 31), (1 << 31), size=100, dtype=np.int64
        )
        for x in (*range(-128, 128), *[int(v) for v in sample]):
            assert fn(x) == execute_program(program.typed_wasm, x).value


def test_nonintegral_spline_or_wrong_domain_refused_by_native_bridge():
    original = graph()
    modified = replace(original, domain=(-128., 128.))
    with pytest.raises(KANAdmissionError, match="domain"):
        lower_exact_polynomial_kan(modified)
    changed = replace(original, nodes=(
        original.nodes[0],
        replace(original.nodes[1], edges=(
            original.nodes[1].edges[0],
            replace(original.nodes[1].edges[1], coefficients=(0., -2.5)),
            original.nodes[1].edges[2],
        )),
    ))
    with pytest.raises(KANAdmissionError, match="integral"):
        lower_exact_polynomial_kan(changed)
    spline = replace(original, nodes=(
        original.nodes[0],
        replace(original.nodes[1], edges=(
            original.nodes[1].edges[0],
            EdgeFunction("linear", "x", "linear_spline",
                         (0., 1.), (-128., 127.)),
            original.nodes[1].edges[2],
        )),
    ))
    with pytest.raises(KANAdmissionError, match="Nonpolynomial"):
        lower_exact_polynomial_kan(spline)
