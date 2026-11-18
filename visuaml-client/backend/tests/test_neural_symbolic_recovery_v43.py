"""v43: recover and verify symbolic rules without source-program hints.

All neural extraction tests use detached weight tensors. The independent Z3
solver and Wasmtime/C++ executions are separate evidence classes.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    AnalyticRelu,
    NeuralSpecializationError,
    compile_bounded_program,
    decompile_function_from_weights,
    permute_hidden_neurons,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    RecoveredPolynomial,
    SymbolicRecoveryError,
    check_all_i32_equivalence,
    compile_polynomial_cpp17,
    compile_polynomial_wasm,
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    decode_module,
    execute_program,
)


def op(name: str, value: int | None = None) -> VMOperation:
    return VMOperation(name, value)


def quadratic() -> I32Program:
    # y = x*x - 3*x + 7 in unsigned-wrapping i32 arithmetic.
    return I32Program(0, (
        op("local.get", 0), op("local.get", 0), op("i32.mul"),
        op("local.get", 0), op("i32.const", 3), op("i32.mul"),
        op("i32.sub"), op("i32.const", 7), op("i32.add"), op("end"),
    ))


def detached(source: I32Program) -> AnalyticRelu:
    net, _ = compile_bounded_program(source)
    # Deliberately discard the source program digest/receipt before extraction.
    return AnalyticRelu(
        net.w_in.copy(), net.b_in.copy(), net.w_out.copy(), net.b_out.copy()
    )


def signed(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def test_extraction_derives_quadratic_from_only_neural_tensors():
    source = quadratic()
    weights = detached(source)
    poly = recover_quadratic_from_weights(weights)
    assert (poly.a, poly.b, poly.c, poly.degree) == (1, -3, 7, 2)
    assert poly.expression() == "(1)*x*x + (-3)*x + (7)"
    assert poly.inference_points == (-128, -127, -126)
    assert poly.domain == (-128, 127)
    assert not poly.source_program_read
    assert poly.recovered_function_sha256 == decompile_function_from_weights(
        weights
    ).function_sha256
    assert len(poly.digest()) == 64
    for x in range(-128, 128):
        assert x * x - 3 * x + 7 == execute_program(source, x).value


def test_manifest_free_decoder_is_permutation_invariant():
    net = detached(quadratic())
    perm = tuple(int(v) for v in np.random.default_rng(43).permutation(255))
    scrambled = permute_hidden_neurons(net, perm)
    scrambled = AnalyticRelu(
        scrambled.w_in, scrambled.b_in, scrambled.w_out, scrambled.b_out
    )
    assert not np.array_equal(scrambled.b_in, net.b_in)
    baseline = recover_quadratic_from_weights(net)
    after = recover_quadratic_from_weights(scrambled)
    assert baseline == after
    assert hashlib.sha256(scrambled.b_in.tobytes()).hexdigest() != hashlib.sha256(
        net.b_in.tobytes()
    ).hexdigest()


def test_late_nonpolynomial_perturbation_cannot_fool_three_point_extraction():
    net = detached(quadratic())
    weights = net.w_out.copy()
    # Hinge at x=126 changes ONLY x=127 in the finite integer domain.
    last_hinge = int(np.flatnonzero(net.b_in == -126)[0])
    weights[0, last_hinge] += np.float64(1)
    corrupted = AnalyticRelu(net.w_in, net.b_in, weights, net.b_out)
    assert decompile_function_from_weights(corrupted).outputs[:255] == (
        decompile_function_from_weights(net).outputs[:255]
    )
    with pytest.raises(SymbolicRecoveryError, match="falsified.*127"):
        recover_quadratic_from_weights(corrupted)


def test_constant_and_affine_mechanisms_are_also_recoverable():
    cases = (
        (I32Program(0, (op("i32.const", 17), op("end"))), (0, 0, 17, 0)),
        (I32Program(0, (
            op("local.get", 0), op("i32.const", 5),
            op("i32.mul"), op("i32.const", -7),
            op("i32.add"), op("end"),
        )), (0, 5, -7, 1)),
    )
    for source, expected in cases:
        net = detached(source)
        recovered = recover_quadratic_from_weights(net)
        assert (recovered.a, recovered.b, recovered.c, recovered.degree) == expected
        wasm, raw = compile_polynomial_wasm(recovered)
        assert decode_module(raw).program == wasm
        for x in range(-128, 128):
            assert execute_program(source, x).value == execute_program(wasm, x).value


def test_coefficients_and_unsupported_neural_layouts_fail_closed():
    net = detached(quadratic())
    with pytest.raises(NeuralSpecializationError):
        recover_quadratic_from_weights(AnalyticRelu(
            net.w_in, net.b_in, net.w_out.astype(np.float32), net.b_out
        ))
    p = recover_quadratic_from_weights(net)
    for invalid in (
        replace(p, a=1 << 31),
        replace(p, b=True),
        replace(p, degree=0),
        replace(p, source_program_read=True),
        replace(p, domain=(-128, 128)),
    ):
        with pytest.raises(SymbolicRecoveryError):
            compile_polynomial_wasm(invalid)


def test_real_wasmtime_executes_recompiled_symbolic_wasm_and_matches_source():
    wasmtime = pytest.importorskip("wasmtime")
    from visuaml.compiler_lab.wasm_state_vm_v40 import encode_program

    original = quadratic()
    poly = recover_quadratic_from_weights(detached(original))
    candidate, candidate_bytes = compile_polynomial_wasm(poly)
    original_bytes = encode_program(original)
    assert original_bytes != candidate_bytes
    assert decode_module(candidate_bytes).program == candidate
    engine = wasmtime.Engine()
    runners = []
    for source in (original_bytes, candidate_bytes):
        wasmtime.Module.validate(engine, source)
        store = wasmtime.Store(engine)
        module = wasmtime.Module(engine, source)
        fn = wasmtime.Instance(store, module, []).exports(store)["run"]
        runners.append((store, fn))
    inputs = (
        *range(-128, 128), -(1 << 31), (1 << 31) - 1,
        -65536, 65535, 1000003,
    )
    for value in inputs:
        observed_original = runners[0][1](runners[0][0], value)
        observed_candidate = runners[1][1](runners[1][0], value)
        assert observed_original == observed_candidate
        assert observed_original == execute_program(original, value).value
        assert observed_candidate == execute_program(candidate, value).value


def test_z3_proves_all_i32_symbolic_equivalence_and_falsifies_wrong_rule():
    pytest.importorskip("z3")
    net = detached(quadratic())
    poly = recover_quadratic_from_weights(net)
    good = check_all_i32_equivalence(quadratic(), poly)
    assert good["solver_outcome"] == "unsat"
    assert good["all_i32_source_vs_symbolic_equivalent"] is True
    assert good["neural_all_i32_equivalent"] is False
    assert good["machine_checked_proof_object"] is False
    wrong = replace(poly, b=-2)
    bad = check_all_i32_equivalence(quadratic(), wrong)
    assert bad["solver_outcome"] == "sat"
    assert bad["all_i32_source_vs_symbolic_equivalent"] is False
    x = bad["counterexample_input_i32"]
    assert signed(x * x - 3 * x + 7) != signed(x * x - 2 * x + 7)
    assert bad["source_output_u32"] != bad["candidate_output_u32"]


def test_native_cpp17_regeneration_matches_independent_typed_wasm():
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No C++17 compiler available")
    poly = recover_quadratic_from_weights(detached(quadratic()))
    code = compile_polynomial_cpp17(poly)
    assert "std::uint32_t" in code and "std::memcpy" in code
    assert "extern \"C\"" in code
    with tempfile.TemporaryDirectory(prefix="visuaml-symbolic-v43-") as folder:
        root = Path(folder)
        src, binary = root / "reconstructed.cpp", root / "reconstructed.so"
        src.write_text(code, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
             "-Wall", "-Wextra", "-Werror", str(src), "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=45,
        )
        native = ctypes.CDLL(str(binary)).visuaml_recovered_i32
        native.argtypes = [ctypes.c_int32]
        native.restype = ctypes.c_int32
        for x in (
            *range(-128, 128), -(1 << 31), (1 << 31) - 1,
            654321, -1001001, 0x12345678,
        ):
            assert native(x) == execute_program(quadratic(), x).value
