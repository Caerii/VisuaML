"""Original v41 archived weights -> v43 symbolic rule -> WASM/C++17 -> SMT.

The extraction routine is called with ONLY detached neural weight arrays. It
does not receive the original WASM, source program, or historical manifest.
The original WASM is subsequently recovered separately from an immutable
archive and used as an independent verification oracle.

All-i32 proof applies to original-vs-reconstructed *symbolic programs only*,
not to the ReLU neural network outside [-128,127].
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import wasmtime
from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    AnalyticRelu,
    decompile_function_from_weights,
    evaluate_bounded,
)
from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    check_all_i32_equivalence,
    compile_polynomial_cpp17,
    compile_polynomial_wasm,
    recover_quadratic_from_weights,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    decode_module,
    execute_program,
)

SCHEMA = "visuaml.verified-neural-symbolic-roundtrip.v43"
OLD_ARCHIVE_SHA256 = "035a85ab65c9ed395e3dcb0824899609a14e20563113f0c6370731efe0a0675e"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_original_analytic_weights() -> tuple[AnalyticRelu, dict, str]:
    # A record check, not a requirement for the decoder. The decoder receives
    # none of the original program, model identity, or receipt.
    root = Path(__file__).resolve().parents[3]
    raw = (root / "benchmarks/analytic_neural_wasm_v41.json").read_bytes()
    origin = json.loads((
        root / "benchmarks/analytic_neural_wasm_v41.origin.json"
    ).read_bytes())
    if sha(raw) != OLD_ARCHIVE_SHA256 or origin["raw_sha256"] != sha(raw):
        raise AssertionError("Historical v41 weight archive identity mismatch")
    record = json.loads(raw)
    if record["schema"] != "visuaml.wasm-neural-specialization.v41":
        raise AssertionError("Historical neural archive schema mismatch")
    if record["checked_inputs"] != 256 or not record["real_wasm_executed"]:
        raise AssertionError("Historical neural witness incomplete")
    weights = record["network"]
    # Do NOT copy any source_program_sha256 or evidence into the model.
    neural = AnalyticRelu(
        np.array(weights["w_in"], dtype=np.float64),
        np.array(weights["b_in"], dtype=np.float64),
        np.array(weights["w_out"], dtype=np.float64),
        np.array(weights["b_out"], dtype=np.float64),
    )
    return neural, record, sha(raw)


def _native_compiler(code: str):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("No C++17 compiler available")
    scratch = tempfile.TemporaryDirectory(prefix="visuaml-v43-")
    root = Path(scratch.name)
    src = root / "symbolic.cpp"
    binary = root / "symbolic.so"
    src.write_text(code, encoding="utf-8")
    cmd = [
        compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
        "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
        str(src), "-o", str(binary),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    lib = ctypes.CDLL(str(binary))
    f = lib.visuaml_recovered_i32
    f.argtypes = [ctypes.c_int32]
    f.restype = ctypes.c_int32
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True,
        text=True, timeout=10
    ).stdout.splitlines()[0]
    return scratch, f, sha(binary.read_bytes()), version


def run() -> dict:
    import z3

    neural, record, archive_sha = load_original_analytic_weights()
    polynomial = recover_quadratic_from_weights(neural)
    recovered_function = decompile_function_from_weights(neural)
    source_bytes = bytes.fromhex(record["source_wasm_hex"])
    source = decode_module(source_bytes).program
    if polynomial.recovered_function_sha256 != record["function_sha256"]:
        raise AssertionError("Symbolic decoder changed underlying neural function")

    generated, generated_bytes = compile_polynomial_wasm(polynomial)
    wasm_generated = decode_module(generated_bytes)
    if generated != wasm_generated.program:
        raise AssertionError("Recompiled WASM does not round-trip through typed decoder")
    # Separate original source from weight-only extraction and prove
    # equivalence across all 2^32 signed i32 inputs, bitvector semantics.
    proof = check_all_i32_equivalence(source, polynomial)
    if proof["solver_outcome"] != "unsat":
        raise AssertionError("Reconstructed polynomial is not i32 equivalent")
    mutated = replace(polynomial, b=polynomial.b + 1)
    counterexample = check_all_i32_equivalence(source, mutated)
    if counterexample["solver_outcome"] != "sat":
        raise AssertionError("Solver missed intentional wrong-symbolic-rule bug")

    engine = wasmtime.Engine()
    checked_wasm = []
    for wasm in (source_bytes, generated_bytes):
        wasmtime.Module.validate(engine, wasm)
        store = wasmtime.Store(engine)
        module = wasmtime.Module(engine, wasm)
        instance = wasmtime.Instance(store, module, [])
        checked_wasm.append((store, instance.exports(store)["run"]))

    cpp = compile_polynomial_cpp17(polynomial)
    tempdir, native, binary_hash, compiler_version = _native_compiler(cpp)
    try:
        rng = np.random.default_rng(43)
        random_i32 = tuple(int(x) for x in rng.integers(
            -(1 << 31), 1 << 31, size=64, dtype=np.int64
        ))
        inputs = (
            *range(-128, 128),
            -(1 << 31), (1 << 31) - 1,
            -65536, 65535, 1000003, 128,
            *random_i32,
        )
        observations = []
        for x in inputs:
            ref = checked_wasm[0][1](checked_wasm[0][0], x)
            rewritten = checked_wasm[1][1](checked_wasm[1][0], x)
            native_out = int(native(x))
            typed_src = execute_program(source, x).value
            typed_rebuilt = execute_program(generated, x).value
            if len({ref, rewritten, native_out, typed_src, typed_rebuilt}) != 1:
                raise AssertionError(f"Native/WASM/typed disagreement at {x}")
            witness = {
                "input_i32": x,
                "source_wasmtime": ref,
                "symbolic_wasmtime": rewritten,
                "generated_cpp17": native_out,
                "source_typed": typed_src,
                "recompiled_typed": typed_rebuilt,
                "pass": True,
            }
            if -128 <= x <= 127:
                neural_out = evaluate_bounded(neural, x)
                extracted = recovered_function.outputs[x + 128]
                if ref != neural_out or ref != extracted:
                    raise AssertionError(f"Neural/symbolic mismatch inside domain at {x}")
                witness["bounded_neural"] = neural_out
                witness["neural_weight_only_extraction"] = extracted
            observations.append(witness)

        # Explicitly falsify the stronger, unsupported neural all-i32 claim.
        outside_x = 128
        neural_extrapolated = float(
            (neural.w_out @ np.maximum(
                np.float64(0),
                neural.w_in @ np.array([float(outside_x)]) + neural.b_in
            ))[0] + neural.b_out[0]
        )
        outside = {
            "input_i32": outside_x,
            "neural_output": int(neural_extrapolated),
            "source_wasmtime_output": checked_wasm[0][1](
                checked_wasm[0][0], outside_x
            ),
        }
        outside["different"] = (
            outside["neural_output"] != outside["source_wasmtime_output"]
        )
        if not outside["different"]:
            raise AssertionError("Out-of-domain neural falsifier disappeared")

        backend = Path(__file__).resolve().parents[1]
        source_paths = (
            "scripts/probe_neural_symbolic_v43.py",
            "visuaml/compiler_lab/neural_symbolic_recovery_v43.py",
            "visuaml/compiler_lab/analytic_relu_specialization_v41.py",
            "visuaml/compiler_lab/wasm_state_vm_v40.py",
        )
        result = {
            "schema": SCHEMA,
            "checkpoint": "original archived v41 analytical ReLU weight tensors",
            "historical_v41_raw_sha256": archive_sha,
            "historical_v41_source_wasm_sha256": sha(source_bytes),
            "original_wasm_hex": source_bytes.hex(),
            "original_wasm_sha256": sha(source_bytes),
            "compiled_wasm_hex": generated_bytes.hex(),
            "compiled_wasm_sha256": sha(generated_bytes),
            "generated_cpp17_sha256": sha(cpp.encode("utf-8")),
            "generated_cpp17_source": cpp,
            "compiled_native_binary_sha256": binary_hash,
            "compiler_version": compiler_version,
            "polynomial": asdict(polynomial),
            "polynomial_description": polynomial.expression(),
            "weight_only_recovered_function_sha256": recovered_function.function_sha256,
            "weight_only_decompiler_read_source": False,
            "exact_finite_neural_domain_verified": [-128, 127],
            "exact_bounded_neural_witness_count": 256,
            "all_i32_symbolic_vs_source_smt": proof,
            "wrong_symbolic_hypothesis_smt": counterexample,
            "outside_domain_neural_counterexample": outside,
            "independent_runtime_wasmtime": True,
            "native_cpp17_executed": True,
            "neural_all_i32_equivalent": False,
            "arbitrary_learned_neural_program_decompiled": False,
            "smt_result_is_machine_checked_proof_object": False,
            "observations": observations,
            "runtime_witness_count": len(observations),
            "original_neural_source_not_read_during_extraction": True,
            "z3_version": z3.get_version_string(),
            "wasmtime_version": __import__("importlib.metadata", fromlist=["version"]).version(
                "wasmtime"
            ),
            "numpy_version": np.__version__,
            "github_source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
            "source_file_sha256": {
                path: sha((backend / path).read_bytes()) for path in source_paths
            },
            "limitations": (
                "Polynomial-integer hypothesis class, finite domain [-128,127] "
                "for the analytic neural circuit. Z3 unsat proves the encoded "
                "source and recovered symbolic expression have no i32 counterexample "
                "under pure 32-bit modular semantics; it does NOT prove the neural "
                "circuit for inputs outside its admitted interval, any arbitrary "
                "neural program, or a machine-checked Lean proof."
            ),
        }
        if not proof["all_i32_source_vs_symbolic_equivalent"]:
            raise AssertionError("All-i32 bitvector proof not established")
        return result
    finally:
        tempdir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "recovered": record["polynomial_description"],
        "bitvector_all_i32": record["all_i32_symbolic_vs_source_smt"][
            "solver_outcome"
        ],
        "deliberate_wrong_rule": record["wrong_symbolic_hypothesis_smt"][
            "solver_outcome"
        ],
        "neural_domain_witnesses": record["exact_bounded_neural_witness_count"],
        "runtime_witnesses": record["runtime_witness_count"],
        "neural_outside_domain_counterexample": record[
            "outside_domain_neural_counterexample"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
