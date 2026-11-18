"""Original v43 neural→symbolic evidence is byte-immutable and claims remain scoped.

Revalidates historical source/target identities and observation consistency.
Does NOT substitute for rerunning independent Z3, native C++ or Wasmtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.neural_symbolic_recovery_v43 import (
    RecoveredPolynomial,
    compile_polynomial_cpp17,
    compile_polynomial_wasm,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module

ROOT = Path(__file__).resolve().parents[3]
ORIGIN = ROOT / "benchmarks/neural_symbolic_roundtrip_v43.origin.json"
RECORD = ROOT / "benchmarks/neural_symbolic_roundtrip_v43.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_original_v43_record_and_origin_are_byte_bound():
    raw = RECORD.read_bytes()
    origin = json.loads(ORIGIN.read_bytes())
    record = json.loads(raw)
    assert len(raw) == origin["raw_bytes"] == 95484
    assert _sha(raw) == origin["raw_sha256"]
    assert origin["schema"] == "visuaml.neural-symbolic-origin.v43"
    assert record["schema"] == "visuaml.verified-neural-symbolic-roundtrip.v43"
    assert origin["source_commit"] == record["github_source_commit"]
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    assert origin["source_commit"] == "9a1556884d1ddf6a964a391dc7cddec6f4732142"
    assert origin["workflow_run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44224761508"
    )
    assert origin["historical_v41_raw_sha256"] == (
        "035a85ab65c9ed395e3dcb0824899609a14e20563113f0c6370731efe0a0675e"
    )
    assert record["historical_v41_raw_sha256"] == origin["historical_v41_raw_sha256"]
    assert origin["smt_outcome"] == "unsat"
    assert origin["negative_control_smt_outcome"] == "sat"
    assert origin["neural_domain_not_extended"] is True
    assert origin["independent_wasmtime_and_cpp"] is True
    assert origin["authentication"].endswith("not a signed proof object")


def test_archived_wasm_cpp_and_symbolic_hashes_reconstruct_exactly():
    report = json.loads(RECORD.read_bytes())
    original = bytes.fromhex(report["original_wasm_hex"])
    generated = bytes.fromhex(report["compiled_wasm_hex"])
    assert _sha(original) == report["original_wasm_sha256"]
    assert _sha(generated) == report["compiled_wasm_sha256"]
    assert original != generated
    assert decode_module(original).program != decode_module(generated).program
    state = report["polynomial"]
    symbolic = RecoveredPolynomial(
        int(state["a"]), int(state["b"]), int(state["c"]),
        int(state["degree"]), tuple(state["domain"]),
        state["recovered_function_sha256"], tuple(state["inference_points"]),
        state["source_program_read"], state["schema"],
    )
    assert (symbolic.a, symbolic.b, symbolic.c) == (1, -3, 7)
    assert symbolic.source_program_read is False
    assert symbolic.domain == (-128, 127)
    assert symbolic.digest() == report["all_i32_symbolic_vs_source_smt"][
        "symbolic_candidate_sha256"
    ]
    assert compile_polynomial_wasm(symbolic)[1] == generated
    assert compile_polynomial_cpp17(symbolic) == report["generated_cpp17_source"]
    assert _sha(report["generated_cpp17_source"].encode()) == report[
        "generated_cpp17_sha256"
    ]
    assert report["compiled_native_binary_sha256"] == json.loads(
        ORIGIN.read_bytes()
    )["native_binary_sha256"]


def test_finite_neural_domain_and_larger_symbolic_domain_stay_distinct():
    record = json.loads(RECORD.read_bytes())
    assert record["exact_finite_neural_domain_verified"] == [-128, 127]
    assert record["exact_bounded_neural_witness_count"] == 256
    assert record["runtime_witness_count"] == len(record["observations"]) == 326
    assert record["neural_all_i32_equivalent"] is False
    assert record["arbitrary_learned_neural_program_decompiled"] is False
    assert record["smt_result_is_machine_checked_proof_object"] is False
    smt = record["all_i32_symbolic_vs_source_smt"]
    wrong = record["wrong_symbolic_hypothesis_smt"]
    assert smt["solver_outcome"] == "unsat"
    assert smt["all_i32_source_vs_symbolic_equivalent"] is True
    assert smt["neural_all_i32_equivalent"] is False
    assert smt["machine_checked_proof_object"] is False
    assert wrong["solver_outcome"] == "sat"
    assert wrong["source_output_u32"] != wrong["candidate_output_u32"]
    assert wrong["all_i32_source_vs_symbolic_equivalent"] is False
    assert record["outside_domain_neural_counterexample"] == {
        "input_i32": 128, "neural_output": 16005,
        "source_wasmtime_output": 16007, "different": True,
    }
    assert [v["input_i32"] for v in record["observations"][:256]] == list(
        range(-128, 128)
    )
    for witness in record["observations"]:
        assert witness["pass"] is True
        x = witness["input_i32"]
        expected = (x * x - 3 * x + 7) & 0xFFFFFFFF
        expected = expected - (1 << 32) if expected >= (1 << 31) else expected
        assert all(witness[key] == expected for key in (
            "source_wasmtime", "symbolic_wasmtime", "generated_cpp17",
            "source_typed", "recompiled_typed",
        ))
        if -128 <= x <= 127:
            assert witness["bounded_neural"] == expected
            assert witness["neural_weight_only_extraction"] == expected
        else:
            assert "bounded_neural" not in witness
            assert "neural_weight_only_extraction" not in witness
