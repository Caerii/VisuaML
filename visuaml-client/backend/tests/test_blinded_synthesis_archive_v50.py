"""Historical v50 blinded-synthesis evidence is immutable, independently typed.

Verifies stored source and regenerated canonical WASM bytes, all heldout
results and the candidate-bound SMT receipt. This is an archival consistency
test, not a fresh proof or a neural weight decompilation test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module, execute_program

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/blinded_neural_cegis_v50.json"
META = ROOT / "benchmarks/blinded_neural_cegis_v50.origin.json"


def _record():
    raw = RAW.read_bytes()
    record = json.loads(raw)
    origin = json.loads(META.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 42402
    assert hashlib.sha256(raw).hexdigest() == origin["raw_sha256"] == (
        "e644c7b6558852b7754b85be0a0a5f735c47e99638a40fd0bffaa486e758fcdd"
    )
    assert origin["source_commit"] == record["source_commit"] == (
        "cf25a876e2971169274df5fca1beb04caff60f0f"
    )
    assert origin["workflow_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44250342787"
    )
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    return record, origin


def test_blinded_synthesis_original_report_claim_boundaries():
    result, origin = _record()
    assert result["schema"] == "visuaml.blinded-neural-behavior-evidence.v50"
    assert origin["schema"] == "visuaml.blinded-cegis-origin.v50"
    assert result["program_count"] == 5
    assert result["heldout_values_per_program"] == 16
    assert result["fixed_bit_weights_sha256"] == origin["fixed_neural_weights_sha256"]
    assert result["true_source_wasm_compiled_by_wasmtime"]
    assert result["cubic_outside_grammar_rejected"]
    assert result["all_i32_source_candidate_smt_checked"]
    assert result["multiple_source_programs_same_behavior_possible"]
    assert result["no_source_passed_into_synthesizer"]
    assert result["no_weights_passed_into_synthesizer"]
    assert not result["unique_historical_program_recovery_claimed"]
    assert not result["arbitrary_neural_model_decompilation_claimed"]
    assert not result["neural_gate_all_i32_equivalence_proved"]
    assert not origin["arbitrary_weight_decompilation_claimed"]
    assert not origin["unique_original_program_recovery_claimed"]
    assert not origin["all_i32_neural_gate_equivalence_proved"]


def test_original_and_synthesized_wasm_witnesses_still_agree():
    result, _ = _record()
    names = {item["name"] for item in result["results"]}
    assert names == {
        "quadratic", "identity_plus_zero", "zero_test",
        "signed_negative", "signed_threshold_5",
    }
    for item in result["results"]:
        original = bytes.fromhex(item["original_wasm_hex"])
        candidate = bytes.fromhex(item["recovered_wasm_hex"])
        assert hashlib.sha256(original).hexdigest() == item["original_wasm_sha256"]
        assert hashlib.sha256(candidate).hexdigest() == item["recovered_wasm_sha256"]
        source = decode_module(original).program
        generated = decode_module(candidate).program
        assert generated.digest() == item["recovered_typed_program_sha256"]
        report = item["synthesis_report"]
        assert report["all_i32_source_vs_candidate_equivalent"]
        assert report["rounds"][-1]["verification"]["solver_outcome"] == "unsat"
        assert not report["source_program_received_by_synthesizer"]
        assert not report["neural_weights_received_by_synthesizer"]
        assert len(item["heldout_witnesses"]) == 16
        for witness in item["heldout_witnesses"]:
            x = witness["input_i32"]
            expected = execute_program(source, x).value
            assert execute_program(generated, x).value == expected
            assert witness["pass"]
            assert all(witness[key] == expected for key in (
                "source_wasmtime", "recompiled_wasmtime",
                "fixed_neural_bit_gate_output", "original_typed_i32",
            ))
    threshold = next(x for x in result["results"]
                     if x["name"] == "signed_threshold_5")
    assert len(threshold["synthesis_report"]["rounds"]) == 3
    assert any(
        x["verification"]["solver_outcome"] == "sat"
        for x in threshold["synthesis_report"]["rounds"][:-1]
    )
