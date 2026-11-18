"""v50 permanent blinded neural CEGIS evidence integrity and source ambiguity.

This verifies the historical raw byte digest, original compiled source and
recovered canonical WASM identities, saved SAT counterexamples, and all
heldout numerical witnesses. It does not re-run Z3/Wasmtime; the dedicated
workflow does that independently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    Candidate,
    emit_canonical_wasm,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    decode_module,
    execute_program,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/blinded_neural_cegis_v50.json"
ORIGIN = ROOT / "benchmarks/blinded_neural_cegis_v50.origin.json"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _original():
    raw = RAW.read_bytes()
    report, origin = json.loads(raw), json.loads(ORIGIN.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 42402
    assert _sha(raw) == origin["raw_sha256"] == (
        "e644c7b6558852b7754b85be0a0a5f735c47e99638a40fd0bffaa486e758fcdd"
    )
    assert origin["source_commit"] == report["source_commit"] == (
        "cf25a876e2971169274df5fca1beb04caff60f0f"
    )
    assert origin["workflow_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44250342787"
    )
    assert origin["source_file_sha256"] == report["source_file_sha256"]
    return report, origin


def test_original_blinded_record_is_source_bound_and_claims_stay_restricted():
    report, origin = _original()
    assert report["schema"] == "visuaml.blinded-neural-behavior-evidence.v50"
    assert origin["schema"] == "visuaml.blinded-cegis-origin.v50"
    assert origin["fixed_neural_weights_sha256"] == report["fixed_bit_weights_sha256"]
    assert report["program_count"] == origin["program_count"] == 5
    assert origin["heldout_case_count"] == 80
    assert report["heldout_values_per_program"] == 16
    assert report["true_source_wasm_compiled_by_wasmtime"]
    assert report["all_i32_source_candidate_smt_checked"]
    assert report["multiple_source_programs_same_behavior_possible"]
    assert report["cubic_outside_grammar_rejected"]
    assert report["no_source_passed_into_synthesizer"]
    assert report["no_weights_passed_into_synthesizer"]
    assert not report["unique_historical_program_recovery_claimed"]
    assert not report["arbitrary_neural_model_decompilation_claimed"]
    assert not report["neural_gate_all_i32_equivalence_proved"]
    assert not origin["arbitrary_weight_decompilation_claimed"]
    assert not origin["unique_original_program_recovery_claimed"]
    assert not origin["all_i32_neural_gate_equivalence_proved"]
    assert not origin["external_signed_attestation"]


def test_original_z3_counterexamples_and_80_wasmtime_witnesses_are_consistent():
    report, _ = _original()
    assert {item["name"] for item in report["results"]} == {
        "quadratic", "identity_plus_zero", "zero_test",
        "signed_negative", "signed_threshold_5",
    }
    for item in report["results"]:
        raw_source = bytes.fromhex(item["original_wasm_hex"])
        raw_candidate = bytes.fromhex(item["recovered_wasm_hex"])
        assert _sha(raw_source) == item["original_wasm_sha256"]
        assert _sha(raw_candidate) == item["recovered_wasm_sha256"]
        source = decode_module(raw_source).program
        candidate = Candidate(**item["candidate"])
        rebuilt, emitted_bytes = emit_canonical_wasm(candidate)
        assert emitted_bytes == raw_candidate
        assert rebuilt.digest() == item["recovered_typed_program_sha256"]
        synth = item["synthesis_report"]
        assert synth["accepted_candidate"] == item["candidate"]
        assert synth["accepted_wasm_hex"] == item["recovered_wasm_hex"]
        assert synth["all_i32_source_vs_candidate_equivalent"]
        assert not synth["arbitrary_unknown_weight_decompiled"]
        assert not synth["historical_source_uniquely_recovered"]
        assert synth["rounds"][-1]["verification"]["solver_outcome"] == "unsat"
        for turn in synth["rounds"]:
            assert turn["verification"]["candidate_sha256"] == (
                turn["selected_hypothesis_sha256"]
            )
        assert len(item["heldout_witnesses"]) == 16
        for witness in item["heldout_witnesses"]:
            x = witness["input_i32"]
            expected = execute_program(source, x).value
            assert witness["pass"]
            assert witness["source_wasmtime"] == expected
            assert witness["recompiled_wasmtime"] == expected
            assert witness["fixed_neural_bit_gate_output"] == expected
            assert witness["original_typed_i32"] == expected
    threshold = next(x for x in report["results"] if x["name"] == "signed_threshold_5")
    outcomes = threshold["synthesis_report"]["rounds"]
    assert [t["verification"]["solver_outcome"] for t in outcomes] == [
        "sat", "sat", "unsat",
    ]
    assert [t["verification"].get("counterexample_i32") for t in outcomes] == [
        5, 4, None,
    ]
    identity = next(x for x in report["results"] if x["name"] == "identity_plus_zero")
    assert identity["original_wasm_sha256"] != identity["recovered_wasm_sha256"]
