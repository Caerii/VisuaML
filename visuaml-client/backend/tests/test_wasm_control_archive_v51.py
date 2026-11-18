"""Historical v51 real WASM structured-control research evidence integrity.

The archived raw 407,330-byte experiment is immutable. This test checks
source commit binding, canonical .wasm bytes, all host transition states,
and the separately reported Wasmtime final output witnesses. It does not
replace a fresh Wasmtime execution, full WASM conformance or all-input proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.wasm_control_flow_v51 import (
    decode_control,
    encode_control,
    execute_control,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/wasm_structured_control_v51.json"
ORIGIN = ROOT / "benchmarks/wasm_structured_control_v51.origin.json"


def _record():
    raw = RAW.read_bytes()
    record = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 407330
    assert hashlib.sha256(raw).hexdigest() == origin["raw_sha256"] == (
        "6b7314490001ead9965647f5a43d8a7ee95735a25023f9eeff2936071479d19b"
    )
    assert origin["source_commit"] == record["source_commit"] == (
        "98891577aab7cc16fc5ca8b61f8739ed5dadf73e"
    )
    assert origin["workflow_run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44252421781"
    )
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    return record, origin


def test_original_report_preserves_fuel_vs_trap_and_neural_limits():
    record, origin = _record()
    assert record["schema"] == "visuaml.structured-control-differential.v51"
    assert origin["schema"] == "visuaml.wasm-structured-control-origin.v51"
    assert record["program_count"] == origin["program_count"] == 5
    assert len(record["witnesses"]) == origin["input_witness_count"] == 34
    assert record["total_control_state_transitions"] == origin["observed_host_steps"] == 1180
    assert record["source_wasm_independently_validated"]
    assert record["real_wasmtime_executed"]
    assert record["all_finite_witnesses_passed"]
    assert record["host_fuel_exhaustion_observed"]
    assert record["independent_wasmtime_fuel_exhaustion_observed"]
    assert record["nonterminating_case_rejected_with_bounded_fuel"]
    assert not record["neural_pc_routing_implemented"]
    assert not record["general_wasm_supported"]
    assert not record["all_possible_input_equivalence_proved"]
    assert not record["standard_transformer_executed"]
    assert not record["arbitrary_decompilation_executed"]
    assert origin["host_pc_control_only"]
    assert not origin["neural_branch_routing_proved"]
    assert not origin["all_input_loop_equivalence_proved"]


def test_original_wasm_bytes_and_complete_machine_traces_replay():
    record, _ = _record()
    originals = {}
    for info in record["programs"]:
        raw = bytes.fromhex(info["canonical_wasm_hex"])
        assert hashlib.sha256(raw).hexdigest() == info["canonical_wasm_sha256"]
        p = decode_control(raw)
        assert p.digest() == info["source_program_sha256"]
        assert encode_control(p) == raw
        originals[info["name"]] = p
    assert len(originals) == 5
    step_count = 0
    for witness in record["witnesses"]:
        p = originals[witness["program"]]
        result = execute_control(p, witness["input_i32"], fuel=500)
        assert result.value == witness["source_wasmtime_i32"]
        assert result.value == witness["host_control_bit_neural_i32"]
        assert result.transitions == witness["instruction_transitions"]
        assert list(result.trace) == witness["complete_host_control_trace"]
        assert witness["passed"]
        step_count += result.transitions
    assert step_count == record["total_control_state_transitions"]
    never = bytes.fromhex(record["nontermination_original_wasm_hex"])
    assert hashlib.sha256(never).hexdigest() == record[
        "bounded_nontermination_example_sha256"
    ]
