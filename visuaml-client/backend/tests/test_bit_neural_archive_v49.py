"""Immutable original v49 exact-i32 bit-neural experiment archival audit.

The record's original byte digest, source-commit fingerprints and all stored
canonical WASM programs must remain identical. This is an independent
historical-record check, not a fresh external Wasmtime execution or an all-i32
formal proof of the neural circuit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from visuaml.compiler_lab.wasm_state_vm_v40 import (
    decode_module,
    execute_program,
    initial_state,
    step,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/bit_neural_i32_v49.json"
ORIGIN = ROOT / "benchmarks/bit_neural_i32_v49.origin.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _original():
    raw = RAW.read_bytes()
    record, origin = json.loads(raw), json.loads(ORIGIN.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 182777
    assert _sha(raw) == origin["raw_sha256"] == (
        "29819fd7117229fa435ff8691703244628072d6835f19efef860e15f23ec29ed"
    )
    assert origin["source_commit"] == record["source_commit"] == (
        "1b9a115d0f1b8032cc1d8498609464ab47b979ed"
    )
    assert origin["run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44248361812"
    )
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    return record, origin


def test_original_full_width_evidence_origin_and_scientific_scope():
    report, origin = _original()
    assert report["schema"] == "visuaml.full-width-i32-neural-circuit.v49"
    assert origin["schema"] == "visuaml.bit-neural-wasm-origin.v49"
    assert report["fixed_interpreter_weights_sha256"] == origin[
        "fixed_neural_weights_sha256"
    ]
    assert report["program_count"] == origin["program_count"] == 11
    assert report["total_output_witnesses"] == origin["input_witness_count"] == 286
    assert report["total_stepwise_state_comparisons"] == origin[
        "operation_step_count"
    ] == 1196
    assert report["inputs_per_program"] == 26
    assert report["input_range_admitted"] == [-(1 << 31), (1 << 31)-1]
    assert report["input_values_exhaustively_tested"] is False
    assert report["modular_add_sub_mul_and_signed_comparisons"] is True
    assert report["independent_original_wasmtime_executed"] is True
    assert report["every_witness_step_state_compared_to_v40"] is True
    assert not report["general_wasm_supported"]
    assert not report["memory_or_control_flow_supported"]
    assert not report["standard_softmax_transformer_executed"]
    assert not report["arbitrary_neural_decompilation_implemented"]
    assert not report["all_i32_neural_circuit_smt_proved"]
    assert not origin["full_i32_domain_exhaustively_proved"]
    assert not origin["complete_wasm_supported"]
    assert not origin["trained_or_arbitrary_neural_decompilation_proved"]


def test_original_raw_wasm_and_all_286_witnesses_are_consistent():
    report, _ = _original()
    programs = {}
    for item in report["programs"]:
        raw = bytes.fromhex(item["canonical_source_wasm_hex"])
        assert _sha(raw) == item["canonical_source_wasm_sha256"]
        program = decode_module(raw).program
        assert program.digest() == item["typed_program_sha256"]
        assert item["fixed_interpreter_weights_sha256"] == report[
            "fixed_interpreter_weights_sha256"
        ]
        programs[item["program"]] = program
    assert len(programs) == 11
    for witness in report["witnesses"]:
        assert witness["pass"] is True
        program = programs[witness["program"]]
        x = witness["input_i32"]
        expected = execute_program(program, x)
        assert witness["wasmtime_i32"] == expected.value
        assert witness["bit_neural_i32"] == expected.value
        assert witness["typed_reference_i32"] == expected.value
        assert witness["original_operation_steps"] == len(program.operations)
        if x in (-(1 << 31), 0, (1 << 31)-1):
            assert len(witness["full_step_trace"]) == len(program.operations)
            state = initial_state(program, x)
            for recorded, operation in zip(
                witness["full_step_trace"], program.operations, strict=True
            ):
                state, transition = step(program, state)
                assert recorded["opcode"] == operation.opcode
                assert recorded["state"] == asdict(state)
                assert recorded["typed_transition"] == asdict(transition)
        else:
            assert witness["full_step_trace"] == []
