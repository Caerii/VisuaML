"""Immutable verification of the original v40 independent Wasmtime record.

Never update old experimental bytes to follow a code change. New experiments
must receive new versioned records and origin manifests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    alpha_rename_locals,
    decode_module,
    encode_program,
)

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "benchmarks/typed_wasm_state_differential_v40.json"
ORIGIN = ROOT / "benchmarks/typed_wasm_state_differential_v40.origin.json"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_original_full_wasm_evidence_is_byte_bound_and_not_a_neural_claim():
    raw = RECORD.read_bytes()
    record = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert record["schema"] == "visuaml.wasm-state-differential.v40"
    assert origin["schema"] == "visuaml.wasm-state-differential-origin.v40"
    assert _sha(raw) == origin["archived_raw_sha256"]
    assert len(raw) == origin["archived_raw_bytes"] == 110785
    assert origin["source_commit"] == record["github_checkout_sha"]
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    assert origin["source_module_sha256"] == record["source_sha256"]
    assert origin["target_module_sha256"] == record["target_sha256"]
    assert origin["independent_wasmtime"] is True
    assert record["neural_weights_executed"] is False
    assert record["all_i32_inputs_proved"] is False
    assert record["complete"] is True
    assert record["independent_wasmtime"] is True
    assert record["wasmtime_version"] == "28.0.0"
    assert record["rewrite_receipt"]["typed_alpha_inverse_validated"] is True
    assert record["rewrite_receipt"]["binary_roundtrip_exact"] is True
    assert record["rewrite_receipt"]["module_typechecked_by_independent_engine"] is False
    assert record["rewrite_receipt"]["observational_parity_tested"] is False
    assert origin["authentication"].startswith("CI-origin metadata")


def test_original_full_wasm_binary_and_finite_witnesses_are_self_consistent():
    record = json.loads(RECORD.read_bytes())
    source = bytes.fromhex(record["source_hex"])
    target = bytes.fromhex(record["target_hex"])
    assert source != target
    assert _sha(source) == record["source_sha256"]
    assert _sha(target) == record["target_sha256"]
    source_ir = decode_module(source).program
    target_ir = decode_module(target).program
    assert source_ir.digest() == record["source_program_sha256"]
    assert target_ir.digest() == record["target_program_sha256"]
    rebuilt, receipt = alpha_rename_locals(source, (2, 0, 1))
    assert rebuilt == target
    assert receipt["source_sha256"] == record["source_sha256"]
    assert receipt["target_sha256"] == record["target_sha256"]
    assert encode_program(source_ir) == source
    assert encode_program(target_ir) == target
    witnesses = record["witnesses"]
    assert record["checked_inputs"] == len(witnesses) == 260
    assert [w["input_i32"] for w in witnesses] == [
        *range(-128, 128), -(1 << 31), (1 << 31) - 1, 65535, -65536,
    ]
    for witness in witnesses:
        x = witness["input_i32"]
        expected = (2 * x + 4) & 0xFFFFFFFF
        expected = expected - (1 << 32) if expected >= (1 << 31) else expected
        assert witness["pass"] is True
        assert witness["step_count"] == 13
        assert witness["typed_source_output"] == expected
        assert witness["typed_target_output"] == expected
        assert witness["wasmtime_source_output"] == expected
        assert witness["wasmtime_target_output"] == expected
        for key in ("source_trace_sha256", "target_trace_sha256"):
            assert len(witness[key]) == 64
            assert all(ch in "0123456789abcdef" for ch in witness[key])


def test_historical_experiment_source_fingerprints_are_explicit():
    origin = json.loads(ORIGIN.read_bytes())
    # Historical source identifiers, not a requirement that current HEAD
    # perpetually contains identical Python files.
    assert origin["source_file_sha256"] == {
        "scripts/probe_wasm_state_vm_v40.py":
            "d84ac14622b22da422cf1039f714d4ae1a0ee68a073c922150842970a523d4fb",
        "visuaml/compiler_lab/wasm_bitstream_v39.py":
            "c30b5db8375cfae9e1bd1ac526414ad2f07f1abcad839790951d7bd9ecb80274",
        "visuaml/compiler_lab/wasm_state_vm_v40.py":
            "410375f7c5ae4e64c2863a8490cc7284eefc33b55eda0f6253bfac99e49b7577",
    }
    assert origin["workflow_run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44219119721"
    )
    assert origin["source_commit"] == (
        "eb85172b2e29c95986da896bcb30619deb285ece"
    )
