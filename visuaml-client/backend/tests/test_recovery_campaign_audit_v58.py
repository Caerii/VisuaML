"""Source-bound recovery campaign replay and tampering tests.

All-rejected fixtures do not assert any actual neural recovery result.
Tests exercise accounting and provenance; true successes additionally
require independent Z3/Wasmtime/v58 replay in the verifier.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import benchmark_recovery_failures_v58 as producer
from scripts import verify_recovery_campaign_v58 as replay
from visuaml.compiler_lab.wasm_state_vm_v40 import encode_program


def fixture(holdout=False):
    attempts = []
    for workload in producer.eligible_workloads(holdout=holdout):
        raw = encode_program(workload.source)
        import hashlib

        attempts.append({
            "name": workload.name,
            "split": workload.split,
            "source_wasm_hex": raw.hex(),
            "source_wasm_sha256": hashlib.sha256(raw).hexdigest(),
            "status": "rejected",
            "failure_class": "grammar_rejection",
            "exception_type": "BehavioralSynthesisError",
            "message": "fixture: not a measured CEGIS run",
            "all_i32_typed_source_equivalence_proved": False,
            "all_i32_neural_weights_equivalence_proved": False,
            "audit": {
                "neural_oracle_queries": 0,
                "independent_verifier_invocations": 0,
            },
        })
    return {
        "schema": producer.SCHEMA,
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "policy": {
            "holdout_enabled": holdout,
            "failures_in_denominator": True,
            "all_i32_neural_weights_proved": False,
            "arbitrary_learned_weight_decompilation": False,
        },
        "attempts": attempts,
        "summary": producer.summarize(attempts),
    }


def check(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(
        replay, "_git",
        lambda root, *args: "a" * 40 if args[-1] == "HEAD" else "b" * 40,
    )
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(archive), encoding="utf-8")
    return replay.review(path, tmp_path)


def test_complete_rejected_inventory_preserved(tmp_path, monkeypatch):
    data = fixture(holdout=True)
    output = check(data, tmp_path, monkeypatch)
    assert output["full_denominator_verified"]
    assert output["source_bytes_verified"]
    assert output["verified_attempts_replayed"] == 0
    assert output["rejected_attempts_preserved"] == 14


def test_missing_failure_cannot_be_silently_dropped(tmp_path, monkeypatch):
    data = fixture()
    data["attempts"].pop(0)
    with pytest.raises(ValueError, match="incomplete"):
        check(data, tmp_path, monkeypatch)


def test_modified_source_bytes_cannot_pass_replay(tmp_path, monkeypatch):
    data = fixture()
    item = data["attempts"][0]
    raw = bytearray.fromhex(item["source_wasm_hex"])
    raw[-2] ^= 1
    item["source_wasm_hex"] = bytes(raw).hex()
    with pytest.raises(ValueError, match="canonical source"):
        check(data, tmp_path, monkeypatch)


def test_modified_success_denominator_cannot_pass_replay(tmp_path, monkeypatch):
    data = fixture()
    data["summary"]["total_rejected"] -= 1
    with pytest.raises(ValueError, match="denominator"):
        check(data, tmp_path, monkeypatch)


def test_overclaimed_rejection_cannot_pass(tmp_path, monkeypatch):
    data = fixture()
    data["attempts"][0]["all_i32_typed_source_equivalence_proved"] = True
    with pytest.raises(ValueError, match="false rejection"):
        check(data, tmp_path, monkeypatch)
