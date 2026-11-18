"""Permanent evidence verification, without rerunning or inventing numerical results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "benchmarks/pretrained_mnist8_onnx_numpy_oracles_v37.json"
ORIGIN = ROOT / "benchmarks/pretrained_mnist8_onnx_numpy_oracles_v37.origin.json"
PRE = ROOT / "benchmarks/pretrained_mnist8_onnx_preflight_v37.json"


def test_archived_pretrained_oracle_receipt_is_byte_bound():
    wire = RECORD.read_bytes()
    manifest = json.loads(ORIGIN.read_bytes())
    obs = json.loads(wire)
    assert hashlib.sha256(wire).hexdigest() == manifest["record_sha256"]
    assert len(wire) == manifest["original_record_bytes"] == 2972
    assert hashlib.sha256(PRE.read_bytes()).hexdigest() == manifest["original_source_preflight_record_sha256"]
    assert manifest["github_actions"]["checkout_commit"] == "fe164cd7ea006a456eb8b8b9e760afc1396e4393"
    assert manifest["github_actions"]["checkout_commit_parents"][1] == manifest["github_actions"]["pull_request_head"]
    assert obs["schema"] == "visuaml.pretrained-cnn-onnx-vs-numpy.v37"
    assert obs["all_passed"] is True
    assert obs["executed_native_semantic_cpp"] is False
    assert obs["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert obs["preflight_schema"] == "visuaml.pretrained-onnx-preflight.v37"
    assert (obs["atol"], obs["rtol"]) == (1e-4, 1e-4)
    assert len(obs["witnesses"]) == 5
    assert {x["name"] for x in obs["witnesses"]} == {
        "zero", "ones", "impulse", "checkerboard", "random_seed_37"
    }
    assert all(x["pass"] and x["max_abs_error"] <= x["max_allowed_error"] for x in obs["witnesses"])
    assert max(x["max_abs_error"] for x in obs["witnesses"]) == 7.152557373046875e-07


def test_provenance_is_not_native_compilation_claim():
    obs = json.loads(RECORD.read_bytes())
    manifest = json.loads(ORIGIN.read_bytes())
    assert manifest["executed_native_semantic_cpp"] is False
    assert "not generated C++ execution" in obs["limitations"]
    assert "fresh-independent-numerical-oracle-replay" == manifest["evidence_class"]
