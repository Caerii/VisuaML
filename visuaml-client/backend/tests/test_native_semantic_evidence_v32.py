"""Negative and positive integrity gates for v32 generated-native receipts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from examples.native_fx_codegen_demo_v32 import run as fx_run
from examples.native_semantic_codegen_demo_v32 import benchmark
from scripts.verify_native_semantic_evidence_v32 import EvidenceError, verify
from visuaml.compiler_lab.native_cpu_v28 import compiler_available

pytestmark = pytest.mark.skipif(
    not compiler_available(), reason="Trusted C++17 toolchain required"
)

SOURCE_COMMIT = "a" * 40
BACKEND = Path(__file__).resolve().parents[1]


def witnesses(monkeypatch):
    monkeypatch.setenv("VISUAML_SOURCE_COMMIT", SOURCE_COMMIT)
    return benchmark(trials=2), fx_run(seed=32)


def test_native_evidence_checks_generated_cpp_binary_and_fx_equivalence(monkeypatch):
    declared, fx = witnesses(monkeypatch)
    result = verify(
        declared, fx, expected_commit=SOURCE_COMMIT, root=BACKEND
    )
    assert result["source_and_cpp_hashes_verified"] is True
    assert result["source_commit"] == SOURCE_COMMIT


def test_native_evidence_rejects_source_and_weight_tampering(monkeypatch):
    declared, fx = witnesses(monkeypatch)
    changed = deepcopy(declared)
    changed["fixtures"][0]["generated_cpp"] += "\n// forged"
    with pytest.raises(EvidenceError, match="content hash mismatch"):
        verify(changed, fx, expected_commit=SOURCE_COMMIT, root=BACKEND)
    changed = deepcopy(fx)
    name = next(iter(changed["fixtures"][0]["weight_sha256"]))
    changed["fixtures"][0]["weight_sha256"][name] = "0" * 64
    with pytest.raises(EvidenceError, match="Bound FX weight"):
        verify(declared, changed, expected_commit=SOURCE_COMMIT, root=BACKEND)
    changed = deepcopy(declared)
    changed["source_commit"] = "b" * 40
    with pytest.raises(EvidenceError, match="source revision mismatch"):
        verify(changed, fx, expected_commit=SOURCE_COMMIT, root=BACKEND)
