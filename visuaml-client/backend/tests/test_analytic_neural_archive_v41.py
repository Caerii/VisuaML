"""Integrity-only audit of the immutable v41 neural-specialization evidence.

This checks original record bytes and constrained reconstructed functions.
It is NOT a substitute for a fresh independent Wasmtime/NumPy replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.analytic_relu_specialization_v41 import (
    AnalyticRelu,
    decompile_function_from_weights,
    evaluate_bounded,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/analytic_neural_wasm_v41.json"
ORIGIN = ROOT / "benchmarks/analytic_neural_wasm_v41.origin.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_neural_archive_has_exact_original_bytes_and_traceable_ci():
    raw = RAW.read_bytes()
    report = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert report["schema"] == "visuaml.wasm-neural-specialization.v41"
    assert origin["schema"] == "visuaml.analytic-neural-wasm-origin.v41"
    assert origin["raw_sha256"] == _sha(raw)
    assert origin["raw_bytes"] == len(raw) == 57435
    assert report["github_checkout_sha"] == origin["source_commit"]
    assert report["source_file_sha256"] == origin["source_file_sha256"]
    assert report["source_wasm_sha256"] == origin["source_wasm_sha256"]
    assert report["function_sha256"] == origin["recovered_function_sha256"]
    assert origin["workflow_run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44220386353"
    )
    assert report["complete"] and report["real_wasm_executed"]
    assert report["analytic_neural_weights_executed"]
    assert report["manifest_free_function_recovered"]
    assert not report["source_program_recovered"]
    assert not report["universal_neural_vm"]
    assert not report["all_i32_inputs_proved"]
    assert origin["authentication"].startswith("CI-origin metadata")
    assert origin["false_claims"] == {
        "all_i32_inputs_proved": False,
        "source_program_decompiled": False,
        "universal_neural_vm": False,
    }


def test_archived_original_weights_reconstruct_same_function_without_manifest():
    report = json.loads(RAW.read_bytes())
    tensors = report["network"]
    net = AnalyticRelu(
        np.array(tensors["w_in"], dtype=np.float64),
        np.array(tensors["b_in"], dtype=np.float64),
        np.array(tensors["w_out"], dtype=np.float64),
        np.array(tensors["b_out"], dtype=np.float64),
    )
    reconstructed = decompile_function_from_weights(net)
    assert reconstructed.function_sha256 == report["function_sha256"]
    assert reconstructed.domain == (-128, 127)
    assert reconstructed.manifest_used is False
    assert report["checked_inputs"] == len(report["witnesses"]) == 256
    assert [item["input_i32"] for item in report["witnesses"]] == list(range(-128, 128))
    for witness in report["witnesses"]:
        value = witness["input_i32"]
        expected = value * value - 3 * value + 7
        assert witness["pass"] is True
        assert witness["wasmtime"] == witness["typed"] == expected
        assert witness["relu"] == witness["extracted_from_weights"] == expected
        assert reconstructed.outputs[value + 128] == expected
        assert evaluate_bounded(net, value) == expected
    assert _sha(bytes.fromhex(report["source_wasm_hex"])) == report["source_wasm_sha256"]
    # The explicit falsifier is part of the research record.
    assert report["outside_domain_counterexample"] == {
        "input": 128, "neural_extrapolation": 16005,
        "wasmtime_result": 16007, "disagree": True,
    }
