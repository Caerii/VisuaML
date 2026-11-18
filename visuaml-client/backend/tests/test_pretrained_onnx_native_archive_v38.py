"""Permanent v38 native-pretrained observation integrity; no rerun or universal proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "benchmarks/pretrained_mnist8_native_semantic_v38.json"
ORIGIN = ROOT / "benchmarks/pretrained_mnist8_native_semantic_v38.origin.json"


def test_native_pretrained_archive_is_exact_and_scoped():
    wire = RECORD.read_bytes()
    doc = json.loads(wire)
    origin = json.loads(ORIGIN.read_bytes())
    assert origin["schema"] == "visuaml.pretrained-native-archive-origin.v38"
    assert origin["evidence_path"] == "benchmarks/pretrained_mnist8_native_semantic_v38.json"
    assert origin["evidence_bytes"] == len(wire)
    assert hashlib.sha256(wire).hexdigest() == origin["evidence_sha256"]
    assert doc["schema"] == "visuaml.pretrained-onnx-native-semantic.v38"
    assert doc["source_commit"] == origin["source_commit"]
    assert doc["checkpoint_sha256"] == origin["checkpoint_sha256"]
    assert doc["checkpoint_sha256"] == "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
    assert doc["native_execution"] is True and doc["all_passed"] is True
    assert doc["graph_operation_count"] == 12
    assert doc["compiler"]["schema"] == "visuaml.semantic-native-cpu.v32"
    assert doc["compiler"]["semantic_sha256"] == origin["semantic_sha256"]
    assert doc["compiler"]["native_binary_sha256"] == origin["native_binary_sha256"]
    assert doc["compiler"]["generated_cpp_sha256"] == origin["generated_cpp_sha256"]
    assert hashlib.sha256(doc["generated_cpp"].encode()).hexdigest() == origin["generated_cpp_sha256"]
    assert len(doc["witnesses"]) == 5
    assert [w["name"] for w in doc["witnesses"]] == [
        "zero", "ones", "impulse", "checkerboard", "random_seed_37",
    ]
    assert len({w["input_sha256"] for w in doc["witnesses"]}) == 5
    assert all(w["input_immutability"] and all(
        v["passed"] for v in w["numerics"].values()
    ) for w in doc["witnesses"])
    assert set(doc["witnesses"][0]["numerics"]) == {"onnxruntime", "scalar_numpy"}
    assert doc["atol"] == doc["rtol"] == 0.0001
    assert doc["compiler"]["output_shape"] == [1, 10]
    assert doc["compiler"]["op_sequence"] == [
        "reshape", "conv2d", "broadcast_add", "relu", "maxpool2d", "conv2d",
        "broadcast_add", "relu", "maxpool2d", "reshape", "matmul", "broadcast_add",
    ]
    assert origin["workflow_run_url"] == "https://github.com/Caerii/VisuaML/actions/runs/44214684259"
