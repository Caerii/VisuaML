"""Versioned v20 real-checkpoint bundle reload and tamper rejection."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from test_hf_llama_checkpoint_v19 import fixture
from visuaml.compiler_lab.hf_llama_checkpoint_v19 import load_hf_llama_local
from visuaml.compiler_lab.llama_bundle_v20 import (
    LlamaBundleError,
    load_llama_bundle,
    write_llama_bundle,
)
from visuaml.compiler_lab.llama_semantic_v20 import (
    lower_llama_semantic_program,
    make_llama_semantic_program,
)


def build(tmp_path, *, layers=2, batch=1):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    fixture(input_dir, layers=layers)
    weights = load_hf_llama_local(input_dir, declared_revision="a" * 40)
    plan = lower_llama_semantic_program(
        make_llama_semantic_program(weights, batch=batch), weights, batch=batch
    )
    output = tmp_path / "output"
    write_llama_bundle(
        plan, input_dir, output, declared_source="verified-local/llama-fixture"
    )
    return weights, plan, output


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_reload_executes_same_numerical_graph(tmp_path, layers):
    weights, plan, output = build(tmp_path, layers=layers)
    restored = load_llama_bundle(output)
    assert restored.semantic_sha256 == plan.semantic_sha256
    assert restored.weights.checkpoint_sha256 == weights.checkpoint_sha256
    data = np.array([[1, 2, 3, 4]], dtype=np.int64)
    left = plan.new_session().step(data)
    right = restored.new_session().step(data)
    np.testing.assert_allclose(left, right, atol=0, rtol=0)
    np.testing.assert_allclose(
        right, weights.forward(data), atol=5e-5, rtol=5e-5
    )


@pytest.mark.parametrize("filename", ("model.safetensors", "config.json", "semantic.json", "bundle.json"))
def test_single_file_mutation_is_rejected(tmp_path, filename):
    _, _, output = build(tmp_path, layers=1)
    target = output / filename
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(LlamaBundleError):
        load_llama_bundle(output)


def test_rehashed_malicious_semantic_wiring_is_still_rejected(tmp_path):
    _, _, output = build(tmp_path, layers=1)
    source = output / "semantic.json"
    graph = json.loads(source.read_text())
    graph["regions"][0]["operations"][0]["op"] = "import_untrusted_code"
    source.write_text(json.dumps(graph))
    manifest_path = output / "bundle.json"
    metadata = json.loads(manifest_path.read_text())
    metadata["file_sha256"]["semantic.json"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    def encode(doc):
        return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()

    unsigned = {name: value for name, value in metadata.items() if name != "manifest_sha256"}
    metadata["manifest_sha256"] = hashlib.sha256(encode(unsigned)).hexdigest()
    manifest_path.write_bytes(encode(metadata))
    with pytest.raises(LlamaBundleError, match="compiler-canonical"):
        load_llama_bundle(output)


def test_symlink_and_repeat_upload_are_rejected(tmp_path):
    _, plan, output = build(tmp_path, layers=1)
    with pytest.raises(LlamaBundleError, match="empty"):
        write_llama_bundle(
            plan, tmp_path / "input", output,
            declared_source="verified-local/llama-fixture",
        )
    weight_path = output / "model.safetensors"
    weight_path.unlink()
    weight_path.symlink_to((tmp_path / "input" / "model.safetensors").resolve())
    with pytest.raises(LlamaBundleError):
        load_llama_bundle(output)


def test_rejected_unknown_source_claim(tmp_path):
    _, plan, _ = build(tmp_path, layers=1)
    with pytest.raises(LlamaBundleError):
        write_llama_bundle(
            plan, tmp_path / "input", tmp_path / "another",
            declared_source="../path-traversal",
        )
