"""v19 independent artifact loading and bounded memory-lifetime verification."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.memory_planner_v19 import (
    MemoryPlanningError,
    plan_decoder_memory,
)
from visuaml.compiler_lab.semantic_bundle_v19 import (
    BundleV19Error,
    load_decoder_bundle,
    write_decoder_bundle,
)
from visuaml.compiler_lab.semantic_decoder_v18 import (
    lower_decoder_semantic_program,
    make_decoder_semantic_program,
)
from visuaml.compiler_lab.stacked_decoder_v14 import StackedWeights
from visuaml.compiler_lab.streaming_decoder_v16 import StreamingStackedDecoder


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_bundle_roundtrip_independent_decoder_execution(tmp_path, layers):
    weights = StackedWeights.seeded(19, layers=layers)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    write_decoder_bundle(plan, tmp_path, provenance="trusted local seeded v19 test")
    loaded = load_decoder_bundle(tmp_path)
    assert loaded.semantic == plan.semantic
    assert loaded.checkpoint_sha256 == plan.checkpoint_sha256
    executor = loaded.new_session()
    original = StreamingStackedDecoder(weights)
    for chunk in (
        np.array([[1, 2, 3]], np.int64),
        np.array([[4]], np.int64),
        np.array([[1, 5, 2, 4]], np.int64),
    ):
        np.testing.assert_allclose(executor.step(chunk), original.step(chunk), atol=3e-5, rtol=3e-5)
    for new, old in zip(executor.snapshot().layers, original.snapshot().layers):
        np.testing.assert_allclose(new.keys, old.keys, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(new.values, old.values, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("filename", ("bundle.json", "semantic.json", "checkpoint.json", "weights.npz"))
def test_tampered_bundle_rejected(tmp_path, filename):
    weights = StackedWeights.seeded(7)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    write_decoder_bundle(plan, tmp_path, provenance="immutable seeded test")
    target = tmp_path / filename
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(BundleV19Error):
        load_decoder_bundle(tmp_path)


def test_missing_and_duplicate_json_are_rejected(tmp_path):
    weights = StackedWeights.seeded(7)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    write_decoder_bundle(plan, tmp_path, provenance="seeded")
    original = (tmp_path / "bundle.json").read_text()
    (tmp_path / "bundle.json").write_text(original[:-2] + ',"schema":"x"}')
    with pytest.raises(BundleV19Error):
        load_decoder_bundle(tmp_path)


def test_manifest_cannot_silently_rewrite_semantic_wiring(tmp_path):
    weights = StackedWeights.seeded(8)
    plan = lower_decoder_semantic_program(make_decoder_semantic_program(weights), weights)
    write_decoder_bundle(plan, tmp_path, provenance="seeded")
    semantic_path = tmp_path / "semantic.json"
    doc = json.loads(semantic_path.read_text())
    doc["regions"][0]["operations"][0]["op"] = "arbitrary_python"
    semantic_path.write_text(json.dumps(doc))
    manifest_path = tmp_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    def encode(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    manifest["files_sha256"]["semantic.json"] = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(encode(unsigned)).hexdigest()
    manifest_path.write_bytes(encode(manifest))
    with pytest.raises(BundleV19Error, match="disagrees"):
        load_decoder_bundle(tmp_path)


def test_liveness_tracks_bound_and_last_uses():
    weights = StackedWeights.seeded(31, layers=4)
    program = make_decoder_semantic_program(weights, batch=2)
    report = plan_decoder_memory(program, maximum_query=32, maximum_context=64)
    assert report.semantic_sha256 == program.digest()
    assert report.peak_logical_bytes > 0
    assert report.peak_logical_values > 0
    assert any(item.value == "logits" for item in report.lifetimes)
    assert all(item.last_used_at >= item.defined_at for item in report.lifetimes)
    assert report.as_dict()["schema"] == "visuaml.semantic-memory-analysis.v19"
    small = plan_decoder_memory(program, maximum_query=1, maximum_context=8)
    assert small.peak_logical_bytes <= report.peak_logical_bytes


def test_liveness_rejects_unadmitted_symbols_and_limits():
    weights = StackedWeights.seeded(3)
    p = make_decoder_semantic_program(weights)
    with pytest.raises(MemoryPlanningError):
        plan_decoder_memory(p, maximum_query=65)
    first = p.regions[0].operations[0]
    invalid = replace(first, result_type=replace(first.result_type, shape=(1, "W", 16)))
    region = replace(p.regions[0], operations=(invalid, *p.regions[0].operations[1:]))
    with pytest.raises(MemoryPlanningError):
        plan_decoder_memory(replace(p, regions=(region,)))
