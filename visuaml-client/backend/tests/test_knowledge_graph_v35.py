"""v35 provenance-first overlay: real v28 checkpoint and replay guardrails."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from visuaml_evidence.knowledge_graph_v35 import (
    KnowledgeGraphError,
    _load_v28,
    knowledge_graph,
    preflight_replay,
    replay_plan_v28,
)
from visuaml_evidence.library_v34 import EvidenceLibrary

ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    "benchmarks/native_online_attention_observation_v31.json",
    "benchmarks/native_semantic_codegen_observation_v32.json",
    "benchmarks/export_evidence_observation_v33.json",
    "benchmarks/export_evidence_observation_v33.origin.json",
    "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json",
)


def copy_evidence(tmp_path: Path) -> Path:
    for relative in SOURCES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_v35_builds_thirteen_immutable_studies_from_four_sources():
    old = EvidenceLibrary.from_repo(ROOT).catalog()
    graph = knowledge_graph(ROOT)
    assert graph["schema"] == "visuaml.knowledge-graph.v35"
    assert graph["parent_catalog_sha256"] == old["catalog_sha256"]
    assert graph["counts"]["studies"] == 13
    assert graph["counts"]["source_artifacts"] == 4
    assert graph["counts"]["pinned_pretrained_witnesses"] == 1
    assert len(graph["nodes"]) == 13
    assert graph == knowledge_graph(ROOT)
    assert len(graph["knowledge_graph_sha256"]) == 64
    old_ids = {item["entry_id"] for item in old["entries"]}
    new_ids = {item["entry_id"] for item in graph["nodes"]}
    assert old_ids <= new_ids
    assert len(new_ids - old_ids) == 1


def test_historical_pretrained_checkpoint_is_connected_to_v31_by_identity():
    graph = knowledge_graph(ROOT)
    pinned = next(
        item for item in graph["nodes"]
        if item["kind"] == "pinned_pretrained_native_witness"
    )
    assert pinned["checkpoint_sha256"] == (
        "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
    )
    assert pinned["model_revision"] == (
        "756e9746eb801752d69771ef3e9bc433b21d853d"
    )
    assert pinned["result"]["numerical_witness_count"] == 2
    assert pinned["source_commit"] is None
    neighbors = [
        edge for edge in graph["edges"]
        if edge["relation"] == "identical_checkpoint_fingerprint"
        and pinned["entry_id"] in (edge["from"], edge["to"])
    ]
    assert len(neighbors) >= 1
    other_ids = {
        edge["to"] if edge["from"] == pinned["entry_id"] else edge["from"]
        for edge in neighbors
    }
    assert any(
        item["entry_id"] in other_ids and item["label"] == "pinned-tiny-llama"
        for item in graph["nodes"]
    )
    assert all("not_transferable" in edge["scope"] for edge in neighbors)


def test_metadata_preflight_fails_closed_on_current_changed_source():
    graph = knowledge_graph(ROOT)
    plan = graph["plans"][0]
    assert plan["schema"] == "visuaml.replay-plan.v35"
    assert plan["historical_source_commit"] is None
    report = preflight_replay(plan, ROOT)
    assert report["result_class"] == "replay_blocked"
    assert report["static_source_and_checkpoint_ready"] is False
    assert report["numerical_replay_performed"] is False
    assert report["performance_replay_performed"] is False
    assert any(
        blocker["kind"] == "pinned_checkpoint_not_provided"
        for blocker in report["blockers"]
    )
    # v28 source was changed by v29-v31; do not silently replay against main.
    assert any(
        blocker["kind"] == "source_sha256_mismatch"
        for blocker in report["blockers"]
    )


def test_replay_cannot_fake_missing_checkpoint_or_modify_plan(tmp_path):
    plan = knowledge_graph(ROOT)["plans"][0]
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"llama"}')
    (checkpoint / "model.safetensors").write_bytes(b"fake weights")
    report = preflight_replay(plan, ROOT, checkpoint=checkpoint)
    assert len([
        item for item in report["blockers"]
        if item["kind"] == "checkpoint_sha256_mismatch"
    ]) == 2
    assert report["numerical_replay_performed"] is False
    modified = json.loads(json.dumps(plan))
    modified["tolerance_atol"] = 0.9
    with pytest.raises(KnowledgeGraphError, match="not bound"):
        preflight_replay(modified, ROOT, checkpoint=checkpoint)
    modified = json.loads(json.dumps(plan))
    modified["source_files_sha256"]["../../arbitrary.py"] = "a" * 64
    with pytest.raises(KnowledgeGraphError, match="not bound"):
        preflight_replay(modified, ROOT)


def test_v28_archive_tampering_does_not_create_new_valid_history(tmp_path):
    repo = copy_evidence(tmp_path)
    before = knowledge_graph(repo)
    file = repo / SOURCES[-1]
    data = json.loads(file.read_text())
    data["chunked_KV_and_greedy_parity"] = False
    file.write_text(json.dumps(data))
    with pytest.raises(KnowledgeGraphError, match="parity/provenance"):
        knowledge_graph(repo)
    file.write_text(json.dumps({
        **data, "chunked_KV_and_greedy_parity": True,
        "checkpoint_sha256": "f" * 64,
    }))
    with pytest.raises(KnowledgeGraphError, match="source identities disagree"):
        knowledge_graph(repo)
    assert len(before["nodes"]) == 13


def test_negative_or_nonfinite_historical_errors_rejected(tmp_path):
    repo = copy_evidence(tmp_path)
    file = repo / SOURCES[-1]
    record = json.loads(file.read_text())
    record["witnesses"][0]["max_abs_error_against_HF"] = 1.0
    file.write_text(json.dumps(record))
    with pytest.raises(KnowledgeGraphError, match="tolerance"):
        _load_v28(repo)
    record["witnesses"][0]["max_abs_error_against_HF"] = float("nan")
    file.write_text(json.dumps(record))
    with pytest.raises(KnowledgeGraphError, match="Nonfinite"):
        _load_v28(repo)


def test_v28_replay_plan_preserves_exact_evidence_identity():
    record, digest = _load_v28(ROOT)
    plan = replay_plan_v28(record, digest)
    assert plan["source_artifact_sha256"] == digest
    assert plan["source_files_sha256"][
        "visuaml/compiler_lab/native_cpu_v28.cpp"
    ] == record["backend"]["cpp_source_sha256"]
    assert plan["checkpoint_files_sha256"]["config.json"] == record["config_sha256"]
    assert plan["witness_count"] == 2
    assert plan["status"] == "preflight_only_no_executable_replay"
    assert "not_replayed" in plan["evidence_class"]
    assert hashlib.sha256(
        (ROOT / SOURCES[-1]).read_bytes()
    ).hexdigest() == digest


def test_no_future_work_is_misrepresented_as_verified():
    graph = knowledge_graph(ROOT)
    assert all(
        "proof" not in item["status"] for item in graph["nodes"]
    )
    assert "numerical" not in graph["plans"][0]["status"]
    assert all("identity_" in edge["scope"] for edge in graph["edges"])
    assert any("not freshly" in issue for issue in graph["limitations"])


def test_historical_native_error_and_token_vocabulary_are_bounded(tmp_path):
    repo = copy_evidence(tmp_path)
    archive = repo / SOURCES[-1]
    original = json.loads(archive.read_text())
    forged = json.loads(json.dumps(original))
    forged["backend"]["independent_max_abs_error"] = 0.5
    archive.write_text(json.dumps(forged))
    with pytest.raises(KnowledgeGraphError, match="native historical numerical tolerance"):
        _load_v28(repo)
    forged = json.loads(json.dumps(original))
    forged["witnesses"][0]["tokens"][0][0] = 512
    archive.write_text(json.dumps(forged))
    with pytest.raises(KnowledgeGraphError, match="witness geometry"):
        _load_v28(repo)
