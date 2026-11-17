"""v36 replay must extend—not rewrite—the historical computational library."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from scripts.archive_pretrained_replay_v36 import ArchivalError
from visuaml_evidence.knowledge_graph_v35 import knowledge_graph as v35_graph
from visuaml_evidence.knowledge_graph_v36 import knowledge_graph as v36_graph

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    "benchmarks/native_online_attention_observation_v31.json",
    "benchmarks/native_semantic_codegen_observation_v32.json",
    "benchmarks/export_evidence_observation_v33.json",
    "benchmarks/export_evidence_observation_v33.origin.json",
    "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json",
    "benchmarks/pretrained_llama_independent_replay_v36.json",
    "benchmarks/pretrained_llama_independent_replay_v36.origin.json",
)


def copied(tmp_path: Path) -> Path:
    for name in ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return tmp_path


def test_v36_keeps_all_historical_study_ids_and_appends_only_one():
    old = v35_graph(ROOT)
    current = v36_graph(ROOT)
    assert current["schema"] == "visuaml.knowledge-graph.v36"
    assert current["parent_graph_sha256"] == old["knowledge_graph_sha256"]
    assert current["counts"]["studies"] == 14
    assert current["counts"]["source_artifacts"] == 5
    assert current["counts"]["fresh_independent_replays"] == 1
    assert current["counts"]["distinct_pinned_pretrained_models"] == 1
    old_ids = {item["entry_id"] for item in old["nodes"]}
    new_ids = {item["entry_id"] for item in current["nodes"]}
    assert old_ids <= new_ids
    assert len(new_ids - old_ids) == 1
    assert current == v36_graph(ROOT)


def test_exact_binary_and_checkpoint_replay_lineage_is_visible():
    graph = v36_graph(ROOT)
    fresh, = [
        node for node in graph["nodes"]
        if node["kind"] == "pinned_pretrained_independent_replay"
    ]
    original, = [
        node for node in graph["nodes"]
        if node["kind"] == "pinned_pretrained_native_witness"
    ]
    assert fresh["status"] == "fresh_independent_numerical_witness_passed"
    assert fresh["native_binary_sha256"] == original["native_binary_sha256"]
    assert fresh["generated_cpp_sha256"] == original["generated_cpp_sha256"]
    assert fresh["semantic_sha256"] == original["semantic_sha256"]
    assert fresh["checkpoint_sha256"] == original["checkpoint_sha256"]
    links = [
        edge for edge in graph["edges"]
        if {edge["from"], edge["to"]} == {
            fresh["entry_id"], original["entry_id"],
        }
    ]
    assert {edge["relation"] for edge in links} == {
        "identical_native_binary_bytes",
        "identical_generated_cpp_bytes",
        "identical_semantic_digest",
        "identical_checkpoint_fingerprint",
    }
    assert all("not_transferable" in edge["scope"] for edge in links)
    assert graph["counts"]["identity_edges"] > 16


def test_replay_result_preserves_provenance_boundary_and_original_plan():
    graph = v36_graph(ROOT)
    old = v35_graph(ROOT)
    assert graph["plans"] == old["plans"]
    assert graph["plans"][0]["status"] == "preflight_only_no_executable_replay"
    replay, = graph["fresh_replays"]
    assert replay["numerical_replay_performed"] is True
    assert replay["passed"] is True
    assert replay["original_experiment_commit"] is None
    assert len(set(replay["source_file_commits"].values())) == 3
    assert replay["producing_ci_run"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44193012326"
    )
    assert replay["record_bytes_sha256"] == (
        "a6b06210cc37a78ded00ab52c201c2c96eea0f4f754b427437b6ded3acce48f0"
    )


def test_original_source_data_and_numerical_archive_remain_unchanged(tmp_path):
    repo = copied(tmp_path)
    before = v36_graph(repo)
    replay = repo / "benchmarks/pretrained_llama_independent_replay_v36.json"
    corrupted = json.loads(replay.read_text())
    corrupted["verification"]["max_abs_error_hf"] = 0.5
    replay.write_text(json.dumps(corrupted))
    with pytest.raises(ArchivalError, match="CI artifact"):
        v36_graph(repo)
    assert len(before["nodes"]) == 14


def test_origin_tampering_cannot_turn_new_replay_into_accepted_evidence(tmp_path):
    repo = copied(tmp_path)
    origin = repo / "benchmarks/pretrained_llama_independent_replay_v36.origin.json"
    forged = json.loads(origin.read_text())
    forged["producing_ci_run"] = "https://github.com/Caerii/VisuaML/actions/runs/0"
    origin.write_text(json.dumps(forged))
    with pytest.raises(ArchivalError, match="origin"):
        v36_graph(repo)
