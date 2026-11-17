"""v34 historical knowledge library: exact provenance, compatibility, negatives."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from visuaml.compiler_lab.evidence_library_v34 import (
    EvidenceLibrary,
    EvidenceLibraryError,
    submit_unreviewed_proposal,
    verify_unreviewed_proposal,
)

ROOT = Path(__file__).resolve().parents[3]
PATHS = (
    "benchmarks/native_online_attention_observation_v31.json",
    "benchmarks/native_semantic_codegen_observation_v32.json",
    "benchmarks/export_evidence_observation_v33.json",
    "benchmarks/export_evidence_observation_v33.origin.json",
)


def temp_repo(tmp_path: Path) -> Path:
    for relative in PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_library_indexes_three_sources_without_elevating_evidence():
    library = EvidenceLibrary.from_repo(ROOT)
    assert len(library.entries) == 11
    catalog = library.catalog()
    assert catalog["schema"] == "visuaml.evidence-library.v34"
    assert catalog["counts"]["source_artifacts"] == 3
    assert catalog["counts"]["kinds"] == {
        "attention_context_sweep": 4,
        "native_exported_graph": 2,
        "native_generated_graph": 4,
        "rewrite_evaluation": 1,
    }
    assert len(catalog["catalog_sha256"]) == 64
    assert all("proof" not in entry.status for entry in library.entries)
    assert all(
        entry.evidence_class != "formal_proof" for entry in library.entries
    )
    assert catalog == EvidenceLibrary.from_repo(ROOT).catalog()


def test_exact_binary_lineage_connects_old_and_new_frontends():
    library = EvidenceLibrary.from_repo(ROOT)
    exported = library.query(
        kind="native_exported_graph",
        op_sequence=("matmul", "relu", "matmul", "add"),
        context={"frontend": "trusted_static_torch_export"},
    )
    assert len(exported) == 1
    source = exported[0]
    neighbors = library.neighbors(source.entry_id)
    matching = {
        item.entry_id: item for item in library.entries
        if item.native_binary_sha256 == source.native_binary_sha256
    }
    assert len(matching) >= 3
    assert {item["entry_id"] for item in neighbors} == (
        set(matching) - {source.entry_id}
    )
    assert {item["relation"] for item in neighbors} == {
        "same_native_binary_hash"
    }


def test_identity_and_hardware_queries_fail_closed():
    library = EvidenceLibrary.from_repo(ROOT)
    record = library.query(kind="native_exported_graph")[0]
    assert library.query(
        semantic_sha256=record.semantic_sha256,
        native_binary_sha256=record.native_binary_sha256,
    ) == (record,)
    assert library.query(
        semantic_sha256=record.semantic_sha256,
        context={"native_target": "nonexistent-cpu"},
    ) == ()
    assert library.query(
        op_sequence=record.op_sequence,
        context={"model": "pretrained"},
    ) == ()
    assert library.query(
        checkpoint_sha256="a" * 64,
    ) == ()
    with pytest.raises(EvidenceLibraryError, match="SHA-256"):
        library.query(semantic_sha256="injected")
    with pytest.raises(EvidenceLibraryError, match="Unknown"):
        library.neighbors("not-an-identity")


def test_library_preserves_negative_attentional_performance():
    library = EvidenceLibrary.from_repo(ROOT)
    sweeps = library.query(kind="attention_context_sweep")
    assert len(sweeps) == 4
    assert all(item.status == "mixed_not_promoted" for item in sweeps)
    assert any(dict(item.result)["min_speed_ratio"] < 1 for item in sweeps)
    assert library.query(
        kind="attention_context_sweep",
        checkpoint_sha256=(
            "d910edc7031721de23a7af111e355e6c09d5f1b3a2c43c0b6b0aa6a9135ffe03"
        ),
    ) == (next(
        item for item in sweeps if item.label == "pinned-tiny-llama"
    ),)


def test_library_preserves_negative_rewrite_timing():
    library = EvidenceLibrary.from_repo(ROOT)
    record, = library.query(kind="rewrite_evaluation")
    assert record.status == "passed_finite_witnesses"
    assert dict(record.result)["before_over_after_latency_ratio"] < 1
    assert record.evidence_class == "source_bound_finite_numeric_witnesses"
    # A passing numerical witness never implies a faster implementation.
    assert "optimization_accepted" not in record.status


def test_source_bytes_and_origin_digest_must_match(tmp_path):
    repo = temp_repo(tmp_path)
    assert len(EvidenceLibrary.from_repo(repo).entries) == 11
    file = repo / "benchmarks/export_evidence_observation_v33.json"
    data = json.loads(file.read_text())
    data["model_witnesses"][0]["max_abs_error_eager_torch"] = 0.5
    file.write_text(json.dumps(data))
    with pytest.raises(EvidenceLibraryError, match="original CI byte digest"):
        EvidenceLibrary.from_repo(repo)


def test_generated_cpp_source_is_checked_even_if_origin_sha_updated(tmp_path):
    repo = temp_repo(tmp_path)
    file = repo / "benchmarks/export_evidence_observation_v33.json"
    origin = repo / "benchmarks/export_evidence_observation_v33.origin.json"
    data = json.loads(file.read_text())
    data["model_witnesses"][0]["generated_cpp"] += "// silent tampering"
    file.write_text(json.dumps(data, sort_keys=True))
    metadata = json.loads(origin.read_text())
    metadata["evidence_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    origin.write_text(json.dumps(metadata))
    with pytest.raises(EvidenceLibraryError, match=r"Generated C\+\+ bytes diverged"):
        EvidenceLibrary.from_repo(repo)


def test_curated_source_observation_does_not_automatically_gain_new_claims(tmp_path):
    repo = temp_repo(tmp_path)
    file = repo / "benchmarks/native_online_attention_observation_v31.json"
    record = json.loads(file.read_text())
    record["synthetic"]["shapes"][0]["ratios"].append(100.0)
    file.write_text(json.dumps(record))
    with pytest.raises(EvidenceLibraryError, match="context study geometry"):
        EvidenceLibrary.from_repo(repo)


def test_unreviewed_proposals_are_content_addressed_idempotent_and_excluded(tmp_path):
    original = ROOT / "benchmarks/export_evidence_observation_v33.json"
    data = original.read_bytes()
    url = "https://github.com/Caerii/VisuaML/actions/runs/44167207128"
    first = submit_unreviewed_proposal(tmp_path, data, source_url=url)
    second = submit_unreviewed_proposal(tmp_path, data, source_url=url)
    assert first == second
    assert first.read_bytes() == data
    assert verify_unreviewed_proposal(first)
    assert first.stem == hashlib.sha256(data).hexdigest()
    assert len(EvidenceLibrary.from_repo(ROOT).entries) == 11
    first.write_bytes(data + b" ")
    assert not verify_unreviewed_proposal(first)
    with pytest.raises(EvidenceLibraryError, match="tampering"):
        submit_unreviewed_proposal(tmp_path, data, source_url=url)


def test_proposals_cannot_become_reviewed_or_executable_by_shape(tmp_path):
    with pytest.raises(EvidenceLibraryError, match="schema"):
        submit_unreviewed_proposal(
            tmp_path, b'{"schema":"arbitrary-python-execution"}',
            source_url="https://github.com/Caerii/VisuaML/actions/runs/1",
        )
    with pytest.raises(EvidenceLibraryError, match="CI provenance"):
        submit_unreviewed_proposal(
            tmp_path, b'{"schema":"visuaml.export-evidence-demonstration.v33"}',
            source_url="https://attacker.invalid/run",
        )
    with pytest.raises(EvidenceLibraryError, match="byte budget"):
        submit_unreviewed_proposal(
            tmp_path, b"a" * (1024 * 1024 + 1),
            source_url="https://github.com/Caerii/VisuaML/actions/runs/1",
        )
