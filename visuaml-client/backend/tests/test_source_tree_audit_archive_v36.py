"""The v36.1 source correction is additive and byte-bound to CI.

Never rewrite the independently executed numerical record. The old archival
claim about no common commit is explicitly superseded, not silently erased.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.audit_full_source_tree_v36 import AuditError, verify_archived_correction

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "benchmarks/pretrained_llama_source_tree_audit_v36_1.json"
ORIGIN = ROOT / "benchmarks/pretrained_llama_source_tree_audit_v36_1.origin.json"
FRESH = ROOT / "benchmarks/pretrained_llama_independent_replay_v36.json"
FRESH_ORIGIN = ROOT / "benchmarks/pretrained_llama_independent_replay_v36.origin.json"
OLD = ROOT / "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json"

EXACT_COMMITS = {
    "12c6070373b92f1a3628581e0d37094ea5cc82db",
    "49313aac64d01c3f0cda175712f179ff526116b5",
    "71b575b2e7e5858bcc1e5beda5304fb884b2ed35",
    "852e2346ff4d67a1ece5e8e373f6b93a3646cc8a",
    "e1e92e8bf168cac89a418384e99e6cfa9176734c",
    "ff773237aa09fee8588733cd8a537ad456fe2202",
}


def source_audit() -> tuple[dict, dict]:
    exact = AUDIT.read_bytes()
    report = json.loads(exact)
    origin = json.loads(ORIGIN.read_bytes())
    if origin["audit_source_bytes_sha256"] != hashlib.sha256(exact).hexdigest():
        raise ValueError("Historical source audit byte identity changed")
    return report, origin


def test_corrective_audit_preserves_original_numerical_artifact():
    report, origin = source_audit()
    assert report["schema"] == "visuaml.exhaustive-source-tree-audit.v36.1"
    assert report["scan_exhaustive_for_reachable_refs"] is True
    assert report["reachable_git_commits_scanned"] == 620
    assert set(report["exact_matching_tree_commits"]) == EXACT_COMMITS
    assert report["matching_tree_count"] == len(EXACT_COMMITS)
    assert report["original_ci_commit_established"] is False
    assert origin["producing_ci_run"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44196790520"
    )
    assert origin["status"] == "corrected_source_provenance_not_signed"
    historical = json.loads(OLD.read_bytes())
    assert report["historical_artifact_sha256"] == hashlib.sha256(
        OLD.read_bytes()
    ).hexdigest()
    for path, digest in historical["matched_cpu_benchmark"]["source_sha256"].items():
        assert report["exact_original_file_sha256"][
            f"visuaml-client/backend/{path}"
        ] == digest
    fresh = json.loads(FRESH.read_bytes())
    fresh_origin = json.loads(FRESH_ORIGIN.read_bytes())
    assert fresh_origin["record_bytes_sha256"] == hashlib.sha256(
        FRESH.read_bytes()
    ).hexdigest()
    assert fresh["status"] == "fresh_independent_numerical_witness_passed"
    assert fresh["verification"]["passed"] is True
    # The erroneous old metadata is kept as historical data, and superseded.
    assert fresh["historical_source_recovery"][
        "single_commit_contains_all_recorded_sources"
    ] is False
    assert report["matching_tree_count"] > 0


def test_archive_verifier_rejects_mutated_audit_bytes(tmp_path):
    import shutil

    root = tmp_path
    for relative in (
        "benchmarks/pretrained_llama_source_tree_audit_v36_1.json",
        "benchmarks/pretrained_llama_source_tree_audit_v36_1.origin.json",
        "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)

    assert verify_archived_correction(root)["matching_tree_count"] == 6
    source = root / "benchmarks/pretrained_llama_source_tree_audit_v36_1.json"
    obj = json.loads(source.read_bytes())
    obj["matching_tree_count"] = 0
    source.write_text(json.dumps(obj))
    with pytest.raises(AuditError, match="origin disagrees"):
        verify_archived_correction(root)


def test_archived_origin_cannot_forge_original_ci_commit(tmp_path):
    import shutil

    root = tmp_path
    for relative in (
        "benchmarks/pretrained_llama_source_tree_audit_v36_1.json",
        "benchmarks/pretrained_llama_source_tree_audit_v36_1.origin.json",
        "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    origin = root / "benchmarks/pretrained_llama_source_tree_audit_v36_1.origin.json"
    obj = json.loads(origin.read_bytes())
    obj["original_ci_commit_established"] = True
    origin.write_text(json.dumps(obj))
    with pytest.raises(AuditError, match="origin disagrees"):
        verify_archived_correction(root)
