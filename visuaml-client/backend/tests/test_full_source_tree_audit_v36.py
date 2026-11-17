"""Regression: file-changing commits cannot substitute for full-tree search."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts.audit_full_source_tree_v36 import (
    AuditError,
    SOURCE_REVISIONS,
    expected_blob_hashes,
    run_audit,
    scan_exact_trees,
)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True, text=True,
    )
    return result.stdout.strip()


def test_tree_snapshot_can_match_when_path_change_commit_intersection_is_empty(tmp_path):
    """Three staggered file introductions yield a matching *later* tree."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Research Test")
    git(tmp_path, "config", "user.email", "research@example.invalid")
    paths = ("a.cpp", "b.py", "c.py")
    commits = []
    expected = {}
    for name in paths:
        (tmp_path / name).write_text(f"verified source for {name}\n")
        git(tmp_path, "add", name)
        git(tmp_path, "commit", "-qm", f"introduce {name}")
        commit = git(tmp_path, "rev-parse", "HEAD")
        commits.append(commit)
        expected[name] = git(tmp_path, "rev-parse", f"HEAD:{name}")

    file_changing = [
        set(git(tmp_path, "log", "--all", "--format=%H", "--", path).splitlines())
        for path in paths
    ]
    assert not set.intersection(*file_changing)
    matches = scan_exact_trees(
        tmp_path,
        git(tmp_path, "rev-list", "--all").splitlines(),
        expected,
    )
    assert matches == [commits[-1]]
    with pytest.raises(AuditError, match="Noncanonical"):
        scan_exact_trees(tmp_path, ["injected; command"], expected)


def test_real_source_blob_contract_is_recovered_from_exact_git_objects():
    root = Path(__file__).resolve().parents[3]
    candidate = next(iter(SOURCE_REVISIONS.values()))
    process = subprocess.run(
        ("git", "cat-file", "-e", f"{candidate}^{{commit}}"),
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode:
        pytest.skip("Full historical Git objects required (fetch-depth=0 CI)")
    blobs, hashes = expected_blob_hashes(root)
    assert len(blobs) == len(hashes) == 3
    assert all(len(value) == 40 for value in blobs.values())
    assert all(len(value) == 64 for value in hashes.values())


def test_exhaustive_real_history_reports_bounded_scope_and_no_ci_attribution():
    root = Path(__file__).resolve().parents[3]
    candidate = next(iter(SOURCE_REVISIONS.values()))
    process = subprocess.run(
        ("git", "cat-file", "-e", f"{candidate}^{{commit}}"),
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode:
        pytest.skip("Full historical Git history required")
    result = run_audit(root)
    assert result["schema"] == "visuaml.exhaustive-source-tree-audit.v36.1"
    assert result["scan_exhaustive_for_reachable_refs"] is True
    assert result["reachable_git_commits_scanned"] >= 3
    assert len(result["source_blob_git_sha1"]) == 3
    assert result["matching_tree_count"] == len(result["exact_matching_tree_commits"])
    assert result["original_ci_commit_established"] is False
