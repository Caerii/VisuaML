"""Exhaustive Git-tree audit of the archived v28 three-file source manifest.

A Git *file-change* log is not enough to show whether a later commit
contains all three file contents. This tool inspects every reachable commit
TREE against exact Git blob identities after checking each SHA-256.
A matching tree still does not prove the original CI ran that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ARCHIVE = "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json"
BACKEND = "visuaml-client/backend"
SOURCE_REVISIONS = {
    "examples/native_cpu_bench_v28.py": "12c6070373b92f1a3628581e0d37094ea5cc82db",
    "visuaml/compiler_lab/native_cpu_v28.cpp": "9afd8ed006ea40b2f1010c0909d510a1a89a476b",
    "visuaml/compiler_lab/native_cpu_v28.py": "ac8ad81bf22b6e8eef8e43f775e7006c2fffa39e",
}
MAX_COMMITS = 25000


class AuditError(ValueError):
    """Incomplete Git history or historical content contract mismatch."""


def git(root: Path, *args: str, timeout: int = 60) -> bytes:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=root, capture_output=True, check=True, timeout=timeout,
        ).stdout
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise AuditError("Historical Git tree query failed") from exc


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expected_blob_hashes(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    try:
        archive = json.loads((root / ARCHIVE).read_bytes())
    except (OSError, ValueError) as exc:
        raise AuditError("Historical numerical archive unavailable") from exc
    expected = archive["matched_cpu_benchmark"]["source_sha256"]
    if not isinstance(expected, dict) or set(expected) != set(SOURCE_REVISIONS):
        raise AuditError("Historical source-file inventory differs from manifest")
    blob_ids = {}
    verified_sha = {}
    for relative, introducing in SOURCE_REVISIONS.items():
        tracked = f"{BACKEND}/{relative}"
        commit = git(root, "rev-parse", "--verify", f"{introducing}^{{commit}}").decode().strip()
        if commit != introducing:
            raise AuditError("Unexpected source blob commit identity")
        data = git(root, "show", f"{introducing}:{tracked}")
        digest = sha256(data)
        if digest != expected[relative]:
            raise AuditError(f"Exact source SHA-256 mismatch: {relative}")
        blob = git(root, "rev-parse", "--verify", f"{introducing}:{tracked}").decode().strip()
        if len(blob) != 40:
            raise AuditError("Unexpected Git blob identity length")
        blob_ids[tracked] = blob
        verified_sha[tracked] = digest
    return blob_ids, verified_sha


def scan_exact_trees(
    repo: Path, commits: list[str], expected_blobs: dict[str, str],
) -> list[str]:
    """Inspect every listed commit's full tree, never only per-path changes."""
    matching = []
    paths = tuple(sorted(expected_blobs))
    if not commits or len(commits) > MAX_COMMITS:
        raise AuditError("Unbounded or empty commit list")
    if not 1 <= len(paths) <= 16:
        raise AuditError("Invalid source inventory")
    for commit in commits:
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise AuditError("Noncanonical Git commit identifier")
        entries = git(repo, "ls-tree", "-z", commit, "--", *paths, timeout=20)
        observed = {}
        for item in entries.split(b"\0"):
            if not item:
                continue
            try:
                meta, filename = item.split(b"\t", 1)
                mode, kind, object_id = meta.decode("ascii").split(" ")
                name = filename.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise AuditError("Malformed Git tree metadata") from exc
            if mode != "100644" or kind != "blob":
                continue
            if name in expected_blobs:
                observed[name] = object_id
        if all(observed.get(name) == expected_blobs[name] for name in paths):
            matching.append(commit)
    return sorted(matching)


def run_audit(repo: Path) -> dict:
    root = Path(repo).resolve()
    expected_blobs, expected_hashes = expected_blob_hashes(root)
    commits = git(root, "rev-list", "--all", timeout=120).decode().splitlines()
    if not commits or len(commits) > MAX_COMMITS:
        raise AuditError("Cannot make exhaustive claim about unbounded history")
    matching = scan_exact_trees(root, commits, expected_blobs)
    return {
        "schema": "visuaml.exhaustive-source-tree-audit.v36.1",
        "historical_artifact_sha256": sha256((root / ARCHIVE).read_bytes()),
        "reachable_git_commits_scanned": len(commits),
        "scan_exhaustive_for_reachable_refs": True,
        "exact_original_file_sha256": expected_hashes,
        "source_blob_git_sha1": expected_blobs,
        "exact_matching_tree_commits": sorted(matching),
        "matching_tree_count": len(matching),
        "original_ci_commit_established": False,
        "statement": (
            "Exact full Git-tree matching is independently audited across "
            "all reachable refs. Matching tree bytes do not prove the old "
            "CI runner used any particular commit."
        ),
        "corrects": (
            "Prior per-file history intersection examined file-changing "
            "commits only; it could not establish the absence of a later "
            "commit containing all three source blobs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(args.repo)
    wire = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(wire, encoding="utf-8")
    print(wire)


if __name__ == "__main__":
    main()
