"""v36 historical source recovery and fresh-run result admission tests.

No Hugging Face downloads, compiler invocation or model execution in this
fast unit suite. The dedicated pinned-run job performs those independently.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
from scripts.replay_pretrained_v36 import (
    CANDIDATE_COMMIT,
    SOURCE_BLOB_COMMITS,
    ReplayError,
    load_archive,
    verify_fresh,
    verify_source,
)

ROOT = Path(__file__).resolve().parents[3]


def _requires_full_history() -> None:
    """Full Git source-history tests run in the dedicated fetch-depth=0 job."""
    found = subprocess.run(
        ("git", "cat-file", "-e", f"{CANDIDATE_COMMIT}^{{commit}}"),
        cwd=ROOT, capture_output=True, check=False,
    )
    if found.returncode:
        pytest.skip("Full Git history required; independently run in v36 source CI")


def _admitted_source_contract() -> dict:
    historical, _ = load_archive(ROOT)
    return {
        "source_sha256": historical["matched_cpu_benchmark"]["source_sha256"],
    }


def test_historical_source_matches_all_archived_sha256_bytes():
    _requires_full_history()
    recovery = verify_source(ROOT)
    assert recovery["schema"] == "visuaml.historical-source-bundle.v36"
    assert recovery["source_verified"] is True
    assert recovery["base_checkout_commit"] == CANDIDATE_COMMIT
    assert recovery["source_file_commits"] == SOURCE_BLOB_COMMITS
    assert len(recovery["source_sha256"]) == 3
    assert all(len(value) == 64 for value in recovery["source_sha256"].values())
    assert recovery["single_commit_contains_all_recorded_sources"] is True
    assert recovery["matching_source_tree_count"] == 6
    assert recovery["original_ci_source_commit"] is None
    assert len(set(recovery["source_file_commits"].values())) == 3
    assert len(recovery["reconstructed_source_bundle_sha256"]) == 64


def test_historical_base_commit_is_explicitly_allowlisted():
    with pytest.raises(ReplayError, match="source allowlist"):
        verify_source(ROOT, "f" * 40)


def test_replay_validation_requires_exact_model_and_source_contract():
    archived, _ = load_archive(ROOT)
    recovery = _admitted_source_contract()
    # Unit exercise only: old data is NOT presented as a fresh run.
    valid = verify_fresh(copy.deepcopy(archived), archived, recovery)
    assert valid["passed"] is True
    assert valid["witness_count"] == 2
    assert valid["max_abs_error_hf"] < 3e-4
    assert valid["tolerance_atol"] == 3e-4

    changed = copy.deepcopy(archived)
    changed["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ReplayError, match="pinned model contract"):
        verify_fresh(changed, archived, recovery)

    changed = copy.deepcopy(archived)
    changed["matched_cpu_benchmark"]["source_sha256"][
        "visuaml/compiler_lab/native_cpu_v28.cpp"
    ] = "a" * 64
    with pytest.raises(ReplayError, match="source or semantic identity|workload"):
        verify_fresh(changed, archived, recovery)


def test_fresh_model_numerics_fail_closed_on_missing_or_wrong_witness():
    archived, _ = load_archive(ROOT)
    recovery = _admitted_source_contract()

    changed = copy.deepcopy(archived)
    changed["witnesses"][0]["tokens"][0][0] = 11
    with pytest.raises(ReplayError, match="tokens or logits shape"):
        verify_fresh(changed, archived, recovery)

    changed = copy.deepcopy(archived)
    changed["witnesses"][0]["max_abs_error_against_HF"] = 1.0
    with pytest.raises(ReplayError, match="numerical tolerance"):
        verify_fresh(changed, archived, recovery)

    changed = copy.deepcopy(archived)
    changed["witnesses"][0]["max_abs_error_against_HF"] = float("nan")
    with pytest.raises(ReplayError, match="nonfinite"):
        verify_fresh(changed, archived, recovery)

    changed = copy.deepcopy(archived)
    changed["chunked_KV_and_greedy_parity"] = False
    with pytest.raises(ReplayError, match="pinned model contract"):
        verify_fresh(changed, archived, recovery)


def test_historical_archive_has_no_fabricated_replay_metadata():
    archive, digest = load_archive(ROOT)
    assert len(digest) == 64
    assert archive["evidence_provenance"]["workflow_url"].startswith(
        "https://github.com/Caerii/VisuaML/actions/runs/"
    )
    assert archive["revision"] == "756e9746eb801752d69771ef3e9bc433b21d853d"
    # An old successful result cannot be transformed into a signed or fresh
    # verification merely by copying its JSON record.
    assert archive.get("schema") != "visuaml.pinned-pretrained-independent-replay.v36"


def test_recovery_distinguishes_matching_trees_from_original_ci_checkout():
    from scripts.replay_pretrained_v36 import discover_historical_sources

    _requires_full_history()
    forensics = discover_historical_sources(ROOT)
    assert forensics["schema"] == "visuaml.historical-source-discovery.v36"
    assert forensics["candidate_is_complete_match"] is False
    assert len(forensics["complete_matching_commits"]) == 6
    assert forensics["commits_scanned_for_complete_source"] >= 620
    assert forensics["found_all_files"] is True
    for path, commit in SOURCE_BLOB_COMMITS.items():
        assert commit in forensics["file_matching_commits"][path]
