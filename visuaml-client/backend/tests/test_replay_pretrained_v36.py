"""v36 historical source recovery and fresh-run result admission tests.

No Hugging Face downloads, compiler invocation or model execution in this
fast unit suite. The dedicated pinned-run job performs those independently.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from scripts.replay_pretrained_v36 import (
    CANDIDATE_COMMIT,
    ReplayError,
    load_archive,
    verify_fresh,
    verify_source,
)

ROOT = Path(__file__).resolve().parents[3]


def test_historical_source_matches_all_archived_sha256_bytes():
    recovery = verify_source(ROOT)
    assert recovery["schema"] == "visuaml.historical-source-recovery.v36"
    assert recovery["source_verified"] is True
    assert recovery["candidate_commit"] == CANDIDATE_COMMIT
    assert len(recovery["source_sha256"]) == 3
    assert all(len(value) == 64 for value in recovery["source_sha256"].values())


def test_historical_source_commit_is_explicitly_allowlisted():
    with pytest.raises(ReplayError, match="source allowlist"):
        verify_source(ROOT, "f" * 40)


def test_replay_validation_requires_exact_model_and_source_contract():
    archived, _ = load_archive(ROOT)
    recovery = verify_source(ROOT)
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
    recovery = verify_source(ROOT)

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
