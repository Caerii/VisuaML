"""v36 durable real-model numerical evidence must be byte- and source-bound."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from scripts.archive_pretrained_replay_v36 import (
    ARCHIVE,
    BUNDLE_SHA256,
    CI_RUN,
    EVIDENCE_BYTES_SHA256,
    ArchivalError,
    origin_manifest,
    prepare,
    verify_artifact,
    verify_checked_in_archive,
)

ROOT = Path(__file__).resolve().parents[3]
OLD = "benchmarks/pretrained_llama_native_cpu_benchmark_v28.json"


def test_fresh_pinned_replay_is_durable_and_has_a_distinct_ci_origin():
    origin = verify_checked_in_archive(ROOT)
    raw = (ROOT / ARCHIVE).read_bytes()
    record = verify_artifact(raw, ROOT)
    assert hashlib.sha256(raw).hexdigest() == EVIDENCE_BYTES_SHA256
    assert origin["producing_ci_run"] == CI_RUN
    assert origin["reconstructed_source_bundle_sha256"] == BUNDLE_SHA256
    assert record["status"] == "fresh_independent_numerical_witness_passed"
    assert record["verification"]["passed"] is True
    assert record["verification"]["witness_count"] == 2
    assert record["verification"]["max_abs_error_hf"] < 3e-4
    assert record["verification"]["max_abs_error_semantic"] < 3e-4
    assert record["historical_source_recovery"]["original_ci_source_commit"] is None
    assert record["historical_source_recovery"][
        "single_commit_contains_all_recorded_sources"
    ] is False


def test_replay_archive_is_copy_exact_and_idempotent(tmp_path):
    old = tmp_path / OLD
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / OLD, old)
    raw = (ROOT / ARCHIVE).read_bytes()
    first = prepare(tmp_path, raw)
    second = prepare(tmp_path, raw)
    assert first == second == origin_manifest(raw)
    assert (tmp_path / ARCHIVE).read_bytes() == raw
    assert verify_checked_in_archive(tmp_path) == first


def test_archival_tampering_and_forged_result_fail_closed(tmp_path):
    old = tmp_path / OLD
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / OLD, old)
    raw = (ROOT / ARCHIVE).read_bytes()
    prepare(tmp_path, raw)
    changed = json.loads(raw)
    changed["fresh_numerical_result"]["witnesses"][0][
        "max_abs_error_against_HF"
    ] = 0.5
    with pytest.raises(ArchivalError, match="CI artifact"):
        verify_artifact(json.dumps(changed).encode(), tmp_path)
    original = tmp_path / ARCHIVE
    original.write_bytes(raw + b" ")
    with pytest.raises(ArchivalError, match="CI artifact"):
        verify_checked_in_archive(tmp_path)
    with pytest.raises(ArchivalError, match="overwrite"):
        prepare(tmp_path, raw)


def test_origin_change_fails_even_when_record_bytes_match(tmp_path):
    old = tmp_path / OLD
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / OLD, old)
    raw = (ROOT / ARCHIVE).read_bytes()
    prepare(tmp_path, raw)
    path = tmp_path / "benchmarks/pretrained_llama_independent_replay_v36.origin.json"
    forged = json.loads(path.read_text())
    forged["producing_ci_run"] = "https://github.com/Caerii/VisuaML/actions/runs/123"
    path.write_text(json.dumps(forged))
    with pytest.raises(ArchivalError, match="origin"):
        verify_checked_in_archive(tmp_path)
