"""Immutable experiment-byte audit; not an independent native execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / 'benchmarks/pretrained_mnist8_onnx_preflight_v37.json'
ORIGIN = ROOT / 'benchmarks/pretrained_mnist8_onnx_preflight_v37.origin.json'


def test_archived_source_bound_model_preflight_remains_preflight_only():
    raw = RECORD.read_bytes()
    manifest = json.loads(ORIGIN.read_text())
    record = json.loads(raw)
    assert manifest['record_sha256'] == hashlib.sha256(raw).hexdigest()
    assert manifest['record_bytes'] == len(raw)
    assert manifest['record_path'] == 'benchmarks/pretrained_mnist8_onnx_preflight_v37.json'
    assert record['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert manifest['github_actions_artifact_reported_digest'].startswith('sha256:')
    assert manifest['github_actions_checkout_merge_commit'] == manifest['source_commit']
    assert record['status'] == 'preflight_only'
    assert record['executed_native'] is False
    assert record['numerical_oracle_executed'] is False
    assert record['op_histogram'] == manifest['observed_operator_histogram']
    assert sorted(record['operators_requiring_extension_or_independent_admission']) == (
        manifest['observed_nonv32_ops']
    )
    assert sum(record['op_histogram'].values()) == len(record['nodes']) == 12
    assert len(record['initializers']) == 8
    assert record['input']['shape'] == [1, 1, 28, 28]
    assert record['output']['shape'] == [1, 10]


def test_drifted_archived_bytes_fail_identity_gate():
    raw = RECORD.read_bytes()
    pinned = json.loads(ORIGIN.read_text())['record_sha256']
    assert hashlib.sha256(raw + b'\n').hexdigest() != pinned
    assert hashlib.sha256(raw.replace(b'preflight_only', b'numerically_verified')).hexdigest() != pinned
