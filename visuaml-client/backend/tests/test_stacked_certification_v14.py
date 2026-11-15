"""Source-bound, claim-limited receipts for compiled multi-layer decoder fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.stacked_certification_v14 import (
    CASES,
    SCHEMA,
    certify_stacked_decoder,
    semantic_source_sha256,
)


def test_stacked_certificate_executes_all_declared_depths():
    report = certify_stacked_decoder()
    assert report["schema"] == SCHEMA
    assert report["source_sha256"] == semantic_source_sha256()
    assert len(report["cases"]) == len(CASES)
    assert {c["layers"] for c in report["cases"]} == {1, 2, 4}
    for case in report["cases"]:
        assert case["witness"]["passed"]
        assert case["witness"]["arrays_checked"] == 3 * (1 + 2 * case["layers"])
        assert case["generation_matches_reference"]
        assert case["all_layer_cache_matches_reference"]
        assert len(case["checkpoint_sha256"]) == 64
    assert report["pretrained_checkpoint_certified"] is False
    assert report["physical_hardware_validated"] is False
    document = {k: v for k, v in report.items() if k != "sha256"}
    digest = hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert report["sha256"] == digest


def test_checked_in_stacked_certificate_has_current_semantic_source():
    public = (
        Path(__file__).resolve().parents[2]
        / "public"
        / "stacked_decoder_certificate.v14.json"
    )
    report = json.loads(public.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    assert report["source_sha256"] == semantic_source_sha256()
    assert {c["layers"] for c in report["cases"]} == {1, 2, 4}
    assert all(c["witness"]["passed"] for c in report["cases"])
    assert report["pretrained_checkpoint_certified"] is False
    assert report["physical_hardware_validated"] is False
