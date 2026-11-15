"""v13 report is cryptographically self-consistent and claim-limited."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.decoder_certification_v13 import certify_decoder_compiler


def test_checked_in_decoder_artifact_certificate_matches_semantic_source():
    public = Path(__file__).resolve().parents[2] / "public" / "decoder_compiler_certificate.v13.json"
    assert public.is_file()
    report = json.loads(public.read_text(encoding="utf-8"))
    payload = {k: v for k, v in report.items() if k != "sha256"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == report["sha256"]
    fresh = certify_decoder_compiler()
    assert report["source_sha256"] == fresh["source_sha256"]
    assert report["claim_level"] == "one-synthetic-tiny-decoder-full-chunk-compiler-execution"
    assert report["family_checkpoint_certified"] is False
    assert report["arbitrary_decoder_supported"] is False
    assert report["physical_accelerator_validated"] is False
    assert len(report["cases"]) == 3
    assert {x["seed"] for x in report["cases"]} == {0, 19, 41}
    assert all(case["witness"]["passed"] for case in fresh["cases"])
    assert all(case["generation_matches_reference"] for case in fresh["cases"])
