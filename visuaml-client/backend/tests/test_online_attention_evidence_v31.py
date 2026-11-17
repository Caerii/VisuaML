"""The v31 evidence snapshot must remain tied to exact tested source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "visuaml-client" / "backend"
EVIDENCE = ROOT / "benchmarks" / "native_online_attention_observation_v31.json"


def test_online_attention_observation_binds_to_source_and_ci():
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert record["schema"] == "visuaml.native-online-attention-observation.v31"
    assert record["evidenceOrigin"]["nativeWorkflowUrl"].startswith(
        "https://github.com/Caerii/VisuaML/actions/runs/"
    )
    for path, expected in record["sourceSha256"].items():
        observed = hashlib.sha256((BACKEND / path).read_bytes()).hexdigest()
        assert observed == expected, f"Evidence source changed: {path}"
    assert len(record["pretrained"]["checkpointSha256"]) == 64
    assert record["pretrained"]["trials"] >= 2
    assert len(record["pretrained"]["pastTokens"]) == len(
        record["pretrained"]["ratios"]
    )
    assert all(x > 0 for x in record["pretrained"]["ratios"])
    for case in record["synthetic"]["shapes"]:
        assert len(case["ratios"]) == len(
            record["synthetic"]["pastTokens"]
        )
        assert all(x > 0 for x in case["ratios"])
    assert "Do not switch the default" in record["decision"]
