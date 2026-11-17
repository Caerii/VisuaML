"""Fail closed if the reviewed, backend-free public evidence snapshot drifts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


def test_published_v33_evidence_is_bounded_and_source_linked():
    asset = (
        Path(__file__).resolve().parents[2]
        / "public" / "evidence" / "v33-observation.json"
    )
    data = json.loads(asset.read_text(encoding="utf-8"))
    assert data["schema"] == "visuaml.public-evidence-snapshot.v1"
    assert data["asOf"] == "2026-11-17"
    assert re.fullmatch(r"[0-9a-f]{40}", data["sourceCommit"])
    assert re.fullmatch(
        r"https://github\.com/Caerii/VisuaML/actions/runs/\d+",
        data["workflowRun"],
    )
    assert len(data["models"]) == 2
    for model in data["models"]:
        assert model["operations"]
        assert all(isinstance(op, str) for op in model["operations"])
        for key in ("semanticSha256", "generatedCppSha256", "binarySha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", model[key])
        for key in (
            "maximumAbsoluteError",
            "secondInputMaximumAbsoluteError",
            "compileMs", "warmMedianMs",
        ):
            assert math.isfinite(model[key]) and model[key] >= 0
    transform = data["transformation"]
    assert transform["rule"] == "relu-idempotence-v1"
    assert transform["numericalStatus"] == "passed_finite_witnesses"
    assert transform["experimentalStatus"] == (
        "exploratory_timing_after_finite_numerical_witnesses"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", transform["evidenceSha256"])
    assert 0 < transform["medianRatioBeforeOverAfter"] < 1
    assert any("synthetic" in x.lower() for x in data["limitations"])
    assert any("not authenticated" in x.lower() for x in data["limitations"])
