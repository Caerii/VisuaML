"""Fail closed if the reviewed, backend-free public evidence snapshot drifts."""

from __future__ import annotations

import hashlib
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


def test_public_summary_replays_the_permanent_source_bound_ci_record():
    """The read-only site cannot invent results absent from archived evidence."""
    root = Path(__file__).resolve().parents[3]
    archive = root / "benchmarks" / "export_evidence_observation_v33.json"
    origin_file = root / "benchmarks" / "export_evidence_observation_v33.origin.json"
    asset = root / "visuaml-client" / "public" / "evidence" / "v33-observation.json"
    exact_bytes = archive.read_bytes()
    full = json.loads(exact_bytes)
    origin = json.loads(origin_file.read_text(encoding="utf-8"))
    visible = json.loads(asset.read_text(encoding="utf-8"))

    assert origin["schema"] == "visuaml.durable-evidence-origin.v1"
    assert origin["evidence_sha256"] == hashlib.sha256(exact_bytes).hexdigest()
    assert origin["source_commit"] == visible["sourceCommit"]
    assert origin["ci_run"] == visible["workflowRun"]
    assert full["schema"] == "visuaml.export-evidence-demonstration.v33"
    assert len(full["model_witnesses"]) == len(visible["models"]) == 2

    for published, witness in zip(
        visible["models"], full["model_witnesses"], strict=True
    ):
        compiler = witness["compiler"]
        assert published["operations"] == compiler["op_sequence"]
        assert published["semanticSha256"] == compiler["semantic_sha256"]
        assert published["generatedCppSha256"] == compiler["generated_cpp_sha256"]
        assert published["binarySha256"] == compiler["native_binary_sha256"]
        assert published["compileMs"] == compiler["compilation_ms"]
        assert published["warmMedianMs"] == witness["warm_native_ms"]["median"]
        assert published["maximumAbsoluteError"] == witness["max_abs_error_eager_torch"]
        assert (
            published["secondInputMaximumAbsoluteError"]
            == witness["max_abs_error_second_input"]
        )
        assert hashlib.sha256(witness["generated_cpp"].encode()).hexdigest() == (
            compiler["generated_cpp_sha256"]
        )

    published_rewrite = visible["transformation"]
    full_rewrite = full["rewrite_witness"]
    full_timing = full["optimization_experiment"]
    assert published_rewrite["numericalStatus"] == full_rewrite["status"]
    assert published_rewrite["evidenceSha256"] == full_rewrite["record_sha256"]
    assert (
        published_rewrite["experimentalStatus"] == full_timing["status"]
    )
    assert published_rewrite["medianRatioBeforeOverAfter"] == (
        full_timing["median_ratio_before_over_after"]
    )
    unsigned = {
        key: value for key, value in full_rewrite.items()
        if key != "record_sha256"
    }
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == full_rewrite["record_sha256"]
