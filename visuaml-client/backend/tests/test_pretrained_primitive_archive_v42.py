"""Immutable historical v42 candidate survey, including failed compression results.

Hash/consistency verification does not independently re-run ONNX inference.
The separate v42 workflow executes the real pinned checkpoint and oracles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "benchmarks/pretrained_primitive_screening_v42.json"
ORIGIN = ROOT / "benchmarks/pretrained_primitive_screening_v42.origin.json"


def test_archival_evidence_and_origin_have_exact_byte_identity():
    raw = RAW.read_bytes()
    record = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == origin["raw_sha256"]
    assert len(raw) == origin["raw_bytes"] == 54727
    assert record["schema"] == "visuaml.pretrained-primitive-screening.v42"
    assert origin["schema"] == "visuaml.pretrained-primitive-screening-origin.v42"
    assert record["source_commit"] == origin["source_commit"]
    assert record["source_file_sha256"] == origin["source_file_sha256"]
    assert record["checkpoint_sha256"] == origin["checkpoint_sha256"]
    assert record["source_weight_sha256"] == origin["original_weight_sha256"]
    assert origin["workflow_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44222189766"
    )
    assert record["checkpoint_sha256"] == (
        "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
    )
    assert record["source_weight_sha256"] == (
        "418379b078799df7956f1bd51e1839a728002f001228aba5b81ac67ad6e26772"
    )
    assert record["source_commit"] == (
        "f58f2965110079f96a3a895aae5e45a9a10f6a59"
    )
    assert origin["authentication"].startswith("CI origin")


def test_negative_result_preserved_without_false_compression_certification():
    record = json.loads(RAW.read_bytes())
    assert record["pretrained_model_accuracy_evaluated"] is False
    assert record["generalizable_mechanism_certified"] is False
    assert record["source_checkpoint_modified"] is False
    assert record["compressed_candidate_accepted"] is False
    assert record["selected_rank"] is None
    assert record["heldout_pass"] is False
    assert record["full_rank_reconstruction_control_pass"] is True
    assert record["discovery_input_count"] == 5
    assert record["heldout_input_count"] == 5
    assert record["threshold_relative_l2"] == 0.05
    ranks = {candidate["name"]: candidate for candidate in record["results"]}
    assert set(ranks) == {
        "rank_2", "rank_4", "rank_8", "rank_10", "rank_4_coordinate_control"
    }
    assert ranks["rank_4"]["retained_spectral_energy"] == (
        ranks["rank_4_coordinate_control"]["retained_spectral_energy"]
    )
    assert ranks["rank_8"]["retained_spectral_energy"] > 0.9
    assert ranks["rank_8"]["retained_spectral_energy"] < 1
    for candidate in ranks.values():
        assert len(candidate["witnesses"]) == 10
        assert candidate["independent_oracle_max_abs_error"] < 1e-4
        discovery = [w for w in candidate["witnesses"] if w["group"] == "discovery"]
        heldout = [w for w in candidate["witnesses"] if w["group"] == "heldout"]
        assert len(discovery) == len(heldout) == 5
        assert all(len(w["input_sha256"]) == 64 for w in candidate["witnesses"])
        assert all(w["numpy"]["relative_l2_logit_change"] >= 0
                   for w in candidate["witnesses"])
    assert sum(w["numpy"]["top1_unchanged"] for w in ranks["rank_8"]["witnesses"]
               if w["group"] == "heldout") == 3
    assert sum(w["numpy"]["top1_unchanged"] for w in ranks["rank_4"]["witnesses"]
               if w["group"] == "heldout") == 1
    assert sum(w["numpy"]["top1_unchanged"] for w in
               ranks["rank_4_coordinate_control"]["witnesses"]
               if w["group"] == "heldout") == 0
