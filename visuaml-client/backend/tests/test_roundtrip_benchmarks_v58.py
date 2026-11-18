"""Contract tests for v58 round-trip and representation-cost reporting.

Synthetic timing fixtures test arithmetic and scope controls, never claim
actual speedup or a completed source-WASM experiment.
"""
from __future__ import annotations

import json

import pytest

from scripts.benchmark_representation_cost_v58 import compare_pair
from scripts.summarize_neural_roundtrips_v58 import summarize
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    SCHEMA, source_workloads,
)


def synthetic_archive(*, all_i32_weight_proof=False):
    cases = [
        {
            "name": "synthetic_not_a_real_result",
            "split": "historical",
            "original_wasm_sha256": "a" * 64,
            "recovered_wasm_sha256": "b" * 64,
            "all_i32_typed_source_equivalence": True,
            "all_i32_neural_weights_equivalence": all_i32_weight_proof,
            "arbitrary_learned_neural_model_decompiled": False,
            "same_source_binary": False,
            "finite_results": [
                {"input_i32": -1, "result_i32": 0},
                {"input_i32": 0, "result_i32": 1},
            ],
            "finite_replay_inputs": 2,
            "cegis_refinements": 2,
            "cegis_observation_count": 9,
            "cegis_recovery_and_smt_wall_ns": 190000,
            "original_wasm_bytes": 70,
            "recovered_wasm_bytes": 35,
        },
    ]
    return {
        "schema": SCHEMA,
        "source": {"commit": "c" * 40, "tree": "d" * 40},
        "inventory": {"source_programs": 1},
        "results": cases,
    }


def test_roundtrip_aggregate_uses_only_real_record_fields():
    raw = json.dumps(synthetic_archive()).encode()
    result = summarize(raw)
    assert result["source_programs"] == 1
    assert result["finite_execution_replays"] == 2
    assert result["claimed_full_domain_constructed_neural_weight_proofs"] == 0
    assert result["claimed_unknown_learned_weight_decompilations"] == 0
    assert result["splits"]["historical"]["median_cegis_refinements"] == 2
    assert result["splits"]["historical"][
        "median_recovered_to_original_size_ratio"
    ] == .5
    assert result["attempted_failed_recoveries_not_scored_by_this_strict_archive"]


def test_roundtrip_aggregate_rejects_overclaimed_neural_full_i32_proof():
    with pytest.raises(ValueError, match="invalid proof-scope"):
        summarize(json.dumps(synthetic_archive(all_i32_weight_proof=True)).encode())


def test_roundtrip_aggregate_rejects_partial_witnesses():
    altered = synthetic_archive()
    altered["results"][0]["finite_replay_inputs"] = 3
    with pytest.raises(ValueError, match="invalid proof-scope"):
        summarize(json.dumps(altered).encode())


def test_roundtrip_source_inventory_has_five_historical_eight_heldout():
    inventory = source_workloads()
    assert len(inventory) == 13
    assert sum(x.split == "historical" for x in inventory) == 5
    assert sum(x.split == "heldout" for x in inventory) == 8


def test_representation_timing_checks_every_warm_and_measured_result():
    measured = compare_pair(
        lambda: 7, lambda: 7, 7, warmups=1, repeats=5,
    )
    assert measured["samples_per_representation"] == 5
    assert len(measured["original_sample_ns"]) == 5
    assert len(measured["recovered_sample_ns"]) == 5
    assert measured["original_median_ns"] >= 0
    assert measured["recovered_median_ns"] >= 0
    assert measured["compilation_excluded"]
    with pytest.raises(AssertionError, match="semantically valid"):
        compare_pair(
            lambda: 8, lambda: 7, 7, warmups=1, repeats=1,
        )
    with pytest.raises(AssertionError, match="wrong result"):
        compare_pair(
            lambda: 7, lambda: 8, 7, warmups=0, repeats=1,
        )
