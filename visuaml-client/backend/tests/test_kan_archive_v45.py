"""v45 immutable KAN mechanism and real-pretrained surrogate provenance.

Evidence audits are NOT substitute for fresh independent ONNX, Wasmtime or
Z3 execution. Historical observation bytes must never be silently rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANNode,
    KANProgram,
    polynomial_identity,
)

ROOT = Path(__file__).resolve().parents[3]


def _archive(base: str, expected_sha: str, expected_bytes: int):
    path = ROOT / f"benchmarks/{base}.json"
    origin = ROOT / f"benchmarks/{base}.origin.json"
    raw = path.read_bytes()
    record, meta = json.loads(raw), json.loads(origin.read_bytes())
    assert len(raw) == expected_bytes == meta["raw_bytes"]
    assert hashlib.sha256(raw).hexdigest() == expected_sha == meta["raw_sha256"]
    assert meta["schema"] == "visuaml.evidence-origin.v45"
    assert meta["record_schema"] == record["schema"]
    assert meta["source_commit"] == record["source_commit"]
    assert meta["source_file_sha256"] == record["source_file_sha256"]
    assert meta["source_commit"] == "3ef0920640dd4a4b34aeb42294f0a9f5445526f9"
    assert meta["workflow_run_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44237380697"
    )
    assert meta["signed_external_attestation"] is False
    assert meta["generic_pretrained_mechanism_certified"] is False
    return record, meta


def test_constructed_kan_raw_archive_verifies_actual_graph_identity():
    record, origin = _archive(
        "kan_mechanism_replay_v45",
        "328a590239f763c8ba4ea440b57a62e3597d5cecc62ea08ded5ea82ae87d2229",
        74762,
    )
    assert record["schema"] == "visuaml.kan-symbolic-construction-replay.v45"
    assert record["full_domain_bounded_witnesses"] == len(record["witnesses"]) == 256
    assert all(item["pass"] for item in record["witnesses"])
    assert record["independently_normalized_polynomial_coefficients"] == ["7", "-3", "1"]
    assert record["all_i32_symbolic_vs_original_wasm"]["solver_outcome"] == "unsat"
    assert record["kan_is_pretrained_mechanism"] is False
    assert record["pretrained_feature_discovery_completed"] is False
    assert record["all_i32_kan_graph_proved"] is False
    assert record["kan_speedup_measured"] is False
    assert origin["actual_pretrained_checkpoint_sha256"] is None
    graph = record["kan_graph"]
    p = KANProgram(
        tuple(graph["inputs"]),
        tuple(KANNode(node["name"], node["combine"], tuple(
            EdgeFunction(
                edge["name"], edge["source"], edge["basis"],
                tuple(edge["coefficients"]), tuple(edge["parameters"])
            ) for edge in node["edges"]
        )) for node in graph["nodes"]),
        tuple(graph["outputs"]), tuple(graph["domain"]),
        graph["schema"],
    )
    assert p.digest() == record["kan_graph_digest"]
    assert tuple(polynomial_identity(p, output="result")) == (7, -3, 1)
    assert len(record["basis_extensions"]) == 4
    assert {v["basis"] for v in record["basis_extensions"]} == {
        "polynomial", "linear_spline", "gaussian_rbf", "fourier"
    }
    assert all(v["all_passed"] for v in record["basis_extensions"])


def test_pretrained_surrogate_archive_is_not_mislabeled_causal_mechanism():
    record, origin = _archive(
        "pretrained_kan_surrogate_v45",
        "0737a6ec5e84e3b5752825734528e5bf5b5ad040413d59c5cf421bff84cef28d",
        46695,
    )
    assert record["schema"] == "visuaml.pretrained-kan-surrogate-slice.v45"
    assert record["source_checkpoint_sha256"] == origin["actual_pretrained_checkpoint_sha256"]
    assert record["source_checkpoint_sha256"] == (
        "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
    )
    assert record["train_count"] == 33
    assert record["heldout_count"] == 32
    assert set(record["train_coordinates"]).isdisjoint(record["heldout_coordinates"])
    assert len(record["independent_scalar_numpy_oracles"]) == 3
    assert record["least_squares_fitted_surrogate"] is True
    assert record["trained_kan"] is False
    assert record["selection_uses_heldout_data"] is False
    assert record["pretrained_internal_mechanism_extracted"] is False
    assert record["teacher_perturbation_ablations_executed"] is False
    assert record["real_digit_accuracy_evaluated"] is False
    assert record["generalization_beyond_one_input_manifold_established"] is False
    assert record["all_variants_finite"]
    variants = {v["family"]: v for v in record["variants"]}
    assert set(variants) == {
        "linear", "polynomial", "linear_spline", "gaussian_rbf", "fourier"
    }
    assert record["selected_by_training_error_only"] == "gaussian_rbf"
    assert variants["linear_spline"]["heldout_mean_abs_logit_error"] < (
        variants["gaussian_rbf"]["heldout_mean_abs_logit_error"]
    )
    for variant in variants.values():
        assert variant["functional_surrogate_only"] is True
        assert variant["pretrained_internal_mechanism_recovered"] is False
        assert len(variant["graph"]["outputs"]) == 10
        assert 0. <= variant["heldout_top1_agreement"] <= 1.
        assert variant["heldout_mean_abs_logit_error"] >= 0.
