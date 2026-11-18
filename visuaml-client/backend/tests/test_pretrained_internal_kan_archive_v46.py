"""v46 immutable original pretrained internal KAN/causal record verification.

A historical record is never rewritten to track later code changes. These
tests check archived bytes, source provenance, numerical summary consistency,
and negative scientific claims; fresh real-model CI is a separate gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from visuaml.compiler_lab.kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANNode,
    KANProgram,
    execute_kan_vector,
)

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "benchmarks/pretrained_internal_kan_causality_v46.json"
ORIGIN = ROOT / "benchmarks/pretrained_internal_kan_causality_v46.origin.json"
HISTORICAL_SHA256 = "180d50d2bc1d444848050a64437860479d657427dffd8dbc42694d9be6977d00"


def _files():
    raw = ARCHIVE.read_bytes()
    report = json.loads(raw)
    origin = json.loads(ORIGIN.read_bytes())
    assert len(raw) == origin["raw_bytes"] == 145852
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_SHA256
    assert origin["raw_sha256"] == HISTORICAL_SHA256
    return report, origin


def _graph(data):
    nodes = []
    for node in data["nodes"]:
        edges = tuple(
            EdgeFunction(
                edge["name"], edge["source"], edge["basis"],
                tuple(edge["coefficients"]), tuple(edge["parameters"])
            )
            for edge in node["edges"]
        )
        nodes.append(KANNode(node["name"], node["combine"], edges))
    return KANProgram(
        tuple(data["inputs"]), tuple(nodes), tuple(data["outputs"]),
        tuple(data["domain"]), data["schema"],
    )


def test_v46_original_source_identity_and_raw_archive_digest():
    record, origin = _files()
    assert origin["schema"] == "visuaml.pretrained-internal-kan-origin.v46"
    assert record["schema"] == "visuaml.pretrained-internal-kan-causality.v46"
    assert origin["source_commit"] == record["source_commit"]
    assert origin["source_file_sha256"] == record["source_file_sha256"]
    assert origin["source_commit"] == "272deae161f1c780c652b419fff8133676abf2a6"
    assert origin["workflow_url"] == (
        "https://github.com/Caerii/VisuaML/actions/runs/44240406427"
    )
    assert origin["original_checkpoint_sha256"] == record["source_checkpoint_sha256"]
    assert record["source_checkpoint_sha256"] == (
        "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
    )
    assert record["reviewed_internal_stage"]["target_shape"] == [1, 16, 14, 14]
    assert origin["instrumented_graph_sha256"] == record["instrumented_original_graph_sha256"]
    assert origin["target_stage"] == record["reviewed_internal_stage"]["target_output"]
    assert record["model_original_bytes_modified"] is False
    assert origin["external_signed_attestation"] is False


def test_each_archived_kan_curve_recomputes_same_and_cross_image_errors():
    record, _ = _files()
    heldout_t = np.array(record["heldout_coordinates"], dtype=np.float64)
    actual_same = np.array(record["heldout_activation_means"], dtype=np.float64)
    actual_cross = np.array(record["cross_activation_means"], dtype=np.float64)
    assert record["train_input_count"] == 17
    assert record["heldout_same_image_input_count"] == 16
    assert record["cross_image_input_count"] == 16
    assert set(record["train_coordinates"]).isdisjoint(record["heldout_coordinates"])
    assert actual_same.shape == actual_cross.shape == (16, 16)
    assert record["base_image_train_sha256"] != record["base_image_cross_sha256"]
    assert len(record["basis_families"]) == 5
    for candidate in record["basis_families"]:
        program = _graph(candidate["edge_graph"])
        assert program.digest() == candidate["kan_graph_sha256"]
        predictions = execute_kan_vector(program, {"t": heldout_t})
        stacked = np.column_stack([
            predictions[name] for name in program.outputs
        ])
        assert stacked.shape == actual_same.shape
        assert np.isclose(
            float(np.mean(np.abs(stacked - actual_same))),
            candidate["heldout_mean_abs_activation_mean_error"],
            rtol=1e-10, atol=1e-12,
        )
        assert np.isclose(
            float(np.mean(np.abs(stacked - actual_cross))),
            candidate["cross_image_mean_abs_activation_mean_error"],
            rtol=1e-10, atol=1e-12,
        )
        assert candidate["data_derived_coefficients"]
        assert candidate["internal_activation_surrogate_not_causal_mechanism"]
    spline = next(x for x in record["basis_families"] if x["basis"] == "linear_spline")
    assert spline["cross_image_mean_abs_activation_mean_error"] > (
        10 * spline["heldout_mean_abs_activation_mean_error"]
    )
    assert sum(x["train_only_selected"] for x in record["basis_families"]) == 1
    assert record["basis_selected_by_train_only"] == "gaussian_rbf"


def test_channel_ablation_is_real_but_does_not_claim_semantic_feature():
    record, origin = _files()
    assert record["selected_causal_channel"] == origin["selected_causal_channel"] == 5
    assert record["negative_control_channel"] == origin["negative_control_channel"] == 3
    assert record["actual_pretrained_internal_interventions_executed"]
    assert record["selected_channel_chosen_without_heldout"]
    assert not record["fitted_kan_edges_are_teacher_internal_mechanisms"]
    assert not record["mechanistic_semantic_feature_identified"]
    assert not record["cross_checkpoint_generalization_proved"]
    assert not record["trained_model_classification_accuracy_measured"]
    assert not origin["pretrained_kan_internal_mechanism_identified"]
    assert not origin["cross_checkpoint_transfer_claimed"]
    ranked = record["all_channel_discovery_effects"]
    assert len(ranked) == 16
    assert {r["channel"] for r in ranked} == set(range(16))
    assert max(ranked, key=lambda r: r["discovery_mean_abs_logit_effect"])[
        "channel"
    ] == 5
    assert min(ranked, key=lambda r: r["discovery_mean_abs_logit_effect"])[
        "channel"
    ] == 3
    for witnesses_name in ("heldout_causal_witnesses", "cross_image_causal_witnesses"):
        witnesses = record[witnesses_name]
        assert len(witnesses) == 16
        assert all(w["identity_mask_numerical_parity"]["passed"] for w in witnesses)
        selected = [
            w["selected_channel_causal_effect"]["mean_abs_logit_change"]
            for w in witnesses
        ]
        controls = [
            w["negative_control_causal_effect"]["mean_abs_logit_change"]
            for w in witnesses
        ]
        assert np.mean(selected) > np.mean(controls)
        # Crucial counterexample to any claim that a large logit perturbation
        # necessarily changed the top predicted class on these probes.
        assert all(w["selected_channel_causal_effect"]["top1_unchanged"]
                   for w in witnesses)
        assert all(w["negative_control_causal_effect"]["top1_unchanged"]
                   for w in witnesses)
    oracle = record["independent_scalar_replay_witnesses"]
    assert len(oracle) == 3
    assert all(all(w[key]["passed"] for key in (
        "stage", "final_logits", "selected_ablation_logits"
    )) for w in oracle)
