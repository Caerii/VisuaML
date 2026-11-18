"""v47 real pretrained spatial donor transplants and KAN predictive sufficiency.

A *frozen* original CNN and preselected original-CNN channel (from v46)
are used. Spatial region selection uses ONLY 3 discovery donor intensities.
KAN fits 16 spatial cells of a donor's internal tensor from 17 training
intensities; held-out donor activations and causal effects are never fitted.

Two explicitly independent dimensions are measured:
 (1) causal output change on the actual patched original CNN;
 (2) whether KAN-predicted donor values reproduce that specific change.
Even successful prediction on one input manifold would not certify a
learned/pretrained internal KAN mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visuaml.compiler_lab.kan_mechanism_ir_v45 import execute_kan_vector
from visuaml.compiler_lab.kan_teacher_slice_v45 import fit_teacher_slice
from visuaml.compiler_lab.onnx_activation_causality_v46 import (
    CHECKPOINT_SHA256,
    instrument_activation,
    numerical_parity,
    reference_trace,
)
from visuaml.compiler_lab.spatial_activation_patching_v47 import (
    DONOR,
    MASK,
    build_patchable_model,
    downstream_reference,
    numeric_effect,
    spatial_mask,
)

SCHEMA = "visuaml.pretrained-spatial-causal-kan.v47"
BASIS = ("linear", "polynomial", "linear_spline", "gaussian_rbf", "fourier")
REGIONS = ((0, 4, 0, 4), (0, 4, 5, 9), (5, 9, 0, 4),
           (5, 9, 5, 9), (10, 14, 10, 14))
DISCOVERY = (0.25, 0.5, 0.75)
TILE_SIZE = 16


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _session(source: bytes):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        source, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _metrics(ref, other):
    delta = other.astype(np.float64) - ref.astype(np.float64)
    return {
        "mean_absolute_logit_error": float(np.mean(np.abs(delta))),
        "max_absolute_logit_error": float(np.max(np.abs(delta))),
        "relative_l2_logit_error": float(
            np.linalg.norm(delta) / max(np.linalg.norm(ref.astype(np.float64)), 1e-12)
        ),
        "top1_agreement": bool(int(np.argmax(ref)) == int(np.argmax(other))),
        "reference_logits_sha256": _sha(ref.tobytes()),
        "predicted_logits_sha256": _sha(other.tobytes()),
    }


def run(model_bytes: bytes) -> dict:
    import onnxruntime as ort

    if _sha(model_bytes) != CHECKPOINT_SHA256:
        raise AssertionError("Original pretrained checkpoint source bytes differ")
    root = Path(__file__).resolve().parents[3]
    historical = json.loads(
        (root / "benchmarks/pretrained_internal_kan_causality_v46.json").read_bytes()
    )
    if (historical["source_checkpoint_sha256"] != CHECKPOINT_SHA256
            or historical["selected_causal_channel"] != 5
            or historical["selected_channel_chosen_without_heldout"] is not True):
        raise AssertionError("Previous channel-selection provenance changed")
    channel = historical["selected_causal_channel"]

    instrumented, review = instrument_activation(model_bytes)
    patched, patch_contract = build_patchable_model(model_bytes)
    trace = _session(instrumented)
    transplant = _session(patched)
    if (review.target_shape != (1, 16, 14, 14)
            or sorted(x.name for x in trace.get_outputs()) !=
            sorted([review.original_output, review.target_output])):
        raise AssertionError("Original pretrained internal-activation ABI drift")
    if sorted(x.name for x in transplant.get_inputs()) != sorted(
        ["Input3", DONOR, MASK]
    ):
        raise AssertionError("Patchable graph introduced unexpected runtime arguments")

    recipient = (
        np.random.default_rng(470).random((1, 1, 28, 28)).astype(np.float32)
        * np.float32(0.65)
    )
    donor_base = np.random.default_rng(471).random(
        (1, 1, 28, 28)
    ).astype(np.float32)
    cross_base = np.random.default_rng(472).random(
        (1, 1, 28, 28)
    ).astype(np.float32)
    recipient_logits, recipient_stage = trace.run(
        [review.original_output, review.target_output], {"Input3": recipient}
    )
    train = np.linspace(0., 1., 17, dtype=np.float64)
    heldout = (np.arange(16, dtype=np.float64) + 0.5) / 16.
    if np.intersect1d(train, heldout).size:
        raise AssertionError("KAN test leakage in scalar intensities")

    def donor_at(base: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        image = (base * np.float32(t)).astype(np.float32)
        logits, activation = trace.run(
            [review.original_output, review.target_output], {"Input3": image}
        )
        if activation.shape != (1, 16, 14, 14) or not np.isfinite(activation).all():
            raise AssertionError("Incorrect real pretrained donor activations")
        return logits, activation

    def patch(donor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return transplant.run(
            [review.original_output],
            {"Input3": recipient, DONOR: donor, MASK: mask},
        )[0]

    # The candidate region is selected on different, fixed discovery donor
    # intensities than the cross-image test; no heldout causal labels enter.
    search = []
    for region in REGIONS:
        mask = spatial_mask(channel, region)
        changes = []
        for t in DISCOVERY:
            _, donor = donor_at(donor_base, t)
            changed = patch(donor, mask)
            changes.append(numeric_effect(
                recipient_logits, changed
            )["mean_absolute_logit_delta"])
        search.append({
            "region": list(region),
            "mean_absolute_logit_effect": float(np.mean(changes)),
            "max_absolute_logit_effect": float(max(changes)),
        })
    selected = max(search, key=lambda x: x["mean_absolute_logit_effect"])
    region = tuple(selected["region"])
    mask = spatial_mask(channel, region)
    y0, y1, x0, x1 = region
    assert (y1 - y0) * (x1 - x0) == TILE_SIZE

    def tile(activation: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            activation[0, channel, y0:y1, x0:x1].reshape(-1),
            dtype=np.float64,
        )

    def measurements(base, coords):
        cases = [donor_at(base, float(t)) for t in coords]
        return cases, np.stack([tile(activation) for _, activation in cases])

    train_cases, train_cells = measurements(donor_base, train)
    same_cases, same_cells = measurements(donor_base, heldout)
    cross_cases, cross_cells = measurements(cross_base, heldout)

    fitted = []
    models = []
    for family in BASIS:
        candidate = fit_teacher_slice(
            train, train_cells, heldout, same_cells, family
        )
        values = execute_kan_vector(candidate.program, {"t": heldout})
        estimates = np.column_stack([
            values[name] for name in candidate.program.outputs
        ])
        cross_abs = float(np.mean(np.abs(estimates - cross_cells)))
        fitted.append(candidate)
        models.append({
            "basis": family,
            "graph_sha256": candidate.program.digest(),
            "graph": asdict(candidate.program),
            "train_max_abs_cell_error": candidate.train_max_abs_logit_error,
            "same_image_heldout_cell_mae": candidate.heldout_mean_abs_logit_error,
            "cross_image_cell_mae": cross_abs,
            "data_fitted_curve_not_original_mechanism": True,
        })
    chosen = min(fitted, key=lambda candidate: candidate.train_max_abs_logit_error)
    predictor = chosen.program
    for model in models:
        model["selected_by_training_only"] = model["basis"] == chosen.family

    # Patch exactly the selected 4x4 donor tile with either actual values or
    # the selected KAN's predictions. All nonselected tensor cells are from
    # the actual donor but masked, thus cannot affect the receiver.
    def predict_tile(t: float) -> np.ndarray:
        result = execute_kan_vector(
            predictor, {"t": np.array([t], dtype=np.float64)}
        )
        return np.asarray(
            [result[name][0] for name in predictor.outputs], dtype=np.float32
        )

    mask_identity = np.zeros_like(mask)
    identity_result = patch(recipient_stage, mask_identity)
    if not numerical_parity(recipient_logits, identity_result)["passed"]:
        raise AssertionError("Zero patch mask must be observational identity")
    same_input = patch(recipient_stage, mask)
    if not numerical_parity(recipient_logits, same_input)["passed"]:
        raise AssertionError("Patching recipient's own stage violated identity")

    # Concrete independent scalar replay of the two nontrivial grafts.
    scalar_witnesses = []
    for label, donor in (
        ("same_seed_t_0.5", same_cases[8][1]),
        ("cross_seed_t_0.5", cross_cases[8][1]),
    ):
        graft, expected = downstream_reference(
            model_bytes, recipient, donor, mask
        )
        observed = patch(donor, mask)
        parity = numerical_parity(expected, observed)
        if not parity["passed"]:
            raise AssertionError("Real patched ONNX deviated from independent scalar oracle")
        scalar_witnesses.append({
            "case": label, "graft_sha256": _sha(graft.tobytes()),
            "logit_oracle": parity,
        })

    def outcomes(cases, group):
        records = []
        for i, ((_, true_stage), t) in enumerate(zip(cases, heldout, strict=True)):
            true = patch(true_stage, mask)
            generated_stage = np.zeros_like(true_stage)
            generated_stage[0, channel, y0:y1, x0:x1] = predict_tile(float(t)).reshape(
                y1-y0, x1-x0
            )
            surrogate = patch(generated_stage, mask)
            mean_stage = np.zeros_like(true_stage)
            mean_stage[0, channel, y0:y1, x0:x1] = float(np.mean(tile(true_stage)))
            mean_result = patch(mean_stage, mask)
            # Shuffled donor tile control preserves its values/mean while
            # destroying spatial correspondence, a falsification baseline.
            shuffled = np.zeros_like(true_stage)
            shuffled[0, channel, y0:y1, x0:x1] = np.asarray(
                tile(true_stage)[::-1].reshape(y1-y0, x1-x0), dtype=np.float32
            )
            permuted_result = patch(shuffled, mask)
            records.append({
                "group": group, "index": i, "t": float(t),
                "actual_donor_tile_sha256": _sha(tile(true_stage).tobytes()),
                "actual_causal_effect": numeric_effect(recipient_logits, true),
                "kan_causal_prediction_error": _metrics(true, surrogate),
                "mean_only_control_causal_prediction_error": _metrics(true, mean_result),
                "spatial_permutation_control_error": _metrics(true, permuted_result),
                "kan_fitted_internal_mechanism_verified": False,
            })
        return records

    same = outcomes(same_cases, "same_donor_image_heldout")
    cross = outcomes(cross_cases, "cross_donor_image_unseen")
    def avg(records, key):
        return float(np.mean([r[key]["mean_absolute_logit_error"] for r in records]))

    paths = (
        "scripts/probe_spatial_causal_kan_v47.py",
        "visuaml/compiler_lab/spatial_activation_patching_v47.py",
        "visuaml/compiler_lab/kan_teacher_slice_v45.py",
    )
    backend = Path(__file__).resolve().parents[1]
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            path: _sha((backend/path).read_bytes()) for path in paths
        },
        "original_checkpoint_sha256": CHECKPOINT_SHA256,
        "original_v46_preselected_channel": channel,
        "reviewed_activation": asdict(review),
        "patchable_model_sha256": patch_contract["patched_sha256"],
        "patch_contract": patch_contract,
        "recipient_sha256": _sha(recipient.tobytes()),
        "training_donor_base_sha256": _sha(donor_base.tobytes()),
        "unseen_donor_base_sha256": _sha(cross_base.tobytes()),
        "mask_sha256": _sha(mask.tobytes()),
        "selected_region": list(region),
        "region_discovery_inputs": list(DISCOVERY),
        "region_discovery_scores": search,
        "selected_from_discovery_only": True,
        "train_coordinates": train.tolist(),
        "heldout_coordinates": heldout.tolist(),
        "train_tile_values": train_cells.tolist(),
        "heldout_tile_values": same_cells.tolist(),
        "unseen_image_tile_values": cross_cells.tolist(),
        "fitted_basis_families": models,
        "selected_basis_train_only": chosen.family,
        "same_image_causal_witnesses": same,
        "cross_image_causal_witnesses": cross,
        "mean_kan_causal_prediction_mae_same": avg(same,"kan_causal_prediction_error"),
        "mean_kan_causal_prediction_mae_cross": avg(cross,"kan_causal_prediction_error"),
        "mean_mean_only_control_mae_same": avg(same, "mean_only_control_causal_prediction_error"),
        "mean_mean_only_control_mae_cross": avg(cross, "mean_only_control_causal_prediction_error"),
        "mean_permuted_control_mae_same": avg(same, "spatial_permutation_control_error"),
        "mean_permuted_control_mae_cross": avg(cross, "spatial_permutation_control_error"),
        "independent_scalar_oracles": scalar_witnesses,
        "original_bytes_modified": False,
        "all_pinned_model_checks_passed": True,
        "pretrained_kan_internal_mechanism_certified": False,
        "cross_checkpoint_transfer_proved": False,
        "spatial_donor_patch_semantics_verified_finitely": True,
        "other_image_is_representative_of_natural_images": False,
        "onnxruntime_version": __import__("importlib.metadata", fromlist=["version"]).version("onnxruntime"),
        "numpy_version": np.__version__,
        "limitations": (
            "One exact pretrained CNN and one fixed recipient, two seeded "
            "donor images. Region and KAN basis selection use only discovery "
            "inputs, but the learned edge curves fit one scalar manifold. "
            "Spatial donor patching shows original-model causal effects; "
            "a KAN surrogate is not proven to be an internal network "
            "mechanism. No natural-image accuracy, general checkpoint "
            "transfer, all-input equivalence or performance speedup."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = run(args.model.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_region": record["selected_region"],
        "selected_basis": record["selected_basis_train_only"],
        "same_kan_causal_mae": record["mean_kan_causal_prediction_mae_same"],
        "cross_kan_causal_mae": record["mean_kan_causal_prediction_mae_cross"],
        "same_mean_only_control": record["mean_mean_only_control_mae_same"],
        "cross_mean_only_control": record["mean_mean_only_control_mae_cross"],
        "pretrained_kan_mechanism_certified": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
