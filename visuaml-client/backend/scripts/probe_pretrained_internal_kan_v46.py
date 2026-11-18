"""v46 pinned pretrained CNN: internal activation tracing and causal KAN probes.

Exactly one vetted ReLU activation tensor is traced from a SHA-pinned CNN.
KAN edges fit per-channel spatial mean activation vs one input-intensity axis.
Channel masking is performed INSIDE the original CNN graph and observed at
final logits. A second frozen image provides an out-of-manifold control.
The KAN curves are descriptive surrogates, NOT recovered teacher mechanisms.
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
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    TARGET,
    ablate_channel,
    instrument_activation,
    numerical_parity,
    original_review,
    reference_trace,
)

SCHEMA = "visuaml.pretrained-internal-kan-causality.v46"
FAMILIES = ("linear", "polynomial", "linear_spline", "gaussian_rbf", "fourier")
TRAIN_SCALARS = (0., .25, .5, .75, 1.)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _session(data: bytes):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        data, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _sample(base: np.ndarray, amount: float) -> np.ndarray:
    return np.ascontiguousarray(base * np.float32(amount), dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> dict:
    if (a.shape != b.shape or a.dtype != np.float32
            or b.dtype != np.float32 or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise AssertionError("Missing finite causal logit tensor")
    delta = (a.astype(np.float64) - b.astype(np.float64))
    return {
        "mean_abs_logit_change": float(np.mean(np.abs(delta))),
        "max_abs_logit_change": float(np.max(np.abs(delta))),
        "top1_unchanged": bool(int(np.argmax(a)) == int(np.argmax(b))),
        "baseline_output_sha256": _sha(a.tobytes()),
        "altered_output_sha256": _sha(b.tobytes()),
    }


def _factor_predictions(fit, coords: np.ndarray) -> np.ndarray:
    outputs = execute_kan_vector(fit.program, {"t": coords})
    return np.column_stack([outputs[name] for name in fit.program.outputs])


def _activation_mean(stage: np.ndarray) -> np.ndarray:
    if stage.ndim != 4 or stage.shape[0] != 1 or stage.shape[1] != 16:
        raise AssertionError("Expected pinned NCHW 16-channel activation")
    if not np.isfinite(stage).all():
        raise AssertionError("Nonfinite intermediate CNN activation")
    return np.mean(stage[0].astype(np.float64), axis=(1, 2))


def run(checkpoint: bytes) -> dict:
    import onnxruntime as ort

    info = original_review(checkpoint)
    traced, traced_info = instrument_activation(checkpoint)
    if info != traced_info or _sha(checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("Original checkpoint identity changed")
    session = _session(traced)
    input_name = session.get_inputs()[0].name
    outputs = [x.name for x in session.get_outputs()]
    if input_name != "Input3" or set(outputs) != {info.original_output, TARGET}:
        raise AssertionError("Internal-trace ONNX runtime ABI drift")

    rng_a = np.random.default_rng(460)
    rng_b = np.random.default_rng(461)
    base_a = rng_a.random((1, 1, 28, 28)).astype(np.float32)
    base_b = rng_b.random((1, 1, 28, 28)).astype(np.float32)
    train = np.linspace(0., 1., 17, dtype=np.float64)
    heldout = (np.arange(16, dtype=np.float64) + .5) / 16.
    if np.intersect1d(train, heldout).size:
        raise AssertionError("Internal KAN probe leakage")

    def fetch(base: np.ndarray, t: float):
        sample = _sample(base, t)
        logits, activation = session.run(
            [info.original_output, TARGET], {input_name: sample}
        )
        if (logits.shape != (1, 10)
                or tuple(activation.shape) != info.target_shape
                or logits.dtype != np.float32
                or activation.dtype != np.float32):
            raise AssertionError("Internal CNN tensor shape/dtype changed")
        return sample, logits, activation

    train_cases = [fetch(base_a, float(t)) for t in train]
    heldout_cases = [fetch(base_a, float(t)) for t in heldout]
    cross_cases = [fetch(base_b, float(t)) for t in heldout]

    def table(cases):
        return np.stack([_activation_mean(activation) for _, _, activation in cases])

    train_means = table(train_cases)
    heldout_means = table(heldout_cases)
    cross_means = table(cross_cases)
    fits = []
    families = []
    for family in FAMILIES:
        fit = fit_teacher_slice(
            train, train_means, heldout, heldout_means, family
        )
        candidate_cross = _factor_predictions(fit, heldout)
        discrepancy = candidate_cross - cross_means
        cross_error = float(np.mean(np.abs(discrepancy)))
        fits.append(fit)
        families.append({
            "basis": family,
            "kan_graph_sha256": fit.program.digest(),
            "edge_graph": asdict(fit.program),
            "train_max_abs_activation_mean_error": fit.train_max_abs_logit_error,
            "heldout_max_abs_activation_mean_error": fit.heldout_max_abs_logit_error,
            "heldout_mean_abs_activation_mean_error": fit.heldout_mean_abs_logit_error,
            "cross_image_mean_abs_activation_mean_error": cross_error,
            "train_only_selected": False,
            "data_derived_coefficients": True,
            "internal_activation_surrogate_not_causal_mechanism": True,
        })
    chosen = min(fits, key=lambda v: v.train_max_abs_logit_error)
    for item in families:
        item["train_only_selected"] = item["basis"] == chosen.family

    # All 16 channel ablations are evaluated in the ORIGINAL CNN downstream
    # computation; selection uses only 5 fixed discovery scalar values.
    source_train = [fetch(base_a, t) for t in TRAIN_SCALARS]
    baseline = [case[1] for case in source_train]
    altered_models: dict[int | None, tuple[bytes, dict]] = {}
    channel_scores = []
    for channel in range(info.target_shape[1]):
        patched, origin = ablate_channel(checkpoint, channel)
        altered_models[channel] = (patched, origin)
        altered_session = _session(patched)
        changes = []
        for (sample, _, _), original in zip(source_train, baseline, strict=True):
            changed = altered_session.run(
                [info.original_output], {input_name: sample}
            )[0]
            changes.append(_distance(original, changed)["mean_abs_logit_change"])
        channel_scores.append({
            "channel": channel,
            "discovery_mean_abs_logit_effect": float(np.mean(changes)),
            "discovery_worst_abs_logit_effect": float(max(changes)),
            "intervened_model_sha256": _sha(patched),
        })
    selected = max(channel_scores, key=lambda entry: (
        entry["discovery_mean_abs_logit_effect"], -entry["channel"]
    ))["channel"]
    control = min(channel_scores, key=lambda entry: (
        entry["discovery_mean_abs_logit_effect"], entry["channel"]
    ))["channel"]
    if selected == control:
        raise AssertionError("No channel-specific causal effect could be identified")

    identity, identity_provenance = ablate_channel(checkpoint, None)
    altered_models[None] = (identity, identity_provenance)
    identity_session = _session(identity)
    selected_session = _session(altered_models[selected][0])
    control_session = _session(altered_models[control][0])

    def heldout_interventions(cases, group):
        witnesses = []
        for idx, (sample, original, stage) in enumerate(cases):
            identity_logits = identity_session.run(
                [info.original_output], {input_name: sample}
            )[0]
            selected_logits = selected_session.run(
                [info.original_output], {input_name: sample}
            )[0]
            control_logits = control_session.run(
                [info.original_output], {input_name: sample}
            )[0]
            identity_check = numerical_parity(original, identity_logits)
            if not identity_check["passed"]:
                raise AssertionError("Identity mask changed original computation")
            witnesses.append({
                "group": group, "index": idx,
                "input_sha256": _sha(sample.tobytes()),
                "stage_sha256": _sha(stage.tobytes()),
                "original_logits_sha256": _sha(original.tobytes()),
                "identity_mask_numerical_parity": identity_check,
                "selected_channel": selected,
                "selected_channel_causal_effect": _distance(
                    original, selected_logits
                ),
                "negative_control_channel": control,
                "negative_control_causal_effect": _distance(
                    original, control_logits
                ),
            })
        return witnesses

    hold_witnesses = heldout_interventions(heldout_cases, "heldout_same_image")
    cross_witnesses = heldout_interventions(cross_cases, "cross_image")

    # Independent scalar reference: original stage, final output, and
    # channel-masked final output; ONNX Runtime used only on the other side.
    independent = []
    for group, sample in (
        ("train_one", _sample(base_a, 1.)),
        ("heldout_first", heldout_cases[0][0]),
        ("cross_image_first", cross_cases[0][0]),
    ):
        _, observed_logits, observed_stage = (
            fetch(base_a, 1.) if group == "train_one"
            else fetch(base_a if group == "heldout_first" else base_b,
                       float(heldout[0]))
        )
        reference_stage, reference_logits = reference_trace(checkpoint, sample)
        trace_check = numerical_parity(reference_stage, observed_stage)
        final_check = numerical_parity(reference_logits, observed_logits)
        scalar_stage, scalar_masked = reference_trace(
            checkpoint, sample, selected
        )
        runtime_masked = selected_session.run(
            [info.original_output], {input_name: sample}
        )[0]
        masked_check = numerical_parity(scalar_masked, runtime_masked)
        if (not trace_check["passed"] or not final_check["passed"]
                or not masked_check["passed"]
                or not np.array_equal(scalar_stage, reference_stage)):
            raise AssertionError("Independent scalar/ONNX Runtime channel parity failed")
        independent.append({
            "group": group, "input_sha256": _sha(sample.tobytes()),
            "stage": trace_check,
            "final_logits": final_check,
            "selected_ablation_logits": masked_check,
        })

    backend = Path(__file__).resolve().parents[1]
    source_paths = (
        "visuaml/compiler_lab/onnx_activation_causality_v46.py",
        "scripts/probe_pretrained_internal_kan_v46.py",
        "visuaml/compiler_lab/kan_teacher_slice_v45.py",
        "scripts/probe_pretrained_mnist8_oracles_v37.py",
    )
    return {
        "schema": SCHEMA,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_revision": CHECKPOINT_REVISION,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            name: _sha((backend / name).read_bytes()) for name in source_paths
        },
        "reviewed_internal_stage": asdict(info),
        "instrumented_original_graph_sha256": _sha(traced),
        "train_input_count": len(train),
        "heldout_same_image_input_count": len(heldout),
        "cross_image_input_count": len(heldout),
        "train_coordinates": train.tolist(),
        "heldout_coordinates": heldout.tolist(),
        "base_image_train_sha256": _sha(base_a.tobytes()),
        "base_image_cross_sha256": _sha(base_b.tobytes()),
        "train_activation_mean_sha256": _sha(train_means.tobytes()),
        "heldout_activation_mean_sha256": _sha(heldout_means.tobytes()),
        "cross_activation_mean_sha256": _sha(cross_means.tobytes()),
        "train_activation_means": train_means.tolist(),
        "heldout_activation_means": heldout_means.tolist(),
        "cross_activation_means": cross_means.tolist(),
        "basis_families": families,
        "basis_selected_by_train_only": chosen.family,
        "ablation_discovery_cases": list(TRAIN_SCALARS),
        "all_channel_discovery_effects": channel_scores,
        "selected_causal_channel": selected,
        "negative_control_channel": control,
        "selected_channel_model_sha256": altered_models[selected][1][
            "altered_checkpoint_sha256"
        ],
        "negative_control_model_sha256": altered_models[control][1][
            "altered_checkpoint_sha256"
        ],
        "identity_model_sha256": identity_provenance[
            "altered_checkpoint_sha256"
        ],
        "heldout_causal_witnesses": hold_witnesses,
        "cross_image_causal_witnesses": cross_witnesses,
        "independent_scalar_replay_witnesses": independent,
        "model_original_bytes_modified": False,
        "actual_pretrained_internal_interventions_executed": True,
        "selected_channel_chosen_without_heldout": True,
        "fitted_kan_edges_are_teacher_internal_mechanisms": False,
        "trained_model_classification_accuracy_measured": False,
        "cross_checkpoint_generalization_proved": False,
        "mechanistic_semantic_feature_identified": False,
        "all_passed": True,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "limitations": (
            "One exact pretrained CNN, one internal ReLU activation tensor, "
            "mean-pooled channels, two synthetic seed-fixed intensity manifolds. "
            "KANs approximate measured activations from one image's train points; "
            "causal masking of an original CNN channel is a separate observation. "
            "No semantic feature labels, genuine image accuracy, cross-model "
            "replication, causal KAN-feature alignment, or universal invariance."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.model.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": result["reviewed_internal_stage"]["target_output"],
        "train": result["train_input_count"],
        "heldout": result["heldout_same_image_input_count"],
        "cross": result["cross_image_input_count"],
        "best_train_basis": result["basis_selected_by_train_only"],
        "causal_channel": result["selected_causal_channel"],
        "negative_control_channel": result["negative_control_channel"],
        "scalar_oracles": len(result["independent_scalar_replay_witnesses"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
