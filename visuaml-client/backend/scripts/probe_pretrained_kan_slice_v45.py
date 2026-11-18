"""Evaluate fitted KAN edge functions against a REAL pinned pretrained CNN.

This is a 1D *functional surrogate* study: the original MNIST-8 checkpoint is
run under ONNX Runtime on a 65-point scalar sweep t*fixed_random_image.
Coefficients are fitted on 33 coordinates only, and the 32 interleaved
heldout coordinates are never used for fitting. No labels, attribution to
internal CNN circuits, parameter intervention or classification accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts.audit_pretrained_onnx_v37 import audit_bytes
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    ATOL,
    RTOL,
    numpy_onnx_interpreter,
    sha,
)
from visuaml.compiler_lab.kan_teacher_slice_v45 import fit_teacher_slice

SCHEMA = "visuaml.pretrained-kan-surrogate-slice.v45"
CHECKPOINT_SHA = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
REVISION = "a19f9a8c2333de1df9b03f10f5739f468b699a1a"
FAMILIES = ("linear", "polynomial", "linear_spline", "gaussian_rbf", "fourier")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(model_data: bytes) -> dict:
    import onnx
    import onnxruntime as ort

    audit = audit_bytes(model_data, expected_sha256=CHECKPOINT_SHA,
                        source_revision=REVISION)
    if audit["input"]["shape"] != [1, 1, 28, 28] or audit["output"]["shape"] != [1, 10]:
        raise AssertionError("Pretrained input/output ABI mismatch")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model_data, providers=["CPUExecutionProvider"], sess_options=options
    )
    if (len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1
            or session.get_inputs()[0].name != audit["input"]["name"]
            or session.get_outputs()[0].name != audit["output"]["name"]):
        raise AssertionError("Independent ONNX Runtime ABI mismatch")
    input_name = audit["input"]["name"]
    output_name = audit["output"]["name"]
    base = np.random.default_rng(450).random((1, 1, 28, 28)).astype(np.float32)
    train = np.linspace(0., 1., 33, dtype=np.float64)
    holdout = (np.arange(32, dtype=np.float64) + .5) / 32.
    if np.intersect1d(train, holdout).size:
        raise AssertionError("Teacher slice heldout data leakage")

    samples = []
    def logits(t: float) -> np.ndarray:
        sample = np.asarray(np.float32(t) * base, dtype=np.float32)
        result = session.run([output_name], {input_name: sample})[0]
        if result.shape != (1, 10) or not np.isfinite(result).all():
            raise AssertionError("Invalid finite trained-model response")
        samples.append((t, sample, result))
        return np.asarray(result[0], dtype=np.float64)
    train_logits = np.stack([logits(float(t)) for t in train], axis=0)
    heldout_logits = np.stack([logits(float(t)) for t in holdout], axis=0)

    # Three independent scalar NumPy oracles check that ONNX Runtime is
    # genuinely executing the original original model semantics.
    parsed = onnx.load_model_from_string(model_data)
    oracle = []
    for i in (0, 16, 64):
        t, sample, ort_logits = samples[i]
        reference = numpy_onnx_interpreter(parsed, sample)
        delta = np.abs(ort_logits - reference)
        allowed = ATOL + RTOL * np.abs(reference)
        if not np.all(delta <= allowed):
            raise AssertionError("Pinned CNN NumPy-vs-ORT oracle drift")
        oracle.append({
            "t": t, "input_sha256": sha(sample),
            "onnxruntime_logits_sha256": sha(ort_logits),
            "numpy_logits_sha256": sha(reference),
            "max_abs_error": float(delta.max()),
        })

    candidates = []
    fitted = []
    for family in FAMILIES:
        fit = fit_teacher_slice(
            train, train_logits, holdout, heldout_logits, family
        )
        candidates.append({
            "family": family,
            "graph_sha256": fit.program.digest(),
            "graph": asdict(fit.program),
            "train_max_abs_logit_error": fit.train_max_abs_logit_error,
            "heldout_max_abs_logit_error": fit.heldout_max_abs_logit_error,
            "heldout_mean_abs_logit_error": fit.heldout_mean_abs_logit_error,
            "heldout_relative_l2_logit_error": fit.heldout_relative_l2_logit_error,
            "heldout_top1_agreement": fit.heldout_top1_agreement,
            "functional_surrogate_only": fit.fit_is_functional_surrogate,
            "pretrained_internal_mechanism_recovered": fit.pretrained_internal_mechanism_recovered,
        })
        fitted.append(fit)
    # Selection uses ONLY the discovery/train error. Heldout metrics
    # are reported afterward and cannot influence this model selection.
    selected = min(fitted, key=lambda x: x.train_max_abs_logit_error)
    selected_report = next(x for x in candidates if x["family"] == selected.family)
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/probe_pretrained_kan_slice_v45.py",
        "visuaml/compiler_lab/kan_teacher_slice_v45.py",
        "visuaml/compiler_lab/kan_mechanism_ir_v45.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            path: _sha((root / path).read_bytes()) for path in paths
        },
        "source_checkpoint_sha256": CHECKPOINT_SHA,
        "source_checkpoint_revision": REVISION,
        "seed": 450,
        "base_input_sha256": sha(base),
        "probe_definition": "t * a single frozen random 28x28 image",
        "input_domain": [0., 1.],
        "train_coordinates": train.tolist(),
        "heldout_coordinates": holdout.tolist(),
        "training_logit_sha256": _sha(np.ascontiguousarray(train_logits).tobytes()),
        "heldout_logit_sha256": _sha(np.ascontiguousarray(heldout_logits).tobytes()),
        "train_count": len(train),
        "heldout_count": len(holdout),
        "independent_scalar_numpy_oracles": oracle,
        "selection_uses_heldout_data": False,
        "selected_by_training_error_only": selected.family,
        "selected_heldout_mean_abs_logit_error": selected_report["heldout_mean_abs_logit_error"],
        "variants": candidates,
        "all_variants_finite": all(
            np.isfinite(v["heldout_mean_abs_logit_error"])
            and np.isfinite(v["heldout_relative_l2_logit_error"])
            for v in candidates
        ),
        "trained_kan": False,
        "least_squares_fitted_surrogate": True,
        "pretrained_internal_mechanism_extracted": False,
        "teacher_perturbation_ablations_executed": False,
        "real_digit_accuracy_evaluated": False,
        "generalization_beyond_one_input_manifold_established": False,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "limitations": (
            "Five 1D KAN basis families fit to trained CNN scalar-sweep logits "
            "by least squares; 33 training/32 interleaved heldout coordinates, "
            "one fixed random base image, no labels or distribution shifts. "
            "Not evidence of internal mechanistic circuit identity, factual "
            "symbolic law recovery, universal KAN advantage, speedup, or "
            "cross-model transfer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.model.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trained_checkpoint_sha256": result["source_checkpoint_sha256"],
        "families_tested": len(result["variants"]),
        "heldout_count": result["heldout_count"],
        "selection_by_training_only": result["selected_by_training_error_only"],
        "selected_heldout_mae": result["selected_heldout_mean_abs_logit_error"],
        "pretrained_internal_mechanism_extracted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
