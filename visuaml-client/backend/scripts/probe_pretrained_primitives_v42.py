"""v42 pinned real CNN: SVD proposals, causal parameter replacements and controls.

This experiment is *mechanism screening*, not proof of a discovered reusable
circuit. Ranking is based on discovery inputs, then tested independently on
five held-out synthetic inputs. No MNIST labels or real-image accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scripts.audit_pretrained_onnx_v37 import audit_bytes
from scripts.probe_pretrained_mnist8_oracles_v37 import (
    ATOL,
    RTOL,
    numpy_onnx_interpreter,
    sha,
)
from visuaml.compiler_lab.pretrained_primitive_candidates_v42 import (
    intervention_metrics,
    propose_low_rank,
    tensor_sha256,
)

SCHEMA = "visuaml.pretrained-primitive-screening.v42"
CHECKPOINT_SHA256 = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
CHECKPOINT_REVISION = "a19f9a8c2333de1df9b03f10f5739f468b699a1a"
TARGET = "Parameter193"
ERROR_THRESHOLD = 0.05


def _cases() -> tuple[tuple[str, str, np.ndarray], ...]:
    sample = np.random.default_rng(37).random((1, 1, 28, 28)).astype(np.float32)
    pulse = np.zeros_like(sample)
    pulse[0, 0, 14, 14] = 1
    checker = (np.indices((28, 28)).sum(axis=0) % 2).astype(np.float32)[None, None]
    stripe = (np.indices((28, 28))[1] % 4 < 2).astype(np.float32)[None, None]
    return (
        ("discovery", "zero", np.zeros_like(sample)),
        ("discovery", "ones", np.ones_like(sample)),
        ("discovery", "impulse", pulse),
        ("discovery", "checker", checker),
        ("discovery", "seed_37", sample),
        ("heldout", "seed_914", np.random.default_rng(914).random(sample.shape).astype(np.float32)),
        ("heldout", "seed_915", np.random.default_rng(915).random(sample.shape).astype(np.float32)),
        ("heldout", "seed_916", np.random.default_rng(916).random(sample.shape).astype(np.float32)),
        ("heldout", "stripes", stripe),
        ("heldout", "negative_0_5", np.full_like(sample, -0.5)),
    )


def _session(data: bytes):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(data, providers=["CPUExecutionProvider"], sess_options=options)


def _outputs(model, session, cases, output_name: str, input_name: str):
    numpy_values = []
    runtime_values = []
    max_oracle_error = 0.0
    for _, _, case in cases:
        expected = numpy_onnx_interpreter(model, case)
        observed = session.run([output_name], {input_name: case})[0]
        if expected.shape != (1, 10) or observed.shape != expected.shape:
            raise AssertionError("Output tensor shape changed")
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise AssertionError("Nonfinite pretrained logits")
        delta = np.abs(expected - observed)
        allowed = ATOL + RTOL * np.abs(expected)
        max_oracle_error = max(max_oracle_error, float(delta.max()))
        if not np.all(delta <= allowed):
            raise AssertionError("Independent modified-checkpoint oracle parity failed")
        numpy_values.append(expected)
        runtime_values.append(observed)
    return numpy_values, runtime_values, max_oracle_error


def run(raw: bytes) -> dict:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    audit = audit_bytes(raw, expected_sha256=CHECKPOINT_SHA256,
                        source_revision=CHECKPOINT_REVISION)
    if audit["input"]["shape"] != [1, 1, 28, 28] or audit["output"]["shape"] != [1, 10]:
        raise AssertionError("Pinned model graph ABI drift")
    model = onnx.load_model_from_string(raw)
    weights = {entry.name: entry for entry in model.graph.initializer}
    if TARGET not in weights:
        raise AssertionError("Target linear operator missing")
    tensor = np.ascontiguousarray(numpy_helper.to_array(weights[TARGET])).copy()
    if tensor.dtype != np.float32 or tensor.shape != (16, 4, 4, 10):
        raise AssertionError("Unreviewed pretrained linear geometry")
    matrix = tensor.reshape(256, 10).copy()
    source_weight_sha256 = tensor_sha256(tensor)
    approved = {item["name"]: item["canonical_array_sha256"]
                for item in audit["initializers"]}
    if source_weight_sha256 != approved[TARGET]:
        raise AssertionError("Source parameter digest is not preflight-bound")
    cases = _cases()
    reference_numpy, reference_ort, ref_error = _outputs(
        model, _session(raw), cases,
        audit["output"]["name"], audit["input"]["name"],
    )
    original_bytes = raw
    variants = (
        ("rank_2", propose_low_rank(matrix, 2)),
        ("rank_4", propose_low_rank(matrix, 4)),
        ("rank_8", propose_low_rank(matrix, 8)),
        ("rank_10", propose_low_rank(matrix, 10)),
        ("rank_4_coordinate_control", propose_low_rank(
            matrix, 4, negative_control=True
        )),
    )
    results = []
    for name, candidate in variants:
        changed = onnx.load_model_from_string(original_bytes)
        target = next(t for t in changed.graph.initializer if t.name == TARGET)
        replacement = candidate.proposed_weight.reshape(tensor.shape)
        target.CopyFrom(numpy_helper.from_array(replacement, name=TARGET))
        onnx.checker.check_model(changed, full_check=True)
        emitted = changed.SerializeToString(deterministic=True)
        actual = {t.name: tensor_sha256(np.array(numpy_helper.to_array(t), copy=True))
                  for t in changed.graph.initializer if t.data_type == onnx.TensorProto.FLOAT}
        if actual[TARGET] != candidate.proposed_weight_sha256:
            raise AssertionError("Candidate target weight not embedded exactly")
        for key, value in actual.items():
            if key != TARGET and value != approved[key]:
                raise AssertionError("Unrelated model parameter modified")
        expected, observed, oracle_error = _outputs(
            changed, _session(emitted), cases,
            audit["output"]["name"], audit["input"]["name"],
        )
        metrics = []
        for (group, case_name, inp), old, new, old_ort, new_ort in zip(
            cases, reference_numpy, expected, reference_ort, observed, strict=True
        ):
            info = intervention_metrics(old, new)
            # Preserve the independent ONNX Runtime result as a distinct
            # witness, not merely another output hash from our interpreter.
            observed_metric = intervention_metrics(old_ort, new_ort)
            metrics.append({
                "group": group, "case": case_name,
                "input_sha256": sha(inp),
                "changed_numpy_logits_sha256": sha(new),
                "changed_onnxruntime_logits_sha256": sha(new_ort),
                "numpy": info, "onnxruntime": observed_metric,
            })
        results.append({
            "name": name, "rank": candidate.rank,
            "construction": candidate.construction,
            "parameter_sha256": candidate.proposed_weight_sha256,
            "changed_checkpoint_sha256": hashlib.sha256(emitted).hexdigest(),
            "retained_spectral_energy": candidate.retained_spectral_energy,
            "relative_frobenius_weight_error": candidate.relative_frobenius_error,
            "independent_oracle_max_abs_error": oracle_error,
            "witnesses": metrics,
        })
    proposals = [result for result in results
                 if result["construction"] == "truncated-svd"]
    eligible = [entry for entry in proposals if all(
        obs["numpy"]["top1_unchanged"]
        and obs["numpy"]["relative_l2_logit_change"] <= ERROR_THRESHOLD
        for obs in entry["witnesses"] if obs["group"] == "discovery"
    )]
    selected = min(eligible, key=lambda entry: entry["rank"]) if eligible else None
    heldout = [
        x for x in selected["witnesses"] if x["group"] == "heldout"
    ] if selected else []
    heldout_pass = bool(selected is not None and all(
        item["numpy"]["top1_unchanged"]
        and item["numpy"]["relative_l2_logit_change"] <= ERROR_THRESHOLD
        for item in heldout
    ))
    backend = Path(__file__).resolve().parents[1]
    files = (
        "scripts/probe_pretrained_primitives_v42.py",
        "visuaml/compiler_lab/pretrained_primitive_candidates_v42.py",
        "scripts/probe_pretrained_mnist8_oracles_v37.py",
    )
    return {
        "schema": SCHEMA, "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_weight_sha256": source_weight_sha256,
        "source_weight_shape": list(tensor.shape),
        "source_weight_interpretation": "ONNX-reshaped 256x10 final projection",
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            key: hashlib.sha256((backend / key).read_bytes()).hexdigest()
            for key in files
        },
        "mode": "fixed_pretrained_weight_surgery_without_training",
        "threshold_relative_l2": ERROR_THRESHOLD,
        "discovery_input_count": 5, "heldout_input_count": 5,
        "independent_baseline_max_abs_error": ref_error,
        "results": results, "selected_rank": selected["rank"] if selected else None,
        "heldout_pass": heldout_pass,
        "source_checkpoint_modified": False,
        "pretrained_model_accuracy_evaluated": False,
        "generalizable_mechanism_certified": False,
        "oracle": "independent scalar NumPy ONNX interpreter and ONNX Runtime CPU",
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "limitations": (
            "One predeclared trained projection and synthetic probes; "
            "matrix low-rank approximation and input interventions are "
            "not human-interpretable circuits or evidence of model transfer. "
            "No labeled MNIST accuracy, out-of-distribution reliability, "
            "formal behavior equivalence or mechanism certification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.model.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        record, indent=2, sort_keys=True, allow_nan=False
    ) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "candidate_screening_complete",
        "selected_rank": record["selected_rank"],
        "heldout_pass": record["heldout_pass"],
        "control_max_error": max(
            x["numpy"]["max_abs_logit_change"]
            for y in record["results"] if y["name"] == "rank_4_coordinate_control"
            for x in y["witnesses"]
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
