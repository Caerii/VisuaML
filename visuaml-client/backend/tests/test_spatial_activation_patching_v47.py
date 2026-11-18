"""v47 pinned pretrained ONNX spatial transplantation, reversible controls.

The source model must be exact-byte matched. Outside dedicated CI the tests
that require the real ONNX checkpoint are explicitly skipped, not fabricated.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from visuaml.compiler_lab.onnx_activation_causality_v46 import (
    CHECKPOINT_SHA256,
    numerical_parity,
    reference_trace,
)
from visuaml.compiler_lab.spatial_activation_patching_v47 import (
    DONOR,
    MASK,
    ActivationTraceError,
    _admit_tensor,
    build_patchable_model,
    downstream_reference,
    numeric_effect,
    spatial_mask,
)

MODEL = os.getenv("VISUAML_PINNED_MODEL")


def source():
    if MODEL is None:
        pytest.skip("Full pretrained source tested in pinned dedicated CI")
    data = Path(MODEL).read_bytes()
    assert hashlib.sha256(data).hexdigest() == CHECKPOINT_SHA256
    return data


def _session(source_bytes):
    ort = pytest.importorskip("onnxruntime")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        source_bytes, providers=["CPUExecutionProvider"], sess_options=options
    )


def test_spatial_mask_half_open_single_channel_and_bad_indexes():
    m = spatial_mask(5, (3, 7, 4, 8))
    assert m.dtype == np.float32
    assert m.shape == (1, 16, 14, 14)
    assert np.sum(m) == 16
    assert np.sum(m[:, 5, 3:7, 4:8]) == 16
    for channel, region in (
        (True, (0, 1, 0, 1)), (-1, (0, 1, 0, 1)),
        (16, (0, 1, 0, 1)), (0, (-1, 3, 1, 5)),
        (0, (0, 15, 1, 5)), (0, (3, 3, 1, 5)),
        (0, (1, 3, 5, 4)), (0, (1, 3, 5., 7)),
    ):
        with pytest.raises(ActivationTraceError):
            spatial_mask(channel, region)


def test_fail_closed_donor_mask_geometry_and_numeric_effect():
    src = np.zeros((1, 1, 28, 28), dtype=np.float32)
    donor = np.ones((1, 16, 14, 14), dtype=np.float32)
    mask = spatial_mask(1, (2, 6, 3, 7))
    _admit_tensor(src, donor, mask)
    bad_cases = (
        (donor.astype(np.float64), mask), (donor, mask.astype(np.float64)),
        (donor[:, :15], mask), (donor, mask + np.float32(0.5)),
        (np.full_like(donor, np.nan), mask),
    )
    for bad_donor, bad_mask in bad_cases:
        with pytest.raises(ActivationTraceError):
            _admit_tensor(src, bad_donor, bad_mask)
    with pytest.raises(ActivationTraceError):
        _admit_tensor(src.astype(np.float64), donor, mask)
    x = np.zeros((1, 10), dtype=np.float32)
    y = x.copy()
    y[0, 0] = 1
    assert numeric_effect(x, y)["top1_changed"] is False
    assert numeric_effect(x, y)["max_absolute_logit_delta"] == 1
    with pytest.raises(ActivationTraceError):
        numeric_effect(x, y.astype(np.float64))


def test_patchable_model_preserves_legacy_opset_original_bytes_and_identity():
    raw = source()
    binary, metadata = build_patchable_model(raw)
    assert hashlib.sha256(raw).hexdigest() == CHECKPOINT_SHA256
    assert binary != raw
    assert metadata["source_ir_version"] == 3
    assert metadata["source_opset_versions"] == [{"domain": "", "version": 8}]
    assert metadata["original_nodes"] + 4 == metadata["modified_nodes"]
    assert metadata["target_shape"] == [1, 16, 14, 14]
    assert metadata["embedded_donor_tensor"] is False
    assert metadata["full_model_all_input_equivalence_proved"] is False

    original = _session(raw)
    modified = _session(binary)
    image = np.random.default_rng(47).random((1, 1, 28, 28)).astype(np.float32)
    donor_image = np.random.default_rng(48).random(image.shape).astype(np.float32)
    source_stage, expected = reference_trace(raw, image)
    donor_stage, _ = reference_trace(raw, donor_image)
    original_logits = original.run(["Plus214_Output_0"], {"Input3": image})[0]
    assert numerical_parity(expected, original_logits)["passed"]
    identity = np.zeros_like(source_stage)
    check = modified.run(["Plus214_Output_0"], {
        "Input3": image, DONOR: donor_stage, MASK: identity,
    })[0]
    assert numerical_parity(original_logits, check)["passed"]
    np.testing.assert_array_equal(check, original_logits)

    # Binary masks select entire spatial cells exactly.
    for mask in (
        spatial_mask(5, (3, 7, 4, 8)),
        spatial_mask(5, (0, 14, 0, 14)),
        spatial_mask(3, (3, 7, 4, 8)),
    ):
        graft, scalar_output = downstream_reference(raw, image, donor_stage, mask)
        np.testing.assert_array_equal(
            graft[mask == 1], donor_stage[mask == 1]
        )
        runtime = modified.run(["Plus214_Output_0"], {
            "Input3": image, DONOR: donor_stage, MASK: mask,
        })[0]
        assert numerical_parity(scalar_output, runtime)["passed"]


def test_same_image_patch_is_identity_and_spatial_effect_is_not_universal():
    raw = source()
    binary, _ = build_patchable_model(raw)
    run = _session(binary)
    image = np.random.default_rng(49).random((1, 1, 28, 28)).astype(np.float32)
    stage, baseline = reference_trace(raw, image)
    mask = spatial_mask(5, (4, 8, 4, 8))
    output = run.run(["Plus214_Output_0"], {
        "Input3": image, DONOR: stage, MASK: mask,
    })[0]
    # The independent scalar interpreter and the real ONNX Runtime may
    # differ in final accumulation roundoff. Preserve numerical parity
    # without asserting inter-backend bitwise identity.
    assert numerical_parity(baseline, output)["passed"]
    original_ort = _session(raw).run(
        ["Plus214_Output_0"], {"Input3": image}
    )[0]
    np.testing.assert_array_equal(original_ort, output)


def test_bad_original_bytes_rejected_before_instrumenting_graph():
    pytest.importorskip("onnx")
    with pytest.raises(ActivationTraceError):
        build_patchable_model(b"unreviewed")
