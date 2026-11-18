"""v46 adversarial tests for source-bound intermediate activation and causal overlay.

A pinned model is supplied only in dedicated CI. Repository-wide CI still
checks rejection, type guards and normal standalone test discovery.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from visuaml.compiler_lab.onnx_activation_causality_v46 import (
    CHECKPOINT_SHA256,
    TARGET,
    ActivationTraceError,
    ablate_channel,
    instrument_activation,
    numerical_parity,
    original_review,
    reference_trace,
)

MODEL = os.environ.get("VISUAML_PINNED_MODEL")


def _checkpoint():
    if MODEL is None:
        pytest.skip("Pinned upstream checkpoint only installed in dedicated workflow")
    raw = Path(MODEL).read_bytes()
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == CHECKPOINT_SHA256
    return raw


def test_numeric_oracle_requires_finite_matching_dtype_shape():
    x = np.array([0., 1., 2.], dtype=np.float32)
    y = x.copy()
    y[2] += 1e-6
    assert numerical_parity(x, y)["passed"] is True
    y[2] += 1
    assert numerical_parity(x, y)["passed"] is False
    with pytest.raises(ActivationTraceError):
        numerical_parity(x, y.astype(np.float64))
    with pytest.raises(ActivationTraceError):
        numerical_parity(x, np.full_like(x, np.nan))
    with pytest.raises(ActivationTraceError):
        numerical_parity(x, np.zeros((1, 3), dtype=np.float32))


def test_reject_wrong_checkpoint_identity_before_onnx_decode():
    with pytest.raises(ActivationTraceError, match="preflight"):
        original_review(b"corrupt or unpinned")
    with pytest.raises(ActivationTraceError, match="preflight"):
        instrument_activation(b"not the trained model")
    with pytest.raises(ActivationTraceError, match="preflight"):
        ablate_channel(b"other model", channel=0)


def test_real_original_cnn_trace_and_exact_internal_mask_semantics():
    import onnxruntime as ort

    raw = _checkpoint()
    info = original_review(raw)
    assert info.target_output == TARGET
    assert info.source_node_op == "Relu"
    assert info.target_shape[:2] == (1, 16)
    assert info.original_output == "Plus214_Output_0"
    traced, reviewed = instrument_activation(raw)
    assert reviewed == info
    assert traced != raw

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        traced, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    sample = np.random.default_rng(46).random((1, 1, 28, 28)).astype(np.float32)
    observed_logits, observed_stage = session.run(
        [info.original_output, TARGET], {"Input3": sample}
    )
    reference_stage, reference_logits = reference_trace(raw, sample)
    assert numerical_parity(reference_stage, observed_stage)["passed"] is True
    assert numerical_parity(reference_logits, observed_logits)["passed"] is True

    identity, rec_identity = ablate_channel(raw, None)
    identity_session = ort.InferenceSession(identity, sess_options=opts)
    same = identity_session.run([info.original_output], {"Input3": sample})[0]
    assert rec_identity["is_identity_control"] is True
    assert numerical_parity(observed_logits, same)["passed"] is True

    channels = (0, 7, 15)
    for channel in channels:
        patched, receipt = ablate_channel(raw, channel)
        assert receipt["channel"] == channel
        assert receipt["modified_graph_nodes"] == receipt["original_graph_nodes"] + 1
        assert receipt["altered_checkpoint_sha256"] != CHECKPOINT_SHA256
        patched_session = ort.InferenceSession(
            patched, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        actual = patched_session.run([info.original_output], {"Input3": sample})[0]
        stage, expected = reference_trace(raw, sample, channel)
        np.testing.assert_array_equal(stage, reference_stage)
        assert numerical_parity(expected, actual)["passed"] is True
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == CHECKPOINT_SHA256


def test_ablation_rejects_out_of_bounds_and_bad_input_abi():
    raw = _checkpoint()
    for channel in (-1, 16, True, 1.5):
        with pytest.raises(ActivationTraceError, match="Channel index"):
            ablate_channel(raw, channel)
    for sample in (
        np.zeros((1, 1, 28, 28), dtype=np.float64),
        np.zeros((1, 3, 28, 28), dtype=np.float32),
        np.full((1, 1, 28, 28), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ActivationTraceError, match="Runtime input"):
            reference_trace(raw, sample)
