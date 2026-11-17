"""Optimization candidates must pass numerical evidence before timing."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from visuaml.compiler_lab.optimization_search_v33 import evaluate_relu_candidate
from visuaml.compiler_lab.semantic_ir import (
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    ValueType,
)


def double_relu() -> SemanticProgram:
    typ = ValueType((2, 4), "float32", 32)
    return SemanticProgram(
        "torch-fx", (("x", typ),), (),
        (SemanticRegion("forward", 1, (
            SemanticOperation("r1", "tensor", "relu", ("x",), "r1", typ),
            SemanticOperation("r2", "tensor", "relu", ("r1",), "r2", typ),
        )),), ("r2",), hashlib.sha256(b"v33 optimizer fixture").hexdigest(),
    )


def input_data():
    return [{"x": np.array([
        [-1, 1, 0, 2],
        [-3, 4, 0.5, -0.25],
    ], dtype=np.float32)}]


def numpy_oracle(*, x):
    return np.maximum(np.maximum(x, 0), 0)


def torch_oracle(*, x):
    return torch.relu(torch.relu(torch.from_numpy(x.copy()))).numpy()


def test_optimization_search_replays_only_numerically_accepted_rewrite():
    result = evaluate_relu_candidate(
        double_relu(), witnesses=input_data(),
        numpy_oracle=numpy_oracle, torch_oracle=torch_oracle, trials=3,
    )
    assert result["status"] == "exploratory_timing_after_finite_numerical_witnesses"
    assert result["numerical_evidence"]["status"] == "passed_finite_witnesses"
    assert result["before"]["op_sequence"] == ["relu", "relu"]
    assert result["after"]["op_sequence"] == ["relu"]
    assert result["median_ratio_before_over_after"] > 0
    assert result["warmup_per_variant"] == 2
    assert len(result["warm_ms"]["before"]["raw"]) == 3
    assert len(result["warm_ms"]["after"]["raw"]) == 3
    assert len(result["record_sha256"]) == 64


def test_optimization_search_preserves_failure_and_skips_timing():
    def wrong_oracle(*, x):
        return np.full_like(x, np.float32(123.0))
    result = evaluate_relu_candidate(
        double_relu(), witnesses=input_data(),
        numpy_oracle=numpy_oracle, torch_oracle=wrong_oracle, trials=3,
    )
    assert result["status"] == "rejected_no_performance_measurement"
    assert result["numerical_evidence"]["status"] == "rejected"
    assert "warm_ms" not in result
    assert result["numerical_evidence"]["comparison"]["counterexample"]


def test_optimization_search_ignores_non_applicable_graph():
    typ = ValueType((2, 4), "float32", 32)
    plain = SemanticProgram(
        "torch-fx", (("x", typ),), (),
        (SemanticRegion("forward", 1, (
            SemanticOperation("r1", "tensor", "relu", ("x",), "r1", typ),
        )),), ("r1",), hashlib.sha256(b"single relu").hexdigest(),
    )
    result = evaluate_relu_candidate(
        plain, witnesses=input_data(),
        numpy_oracle=numpy_oracle, torch_oracle=torch_oracle, trials=3,
    )
    assert result["status"] == "not_applicable"
    assert "warm_ms" not in result


def test_optimization_search_bounds_repetitions():
    with pytest.raises(ValueError, match="3..51"):
        evaluate_relu_candidate(
            double_relu(), witnesses=input_data(),
            numpy_oracle=numpy_oracle, torch_oracle=torch_oracle, trials=10000,
        )
