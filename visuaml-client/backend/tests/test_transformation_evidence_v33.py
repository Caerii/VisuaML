"""Research evidence cannot substitute a passing test for a formal proof."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from math import prod

import numpy as np
import pytest
import torch
from visuaml.compiler_lab.native_semantic_codegen_v32 import NativeSemanticCompiler
from visuaml.compiler_lab.semantic_ir import (
    SemanticOperation, SemanticProgram, SemanticRegion, ValueType,
)
from visuaml.compiler_lab.transformation_evidence_v33 import (
    TransformationEvidenceError,
    certify_native_rewrite,
    eliminate_idempotent_relu,
    verify_record_integrity,
)


def ty(*shape: int) -> ValueType:
    return ValueType(shape, "float32", 4 * prod(shape))


def relu_program() -> SemanticProgram:
    return SemanticProgram(
        "torch-fx", (("x", ty(3, 4)),), (),
        (SemanticRegion("forward", 1, (
            SemanticOperation("first", "tensor", "relu", ("x",), "first", ty(3, 4)),
            SemanticOperation(
                "second", "tensor", "relu", ("first",), "second", ty(3, 4),
            ),
        )),), ("second",), hashlib.sha256(b"v33-idempotent-relu").hexdigest(),
    )


def numpy_oracle(*, x: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(x, 0), 0).astype(np.float32)


def torch_oracle(*, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.relu(torch.relu(torch.from_numpy(x.copy()))).numpy()


def witnesses() -> list[dict[str, np.ndarray]]:
    return [
        {"x": np.array([
            [-3.0, -0.2, 0.0, 0.5],
            [0.001, 2.0, -1.5, 7.0],
            [1.2, -2.4, 3.4, 0.0],
        ], dtype=np.float32)},
        {"x": np.random.default_rng(3301).standard_normal(
            (3, 4)
        ).astype(np.float32)},
    ]


def test_relu_idempotence_is_graph_level_ssa_rewrite():
    original = relu_program()
    new = eliminate_idempotent_relu(original)
    assert len(original.regions[0].operations) == 2
    assert len(new.regions[0].operations) == 1
    assert new.regions[0].operations[0].operands == ("x",)
    assert new.outputs == original.outputs
    assert original.source_sha256 != new.source_sha256
    assert NativeSemanticCompiler(original).source != NativeSemanticCompiler(new).source
    with pytest.raises(TransformationEvidenceError, match="No admissible"):
        eliminate_idempotent_relu(new)


def test_finite_evidence_registry_accepts_only_independently_checked_witnesses():
    original = relu_program()
    candidate = eliminate_idempotent_relu(original)
    record = certify_native_rewrite(
        original, candidate, rule="relu-idempotence-v1",
        witnesses=witnesses(), numpy_oracle=numpy_oracle,
        torch_oracle=torch_oracle,
    )
    assert record["schema"] == "visuaml.transformation-evidence.v33"
    assert record["status"] == "passed_finite_witnesses"
    assert record["comparison"]["counterexample"] is None
    assert record["evidence_class"] == "finite_numeric_witnesses_not_formal_proof"
    assert len(record["comparison"]["witnesses"]) == 2
    assert verify_record_integrity(record)
    assert all(row["passed"] for row in record["comparison"]["witnesses"])
    assert len(record["before"]["generated_cpp_sha256"]) == 64
    assert record["after"]["op_sequence"] == ["relu"]
    assert record["before"]["op_sequence"] == ["relu", "relu"]


def test_failed_oracle_must_preserve_counterexample_not_mark_verified():
    original = relu_program()
    candidate = eliminate_idempotent_relu(original)

    def wrong_oracle(*, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) * np.float32(0.123)

    record = certify_native_rewrite(
        original, candidate, rule="relu-idempotence-v1",
        witnesses=witnesses()[:1], numpy_oracle=numpy_oracle,
        torch_oracle=wrong_oracle,
    )
    assert record["status"] == "rejected"
    assert record["comparison"]["counterexample"]["witness_index"] == 0
    assert record["comparison"]["counterexample"]["inputs"]["x"]
    assert verify_record_integrity(record)
    record["comparison"]["witnesses"][0]["passed"] = True
    assert not verify_record_integrity(record)


def test_illegal_rewrite_and_missing_rule_fail_closed():
    original = relu_program()
    bad = replace(original, regions=(replace(
        original.regions[0],
        operations=(replace(
            original.regions[0].operations[0], op="remote_eval"
        ), *original.regions[0].operations[1:]),
    ),))
    with pytest.raises(ValueError):
        eliminate_idempotent_relu(bad)
    with pytest.raises(TransformationEvidenceError, match="Unknown"):
        certify_native_rewrite(
            original, eliminate_idempotent_relu(original),
            rule="invented-perfect-proof",
            witnesses=witnesses(), numpy_oracle=numpy_oracle,
            torch_oracle=torch_oracle,
        )


@pytest.mark.parametrize("atol,rtol", ((1.0, 2e-5), (float("nan"), 2e-5)))
def test_rewrite_rejects_invalid_precision_budgets(atol, rtol):
    original = relu_program()
    with pytest.raises(TransformationEvidenceError, match="tolerance"):
        certify_native_rewrite(
            original, eliminate_idempotent_relu(original),
            rule="relu-idempotence-v1", witnesses=witnesses(),
            numpy_oracle=numpy_oracle, torch_oracle=torch_oracle,
            atol=atol, rtol=rtol,
        )


def test_rewrite_refuses_uses_of_intermediate_result():
    original = relu_program()
    region = original.regions[0]
    op = SemanticOperation(
        "third", "tensor", "add", ("first", "second"),
        "third", ty(3, 4),
    )
    mutated = replace(
        original, regions=(replace(region, operations=(
            *region.operations, op,
        )),), outputs=("third",),
    )
    with pytest.raises(TransformationEvidenceError, match="No admissible"):
        eliminate_idempotent_relu(mutated)
