"""v33: reproducible, evidence-carrying pure Semantic IR rewrites.

Digest-linked numerical records are NOT signatures, formal proofs, or trusted
authentication. A rule has only passed explicit finite witnesses if *two*
independent reference implementations and both generated native binaries
agree within recorded tolerances. Preserve counterexamples and never turn a
missing or failed test into a verified optimization.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import replace
from typing import Callable, Mapping, Sequence

import numpy as np

from .native_semantic_codegen_v32 import NativeIRCodegenError, NativeSemanticCompiler
from .semantic_ir import SemanticProgram, validate_semantic_program

SCHEMA = "visuaml.transformation-evidence.v33"


class TransformationEvidenceError(ValueError):
    """An invalid source graph, witness or integrity record."""


def _digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def verify_record_integrity(record: dict) -> bool:
    """Hash check only: does not authenticate authorship or prove semantics."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return False
    actual = record.get("record_sha256")
    if not isinstance(actual, str) or len(actual) != 64:
        return False
    try:
        computed = _digest({
            key: value for key, value in record.items() if key != "record_sha256"
        })
    except (TypeError, ValueError):
        return False
    return actual == computed


def eliminate_idempotent_relu(program: SemanticProgram) -> SemanticProgram:
    """Rewrite relu(relu(x)) -> relu(x) only with no intervening uses.

    Pure, single-region v32 graphs only.  The second result name survives,
    which preserves SSA consumers and outputs; the first name is eliminated
    only if it has no external use. No compiler pass claims proof by itself.
    """
    validate_semantic_program(program)
    if program.state_types or len(program.regions) != 1:
        raise TransformationEvidenceError("Pure single-region program required")
    region = program.regions[0]
    operations = list(region.operations)
    for i in range(len(operations) - 1):
        earlier, later = operations[i], operations[i + 1]
        if (
            earlier.op != "relu" or later.op != "relu"
            or earlier.effect != "pure" or later.effect != "pure"
            or earlier.result is None or later.result is None
            or len(earlier.operands) != 1
            or later.operands != (earlier.result,)
            or earlier.result in program.outputs
            or earlier.state_cell is not None
            or later.state_cell is not None
        ):
            continue
        # Rewriting only a single-use SSA value prevents changing another
        # user's dependency or leaving an unresolved reference.
        used_elsewhere = any(
            earlier.result in op.operands
            for j, op in enumerate(operations) if j != i + 1
        )
        if used_elsewhere:
            continue
        if earlier.result_type != later.result_type:
            continue
        replacement = replace(later, operands=earlier.operands)
        rewritten = tuple(
            op for j, op in enumerate(operations) if j != i
        )
        # After deletion, the next op is at index i.
        rewritten = (
            *rewritten[:i], replacement, *rewritten[i + 1:]
        )
        new_region = replace(region, operations=rewritten)
        lineage = hashlib.sha256(
            (program.digest() + ":relu-idempotence-v1").encode()
        ).hexdigest()
        result = replace(
            program, regions=(new_region,), source_sha256=lineage
        )
        validate_semantic_program(result)
        return result
    raise TransformationEvidenceError("No admissible single-use relu pair")


def _numeric(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise TransformationEvidenceError("Oracle or witness shape/dtype/nonfinite")
    return np.ascontiguousarray(value)


def _difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def certify_native_rewrite(
    before: SemanticProgram,
    after: SemanticProgram,
    *,
    rule: str,
    witnesses: Sequence[Mapping[str, np.ndarray]],
    numpy_oracle: Callable[..., np.ndarray],
    torch_oracle: Callable[..., np.ndarray],
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> dict:
    """Run independent native and oracle computations; record every witness.

    Callable references are trusted local experiments, not sandboxed plugins.
    Status is finite-witness success or rejection, never mathematical proof.
    """
    if not isinstance(rule, str) or rule != "relu-idempotence-v1":
        raise TransformationEvidenceError("Unknown unreviewed rewrite rule")
    if (
        type(atol) is not float or type(rtol) is not float
        or not math.isfinite(atol) or not math.isfinite(rtol)
        or not 0 <= atol <= 1e-3 or not 0 <= rtol <= 1e-3
    ):
        raise TransformationEvidenceError("Invalid numerical tolerance")
    if not isinstance(witnesses, (tuple, list)) or not 1 <= len(witnesses) <= 32:
        raise TransformationEvidenceError("Need 1..32 bounded witnesses")
    canonical = eliminate_idempotent_relu(before)
    if canonical.digest() != after.digest():
        raise TransformationEvidenceError("Candidate is not the declared rewrite")
    # Admitting both programs before execution prevents tests from implicitly
    # endorsing unsupported effects, free symbols, or unbounded allocations.
    baseline = NativeSemanticCompiler(before)
    candidate = NativeSemanticCompiler(after)
    if baseline.plan.input_names != candidate.plan.input_names:
        raise TransformationEvidenceError("Rewrite changed input bindings")
    if baseline.plan.input_shapes != candidate.plan.input_shapes:
        raise TransformationEvidenceError("Rewrite changed input shapes")
    if baseline.plan.output_shape != candidate.plan.output_shape:
        raise TransformationEvidenceError("Rewrite changed output contract")
    expected_keys = set(baseline.plan.input_names)
    shape = baseline.plan.output_shape
    checks = []
    rejection = None
    for number, witness in enumerate(witnesses):
        if not isinstance(witness, Mapping) or set(witness) != expected_keys:
            raise TransformationEvidenceError("Incomplete or extra witness inputs")
        inputs = {
            key: _numeric(witness[key], input_shape).copy()
            for key, input_shape in zip(
                baseline.plan.input_names,
                baseline.plan.input_shapes, strict=True,
            )
        }
        if sum(value.size for value in inputs.values()) > 4096:
            raise TransformationEvidenceError("Counterexample artifact element budget exceeded")
        expected_numpy = _numeric(numpy_oracle(**inputs), shape)
        expected_torch = _numeric(torch_oracle(**inputs), shape)
        actual_before = baseline.run(**inputs)
        actual_after = candidate.run(**inputs)
        readings = {
            "oracle_numpy_vs_torch": _difference(expected_numpy, expected_torch),
            "baseline_vs_numpy": _difference(actual_before, expected_numpy),
            "candidate_vs_numpy": _difference(actual_after, expected_numpy),
            "candidate_vs_baseline": _difference(actual_after, actual_before),
        }
        valid = all(
            np.allclose(a, b, atol=atol, rtol=rtol)
            for a, b in (
                (expected_numpy, expected_torch),
                (actual_before, expected_numpy),
                (actual_after, expected_numpy),
                (actual_after, actual_before),
            )
        )
        check = {
            "witness_index": number,
            "passed": valid,
            "input_sha256": {
                key: hashlib.sha256(value.tobytes()).hexdigest()
                for key, value in inputs.items()
            },
            "max_abs_error": readings,
        }
        checks.append(check)
        if not valid and rejection is None:
            # Preserve a small bounded counterexample, not just an opaque hash.
            rejection = {
                "witness_index": number,
                "inputs": {
                    key: value.tolist() for key, value in inputs.items()
                },
                "expected_numpy": expected_numpy.tolist(),
                "candidate": actual_after.tolist(),
                "max_abs_error": readings,
            }
    record = {
        "schema": SCHEMA,
        "rule": rule,
        "evidence_class": "finite_numeric_witnesses_not_formal_proof",
        "status": (
            "passed_finite_witnesses"
            if all(check["passed"] for check in checks) else "rejected"
        ),
        "before": baseline.report(),
        "after": candidate.report(),
        "comparison": {
            "atol": atol, "rtol": rtol,
            "independent_oracles": ["trusted_numpy", "trusted_pytorch"],
            "witnesses": checks,
            "counterexample": rejection,
        },
        "platform": platform.platform(),
        "limits": (
            "These checks only admit finite tested inputs; compiler changes, "
            "new model shapes, precisions or target hardware require new evidence. "
            "SHA256 integrity does not authenticate a signer."
        ),
    }
    return {**record, "record_sha256": _digest(record)}
