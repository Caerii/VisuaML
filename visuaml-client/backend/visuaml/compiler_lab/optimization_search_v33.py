"""v33: gate candidate timing on source-bound independent numerical evidence.

This is an experiment harness, not an optimizer that selects production code.
It never times a rejected candidate; each accepted rule remains bounded by
its finite input witnesses and exact source fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .native_semantic_codegen_v32 import NativeSemanticCompiler
from .semantic_ir import SemanticProgram
from .transformation_evidence_v33 import (
    TransformationEvidenceError,
    certify_native_rewrite,
    eliminate_idempotent_relu,
    verify_record_integrity,
)

SCHEMA = "visuaml.optimization-experiment.v33"


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def evaluate_relu_candidate(
    program: SemanticProgram,
    *,
    witnesses: Sequence[Mapping[str, np.ndarray]],
    numpy_oracle: Callable[..., np.ndarray],
    torch_oracle: Callable[..., np.ndarray],
    trials: int = 9,
) -> dict:
    if type(trials) is not int or not 3 <= trials <= 51:
        raise ValueError("Expected 3..51 measured repetitions")
    try:
        after = eliminate_idempotent_relu(program)
    except TransformationEvidenceError as exc:
        result = {
            "schema": SCHEMA,
            "status": "not_applicable",
            "source_semantic_sha256": program.digest(),
            "rule": "relu-idempotence-v1",
            "reason": str(exc),
            "limitations": "No native candidate was generated or measured.",
        }
        return {**result, "record_sha256": _hash(result)}
    evidence = certify_native_rewrite(
        program, after, rule="relu-idempotence-v1",
        witnesses=witnesses, numpy_oracle=numpy_oracle,
        torch_oracle=torch_oracle,
    )
    if not verify_record_integrity(evidence):
        raise ValueError("Numerical witness evidence checksum failed")
    if evidence["status"] != "passed_finite_witnesses":
        rejected = {
            "schema": SCHEMA,
            "status": "rejected_no_performance_measurement",
            "rule": "relu-idempotence-v1",
            "numerical_evidence": evidence,
            "limitations": "No speedup can be accepted after rejected numerical evidence.",
        }
        return {**rejected, "record_sha256": _hash(rejected)}
    baseline = NativeSemanticCompiler(program)
    candidate = NativeSemanticCompiler(after)
    if not witnesses:
        raise ValueError("No validated input witnesses")
    # Timing only replays a previously checked witness. No new unverified
    # geometry, numeric behavior, dtype or state effect is admitted.
    witness = {
        key: np.ascontiguousarray(value).copy()
        for key, value in witnesses[0].items()
    }
    original = baseline.run(**witness)
    transformed = candidate.run(**witness)
    if not np.allclose(original, transformed, atol=2e-5, rtol=2e-5):
        raise ValueError("Witness changed between admission and benchmark")
    readings: dict[str, list[float]] = {
        "before": [], "after": [],
    }
    # Explicit warm-up, excluded from latency samples.
    for _ in range(2):
        baseline.run(**witness)
        candidate.run(**witness)
    for trial in range(trials):
        order = ("before", "after") if trial % 2 == 0 else ("after", "before")
        for label in order:
            native = baseline if label == "before" else candidate
            started = time.perf_counter_ns()
            native.run(**witness)
            readings[label].append((time.perf_counter_ns() - started) / 1e6)
    before_median = statistics.median(readings["before"])
    after_median = statistics.median(readings["after"])
    payload = {
        "schema": SCHEMA,
        "status": "exploratory_timing_after_finite_numerical_witnesses",
        "rule": "relu-idempotence-v1",
        "numerical_evidence": evidence,
        "before": baseline.report(),
        "after": candidate.report(),
        "trials": trials,
        "warmup_per_variant": 2,
        "alternating_order": True,
        "warm_ms": {
            label: {
                "median": round(statistics.median(values), 6),
                "raw": [round(v, 6) for v in values],
            }
            for label, values in readings.items()
        },
        "median_ratio_before_over_after": round(
            before_median / after_median, 6
        ),
        "input_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in witness.items()
        },
        "limitations": (
            "Tiny native CPU warm execution, only one bounded input geometry; "
            "excludes compile/model load, energy and serving concurrency. "
            "Finite checks are not proofs and speedup is not globally accepted."
        ),
    }
    return {**payload, "record_sha256": _hash(payload)}
