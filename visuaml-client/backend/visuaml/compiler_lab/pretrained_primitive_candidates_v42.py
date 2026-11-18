"""Bounded, falsifiable low-rank candidate discovery from pretrained tensor bytes.

This is *not* discovery of a semantic feature or an exact neural circuit.
SVD is a mathematical weight factorization. Causal relevance requires replacing
the original parameter in the real model and measuring held-out behavior.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

MAX_ELEMENTS = 65536
SCHEMA = "visuaml.pretrained-primitive-candidate.v42"


class PrimitiveCandidateError(ValueError):
    """Unsafe, nonfinite, unsupported or ambiguous weight candidate."""


@dataclass(frozen=True)
class RankCandidate:
    rank: int
    matrix_shape: tuple[int, int]
    proposed_weight: np.ndarray
    original_weight_sha256: str
    proposed_weight_sha256: str
    relative_frobenius_error: float
    retained_spectral_energy: float
    construction: str
    schema: str = SCHEMA


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _admit(matrix: np.ndarray) -> np.ndarray:
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or matrix.dtype != np.float32 or matrix.size > MAX_ELEMENTS
            or any(not 1 <= dim <= 512 for dim in matrix.shape)
            or not np.isfinite(matrix).all()):
        raise PrimitiveCandidateError("Expected bounded, finite float32 matrix")
    if not np.any(matrix):
        raise PrimitiveCandidateError("Zero matrix has ambiguous spectral decomposition")
    return np.ascontiguousarray(matrix).copy()


def propose_low_rank(matrix: np.ndarray, rank: int, *,
                     negative_control: bool = False) -> RankCandidate:
    """Deterministic SVD candidate and matched-spectrum coordinate-scramble control.

    This factorization alone establishes neither mechanism nor transfer.
    """
    original = _admit(matrix)
    if type(rank) is not int or not 1 <= rank <= min(original.shape):
        raise PrimitiveCandidateError("Invalid integer rank")
    u, s, vh = np.linalg.svd(original.astype(np.float64), full_matrices=False)
    if negative_control:
        # Right-coordinate reversal preserves singular values but changes the
        # represented linear function. It is NOT a semantics-preserving rewrite.
        right = vh[:rank, ::-1]
    else:
        right = vh[:rank, :]
    restored = (u[:, :rank] * s[:rank]) @ right
    if not np.isfinite(restored).all():
        raise PrimitiveCandidateError("Nonfinite reconstruction")
    output = np.ascontiguousarray(restored.astype(np.float32))
    ratio = float(np.linalg.norm(
        original.astype(np.float64) - restored, ord="fro"
    ) / np.linalg.norm(original.astype(np.float64), ord="fro"))
    energy = float(np.sum(s[:rank] ** 2) / np.sum(s ** 2))
    if not math.isfinite(ratio) or not 0 <= energy <= 1 + 1e-12:
        raise PrimitiveCandidateError("Invalid finite singular-value evidence")
    return RankCandidate(
        rank, tuple(original.shape), output,
        tensor_sha256(original), tensor_sha256(output), ratio, min(1., energy),
        "matched-spectrum-coordinate-scramble" if negative_control
        else "truncated-svd",
    )


def intervention_metrics(base: np.ndarray, altered: np.ndarray) -> dict:
    """Compare observable logits, never pretend to measure labeled accuracy."""
    if (not isinstance(base, np.ndarray) or not isinstance(altered, np.ndarray)
            or base.dtype != np.float32 or altered.dtype != np.float32
            or base.shape != altered.shape or base.ndim != 2
            or not np.isfinite(base).all() or not np.isfinite(altered).all()):
        raise PrimitiveCandidateError("Incompatible finite float32 logit witnesses")
    delta = altered.astype(np.float64) - base.astype(np.float64)
    normalizer = float(np.linalg.norm(base.astype(np.float64)))
    return {
        "max_abs_logit_change": float(np.max(np.abs(delta))),
        "relative_l2_logit_change": float(
            np.linalg.norm(delta) / max(normalizer, 1e-12)
        ),
        "top1_unchanged": bool(np.argmax(base, axis=-1).tolist()
                               == np.argmax(altered, axis=-1).tolist()),
        "baseline_top1": [int(i) for i in np.argmax(base, axis=-1)],
        "modified_top1": [int(i) for i in np.argmax(altered, axis=-1)],
    }
