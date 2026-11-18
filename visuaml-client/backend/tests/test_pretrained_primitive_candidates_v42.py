"""v42 primitive-screening contract: weight geometry is not mechanism evidence."""
from __future__ import annotations

import hashlib
from itertools import pairwise

import numpy as np
import pytest
from visuaml.compiler_lab.pretrained_primitive_candidates_v42 import (
    PrimitiveCandidateError,
    intervention_metrics,
    propose_low_rank,
    select_compressed_rank,
    tensor_sha256,
)


def test_svd_candidates_are_finite_rank_bounded_and_best_frobenius_approximation():
    w = np.random.default_rng(42).normal(size=(12, 7)).astype(np.float32)
    original = w.copy()
    candidates = [propose_low_rank(w, rank) for rank in (1, 2, 4, 7)]
    assert np.array_equal(w, original)
    assert all(x.proposed_weight.shape == w.shape for x in candidates)
    assert all(x.original_weight_sha256 == tensor_sha256(w) for x in candidates)
    assert all(
        x.proposed_weight_sha256 == hashlib.sha256(x.proposed_weight.tobytes()).hexdigest()
        for x in candidates
    )
    assert all(x.construction == "truncated-svd" for x in candidates)
    assert all(a.relative_frobenius_error >= b.relative_frobenius_error
               for a, b in pairwise(candidates))
    assert all(a.retained_spectral_energy <= b.retained_spectral_energy
               for a, b in pairwise(candidates))
    np.testing.assert_allclose(candidates[-1].proposed_weight, w, rtol=1e-5, atol=1e-5)


def test_identical_singular_spectrum_does_not_imply_identical_function():
    w = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
    candidate = propose_low_rank(w, 2)
    control = propose_low_rank(w, 2, negative_control=True)
    assert candidate.retained_spectral_energy == control.retained_spectral_energy
    assert candidate.proposed_weight_sha256 != control.proposed_weight_sha256
    x = np.array([1, 2, 4], dtype=np.float32)
    assert not np.allclose(candidate.proposed_weight @ x,
                           control.proposed_weight @ x)
    assert candidate.relative_frobenius_error <= control.relative_frobenius_error


@pytest.mark.parametrize("bad_rank", [0, 4, True, 1.5, -1])
def test_rank_admission_fails_closed(bad_rank):
    with pytest.raises(PrimitiveCandidateError, match="rank"):
        propose_low_rank(np.ones((3, 3), dtype=np.float32), bad_rank)


def test_bad_weights_are_rejected():
    base = np.ones((2, 3), dtype=np.float32)
    for bad in (
        base.astype(np.float64), base.reshape((1, 2, 3)),
        np.full_like(base, np.nan), np.zeros_like(base),
        np.ones((512, 512), dtype=np.float32),
    ):
        with pytest.raises(PrimitiveCandidateError):
            propose_low_rank(bad, 1)


def test_behavioral_logit_witnesses_do_not_conflate_similarity_with_accuracy():
    original = np.array([[3, 2, 1], [1, 2, 3]], dtype=np.float32)
    changed = np.array([[2.9, 2.1, 1], [1, 3, 2]], dtype=np.float32)
    info = intervention_metrics(original, changed)
    assert info["baseline_top1"] == [0, 2]
    assert info["modified_top1"] == [0, 1]
    assert not info["top1_unchanged"]
    assert info["max_abs_logit_change"] == 1
    assert info["relative_l2_logit_change"] > 0
    with pytest.raises(PrimitiveCandidateError):
        intervention_metrics(original, changed.astype(np.float64))
    with pytest.raises(PrimitiveCandidateError):
        intervention_metrics(original, np.full_like(changed, np.nan))


def test_full_rank_is_a_numerical_baseline_not_exact_bitwise_certificate():
    rng = np.random.default_rng(77)
    w = rng.standard_normal((10, 10)).astype(np.float32)
    candidate = propose_low_rank(w, 10)
    assert candidate.relative_frobenius_error < 1e-12
    assert candidate.retained_spectral_energy > 1 - 1e-12
    assert np.isfinite(candidate.proposed_weight).all()
    np.testing.assert_allclose(candidate.proposed_weight, w, atol=1e-6, rtol=1e-6)


def test_full_rank_identity_does_not_count_as_compression():
    def candidate(rank, *, top1=True, error=0.0):
        cases = [
            {
                "group": group,
                "numpy": {
                    "top1_unchanged": top1,
                    "relative_l2_logit_change": error,
                },
            }
            for group in ("discovery", "heldout")
            for _ in range(5)
        ]
        return {
            "construction": "truncated-svd",
            "rank": rank,
            "witnesses": cases,
        }

    only_baseline = select_compressed_rank(
        [candidate(4, top1=False), candidate(10)], full_rank=10
    )
    assert only_baseline == {
        "selected_rank": None,
        "heldout_pass": False,
        "compressed_candidate_accepted": False,
        "full_rank_reconstruction_control_pass": True,
    }
    reduced = select_compressed_rank(
        [candidate(8), candidate(10)], full_rank=10
    )
    assert reduced["selected_rank"] == 8
    assert reduced["compressed_candidate_accepted"] is True
    with pytest.raises(PrimitiveCandidateError, match="Full-rank"):
        select_compressed_rank([candidate(8)], full_rank=10)
