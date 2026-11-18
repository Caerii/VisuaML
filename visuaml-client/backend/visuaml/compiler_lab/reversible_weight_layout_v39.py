"""Exact invertible hidden-channel reordering for a bounded ReLU MLP.

For P a permutation, W1'=P W1, b1'=P b1, W2'=W2 P^{-1}.
This is an exact algebraic representation transformation in real arithmetic.
In floating-point, summation order changes can change last bits; no blanket
bitwise-output-equivalence or equivalence of arbitrary stateful graphs is claimed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

MAX_ELEMENTS = 65_536


class LayoutRewriteError(ValueError):
    """Unsupported tensor, axis, permutation or resource geometry."""


@dataclass(frozen=True)
class ReorderedMLP:
    first: np.ndarray
    bias: np.ndarray
    second: np.ndarray
    permutation: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    receipt: dict


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _strict_float(value: np.ndarray, rank: int) -> np.ndarray:
    if (not isinstance(value, np.ndarray)
            or value.dtype != np.float32 or value.ndim != rank
            or any(d <= 0 or d > 512 for d in value.shape)
            or value.size > MAX_ELEMENTS or not np.isfinite(value).all()):
        raise LayoutRewriteError("Only bounded finite float32 dense tensors")
    return np.ascontiguousarray(value).copy()


def permute_relu_hidden(
    first: np.ndarray, bias: np.ndarray, second: np.ndarray,
    permutation: tuple[int, ...],
) -> ReorderedMLP:
    """Permute independent hidden units, remapping *both* adjacent weights."""
    w1, b1, w2 = (_strict_float(first, 2),
                  _strict_float(bias, 1), _strict_float(second, 2))
    hidden = w1.shape[0]
    if b1.shape != (hidden,) or w2.shape[1] != hidden:
        raise LayoutRewriteError("Inconsistent two-linear ReLU hidden geometry")
    if (not isinstance(permutation, tuple)
            or len(permutation) != hidden
            or any(type(i) is not int for i in permutation)
            or set(permutation) != set(range(hidden))):
        raise LayoutRewriteError("Expected a unique in-range hidden-axis permutation")
    indices = np.array(permutation, dtype=np.int64)
    inverse = np.argsort(indices)
    # Given (P v)[j] = v[permutation[j]], (W2 P^-1)[:,j] = W2[:,permutation[j]].
    new_first = np.ascontiguousarray(w1[indices])
    new_bias = np.ascontiguousarray(b1[indices])
    new_second = np.ascontiguousarray(w2[:, indices])
    # Exact byte-for-byte round trip, independent of any numerical execution.
    if (not np.array_equal(new_first[inverse], w1)
            or not np.array_equal(new_bias[inverse], b1)
            or not np.array_equal(new_second[:, inverse], w2)):
        raise LayoutRewriteError("Permutation failed exact reversible weight check")
    original = {"first": _sha(w1), "bias": _sha(b1), "second": _sha(w2)}
    transformed = {
        "first": _sha(new_first), "bias": _sha(new_bias), "second": _sha(new_second)
    }
    receipt = {
        "schema": "visuaml.reversible-hidden-layout.v39",
        "algebraic_contract": "W2@relu(W1@x+b) == W2'@relu(W1'@x+b') over exact reals",
        "original_sha256": original, "transformed_sha256": transformed,
        "hidden_count": hidden, "permutation": list(permutation),
        "inverse_permutation": inverse.tolist(), "weights_recovered_byte_exact": True,
        "general_graph_equivalence_proved": False,
        "floating_point_output_bitwise_equivalence_proved": False,
        "limitations": (
            "Two adjacent dense linear layers with pointwise ReLU, no residual "
            "fanout, batch normalization, shared weights, mutable state, "
            "quantization or untracked aliases; float32 execution may round differently."
        ),
    }
    return ReorderedMLP(new_first, new_bias, new_second,
                        tuple(permutation), tuple(int(x) for x in inverse), receipt)
