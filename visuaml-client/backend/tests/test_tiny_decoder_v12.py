"""Tiny autoregressive decoder inference / cache and deterministic generation."""

from __future__ import annotations

import numpy as np
import pytest
from visuaml.compiler_lab.family_blocks_v12 import BlockContractError, empty_cache
from visuaml.compiler_lab.tiny_decoder_v12 import (
    TinyDecoderWeights,
    decoder_tokens_numpy,
    decoder_tokens_torch,
    greedy_generate,
)


@pytest.mark.parametrize("seed", [0, 8, 19])
def test_complete_tiny_decoder_prefill_and_decode_parity(seed):
    w = TinyDecoderWeights.seeded(seed=seed)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 19, size=(2, 6), dtype=np.int64)
    np_state = empty_cache(2, 2, 2, 16)
    torch_state = empty_cache(2, 2, 2, 16)
    outs = []
    for part in (tokens[:, :3], tokens[:, 3:5], tokens[:, 5:]):
        n, np_state = decoder_tokens_numpy(part, w, np_state)
        t, torch_state = decoder_tokens_torch(part, w, torch_state)
        np.testing.assert_allclose(n, t, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(np_state.keys, torch_state.keys, atol=1e-6)
        outs.append(n)
    full, fs = decoder_tokens_numpy(tokens, w, empty_cache(2, 2, 2, 16))
    np.testing.assert_allclose(np.concatenate(outs, axis=1), full, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(fs.values, np_state.values, atol=1e-6)
    assert full.shape == (2, 6, 19)


@pytest.mark.parametrize("seed", [0, 19, 45])
def test_greedy_autoregressive_generation_matches_and_is_reproducible(seed):
    w = TinyDecoderWeights.seeded(seed=seed)
    tokens = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64)
    a, as_ = greedy_generate(tokens, w, steps=4, backend="numpy")
    b, bs = greedy_generate(tokens, w, steps=4, backend="torch")
    assert a.shape == (2, 4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(as_.keys, bs.keys, atol=1e-6)
    np.testing.assert_array_equal(
        greedy_generate(tokens, w, steps=4, backend="numpy")[0], a
    )
    assert as_.keys.shape[1] == 7


def test_tokens_reject_invalid_ids_and_context_overflow():
    w = TinyDecoderWeights.seeded()
    with pytest.raises(BlockContractError):
        decoder_tokens_numpy(np.array([[99]], np.int64), w, empty_cache(1, 2, 2, 16))
    with pytest.raises(BlockContractError):
        greedy_generate(np.array([[1, 2]], np.int64), w, steps=4, max_context=5)
