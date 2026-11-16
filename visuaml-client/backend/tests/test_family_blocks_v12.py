"""Fixed-seed cross-framework, stateful and negative contracts for six v12 blocks."""

from __future__ import annotations

import numpy as np
import pytest
from visuaml.compiler_lab.family_blocks_v12 import (
    BlockContractError,
    DecoderWeights,
    ScanState,
    ScanWeights,
    decoder_step_numpy,
    decoder_step_torch,
    empty_cache,
    flow_numpy,
    flow_torch,
    gnn_numpy,
    gnn_torch,
    moe_numpy,
    moe_torch,
    quantize_per_channel_numpy,
    quantize_per_channel_torch,
    selective_conv_scan_numpy,
    selective_conv_scan_torch,
    sgd_numpy,
    sgd_torch,
)


@pytest.mark.parametrize("seed", [0, 1, 39])
def test_gqa_rope_decoder_prefill_decode_matches_torch_and_chunking(seed):
    rng = np.random.default_rng(seed)
    weights = DecoderWeights.seeded(8, 4, 2, seed=seed)
    sequence = rng.normal(size=(2, 5, 8)).astype(np.float32)
    np_cache = empty_cache(2, 2, 2, 16)
    th_cache = empty_cache(2, 2, 2, 16)
    outputs = []
    for chunk in (sequence[:, :3], sequence[:, 3:4], sequence[:, 4:]):
        a, np_cache = decoder_step_numpy(chunk, weights, np_cache, heads=4)
        b, th_cache = decoder_step_torch(chunk, weights, th_cache, heads=4)
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(np_cache.keys, th_cache.keys, atol=1e-6)
        outputs.append(a)
    full, full_cache = decoder_step_numpy(
        sequence, weights, empty_cache(2, 2, 2, 16), heads=4
    )
    np.testing.assert_allclose(
        np.concatenate(outputs, axis=1), full, atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(full_cache.keys, np_cache.keys, atol=1e-6)
    assert np_cache.keys.shape == (2, 5, 2, 2)


def test_decoder_rejects_cache_overrun_and_invalid_head_config():
    w = DecoderWeights.seeded(8, 4, 2, seed=3)
    x = np.ones((1, 2, 8), np.float32)
    with pytest.raises(BlockContractError, match="overflow"):
        decoder_step_numpy(x, w, empty_cache(1, 2, 2, 1), heads=4)
    with pytest.raises(BlockContractError):
        decoder_step_numpy(x, w, empty_cache(1, 2, 2, 8), heads=3)


@pytest.mark.parametrize("seed", [0, 4, 23])
def test_causal_conv_selective_scan_torch_and_chunking(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 6, 4)).astype(np.float32)
    w = ScanWeights(
        rng.normal(scale=0.1, size=(4, 3)).astype(np.float32),
        rng.uniform(0.1, 0.9, 4).astype(np.float32),
        rng.uniform(0.05, 0.7, 4).astype(np.float32),
        rng.normal(scale=0.5, size=4).astype(np.float32),
        rng.normal(size=4).astype(np.float32),
    )
    state = ScanState(np.zeros((2, 2, 4), np.float32), np.zeros((2, 4), np.float32))
    n, s = selective_conv_scan_numpy(x, w, state)
    t, st = selective_conv_scan_torch(x, w, state)
    np.testing.assert_allclose(n, t, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(s.recurrent, st.recurrent, atol=2e-6, rtol=2e-6)
    first, part = selective_conv_scan_numpy(x[:, :2], w, state)
    second, part = selective_conv_scan_numpy(x[:, 2:], w, part)
    np.testing.assert_allclose(np.concatenate([first, second], axis=1), n, atol=2e-6)
    np.testing.assert_array_equal(x, x)
    assert not np.shares_memory(s.convolution, state.convolution)


@pytest.mark.parametrize("seed", [0, 1, 11])
def test_capacity_bounded_topk_moe_routing_parity(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 6)).astype(np.float32)
    logits = rng.normal(size=(8, 4)).astype(np.float32)
    logits[0] = np.float32(0)  # deterministic equal-logit tie
    wi = rng.normal(scale=0.2, size=(4, 6, 9)).astype(np.float32)
    wo = rng.normal(scale=0.2, size=(4, 9, 6)).astype(np.float32)
    a, load, index = moe_numpy(x, logits, wi, wo, top_k=2, capacity=3)
    b, tload, tindex = moe_torch(x, logits, wi, wo, top_k=2, capacity=3)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(load, tload)
    np.testing.assert_array_equal(index, tindex)
    np.testing.assert_array_equal(index[0], [0, 1])
    assert load.max() <= 3


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_time_conditioned_flow_sampler_parity(method):
    rng = np.random.default_rng(92)
    x = rng.normal(size=(3, 8)).astype(np.float32)
    condition = rng.normal(size=(3, 4)).astype(np.float32)
    w = rng.normal(scale=0.1, size=(13, 8)).astype(np.float32)
    before = x.copy()
    a = flow_numpy(x, condition, w, steps=7, method=method)
    b = flow_torch(x, condition, w, steps=7, method=method)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(x, before)
    assert not np.array_equal(a, x)


@pytest.mark.parametrize("edges", [0, 3, 9])
def test_graph_segment_attention_duplicate_and_empty_edges(edges):
    rng = np.random.default_rng(2)
    nodes = rng.normal(size=(5, 4)).astype(np.float32)
    src = np.array([0, 1, 1, 3, 4, 2, 0, 0, 1], np.int64)[:edges]
    dst = np.array([1, 1, 1, 0, 0, 0, 3, 3, 3], np.int64)[:edges]
    w = rng.normal(scale=0.1, size=(4, 6)).astype(np.float32)
    a = rng.normal(scale=0.2, size=(12,)).astype(np.float32)
    n = gnn_numpy(nodes, src, dst, w, a)
    t = gnn_torch(nodes, src, dst, w, a)
    np.testing.assert_allclose(n, t, atol=3e-6, rtol=3e-6)
    if not edges:
        assert not n.any()


def test_graph_rejects_out_of_bounds_edges():
    nodes = np.zeros((3, 2), np.float32)
    with pytest.raises(BlockContractError, match="out-of-range"):
        gnn_numpy(
            nodes,
            np.array([5], np.int64),
            np.array([0], np.int64),
            np.ones((2, 2), np.float32),
            np.ones(4, np.float32),
        )


@pytest.mark.parametrize("seed", [0, 11])
def test_per_channel_int8_quantization_parity_with_zero_channel(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(9, 5)).astype(np.float32)
    w[:, 2] = 0
    q1, s1 = quantize_per_channel_numpy(w)
    q2, s2 = quantize_per_channel_torch(w)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_allclose(s1, s2, rtol=1e-6)
    assert s1[2] == 1 and not q1[:, 2].any()
    assert np.abs((q1 * s1) - w).max() < 0.025


@pytest.mark.parametrize("seed", [0, 2, 4])
def test_autograd_sgd_vs_independent_analytic_gradient(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 3)).astype(np.float32)
    y = rng.normal(size=(6, 2)).astype(np.float32)
    w = rng.normal(scale=0.2, size=(3, 2)).astype(np.float32)
    np_weights, loss_a = sgd_numpy(x, y, w, rate=0.04, steps=5)
    t_weights, loss_b = sgd_torch(x, y, w, rate=0.04, steps=5)
    np.testing.assert_allclose(np_weights, t_weights, atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(loss_a, loss_b, atol=3e-7, rtol=1e-6)
    assert loss_a[-1] < loss_a[0]
