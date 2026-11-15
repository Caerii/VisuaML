"""Cross-framework reference semantics for narrow exotic neural primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from visuaml.compiler_lab.exotic_references import (
    REFERENCE_KERNELS,
    ReferenceKernelError,
    diagonal_scan_numpy,
    diagonal_scan_torch,
    graph_messages_numpy,
    graph_messages_torch,
    lif_numpy,
    lif_torch,
    routed_moe_numpy,
    routed_moe_torch,
)


def test_four_reference_kernels_are_explicitly_not_aot_backends():
    assert REFERENCE_KERNELS == {"diagonal_selective_scan", "stable_topk_moe",
                                  "graph_additive_scatter", "lif_discrete"}


@pytest.mark.parametrize("seed", [0, 19, 77])
def test_diagonal_ssm_scan_final_state_and_outputs(seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(2, 11, 7)).astype(np.float32)
    a = rng.uniform(0, 0.95, size=u.shape).astype(np.float32)
    b = rng.normal(size=u.shape).astype(np.float32)
    c = rng.normal(size=u.shape).astype(np.float32)
    state = rng.normal(size=(2, 7)).astype(np.float32)
    snapshot = state.copy()
    ref = diagonal_scan_numpy(u, a, b, c, state)
    expected, after = diagonal_scan_torch(*(torch.from_numpy(z) for z in (u, a, b, c, state)))
    np.testing.assert_allclose(ref.outputs, expected.numpy(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(ref.final_state, after.numpy(), atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(state, snapshot)


def test_diagonal_ssm_rejects_invalid_shapes_and_nan():
    u = np.zeros((1, 3, 2), dtype=np.float32)
    with pytest.raises(ReferenceKernelError):
        diagonal_scan_numpy(u, u, u, u, np.zeros((1, 3), dtype=np.float32))
    dirty = u.copy()
    dirty[0, 1, 0] = np.nan
    with pytest.raises(ReferenceKernelError):
        diagonal_scan_numpy(dirty, u, u, u, np.zeros((1, 2), dtype=np.float32))


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_sparse_expert_routing_matches_torch(top_k):
    rng = np.random.default_rng(70 + top_k)
    x = rng.normal(size=(8, 6)).astype(np.float32)
    logits = rng.normal(size=(8, 4)).astype(np.float32)
    weight = rng.normal(size=(4, 6, 5)).astype(np.float32)
    result = routed_moe_numpy(x, logits, weight, top_k=top_k)
    value, ids, gates = routed_moe_torch(*(torch.from_numpy(x0) for x0 in (x, logits, weight)), top_k=top_k)
    np.testing.assert_array_equal(result.selected_experts, ids.numpy())
    np.testing.assert_allclose(result.selected_gates, gates.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.output, value.numpy(), rtol=1e-5, atol=1e-5)


def test_router_tie_policy_lowest_expert_id():
    x = np.ones((2, 3), dtype=np.float32)
    logits = np.array([[3, 3, 1], [0, 0, 0]], dtype=np.float32)
    w = np.ones((3, 3, 2), dtype=np.float32)
    out = routed_moe_numpy(x, logits, w, top_k=2)
    np.testing.assert_array_equal(out.selected_experts, [[0, 1], [0, 1]])


def test_router_rejects_wrong_topk_and_expert_shape():
    x = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ReferenceKernelError):
        routed_moe_numpy(x, np.ones((2, 4), dtype=np.float32), np.ones((4, 4, 2), dtype=np.float32), top_k=2)
    with pytest.raises(ReferenceKernelError):
        routed_moe_numpy(x, np.ones((2, 4), dtype=np.float32), np.ones((4, 3, 2), dtype=np.float32), top_k=0)


def test_graph_scatter_duplicate_edges_counted_and_parity():
    rng = np.random.default_rng(64)
    nodes = rng.normal(size=(7, 5)).astype(np.float32)
    matrix = rng.normal(size=(5, 3)).astype(np.float32)
    edges = np.array([[0, 0, 1, 5, 5], [2, 2, 2, 6, 0]], dtype=np.int64)
    ref = graph_messages_numpy(nodes, edges, matrix)
    got, degree = graph_messages_torch(*(torch.from_numpy(t) for t in (nodes, edges, matrix)))
    np.testing.assert_allclose(ref.aggregated, got.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(ref.degree, degree.numpy())
    assert ref.degree[2] == 3


def test_graph_scatter_rejects_negative_indices_and_supports_empty_edges():
    x = np.ones((3, 2), dtype=np.float32)
    w = np.eye(2, dtype=np.float32)
    r = graph_messages_numpy(x, np.empty((2, 0), dtype=np.int64), w)
    assert np.count_nonzero(r.aggregated) == 0
    with pytest.raises(ReferenceKernelError):
        graph_messages_numpy(x, np.array([[0, -1], [1, 2]], dtype=np.int64), w)


@pytest.mark.parametrize("refractory_steps", [0, 1, 3])
def test_lif_discrete_reset_and_refractory_parity(refractory_steps):
    rng = np.random.default_rng(11 + refractory_steps)
    currents = rng.uniform(-0.1, 1.5, size=(12, 2, 5)).astype(np.float32)
    initial = rng.uniform(0, 0.8, size=(2, 5)).astype(np.float32)
    reference = lif_numpy(currents, initial, leak=0.9, threshold=1.0,
                          refractory_steps=refractory_steps)
    spikes, final, refr = lif_torch(torch.from_numpy(currents), torch.from_numpy(initial),
                                   leak=0.9, threshold=1.0, refractory_steps=refractory_steps)
    np.testing.assert_array_equal(reference.spikes, spikes.numpy())
    np.testing.assert_allclose(reference.membrane, final.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(reference.refractory, refr.numpy())


def test_lif_rejects_unphysical_leak_and_nans():
    currents = np.ones((2, 1, 3), dtype=np.float32)
    initial = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ReferenceKernelError):
        lif_numpy(currents, initial, leak=1.2, threshold=1.0)
    with pytest.raises(ReferenceKernelError):
        lif_numpy(currents, initial, leak=0.9, threshold=np.nan)
