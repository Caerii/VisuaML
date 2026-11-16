"""Cross-framework, explicit reference kernels for four architecture primitives.

These ARE NOT v11 NumPy-AOT operators, native accelerator kernels, or complete
Mamba/MoE/GNN/spiking models. They establish restricted, deterministic numerical
contracts to serve as oracles for future legalizations and optimization tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_ELEMENTS = 2_000_000


class ReferenceKernelError(ValueError):
    """Invalid reference-kernel input or unsupported semantics."""


def _fp32(value: Any, name: str, *, rank: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != rank
    ):
        raise ReferenceKernelError(f"{name} requires a float32 rank-{rank} ndarray")
    if value.size > MAX_ELEMENTS or not bool(np.all(np.isfinite(value))):
        raise ReferenceKernelError(
            f"{name} exceeds budget or contains non-finite values"
        )
    return value


def _int64(value: Any, name: str, *, rank: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.int64
        or value.ndim != rank
    ):
        raise ReferenceKernelError(f"{name} requires an int64 rank-{rank} ndarray")
    if value.size > MAX_ELEMENTS:
        raise ReferenceKernelError(f"{name} exceeds element budget")
    return value


@dataclass(frozen=True)
class ScanResult:
    outputs: np.ndarray
    final_state: np.ndarray


def diagonal_scan_numpy(
    u: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, initial: np.ndarray
) -> ScanResult:
    """Diagonal selective-scan microkernel: s=a*s+b*u, y=c*s, sequential f32.

    No Mamba convolution, input-dependent parameter projection, gating, or
    parallel associative scan is claimed. State is not mutated in-place.
    """
    for name, value in (("u", u), ("a", a), ("b", b), ("c", c)):
        _fp32(value, name, rank=3)
    _fp32(initial, "initial", rank=2)
    if not u.shape == a.shape == b.shape == c.shape or initial.shape != (
        u.shape[0],
        u.shape[2],
    ):
        raise ReferenceKernelError(
            "Diagonal scan requires matching [batch,time,channels] tensors"
        )
    if u.shape[1] > 2048:
        raise ReferenceKernelError(
            "Sequence length exceeds sequential reference budget"
        )
    state = initial.copy()
    out = np.empty_like(u)
    for t in range(u.shape[1]):
        state = a[:, t, :] * state + b[:, t, :] * u[:, t, :]
        out[:, t, :] = c[:, t, :] * state
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(state)):
        raise ReferenceKernelError("Scan overflowed float32")
    return ScanResult(out, state)


def diagonal_scan_torch(u: Any, a: Any, b: Any, c: Any, initial: Any) -> Any:
    import torch

    tensors = (u, a, b, c, initial)
    if any(
        not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32
        or v.device.type != "cpu"
        for v in tensors
    ):
        raise ReferenceKernelError("Torch reference requires CPU float32 tensors")
    # Use the same strict admission as NumPy before computation.
    diagonal_scan_numpy(*(v.detach().numpy() for v in tensors))
    state = initial.clone()
    ys = []
    for t in range(u.shape[1]):
        state = a[:, t, :] * state + b[:, t, :] * u[:, t, :]
        ys.append(c[:, t, :] * state)
    return torch.stack(ys, dim=1) if ys else u.clone(), state


@dataclass(frozen=True)
class MoEResult:
    output: np.ndarray
    selected_experts: np.ndarray
    selected_gates: np.ndarray


def routed_moe_numpy(
    x: np.ndarray, logits: np.ndarray, weights: np.ndarray, *, top_k: int
) -> MoEResult:
    """Stable top-k token routing, selected-softmax weights, dense linear experts.

    No expert capacity, dropout, load balancing, distributed all-to-all, or
    non-linear FFN is included. Index ties choose the lowest expert ID.
    """
    _fp32(x, "tokens", rank=2)
    _fp32(logits, "router", rank=2)
    _fp32(weights, "experts", rank=3)
    n, d = x.shape
    e, weight_d, h = weights.shape
    if logits.shape != (n, e) or weight_d != d or not (1 <= top_k <= e <= 256):
        raise ReferenceKernelError("Invalid MoE expert count, shape, or top-k")
    if n * top_k * h > MAX_ELEMENTS:
        raise ReferenceKernelError("MoE routed activation budget exceeded")
    selected = np.argsort(-logits, axis=-1, kind="stable")[:, :top_k].astype(np.int64)
    selected_scores = np.take_along_axis(logits, selected, axis=1)
    centered = selected_scores - selected_scores.max(axis=1, keepdims=True)
    e_scores = np.exp(centered)
    gates = e_scores / e_scores.sum(axis=1, keepdims=True)
    output = np.zeros((n, h), dtype=np.float32)
    for k in range(top_k):
        wk = weights[selected[:, k]]
        projected = np.einsum("nd,ndh->nh", x, wk, optimize=False)
        output += gates[:, k, None] * projected
    if not np.all(np.isfinite(output)):
        raise ReferenceKernelError("MoE output is non-finite")
    return MoEResult(output, selected, gates.astype(np.float32))


def routed_moe_torch(x: Any, logits: Any, weights: Any, *, top_k: int) -> Any:
    import torch

    if any(
        not isinstance(v, torch.Tensor)
        or v.device.type != "cpu"
        or v.dtype != torch.float32
        for v in (x, logits, weights)
    ):
        raise ReferenceKernelError("Torch MoE reference requires CPU float32 tensors")
    routed_moe_numpy(
        x.detach().numpy(),
        logits.detach().numpy(),
        weights.detach().numpy(),
        top_k=top_k,
    )
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    gates = torch.softmax(torch.gather(logits, 1, selected), dim=-1)
    output = torch.zeros((x.shape[0], weights.shape[2]), dtype=torch.float32)
    for j in range(top_k):
        picked = weights[selected[:, j]]
        values = torch.einsum("nd,ndh->nh", x, picked)
        output = output + gates[:, j, None] * values
    return output, selected, gates


@dataclass(frozen=True)
class GraphResult:
    aggregated: np.ndarray
    degree: np.ndarray


def graph_messages_numpy(
    x: np.ndarray, edges: np.ndarray, weight: np.ndarray
) -> GraphResult:
    """Directed source->destination dense messages, additive scatter, duplicates count."""
    _fp32(x, "node features", rank=2)
    _fp32(weight, "message projection", rank=2)
    _int64(edges, "edge indices", rank=2)
    if edges.shape[0] != 2 or x.shape[1] != weight.shape[0]:
        raise ReferenceKernelError(
            "Graph edges must be [2,edge_count] and projection dimensions must match"
        )
    if edges.size and (np.min(edges) < 0 or np.max(edges) >= x.shape[0]):
        raise ReferenceKernelError("Graph edge index out of bounds")
    projected = x @ weight
    out = np.zeros((x.shape[0], weight.shape[1]), dtype=np.float32)
    degree = np.zeros(x.shape[0], dtype=np.int64)
    source, target = edges
    np.add.at(out, target, projected[source])
    np.add.at(degree, target, 1)
    return GraphResult(out, degree)


def graph_messages_torch(x: Any, edges: Any, weight: Any) -> Any:
    import torch

    if any(
        not isinstance(v, torch.Tensor) or v.device.type != "cpu"
        for v in (x, edges, weight)
    ):
        raise ReferenceKernelError("Torch graph reference requires CPU tensors")
    graph_messages_numpy(
        x.detach().numpy(), edges.detach().numpy(), weight.detach().numpy()
    )
    messages = (x @ weight).index_select(0, edges[0])
    out = torch.zeros((x.shape[0], weight.shape[1]), dtype=torch.float32)
    out.index_add_(0, edges[1], messages)
    degree = torch.zeros(x.shape[0], dtype=torch.int64)
    degree.index_add_(0, edges[1], torch.ones(edges.shape[1], dtype=torch.int64))
    return out, degree


@dataclass(frozen=True)
class SpikeResult:
    spikes: np.ndarray
    membrane: np.ndarray
    refractory: np.ndarray


def lif_numpy(
    currents: np.ndarray,
    initial: np.ndarray,
    *,
    leak: float,
    threshold: float,
    refractory_steps: int = 0,
) -> SpikeResult:
    """Discrete leaky integrate-and-fire with reset-to-zero and absolute refractory period.

    During refractory steps the membrane holds its last value (zero after a
    spike); it does not integrate incoming current. No surrogate gradient.
    """
    _fp32(currents, "currents", rank=3)
    _fp32(initial, "initial voltage", rank=2)
    if currents.shape[1:] != initial.shape or currents.shape[0] > 2048:
        raise ReferenceKernelError("Invalid [time,batch,neurons] LIF dimensions")
    if (
        type(leak) not in (int, float)
        or type(threshold) not in (int, float)
        or not np.isfinite(leak)
        or not np.isfinite(threshold)
        or not 0 <= leak <= 1
        or threshold <= 0
        or type(refractory_steps) is not int
        or not 0 <= refractory_steps <= 1024
    ):
        raise ReferenceKernelError("Invalid LIF physical/numeric parameters")
    state = initial.copy()
    refractory = np.zeros(initial.shape, dtype=np.int64)
    spikes = np.zeros_like(currents, dtype=np.bool_)
    for t in range(currents.shape[0]):
        ready = refractory == 0
        state = np.where(ready, leak * state + currents[t], state)
        triggered = ready & (state >= threshold)
        spikes[t] = triggered
        state = np.where(triggered, np.float32(0.0), state)
        refractory = np.maximum(refractory - 1, 0)
        refractory[triggered] = refractory_steps
    if not np.all(np.isfinite(state)):
        raise ReferenceKernelError("LIF dynamics overflowed")
    return SpikeResult(spikes, state, refractory)


def lif_torch(
    currents: Any,
    initial: Any,
    *,
    leak: float,
    threshold: float,
    refractory_steps: int = 0,
) -> Any:
    import torch

    if any(
        not isinstance(v, torch.Tensor)
        or v.device.type != "cpu"
        or v.dtype != torch.float32
        for v in (currents, initial)
    ):
        raise ReferenceKernelError("Torch LIF reference requires CPU float32 tensors")
    lif_numpy(
        currents.detach().numpy(),
        initial.detach().numpy(),
        leak=leak,
        threshold=threshold,
        refractory_steps=refractory_steps,
    )
    state = initial.clone()
    refractory = torch.zeros_like(state, dtype=torch.int64)
    spikes = []
    for t in range(currents.shape[0]):
        ready = refractory == 0
        state = torch.where(ready, leak * state + currents[t], state)
        hit = ready & (state >= threshold)
        spikes.append(hit)
        state = torch.where(hit, torch.zeros_like(state), state)
        refractory = torch.clamp(refractory - 1, min=0)
        refractory[hit] = refractory_steps
    return torch.stack(spikes, dim=0), state, refractory


REFERENCE_KERNELS = frozenset(
    {
        "diagonal_selective_scan",
        "stable_topk_moe",
        "graph_additive_scatter",
        "lif_discrete",
    }
)
