"""Closed-world TorchScript lowering of a pure Llama SwiGLU feed-forward region.

The admitted region is RMSNorm -> gate/up projections -> SiLU/multiply -> down
projection. A standalone execution witness must be checked against NumPy.
The compiler does not reorder any KV state effects or claim full model AOT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class FusedFFNError(ValueError):
    """A proposed feed-forward fusion violates the numeric or shape contract."""


def _fused_ffn(
    hidden: torch.Tensor, norm: torch.Tensor, gate: torch.Tensor,
    up: torch.Tensor, down: torch.Tensor, eps: torch.Tensor,
) -> torch.Tensor:
    x = hidden / torch.sqrt(
        torch.mean(hidden * hidden, dim=-1, keepdim=True) + eps
    ) * norm
    a = torch.matmul(x, gate.T)
    b = torch.matmul(x, up.T)
    activation = a / (1.0 + torch.exp(-a))
    return torch.matmul(activation * b, down.T)


@dataclass(frozen=True)
class FusedFFNReport:
    schema: str
    specialization_count: int
    graph_nodes: int
    source: str = "allowlisted PyTorch function traced to TorchScript CPU"


class TorchScriptFusedFFN:
    """Shape-specialized fused feed-forward region, independent of model source."""

    def __init__(self) -> None:
        self._kernels: dict[tuple, torch.jit.ScriptFunction] = {}

    def run(
        self, hidden: np.ndarray, norm: np.ndarray, gate: np.ndarray,
        up: np.ndarray, down: np.ndarray, *, eps: float,
    ) -> tuple[np.ndarray, FusedFFNReport]:
        values = (hidden, norm, gate, up, down)
        if any(
            not isinstance(a, np.ndarray)
            or a.dtype != np.float32 or not np.isfinite(a).all()
            for a in values
        ):
            raise FusedFFNError("Only finite float32 feed-forward arrays are admitted")
        if hidden.ndim != 3 or norm.ndim != 1 or any(
            a.ndim != 2 for a in (gate, up, down)
        ):
            raise FusedFFNError("Invalid Llama FFN rank")
        h = hidden.shape[-1]
        inner = gate.shape[0]
        if (
            not 1 <= hidden.shape[0] <= 2
            or not 1 <= hidden.shape[1] <= 32
            or not 8 <= h <= 384
            or not 8 <= inner <= 2048
            or norm.shape != (h,)
            or gate.shape != up.shape or gate.shape != (inner, h)
            or down.shape != (h, inner)
            or sum(a.nbytes for a in values) > 32 * 1024 * 1024
            or type(eps) not in (float, int) or not 0 < eps <= 1e-3
        ):
            raise FusedFFNError("Invalid bounded SwiGLU operand geometry or epsilon")
        key = (tuple(a.shape for a in values), float(eps))
        tensors = tuple(torch.from_numpy(a.copy(order="C")) for a in values)
        inputs = (*tensors, torch.tensor(float(eps), dtype=torch.float32))
        if key not in self._kernels:
            if len(self._kernels) >= 32:
                raise FusedFFNError("Too many fused feed-forward shapes")
            self._kernels[key] = torch.jit.trace(
                _fused_ffn, inputs, check_trace=True
            )
        output = self._kernels[key](*inputs).detach().cpu().numpy().copy()
        if (
            output.shape != hidden.shape
            or output.dtype != np.float32 or not np.isfinite(output).all()
        ):
            raise FusedFFNError("Fused FFN violated the result contract")
        return output, FusedFFNReport(
            "visuaml.fused-llama-ffn.v20", len(self._kernels),
            len(tuple(self._kernels[key].graph.nodes())),
        )

    @property
    def specializations(self) -> int:
        return len(self._kernels)
