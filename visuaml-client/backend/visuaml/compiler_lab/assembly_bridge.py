"""Strictly scoped bridge to neural-assemblies' k-WTA selection semantics.

This is a tensor *subprogram*, not the full neural-assemblies Brain/Area model:
no recurrent projection schedules, connectome generation, Hebbian plasticity,
refraction updates, LRI, hashed GPU substrate or recorded research claims.

It aligns with the project's declared 'lowest-neuron-id' tie break for its
explicit/deterministic path, unlike torch.topk's unspecified tie ordering.
Use against the actual library's reference corpus before claiming full parity.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CanonicalAssemblySelection(nn.Module):
    """Dense read of one area: drive = winners @ W - optional inhibition.

    Selected winner set uses descending score, stable ties -> low column index.
    No synapses are updated and weights are assumed explicitly materialized.
    """
    def __init__(self, k: int):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def forward(self, active: torch.Tensor, weights: torch.Tensor,
                inhibition: torch.Tensor) -> torch.Tensor:
        drive = active @ weights - inhibition
        # Stable descending sort gives the lower original column index on ties.
        ix = torch.argsort(drive, dim=-1, descending=True, stable=True)[:, :self.k]
        return torch.zeros_like(drive).scatter(-1, ix, 1.0)


def numpy_selection_reference(active: np.ndarray, weights: np.ndarray,
                              inhibition: np.ndarray, k: int) -> np.ndarray:
    if active.ndim != 2 or weights.ndim != 2 or inhibition.ndim != 2:
        raise ValueError("Inputs must be [B,n_pre], [n_pre,n_post], [B,n_post]")
    if active.shape[1] != weights.shape[0] or active.shape[0] != inhibition.shape[0] or weights.shape[1] != inhibition.shape[1]:
        raise ValueError("Invalid area dimensions")
    if not (0 < k <= weights.shape[1]):
        raise ValueError("k outside neuron count")
    drive = np.matmul(active, weights) - inhibition
    chosen = np.argsort(-drive, axis=-1, kind="stable")[:, :k]
    output = np.zeros_like(drive)
    np.put_along_axis(output, chosen, 1.0, axis=-1)
    return output
