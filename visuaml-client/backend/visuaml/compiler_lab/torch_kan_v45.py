"""Independent differentiable PyTorch oracle for the bounded typed-KAN dialect.

This module has no training-loop dependencies and neither downloads pretrained
weights nor runs arbitrary callbacks. Only declared analytical edge basis
parameters are trainable. All numerical tensors are float64 and graph domain
checks are fail closed.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .typed_kan_v45 import MAX_ROWS, KANAdmissionError, KANProgram, validate_kan


class TorchKAN(nn.Module):
    def __init__(self, program: KANProgram, *, trainable: bool = False):
        super().__init__()
        validate_kan(program)
        self.program = program
        self.coefficients = nn.ParameterList([
            nn.Parameter(
                torch.tensor(edge.coefficients, dtype=torch.float64),
                requires_grad=trainable,
            )
            for stage in program.stages
            for node in stage
            for edge in node.edges
        ])

    @staticmethod
    def _basis(edge, coefficient: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(x).all()):
            raise KANAdmissionError("Nonfinite differentiable KAN basis input")
        if bool(torch.any((x < edge.domain[0]) | (x > edge.domain[1]))):
            raise KANAdmissionError("Differentiable basis input outside declared domain")
        if edge.family == "power":
            value = torch.zeros_like(x)
            for coeff in reversed(coefficient.unbind()):
                value = value * x + coeff
        elif edge.family == "chebyshev":
            u = (2 * x - (edge.domain[0] + edge.domain[1])) / (
                edge.domain[1] - edge.domain[0]
            )
            t0, t1 = torch.ones_like(u), u
            value = coefficient[0] * t0
            if coefficient.numel() > 1:
                value = value + coefficient[1] * t1
            for coeff in coefficient[2:]:
                t0, t1 = t1, 2 * u * t1 - t0
                value = value + coeff * t1
        elif edge.family == "fourier":
            theta = (2 * math.pi / edge.scale) * (x - edge.domain[0])
            value = coefficient[0] * torch.ones_like(x)
            for i in range(1, (coefficient.numel() + 1) // 2):
                value = (value + coefficient[2*i-1] * torch.cos(i * theta)
                         + coefficient[2*i] * torch.sin(i * theta))
        elif edge.family == "gaussian":
            value = torch.zeros_like(x)
            for coefficient_value, center in zip(
                coefficient.unbind(), edge.knots, strict=True
            ):
                value = value + coefficient_value * torch.exp(
                    -((x - center) / edge.scale) ** 2
                )
        else:
            # One-dimensional degree-one B-spline; interpolation basis is
            # differentiable w.r.t. coefficients inside each fixed knot cell.
            knots = torch.tensor(edge.knots, dtype=x.dtype, device=x.device)
            indices = torch.searchsorted(knots, x.contiguous(), right=True) - 1
            indices = indices.clamp(0, len(edge.knots) - 2)
            left, right = knots[indices], knots[indices + 1]
            t = (x - left) / (right - left)
            value = (1 - t) * coefficient[indices] + t * coefficient[indices + 1]
        if not bool(torch.isfinite(value).all()) or bool(torch.any(value.abs() > 1e12)):
            raise KANAdmissionError("Differentiable KAN basis overflow")
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float64
                or x.ndim != 2 or x.shape[1] != self.program.input_dim
                or not 1 <= x.shape[0] <= MAX_ROWS
                or not bool(torch.isfinite(x).all())):
            raise KANAdmissionError("Expected finite float64 [batch,features]")
        prev = x
        p = 0
        for stage in self.program.stages:
            nodes = []
            for node in stage:
                parts = []
                for edge in node.edges:
                    parts.append(self._basis(
                        edge, self.coefficients[p], prev[:, edge.source]
                    ))
                    p += 1
                if node.aggregation == "sum":
                    result = torch.stack(parts, dim=-1).sum(dim=-1) + node.bias
                else:
                    result = torch.stack(parts, dim=-1).prod(dim=-1) + node.bias
                nodes.append(result)
            prev = torch.stack(nodes, dim=-1)
            if not bool(torch.isfinite(prev).all()) or bool(torch.any(prev.abs() > 1e12)):
                raise KANAdmissionError("Differentiable KAN composition overflow")
        return prev
