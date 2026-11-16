"""Second compiled CPU backend for selected canonical semantic tensor operations.

TorchScript traces allowlisted numerical primitives into separate executable
graphs with concrete guards. This does not compile an entire stateful decoder,
nor claim CUDA/Triton/IREE portability or generic model checkpoint support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

_SUPPORTED = frozenset({"matmul", "add", "multiply", "silu", "rmsnorm"})
_MAX_INPUT_BYTES = 32 * 1024 * 1024


class TorchScriptBackendError(ValueError):
    """A primitive is outside the checked CPU TorchScript ABI."""


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * w


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))


_KERNELS: dict[str, Callable] = {
    "matmul": torch.matmul,
    "add": torch.add,
    "multiply": torch.mul,
    "silu": _silu,
    "rmsnorm": _rmsnorm,
}


@dataclass(frozen=True)
class KernelReport:
    operator: str
    input_shapes: tuple[tuple[int, ...], ...]
    executable: str
    graph_nodes: int
    source: str = "strict closed-world TorchScript trace of shipped primitive"


class TorchScriptPrimitives:
    """Immutable-shape executable specialization cache; strictly float32 CPU."""

    def __init__(self) -> None:
        self._cache: dict[tuple, torch.jit.ScriptFunction] = {}

    def run(self, operator: str, *values: np.ndarray) -> tuple[np.ndarray, KernelReport]:
        if operator not in _SUPPORTED:
            raise TorchScriptBackendError("No registered TorchScript primitive")
        expected_arity = 1 if operator == "silu" else 2
        if len(values) != expected_arity:
            raise TorchScriptBackendError("Wrong operand arity")
        if any(
            not isinstance(a, np.ndarray) or a.dtype != np.float32
            or a.ndim > 6 or a.size == 0 or not np.isfinite(a).all()
            for a in values
        ):
            raise TorchScriptBackendError("Inputs must be finite float32 tensors")
        if sum(a.nbytes for a in values) > _MAX_INPUT_BYTES:
            raise TorchScriptBackendError("Input byte budget exceeded")
        if operator == "rmsnorm" and (
            values[1].ndim != 1 or values[1].shape[0] != values[0].shape[-1]
        ):
            raise TorchScriptBackendError("Invalid RMSNorm vector dimensions")
        if operator == "matmul" and (
            values[0].ndim != 3 or values[1].ndim != 2
            or values[0].shape[-1] != values[1].shape[0]
        ):
            raise TorchScriptBackendError("Expected a batched projection and 2D weight")
        if operator in ("add", "multiply") and values[0].shape != values[1].shape:
            raise TorchScriptBackendError("Implicit broadcasting is not admitted")
        key = (operator, tuple(a.shape for a in values))
        tensors = tuple(torch.from_numpy(a.copy()) for a in values)
        if key not in self._cache:
            if len(self._cache) >= 64:
                raise TorchScriptBackendError("Too many concrete kernel specializations")
            try:
                # Trace only the statically registered function; user functions
                # and Python source are not accepted anywhere in this API.
                graph = torch.jit.trace(_KERNELS[operator], tensors, check_trace=True)
            except (RuntimeError, ValueError, TypeError) as exc:
                raise TorchScriptBackendError("TorchScript trace failed") from exc
            self._cache[key] = graph
        try:
            result = self._cache[key](*tensors).detach().cpu().numpy().copy()
        except (RuntimeError, ValueError) as exc:
            raise TorchScriptBackendError("Compiled primitive execution failed") from exc
        if result.dtype != np.float32 or not np.isfinite(result).all() or result.nbytes > _MAX_INPUT_BYTES:
            raise TorchScriptBackendError("Invalid compiled primitive result")
        nodes = len(tuple(self._cache[key].graph.nodes()))
        return result, KernelReport(
            operator, tuple(tuple(a.shape) for a in values),
            "TorchScript compiled CPU graph", nodes,
        )

    @property
    def specializations(self) -> int:
        return len(self._cache)
