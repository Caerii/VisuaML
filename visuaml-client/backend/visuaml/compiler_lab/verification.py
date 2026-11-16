"""Fail-closed numeric witness checks and auditable graph pass records.

Witnesses demonstrate behavior on tested inputs ONLY; they are never proofs
of semantic equivalence for all possible inputs or floating point environments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from visuaml.compiler_spike import (
    CapturedProgram,
    decompose_linear,
    fuse_linear_pattern,
)

from .analysis import graph_digest, validate_graph
from .passes import decompose_sdpa, decompose_softmax


class VerificationFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class LeafCheck:
    path: str
    shape: tuple[int, ...]
    dtype: str
    max_abs_error: float
    max_rel_error: float
    passed: bool


@dataclass(frozen=True)
class WitnessReport:
    passed: bool
    leaves: tuple[LeafCheck, ...]
    atol: float
    rtol: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def verify_outputs(
    expected: Any,
    actual: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    check_dtype: bool = True,
) -> WitnessReport:
    leaves: list[LeafCheck] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) or isinstance(b, dict):
            if (
                not isinstance(a, dict)
                or not isinstance(b, dict)
                or a.keys() != b.keys()
            ):
                raise VerificationFailure(f"Output dictionary mismatch at {path}")
            for key in sorted(a):
                walk(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if (
                not isinstance(a, (list, tuple))
                or not isinstance(b, (list, tuple))
                or len(a) != len(b)
            ):
                raise VerificationFailure(f"Output sequence mismatch at {path}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
            return
        aa, bb = _as_np(a), _as_np(b)
        if aa.shape != bb.shape:
            raise VerificationFailure(
                f"Output shape mismatch at {path}: {aa.shape} != {bb.shape}"
            )
        if check_dtype and aa.dtype != bb.dtype:
            raise VerificationFailure(
                f"Output dtype mismatch at {path}: {aa.dtype} != {bb.dtype}"
            )
        try:
            finite = bool(np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)))
            if aa.dtype.kind in "biu":
                # Converting int64 to float64 before comparison can hide 1-unit
                # errors above 2**53. Integer and Boolean outputs are exact.
                good = finite and bool(np.array_equal(aa, bb))
                max_abs = 0.0 if good else float("inf")
                max_rel = 0.0 if good else float("inf")
            else:
                abs_err = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
                denom = np.maximum(np.abs(aa.astype(np.float64)), 1e-12)
                rel_err = abs_err / denom
                good = finite and bool(
                    np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=False)
                )
                max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
                max_rel = float(np.max(rel_err)) if rel_err.size else 0.0
        except (TypeError, ValueError) as exc:
            raise VerificationFailure(f"Non-numeric result at {path}: {exc}") from exc
        leaves.append(
            LeafCheck(path, tuple(aa.shape), str(aa.dtype), max_abs, max_rel, good)
        )

    walk(expected, actual, "$output")
    return WitnessReport(
        bool(leaves) and all(x.passed for x in leaves), tuple(leaves), atol, rtol
    )


@dataclass(frozen=True)
class PassReceipt:
    pass_name: str
    before_sha256: str
    after_sha256: str
    before_nodes: int
    after_nodes: int
    witnesses_checked: int
    max_abs_error: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    captured: CapturedProgram
    receipts: tuple[PassReceipt, ...]
    stages: tuple[CapturedProgram, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_graph_sha256": graph_digest(self.captured.graph_module),
            "passes": [p.as_dict() for p in self.receipts],
        }


PASSES: dict[str, Callable[..., CapturedProgram]] = {
    "decompose_linear": decompose_linear,
    "fuse_linear_pattern": fuse_linear_pattern,
    "decompose_softmax": decompose_softmax,
    "decompose_sdpa": decompose_sdpa,
}


def apply_verified_passes(
    program: CapturedProgram,
    witness_sets: Sequence[tuple[torch.Tensor, ...]],
    pass_names: Sequence[str],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    registry: dict[str, Callable[..., CapturedProgram]] | None = None,
) -> PipelineResult:
    """Run selected passes, checking every witness after every pass.

    The new graph is not returned if any validation fails, so callers cannot
    silently promote a failed transform into a compiled artifact.
    """
    if not witness_sets:
        raise ValueError("At least one concrete witness set is required")
    current = program
    validate_graph(current.graph_module)
    receipts: list[PassReceipt] = []
    stages: list[CapturedProgram] = [program]
    transforms = PASSES if registry is None else registry
    for name in pass_names:
        if name not in transforms:
            raise KeyError(f"Unknown registered pass: {name}")
        before = graph_digest(current.graph_module)
        with torch.no_grad():
            candidate = transforms[name](current, *witness_sets[0])
        validate_graph(candidate.graph_module)
        after = graph_digest(candidate.graph_module)
        if before != graph_digest(current.graph_module):
            raise VerificationFailure(f"Pass {name} mutated its input graph")
        max_error = 0.0
        for witness_idx, case in enumerate(witness_sets):
            with torch.no_grad():
                expected = current.graph_module(*case)
                observed = candidate.graph_module(*case)
            result = verify_outputs(expected, observed, atol=atol, rtol=rtol)
            max_error = max(max_error, *(leaf.max_abs_error for leaf in result.leaves))
            if not result.passed:
                raise VerificationFailure(
                    f"Pass {name} changed behavior on witness #{witness_idx}: "
                    f"max_abs_error={max_error:.6g}"
                )
        receipts.append(
            PassReceipt(
                name,
                before,
                after,
                len(tuple(current.graph_module.graph.nodes)),
                len(tuple(candidate.graph_module.graph.nodes)),
                len(witness_sets),
                max_error,
                True,
            )
        )
        current = candidate
        stages.append(current)
    return PipelineResult(current, tuple(receipts), tuple(stages))
