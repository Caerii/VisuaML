"""Numerical witness verification for effect-preserving program rewrites.

This is NOT a proof of all-input equivalence. A pass is admitted only on
specific caller-supplied states, and every state cell is checked after all
loops. Structural effect trace identity is independently required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .effect_ir import EffectIRError, EffectProgram, assert_effect_trace_preserved
from .effect_runtime import execute_effect_program


@dataclass(frozen=True)
class RewriteReceipt:
    schema: str
    before_sha256: str
    after_sha256: str
    witnesses_checked: int
    max_abs_error: float
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def verify_effect_rewrite(
    before: EffectProgram,
    after: EffectProgram,
    witnesses: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> RewriteReceipt:
    """Fail closed when a stateful rewrite changes any state on a witness."""
    assert_effect_trace_preserved(before, after)
    if not isinstance(witnesses, (list, tuple)) or not 1 <= len(witnesses) <= 64:
        raise EffectIRError("Must supply 1..64 concrete stateful rewrite witnesses")
    if any(
        type(x) not in (float, int) or not np.isfinite(x) or x < 0 for x in (atol, rtol)
    ):
        raise EffectIRError("Witness tolerances must be finite and nonnegative")
    worst = 0.0
    for i, (state, inputs) in enumerate(witnesses):
        a = execute_effect_program(before, state, inputs)
        b = execute_effect_program(after, state, inputs)
        if set(a.final_state) != set(b.final_state):
            raise EffectIRError(f"Rewrite changed state cells for witness {i}")
        for cell in a.final_state:
            x, y = a.final_state[cell], b.final_state[cell]
            if x.dtype != y.dtype or x.shape != y.shape:
                raise EffectIRError(f"Rewrite changed type of {cell} on witness {i}")
            if x.dtype.kind in "iu":
                passed = bool(np.array_equal(x, y))
                error = 0.0 if passed else float("inf")
            else:
                passed = bool(np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=False))
                error = float(
                    np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))
                )
            worst = max(worst, error)
            if not passed:
                raise EffectIRError(
                    f"Rewrite changed cell {cell} on witness {i}, max_abs_error={worst:.6g}"
                )
    return RewriteReceipt(
        "visuaml.effect-rewrite-receipt.v1",
        before.digest(),
        after.digest(),
        len(witnesses),
        worst,
        True,
    )
