"""Explicit, reproducible one-area assembly dynamics (restricted research oracle).

No implicit connectome generation, LRI, column scaling, normalization, noise,
GPU kernels or multi-area scheduling. This is NOT full neural-assemblies parity.
Semantics derive from Caerii/assemblies' explicit-weight projection and
refraction contracts; see docs/STATEFUL_ASSEMBLY_V6.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .stateful_ir import StatefulProgram, make_assembly_step_ir, validate_stateful_ir


class AssemblyStateError(ValueError):
    """Invalid state, input, or unsupported state transition."""


def _finite_float32(array: np.ndarray, label: str, ndim: int) -> np.ndarray:
    value = np.asarray(array)
    if (
        value.ndim != ndim
        or value.dtype != np.float32
        or not np.all(np.isfinite(value))
    ):
        raise AssemblyStateError(f"{label} requires finite float32 rank-{ndim} values")
    return value.copy()


def _indices(
    value: np.ndarray, length: int, *, allow_empty: bool = False
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype.kind not in "iu" or arr.dtype.kind == "b":
        raise AssemblyStateError(
            "Winner indices require a one-dimensional integer array"
        )
    if (not allow_empty and arr.size == 0) or np.any(arr < 0) or np.any(arr >= length):
        raise AssemblyStateError("Winner indices must be nonempty and in range")
    if arr.size and np.unique(arr).size != arr.size:
        raise AssemblyStateError("Winner indices must be unique")
    return np.sort(arr.astype(np.int64, copy=True))


@dataclass(frozen=True)
class AssemblyState:
    weights: np.ndarray  # explicit n_pre x n_post
    bias: np.ndarray  # refractory/refraction n_post
    winners: np.ndarray  # sorted target neuron IDs, valid as next recurrent source
    step_index: int = 0

    def __post_init__(self) -> None:
        weights = _finite_float32(self.weights, "weights", 2)
        bias = _finite_float32(self.bias, "bias", 1)
        if (
            weights.shape[0] == 0
            or weights.shape[1] == 0
            or weights.shape[1] != bias.size
        ):
            raise AssemblyStateError("Incompatible weight and bias dimensions")
        winners = _indices(self.winners, weights.shape[1])
        if type(self.step_index) is not int or self.step_index < 0:
            raise AssemblyStateError("step_index must be a nonnegative integer")
        for value in (weights, bias, winners):
            value.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "winners", winners)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(b"visuaml.assembly-state.v1")
        for array in (self.weights, self.bias, self.winners):
            h.update(str(array.shape).encode())
            h.update(array.dtype.str.encode())
            h.update(array.tobytes(order="C"))
        h.update(str(self.step_index).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class StepConfig:
    k: int
    beta: float = 0.0
    refraction_strength: float = 0.0
    weight_ceiling: float | None = 20.0
    learning_enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k < 1:
            raise AssemblyStateError("k must be a positive integer")
        for name in ("beta", "refraction_strength"):
            v = getattr(self, name)
            if isinstance(v, bool) or not np.isfinite(v) or v < 0:
                raise AssemblyStateError(f"{name} must be finite and nonnegative")
        if self.weight_ceiling is not None and (
            isinstance(self.weight_ceiling, bool)
            or not np.isfinite(self.weight_ceiling)
            or self.weight_ceiling <= 0
        ):
            raise AssemblyStateError("weight ceiling must be positive and finite")
        if type(self.learning_enabled) is not bool:
            raise AssemblyStateError("learning_enabled must be a boolean")


@dataclass(frozen=True)
class AssemblyReceipt:
    step_index: int
    source: tuple[int, ...]
    winners: tuple[int, ...]
    raw_drive: tuple[float, ...]
    net_drive: tuple[float, ...]
    original_sha256: str
    result_sha256: str
    ir_sha256: str
    learning_applied: bool
    committed: bool

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def step_assembly(
    state: AssemblyState,
    config: StepConfig,
    *,
    source_mode: Literal["external", "recurrent"] = "recurrent",
    external_source: np.ndarray | None = None,
    commit: bool = True,
    program: StatefulProgram | None = None,
) -> tuple[AssemblyState, AssemblyReceipt]:
    """Run ONE explicit-weight step with copy-on-write and fail-closed input admission.

    Source winners are snapshotted BEFORE publishing the next target winners;
    otherwise recurrent Hebbian updates silently learn the wrong connections.
    The refraction charge uses pre-update RAW drive, and is learning-gated.
    """
    if type(commit) is not bool:
        raise AssemblyStateError("commit must be a boolean")
    ir = program if program is not None else make_assembly_step_ir(source_mode)
    validate_stateful_ir(ir)
    if ir.source_mode != source_mode:
        raise AssemblyStateError("IR and execution source modes disagree")
    if config.k > state.weights.shape[1]:
        raise AssemblyStateError("k exceeds target population")
    if source_mode == "recurrent":
        if (
            external_source is not None
            or state.weights.shape[0] != state.weights.shape[1]
        ):
            raise AssemblyStateError(
                "Recurrent mode requires square weights and no external source"
            )
        source = state.winners.copy()
    elif source_mode == "external":
        if external_source is None:
            raise AssemblyStateError("External source indices are required")
        source = _indices(external_source, state.weights.shape[0])
    else:
        raise AssemblyStateError("Unsupported source mode")
    # Do not reorder the sequence: the exact engine uses summed active rows,
    # then subtracts bias and ranks the result; learning is post-selection.
    raw = np.sum(state.weights[source], axis=0, dtype=np.float32)
    net = raw - state.bias
    if not np.isfinite(net).all():
        raise AssemblyStateError("Net drive is non-finite")
    # Canonical tie policy: highest score, lowest neuron ID on exact ties.
    selected = np.sort(np.lexsort((np.arange(net.size), -net))[: config.k])
    if not commit:
        output_state = state
    else:
        weights = state.weights.copy()
        bias = state.bias.copy()
        if config.learning_enabled:
            ix = np.ix_(source, selected)
            weights[ix] *= np.float32(1.0 + config.beta)
            if config.weight_ceiling is not None:
                np.clip(weights, 0, config.weight_ceiling, out=weights)
            if config.refraction_strength:
                # Refraction charges raw drive, not net drive and not post-Hebb drive.
                bias[selected] += raw[selected] * np.float32(config.refraction_strength)
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise AssemblyStateError("Update produced non-finite state")
        output_state = AssemblyState(weights, bias, selected, state.step_index + 1)
    receipt = AssemblyReceipt(
        state.step_index,
        tuple(int(x) for x in source),
        tuple(int(x) for x in selected),
        tuple(float(x) for x in raw),
        tuple(float(x) for x in net),
        state.digest(),
        output_state.digest(),
        ir.fingerprint(),
        bool(commit and config.learning_enabled),
        commit,
    )
    return output_state, receipt


def run_assembly_schedule(
    state: AssemblyState,
    config: StepConfig,
    steps: int,
    *,
    source_mode: Literal["external", "recurrent"] = "recurrent",
    external_source: np.ndarray | None = None,
) -> tuple[AssemblyState, tuple[AssemblyReceipt, ...]]:
    if type(steps) is not int or not 0 <= steps <= 10_000:
        raise AssemblyStateError("Steps outside supported 0..10000 range")
    receipts: list[AssemblyReceipt] = []
    for _ in range(steps):
        state, receipt = step_assembly(
            state, config, source_mode=source_mode, external_source=external_source
        )
        receipts.append(receipt)
    return state, tuple(receipts)
