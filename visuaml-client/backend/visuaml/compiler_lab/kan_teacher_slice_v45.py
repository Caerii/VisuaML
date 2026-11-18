"""v45 data-derived KAN surrogate on a declared one-dimensional teacher slice.

This module fits edge coefficients by deterministic least squares to *outputs*
of a teacher function. It never imports hidden teacher features or establishes
causal internal mechanisms. Distinct train/test coordinates are enforced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kan_mechanism_ir_v45 import (
    EdgeFunction,
    KANAdmissionError,
    KANNode,
    KANProgram,
    execute_kan_vector,
)

MAX_OUTPUTS = 16


@dataclass(frozen=True)
class TeacherSliceFit:
    program: KANProgram
    family: str
    train_max_abs_logit_error: float
    heldout_max_abs_logit_error: float
    heldout_mean_abs_logit_error: float
    heldout_relative_l2_logit_error: float
    heldout_top1_agreement: float
    fit_is_functional_surrogate: bool = True
    pretrained_internal_mechanism_recovered: bool = False


def _coordinates(train: np.ndarray, test: np.ndarray, targets: np.ndarray,
                 holdout: np.ndarray) -> None:
    if (not all(isinstance(x, np.ndarray) and x.dtype == np.float64
                for x in (train, test, targets, holdout))
            or train.ndim != 1 or test.ndim != 1
            or not 12 <= len(train) <= 1024 or not 8 <= len(test) <= 1024
            or targets.ndim != 2 or holdout.ndim != 2
            or targets.shape[0] != len(train) or holdout.shape[0] != len(test)
            or targets.shape[1] != holdout.shape[1]
            or not 1 <= targets.shape[1] <= MAX_OUTPUTS):
        raise KANAdmissionError("Unadmitted training/heldout teacher geometry")
    if (not all(np.isfinite(z).all() for z in (train, test, targets, holdout))
            or not all(np.logical_and(z >= 0., z <= 1.).all() for z in (train, test))
            or len(np.unique(train)) != len(train)
            or len(np.unique(test)) != len(test)
            or np.intersect1d(train, test).size):
        raise KANAdmissionError("Teacher-slice domain, finiteness or leakage violation")


def _design(x: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Separate fixed basis matrices and their exact KAN edge parameters."""
    if family == "linear":
        return np.column_stack((np.ones_like(x), x)), np.array([], np.float64)
    if family == "polynomial":
        return np.polynomial.polynomial.polyvander(x, 4), np.array([], np.float64)
    if family == "linear_spline":
        knots = np.linspace(0., 1., 9)
        basis = np.zeros((len(x), len(knots)), dtype=np.float64)
        indices = np.clip(np.searchsorted(knots, x, side="right") - 1, 0,
                          len(knots) - 2)
        t = (x - knots[indices]) / (knots[indices + 1] - knots[indices])
        for row, (index, amount) in enumerate(zip(indices, t, strict=True)):
            basis[row, index] = 1. - amount
            basis[row, index + 1] = amount
        return basis, knots
    if family == "gaussian_rbf":
        centers = np.linspace(0., 1., 9)
        widths = np.full_like(centers, 0.16)
        basis = np.exp(-0.5 * ((x[:, None] - centers[None]) /
                               widths[None])**2)
        return basis, np.concatenate((centers, widths))
    if family == "fourier":
        frequency = np.float64(2 * np.pi)
        cols = [np.ones_like(x)]
        for harmonic in range(1, 5):
            cols.extend([
                np.cos(harmonic * frequency * x),
                np.sin(harmonic * frequency * x),
            ])
        return np.column_stack(cols), np.array([frequency])
    raise KANAdmissionError("Unknown fitted KAN family")


def fit_teacher_slice(train: np.ndarray, targets: np.ndarray, test: np.ndarray,
                      heldout: np.ndarray, family: str) -> TeacherSliceFit:
    _coordinates(train, test, targets, heldout)
    design, parameters = _design(train, family)
    # No gradient descent. Least-squares coefficients are DATA-DERIVED, not
    # analytically known a priori or causally extracted from teacher weights.
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < design.shape[1] or not np.isfinite(coefficients).all():
        raise KANAdmissionError("Basis is rank-deficient/nonfinite on training probes")
    kind = "polynomial" if family == "linear" else family
    nodes = []
    for j in range(targets.shape[1]):
        params = tuple(float(p) for p in parameters)
        coeffs = tuple(float(v) for v in coefficients[:, j])
        nodes.append(KANNode(
            f"logit_{j}", "sum",
            (EdgeFunction(f"edge_{j}", "t", kind, coeffs, params),),
        ))
    program = KANProgram(
        inputs=("t",),
        nodes=tuple(nodes),
        outputs=tuple(n.name for n in nodes),
        domain=(0., 1.),
    )
    # Deliberately evaluate the actual KAN IR, not just the matrix projection,
    # when making any claim about the fitted edge-function model.
    train_result = execute_kan_vector(program, {"t": train})
    test_result = execute_kan_vector(program, {"t": test})
    train_pred = np.column_stack([train_result[name] for name in program.outputs])
    test_pred = np.column_stack([test_result[name] for name in program.outputs])
    delta = test_pred - heldout
    denom = max(float(np.linalg.norm(heldout)), 1e-12)
    return TeacherSliceFit(
        program, family,
        float(np.max(np.abs(train_pred - targets))),
        float(np.max(np.abs(delta))),
        float(np.mean(np.abs(delta))),
        float(np.linalg.norm(delta) / denom),
        float(np.mean(
            np.argmax(test_pred, axis=1) == np.argmax(heldout, axis=1)
        )),
    )
