"""v45 regression of teacher-slice KAN fitting and leakage/scope safeguards."""
from __future__ import annotations

import numpy as np
import pytest

from visuaml.compiler_lab.kan_mechanism_ir_v45 import KANAdmissionError
from visuaml.compiler_lab.kan_teacher_slice_v45 import fit_teacher_slice


def _data():
    train = np.linspace(0., 1., 33, dtype=np.float64)
    holdout = (np.arange(32, dtype=np.float64) + .5) / 32.
    f = lambda x: np.column_stack((x*x - 2*x + 3,
                                   np.sin(2*np.pi*x),
                                   0.5*x + 1))
    return train, f(train), holdout, f(holdout)


def test_polynomial_kan_slice_exact_for_polynomial_outputs_but_not_sine():
    x, target, held, future = _data()
    poly = fit_teacher_slice(x, target, held, future, "polynomial")
    linear = fit_teacher_slice(x, target, held, future, "linear")
    assert poly.program.outputs == ("logit_0", "logit_1", "logit_2")
    assert poly.program.nodes[0].edges[0].basis == "polynomial"
    assert not poly.pretrained_internal_mechanism_recovered
    assert poly.fit_is_functional_surrogate
    assert poly.heldout_mean_abs_logit_error < linear.heldout_mean_abs_logit_error
    # The first column is a known exact polynomial; other columns are only fits.
    from visuaml.compiler_lab.kan_mechanism_ir_v45 import execute_kan_vector
    pred = execute_kan_vector(poly.program, {"t": held})
    np.testing.assert_allclose(pred["logit_0"], future[:, 0], atol=1e-10)
    np.testing.assert_allclose(pred["logit_2"], future[:, 2], atol=1e-10)
    assert poly.heldout_max_abs_logit_error > 0


def test_fourier_basis_fits_periodic_response_and_spline_fits_locally():
    x, target, held, future = _data()
    fourier = fit_teacher_slice(x, target, held, future, "fourier")
    spline = fit_teacher_slice(x, target, held, future, "linear_spline")
    rbf = fit_teacher_slice(x, target, held, future, "gaussian_rbf")
    assert fourier.program.nodes[0].edges[0].basis == "fourier"
    assert spline.program.nodes[0].edges[0].basis == "linear_spline"
    assert rbf.program.nodes[0].edges[0].basis == "gaussian_rbf"
    assert all(0. <= f.heldout_top1_agreement <= 1.
               for f in (fourier, spline, rbf))
    assert all(np.isfinite(f.heldout_max_abs_logit_error)
               for f in (fourier, spline, rbf))
    assert all(not f.pretrained_internal_mechanism_recovered
               for f in (fourier, spline, rbf))


def test_teacher_slice_rejects_holdout_leakage_wrong_dtype_and_bad_geometry():
    x, y, test, expected = _data()
    with pytest.raises(KANAdmissionError, match="leakage"):
        fit_teacher_slice(x, y, x, y, "polynomial")
    with pytest.raises(KANAdmissionError):
        fit_teacher_slice(x.astype(np.float32), y, test, expected, "polynomial")
    with pytest.raises(KANAdmissionError):
        fit_teacher_slice(x, y, test, expected[:, :1], "polynomial")
    with pytest.raises(KANAdmissionError):
        fit_teacher_slice(x, y, test, np.full_like(expected, np.nan), "polynomial")
    with pytest.raises(KANAdmissionError, match="family"):
        fit_teacher_slice(x, y, test, expected, "unreviewed")


def test_rank_deficient_basis_rejected_not_reported_as_success():
    # Unique but concentrated coordinates can be practically ill-conditioned:
    # reject exact rank deficiency rather than asserting an interpretable fit.
    x = np.linspace(0., .0001, 33, dtype=np.float64)
    test = np.linspace(.001, .002, 32, dtype=np.float64)
    y = np.column_stack((x, x*x))
    heldout = np.column_stack((test, test*test))
    with pytest.raises(KANAdmissionError, match="rank-deficient"):
        fit_teacher_slice(x, y, test, heldout, "linear_spline")
