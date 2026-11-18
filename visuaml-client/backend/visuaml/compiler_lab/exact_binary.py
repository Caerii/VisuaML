"""Membership in the binary coefficient domain for admitted float64 tensors."""
from __future__ import annotations

import numpy as np


def is_exact_binary(value: np.ndarray) -> bool:
    """Call after the owning gate's shape and dtype checks."""
    if type(value) is not np.ndarray:
        # Preserve NumPy dispatch and masks for legacy array subclasses.
        return bool(np.isfinite(value).all() and np.isin(value, (0., 1.)).all())
    return bool(np.count_nonzero((value == 0.) | (value == 1.)) == value.size)
