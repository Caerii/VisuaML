"""Membership in the binary coefficient domain for admitted float64 tensors."""
from __future__ import annotations

import numpy as np


def is_exact_binary(value: np.ndarray) -> bool:
    """Call after the owning gate's shape and dtype checks."""
    if type(value) is not np.ndarray:
        # Preserve NumPy dispatch and masks for legacy array subclasses.
        return bool(np.isfinite(value).all() and np.isin(value, (0., 1.)).all())
    matches = (value == 0.) | (value == 1.)
    # Counting reduces small-tensor overhead. The boolean reduction remains
    # faster on the larger admission tables measured by the routing suite.
    if value.size <= 128:
        return bool(np.count_nonzero(matches) == value.size)
    return bool(matches.all())
