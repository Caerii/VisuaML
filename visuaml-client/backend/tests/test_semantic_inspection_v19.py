"""v19 safe live semantic IR inspection and failure gates."""

import pytest
from scripts.semantic_inspection_v19 import inspect


@pytest.mark.parametrize("layers", (1, 2, 4))
def test_inspector_witness(layers):
    report = inspect(layers=layers, seed=19, batch=1)
    assert report["schema"] == "visuaml.semantic-inspection.v19"
    assert report["verification"]["passed"]
    assert report["verification"]["worst_abs_error"] <= 3e-5
    assert report["memory"]["peak_logical_bytes"] > 0
    assert len(report["operations"]) > 20 * layers
    assert any(op["effect"] == "write" for op in report["operations"])


@pytest.mark.parametrize("layers", (0, -1, 5, 1.5, True))
def test_bad_depth_rejected(layers):
    with pytest.raises(ValueError):
        inspect(layers=layers)


def test_batch_two_is_admitted():
    report = inspect(layers=2, seed=5, batch=2)
    assert report["fixture"]["batch"] == 2
