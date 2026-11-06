"""Read-only route payload, validated scientific selection contract and parity bridge."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compiler_lab_service.py"


def test_compiler_lab_service_end_to_end_mlp():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--scenario", "mlp", "--seed", "21"],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    data = json.loads(completed.stdout)
    assert data["api_contract"]["schema"] == "visuaml.compiler-lab.api.v1"
    assert data["scenario"] == "mlp"
    assert len(data["stages"]) == len(data["passes"]) + 1
    assert all(stage["canvas"]["nodes"] for stage in data["stages"])
    assert data["witness"]["passed"] is True
    assert data["torch_export_roundtrip"]["passed"] is True
    assert all(
        step["passed"] and step["witnesses_checked"] > 0 for step in data["passes"]
    )
    assert len(data["hypothetical_partition_sweep"]) == 5
    for item in data["hypothetical_partition_sweep"]:
        part = item["partition"]
        assert part["estimated_total_cost"] >= 0
        assert part["all_host_baseline"] >= 0


@pytest.mark.parametrize(
    "args",
    [
        ["--scenario", "../../evil"],
        ["--scenario", "mlp", "--seed", "-1"],
        ["--scenario", "mlp", "--seed", "100001"],
    ],
)
def test_compiler_lab_rejects_unbounded_or_unknown_scenarios(args):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode != 0
    assert not completed.stdout.strip().startswith("{")


def test_compiler_lab_fixed_fixture_is_not_arbitrary_code_loader():
    from scripts.compiler_lab_service import analyze

    with pytest.raises(ValueError, match="Scenario"):
        analyze("user_models.CustomMaliciousModule")
    with pytest.raises(ValueError, match="seed"):
        analyze("mlp", True)


def test_neural_assembly_winner_masks_match_independent_contract():
    """Same deterministic rule as `neural_assemblies` NumpyExactEngine's plain k-WTA selector.

    This verifies the *winner SET*, not the ordered winner list or stateful
    plasticity/recurrent engine, and does not imply full package parity.
    """
    from visuaml.compiler_lab.assembly_bridge import numpy_selection_reference

    rng = np.random.default_rng(37)
    for n_pre, n_post, k in [(4, 11, 3), (12, 45, 7), (8, 8, 8)]:
        for _ in range(12):
            active = rng.integers(0, 2, size=(2, n_pre)).astype(np.float32)
            weights = rng.integers(-1, 4, size=(n_pre, n_post)).astype(np.float32)
            inhibition = rng.integers(0, 3, size=(2, n_post)).astype(np.float32)
            drive = active @ weights - inhibition
            selected = numpy_selection_reference(active, weights, inhibition, k)
            expected = np.zeros_like(drive)
            for row in range(drive.shape[0]):
                # Independent threshold-based formulation: all above cutoff,
                # ties filled in the lowest original column-index order.
                threshold = np.partition(drive[row], len(drive[row]) - k)[
                    len(drive[row]) - k
                ]
                above = np.flatnonzero(drive[row] > threshold)
                tied = np.flatnonzero(drive[row] == threshold)
                keep = np.r_[above, tied[: k - len(above)]]
                expected[row, keep] = 1
            np.testing.assert_array_equal(selected, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("neural_assemblies") is None,
    reason="Install the separate Caerii/assemblies package to run actual-engine parity",
)
def test_neural_assemblies_exact_engine_selection_parity_when_installed():
    from neural_assemblies.core.numpy_engine._exact import NumpyExactEngine
    from visuaml.compiler_lab.assembly_bridge import numpy_selection_reference

    rng = np.random.default_rng(11)
    for n in (5, 16, 64):
        for k in (1, min(3, n), n):
            for _ in range(8):
                drive = rng.integers(0, 4, size=n).astype(np.float32)
                # _exact_topk returns an index-sorted winner SET, not sorted by score.
                indices = NumpyExactEngine._exact_topk(drive, k)
                reference = np.zeros((1, n), dtype=np.float32)
                reference[0, indices] = 1
                selected = numpy_selection_reference(
                    np.ones((1, 1), dtype=np.float32),
                    drive.reshape(1, -1),
                    np.zeros((1, n), dtype=np.float32),
                    k,
                )
                np.testing.assert_array_equal(selected, reference)
