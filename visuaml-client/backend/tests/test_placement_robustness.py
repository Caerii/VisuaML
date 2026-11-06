"""V5: rigorous minimax-regret placement under hypothetical cost uncertainty."""

import itertools
import json
from dataclasses import replace

import pytest
from visuaml.compiler_lab.backends import BackendCapabilities
from visuaml.compiler_lab.placement_optimizer import (
    HypotheticalCostModel,
    PlacementError,
    optimize_binary_placement,
)
from visuaml.compiler_lab.placement_robustness import analyze_placement_sensitivity
from visuaml.compiler_spike import OpRecord, ProgramIR, TensorSpec


def setup():
    spec = TensorSpec(("1", "4"), "torch.float32", 16)
    ir = ProgramIR(
        (
            OpRecord("x", "input", "", (), (spec,), 0, ()),
            OpRecord("a", "matmul", "", ("x",), (spec,), 600, ()),
            OpRecord("b", "relu", "", ("a",), (spec,), 40, ()),
            OpRecord("c", "matmul", "", ("b",), (spec,), 600, ()),
            OpRecord("out", "output", "", ("c",), (), 0, ()),
        ),
        ("x",),
        ("c",),
    )
    caps = BackendCapabilities(
        "hypothetical",
        "1",
        frozenset({"matmul"}),
        frozenset({"relu"}),
        frozenset({"torch.float32"}),
    )
    profiles = [
        HypotheticalCostModel(
            name=f"transfer-{v}",
            host_op_costs={"matmul": 30, "relu": 3},
            device_op_costs={"matmul": 2},
            transfer_bytes_per_unit=16,
            transfer_overhead=v,
        )
        for v in (0.0, 1.0, 5.0, 80.0)
    ]
    return ir, caps, profiles


def test_sensitivity_aggregate_has_reproducible_witnesses():
    ir, caps, profiles = setup()
    report = analyze_placement_sensitivity(ir, caps, profiles)
    assert report.search == "exhaustive-minimax-regret"
    assert report.candidates_checked == 4  # only two legal device nodes
    assert [s.profile for s in report.scenarios] == [p.name for p in profiles]
    for sc, profile in zip(report.scenarios, profiles):
        optimum = optimize_binary_placement(ir, caps, profile)
        assert sc.optimum_cost == pytest.approx(optimum.estimated_total_cost)
        assert sc.all_host_cost == pytest.approx(optimum.all_host_baseline)
    assert report.preferred.worst_regret >= 0
    assert report.preferred.worst_cost >= report.preferred.scenario_costs[0]
    assert all(0 <= count <= len(profiles) for _, count in report.node_device_frequency)
    json.dumps(report.as_dict(), allow_nan=False)


def test_minimax_regret_agrees_with_independent_exhaustive_cost():
    ir, caps, profiles = setup()
    got = analyze_placement_sensitivity(ir, caps, profiles)
    optima = [optimize_binary_placement(ir, caps, p) for p in profiles]
    brute = float("inf")
    for use_a, use_c in itertools.product((False, True), repeat=2):
        plan = {
            "x": "host",
            "a": "device" if use_a else "host",
            "b": "host",
            "c": "device" if use_c else "host",
            "out": "host",
        }
        regrets = []
        for profile, optimum in zip(profiles, optima):
            cost = 0.0
            for a in optimum.assignments:
                if a.assigned is None:
                    raise AssertionError
                cost += a.device_cost if plan[a.node] == "device" else a.host_cost
            for node in ir.nodes:
                for dep in node.inputs:
                    if plan[dep] != plan[node.id]:
                        cost += (
                            profile.transfer_overhead
                            + 16 / profile.transfer_bytes_per_unit
                        )
            regrets.append(cost - optimum.estimated_total_cost)
        brute = min(brute, max(regrets))
    assert got.preferred.worst_regret == pytest.approx(brute)


def test_high_transfer_cost_changes_optimal_plan():
    ir, caps, profiles = setup()
    result = analyze_placement_sensitivity(ir, caps, profiles)
    assert result.scenarios[0].device_nodes
    assert result.scenarios[-1].device_nodes == ()


def test_restricted_candidate_mode_explicitly_not_exact():
    ir, caps, profiles = setup()
    result = analyze_placement_sensitivity(ir, caps, profiles, max_exact_device_nodes=1)
    assert result.search == "restricted-scenario-candidates"
    assert result.candidates_checked <= len(profiles) + 1
    assert any("not a globally optimal" in x for x in result.warnings)


def test_duplicate_and_invalid_profiles_rejected():
    ir, caps, profiles = setup()
    with pytest.raises(PlacementError, match="distinct names"):
        analyze_placement_sensitivity(ir, caps, [profiles[0], profiles[0]])
    with pytest.raises(PlacementError, match="between 2"):
        analyze_placement_sensitivity(ir, caps, profiles[:1])
    with pytest.raises(PlacementError, match="admission"):
        analyze_placement_sensitivity(ir, caps, profiles, max_exact_device_nodes=20)


def test_illegal_device_assignments_never_chosen():
    ir, caps, profiles = setup()
    result = analyze_placement_sensitivity(ir, caps, profiles)
    assert "b" not in result.preferred.device_nodes
    assert "x" not in result.preferred.device_nodes
    assert "out" not in result.preferred.device_nodes


def test_missing_tensor_bytes_fail_closed():
    ir, caps, profiles = setup()
    nodes = list(ir.nodes)
    nodes[1] = replace(nodes[1], tensors=(TensorSpec(("?",), "torch.float32", None),))
    with pytest.raises(PlacementError, match="Unknown output bytes"):
        analyze_placement_sensitivity(replace(ir, nodes=tuple(nodes)), caps, profiles)


def test_per_profile_optimum_always_zero_regret():
    ir, caps, profiles = setup()
    result = analyze_placement_sensitivity(ir, caps, profiles)
    # Existence of a separate zero-regret optimum for each scenario does not mean
    # that one assignment achieves zero regret across all of them.
    assert all(s.optimum_cost <= s.all_host_cost for s in result.scenarios)
    assert result.preferred.worst_regret > 0
