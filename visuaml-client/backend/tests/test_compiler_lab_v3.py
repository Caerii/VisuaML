"""v3: adversarial IR, exact graph cut, and convolution regression gates."""

import copy
import hashlib
import itertools
import json
import subprocess
import sys
import os

import numpy as np
import pytest
import torch
from torch import nn

from visuaml.compiler_spike import capture, TensorSpec, OpRecord, ProgramIR
from visuaml.compiler_lab.aot_compile import compile_numpy_aot
from visuaml.compiler_lab.backends import BackendCapabilities
from visuaml.compiler_lab.ir_contract import (
    IRContractError,
    ResourcePolicy,
    validate_portable_ir,
)
from visuaml.compiler_lab.numpy_runtime import BundleError, execute, load_bundle
from visuaml.compiler_lab.placement_optimizer import (
    HypotheticalCostModel,
    PlacementError,
    optimize_binary_placement,
)
from visuaml.compiler_lab.export_frontend import capture_export
from visuaml.compiler_lab.verification import verify_outputs


class ConvNet(nn.Module):
    def __init__(self, *, groups=1, stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            4,
            8,
            kernel_size=(3, 2),
            padding=(2, 1),
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return torch.relu(self.conv(x))


def _signed(mutator, artifact):
    m = copy.deepcopy(artifact.manifest)
    mutator(m)
    unsigned = {k: v for k, v in m.items() if k != "manifest_sha256"}
    m["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return m


@pytest.fixture
def sample_artifact():
    torch.manual_seed(61)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    x = torch.randn(2, 3)
    return compile_numpy_aot(capture(model, x)), x


def test_v3_manifest_topological_validation(sample_artifact):
    artifact, x = sample_artifact
    result = validate_portable_ir(artifact.manifest, artifact.constants)
    assert result.schema.endswith("v1") and len(result.calls) == 3
    assert result.declared_constant_bytes > 0
    np.testing.assert_allclose(artifact.run(x.numpy()), artifact.run(x.numpy()))


def test_v3_adversarial_forward_reference_rejected_even_when_resigned(sample_artifact):
    artifact, x = sample_artifact

    def poison(m):
        call = next(n for n in m["nodes"] if n["kind"] == "call")
        call["args"] = {"tuple": [{"ref": "future_value"}]}

    with pytest.raises(IRContractError, match="Use before definition"):
        execute(_signed(poison, artifact), artifact.constants, x.numpy())


def test_v3_adversarial_duplicate_and_unknown_opcode(sample_artifact):
    artifact, x = sample_artifact

    def duplicate(m):
        m["nodes"][1]["name"] = m["nodes"][0]["name"]

    with pytest.raises(IRContractError, match="Duplicate"):
        execute(_signed(duplicate, artifact), artifact.constants, x.numpy())

    def unknown(m):
        next(n for n in m["nodes"] if n["kind"] == "call")["op"] = "eval"

    with pytest.raises(IRContractError, match="Unsupported"):
        execute(_signed(unknown, artifact), artifact.constants, x.numpy())


def test_v3_manifest_policy_denies_excessive_budget(sample_artifact):
    artifact, x = sample_artifact
    with pytest.raises(IRContractError, match="node or input budget"):
        execute(
            artifact.manifest,
            artifact.constants,
            x.numpy(),
            policy=ResourcePolicy(max_nodes=1),
        )
    with pytest.raises(IRContractError, match="Constant memory budget"):
        execute(
            artifact.manifest,
            artifact.constants,
            x.numpy(),
            policy=ResourcePolicy(max_constant_bytes=4),
        )
    with pytest.raises(IRContractError, match="Declared input byte budget"):
        execute(
            artifact.manifest,
            artifact.constants,
            x.numpy(),
            policy=ResourcePolicy(max_input_bytes=4),
        )


def test_v3_loader_archive_and_manifest_budget(sample_artifact, tmp_path):
    artifact, x = sample_artifact
    root = artifact.write(tmp_path / "artifact")
    with pytest.raises(BundleError, match="Manifest byte budget"):
        load_bundle(root, policy=ResourcePolicy(max_manifest_bytes=1))
    with pytest.raises((BundleError, IRContractError), match="budget"):
        load_bundle(root, policy=ResourcePolicy(max_constant_bytes=1))


def test_v3_standalone_runner_carries_validator(sample_artifact, tmp_path):
    artifact, x = sample_artifact
    root = artifact.write(tmp_path / "artifact")
    runtime = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "visuaml"
        / "compiler_lab"
    )
    script = """import sys
import numpy as np
assert 'torch' not in sys.modules
from numpy_runtime import run_bundle
x=np.load(sys.argv[2]); result=run_bundle(sys.argv[1], x)
assert result.shape==(2,2)
assert 'torch' not in sys.modules
"""
    np.save(tmp_path / "input.npy", x.numpy())
    subprocess.run(
        [sys.executable, "-c", script, str(root), str(tmp_path / "input.npy")],
        env={**os.environ, "PYTHONPATH": str(runtime)},
        check=True,
        timeout=25,
    )


@pytest.mark.parametrize(
    "groups,stride,dilation,bias",
    [
        (1, 1, 1, True),
        (2, 2, 1, True),
        (4, 1, 2, False),
    ],
)
def test_v3_conv2d_fx_cross_runtime(groups, stride, dilation, bias):
    torch.manual_seed(33 + groups)
    x = torch.randn(2, 4, 9, 8)
    model = ConvNet(groups=groups, stride=stride, dilation=dilation, bias=bias)
    p = capture(model, x)
    assert any(v.op == "conv2d" and (v.estimated_flops or 0) > 0 for v in p.ir.nodes)
    art = compile_numpy_aot(p)
    diff = verify_outputs(p.graph_module(x), art.run(x.numpy()), atol=3e-5, rtol=3e-5)
    assert diff.passed, diff


def test_v3_conv2d_torch_export_frontend():
    torch.manual_seed(777)
    x = torch.randn(1, 4, 9, 8)
    model = ConvNet(groups=2, stride=2)
    exported = capture_export(model, x)
    assert "conv2d" in {v.op for v in exported.graph.ir.nodes}
    art = exported.compile_numpy()
    assert verify_outputs(
        exported.run_reference(x), art.run(x.numpy()), atol=3e-5, rtol=3e-5
    ).passed


def _tiny_ir():
    spec = TensorSpec(("1", "4"), "torch.float32", 16)
    return ProgramIR(
        (
            OpRecord("x", "input", "", (), (spec,), 0, ()),
            OpRecord("a", "matmul", "", ("x",), (spec,), 1000, ()),
            OpRecord("b", "relu", "", ("a",), (spec,), 100, ()),
            OpRecord("c", "matmul", "", ("b",), (spec,), 1000, ()),
            OpRecord("out", "output", "", ("c",), (), 0, ()),
        ),
        ("x",),
        ("c",),
    )


def _brute(ir, caps, profile):
    # Enumerate only the three computational operations.
    labels = ["a", "b", "c"]
    nodes = {n.id: n for n in ir.nodes}
    edges = [(a, n.id) for n in ir.nodes for a in n.inputs]
    costs = profile
    best = float("inf")
    for choice in itertools.product(("host", "device"), repeat=3):
        assignment = {"x": "host", "out": "host", **dict(zip(labels, choice))}
        if any(
            assignment[n] == "device" and nodes[n].op not in caps.native_ops
            for n in labels
        ):
            continue
        price = 0
        for n in labels:
            op = nodes[n].op
            flops = nodes[n].estimated_flops
            if assignment[n] == "host":
                price += (costs.host_op_costs or {}).get(
                    op, costs.host_overhead + flops / costs.host_flops_per_unit
                )
            else:
                price += (costs.device_op_costs or {}).get(
                    op, costs.device_overhead + flops / costs.device_flops_per_unit
                )
        for a, b in edges:
            if assignment[a] != assignment[b]:
                price += (
                    costs.transfer_overhead
                    + (16 if a != "out" else 0) / costs.transfer_bytes_per_unit
                )
        best = min(best, price)
    return best


def test_v3_graph_cut_matches_exhaustive_search():
    ir = _tiny_ir()
    caps = BackendCapabilities(
        "optical-hypothesis",
        "1",
        frozenset({"matmul"}),
        frozenset({"relu"}),
        frozenset({"torch.float32"}),
    )
    costs = HypotheticalCostModel(
        host_op_costs={"matmul": 100.0, "relu": 10.0},
        device_op_costs={"matmul": 2.0},
        transfer_bytes_per_unit=16.0,
        transfer_overhead=1.0,
    )
    result = optimize_binary_placement(ir, caps, costs)
    assert abs(result.estimated_total_cost - _brute(ir, caps, costs)) < 1e-7
    assert result.estimated_total_cost <= result.all_host_baseline
    assert result.estimated_total_cost <= result.unconstrained_greedy_cost + 1e-7
    assert {x.node: x.assigned for x in result.assignments}["b"] == "host"
    assert len(result.boundaries) > 0
    json.dumps(result.as_dict())


def test_v3_graph_cut_all_host_when_transfer_is_expensive():
    ir = _tiny_ir()
    caps = BackendCapabilities(
        "hypo", "1", frozenset({"matmul"}), frozenset(), frozenset({"torch.float32"})
    )
    costs = HypotheticalCostModel(
        host_op_costs={"matmul": 30, "relu": 2},
        device_op_costs={"matmul": 1},
        transfer_bytes_per_unit=16,
        transfer_overhead=1000,
    )
    result = optimize_binary_placement(ir, caps, costs)
    assert all(v.assigned == "host" for v in result.assignments)
    assert result.estimated_total_cost == result.all_host_baseline


def test_v3_graph_cut_large_explicit_cost_never_forces_unsupported_device():
    """A finite-infinity sentinel must account for explicit costs >1e12."""
    ir = _tiny_ir()
    caps = BackendCapabilities(
        name="only-matmul",
        revision="1",
        native_ops=frozenset({"matmul"}),
        fallback_ops=frozenset(),
        supported_dtypes=frozenset({"torch.float32"}),
    )
    model = HypotheticalCostModel(
        host_op_costs={"matmul": 2e13, "relu": 4e13},
        device_op_costs={"matmul": 1.0},
        transfer_overhead=0.0,
        transfer_bytes_per_unit=16.0,
    )
    got = optimize_binary_placement(ir, caps, model)
    assignments = {a.node: a.assigned for a in got.assignments}
    assert assignments["b"] == "host"
    assert got.estimated_total_cost <= got.all_host_baseline
    assert (
        abs(got.estimated_total_cost - _brute(ir, caps, model))
        <= 1e-12 * got.all_host_baseline
    )


def test_v3_graph_cut_rejects_missing_cost_and_invalid_graph():
    ir = _tiny_ir()
    caps = BackendCapabilities(
        "hypo", "1", frozenset({"matmul"}), frozenset(), frozenset({"torch.float32"})
    )
    with pytest.raises(PlacementError, match="positive"):
        HypotheticalCostModel(transfer_bytes_per_unit=0)
    with pytest.raises(PlacementError, match="size admission"):
        optimize_binary_placement(ir, caps, max_nodes=1)
    bad = ProgramIR(
        (
            OpRecord(
                "z",
                "matmul",
                "",
                ("missing",),
                (TensorSpec(("2",), "torch.float32", 8),),
                100,
                (),
            ),
        ),
        (),
        ("z",),
    )
    with pytest.raises(PlacementError, match="Forward reference"):
        optimize_binary_placement(bad, caps)
    unknown = ProgramIR(
        (
            OpRecord("x", "input", "", (), (), 0, ()),
            OpRecord(
                "z",
                "matmul",
                "",
                ("x",),
                (TensorSpec(("?",), "torch.float32", None),),
                None,
                (),
            ),
        ),
        ("x",),
        ("z",),
    )
    with pytest.raises(PlacementError, match="Unknown output bytes"):
        optimize_binary_placement(unknown, caps)


@pytest.mark.parametrize("seed", range(12))
def test_v3_randomized_graph_cut_equals_brute_force(seed):
    """Enumerate every legal labeling for small random DAGs and check min-cut."""
    rng = np.random.default_rng(seed)
    spec = TensorSpec(("8",), "torch.float32", 32)
    operations = ["matmul", "relu", "conv2d", "add"]
    nodes = [OpRecord("x", "input", "", (), (spec,), 0, ())]
    for j, op in enumerate(operations):
        candidates = [n.id for n in nodes]
        deps = tuple(
            rng.choice(
                candidates,
                size=int(rng.integers(1, min(3, len(candidates)) + 1)),
                replace=False,
            )
        )
        nodes.append(
            OpRecord(
                "z" + str(j), op, "", deps, (spec,), int(rng.integers(20, 1000)), ()
            )
        )
    nodes.append(OpRecord("out", "output", "", ("z3",), (), 0, ()))
    ir = ProgramIR(tuple(nodes), ("x",), ("z3",))
    supported = frozenset(op for op in operations if rng.random() > 0.35)
    caps = BackendCapabilities(
        "hypo", "1", supported, frozenset(), frozenset({"torch.float32"})
    )
    hostcost = {op: float(rng.integers(5, 100)) for op in operations}
    devicecost = {op: float(rng.integers(1, 45)) for op in operations}
    model = HypotheticalCostModel(
        host_op_costs=hostcost,
        device_op_costs=devicecost,
        transfer_bytes_per_unit=32.0,
        transfer_overhead=float(rng.integers(0, 8)),
    )
    solved = optimize_binary_placement(ir, caps, model)
    best = float("inf")
    for labels in itertools.product(("host", "device"), repeat=4):
        assigned = {
            "x": "host",
            "out": "host",
            **{"z" + str(j): labels[j] for j in range(4)},
        }
        if any(
            assigned["z" + str(j)] == "device" and op not in supported
            for j, op in enumerate(operations)
        ):
            continue
        unary = sum(
            (hostcost if assigned["z" + str(j)] == "host" else devicecost)[op]
            for j, op in enumerate(operations)
        )
        crossing = sum(
            model.transfer_overhead + 32 / model.transfer_bytes_per_unit
            for n in nodes
            for dep in n.inputs
            if assigned[n.id] != assigned[dep]
        )
        best = min(best, unary + crossing)
    assert abs(solved.estimated_total_cost - best) < 1e-6, (
        seed,
        solved.estimated_total_cost,
        best,
    )
