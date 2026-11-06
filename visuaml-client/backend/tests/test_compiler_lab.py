"""Validation-focused tests for portable compilation, passes and provenance."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from visuaml.compiler_spike import (
    CapturedProgram,
    UnsupportedOperation,
    capture,
    decompose_linear,
    inspect_fx,
)
from visuaml.compiler_lab.aot_compile import compile_numpy_aot
from visuaml.compiler_lab.numpy_runtime import BundleError, load_bundle, run_bundle
from visuaml.compiler_lab.analysis import graph_digest, cost_report
from visuaml.compiler_lab.backends import (
    BackendCapabilities,
    BackendRegistry,
    NumpyAOTBackend,
    default_registry,
    partition,
)
from visuaml.compiler_lab.export_frontend import capture_export
from visuaml.compiler_lab.verification import (
    VerificationFailure,
    apply_verified_passes,
    verify_outputs,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 7)
        self.b = nn.Linear(7, 3)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class SparseK(nn.Module):
    def forward(self, x, w):
        z = x @ w
        _, inds = torch.topk(z, 2, dim=-1)
        return torch.zeros_like(z).scatter(-1, inds, 1.0)


@pytest.fixture
def small():
    torch.manual_seed(137)
    x = torch.randn(5, 4)
    return capture(Tiny(), x), x


def test_aot_mlp_torch_free_run(small, tmp_path):
    program, x = small
    target = compile_numpy_aot(program)
    directory = target.write(tmp_path / "compile")
    actual = run_bundle(directory, x.numpy())
    expected = program.graph_module(x)
    assert verify_outputs(expected, actual).passed
    assert (directory / "manifest.json").is_file()
    assert (directory / "constants.npz").is_file()


def test_standalone_runtime_imports_without_torch(small, tmp_path):
    program, x = small
    directory = compile_numpy_aot(program).write(tmp_path / "compile")
    runtime = Path(__file__).resolve().parents[1] / "visuaml" / "compiler_lab"
    script = """import sys, numpy as np
assert 'torch' not in sys.modules
from numpy_runtime import run_bundle
x=np.load(sys.argv[2]); y=run_bundle(sys.argv[1],x)
assert y.shape == (5,3)
assert np.isfinite(y).all()
assert 'torch' not in sys.modules
"""
    np.save(tmp_path / "input.npy", x.numpy())
    subprocess.run(
        [sys.executable, "-c", script, str(directory), str(tmp_path / "input.npy")],
        env={**os.environ, "PYTHONPATH": str(runtime)},
        check=True,
        timeout=30,
    )


def test_aot_rejects_shape_and_dtype(small, tmp_path):
    program, x = small
    target = compile_numpy_aot(program).write(tmp_path / "compile")
    with pytest.raises(BundleError, match="requires"):
        run_bundle(target, x.numpy().astype(np.float64))
    with pytest.raises(BundleError, match="requires"):
        run_bundle(target, x[:2].numpy())


def test_aot_fails_closed_unknown_operator():
    class Sine(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    with pytest.raises(UnsupportedOperation, match="No portable NumPy semantics"):
        compile_numpy_aot(capture(Sine(), torch.randn(2, 3)))


def test_aot_hash_checks_detect_constant_corruption(small, tmp_path):
    p, x = small
    directory = compile_numpy_aot(p).write(tmp_path / "compile")
    _, old = load_bundle(directory)
    key = sorted(old)[0]
    old[key] = old[key].copy()
    old[key].flat[0] += 1
    np.savez_compressed(directory / "constants.npz", **old)
    with pytest.raises(BundleError, match="digest"):
        run_bundle(directory, x.numpy())


def test_aot_hash_checks_detect_manifest_mutation(small, tmp_path):
    p, x = small
    directory = compile_numpy_aot(p).write(tmp_path / "compile")
    fn = directory / "manifest.json"
    data = json.loads(fn.read_text())
    data["nodes"][-1]["op"] = "mul"
    fn.write_text(json.dumps(data))
    with pytest.raises(BundleError, match="Manifest integrity"):
        run_bundle(directory, x.numpy())


def test_sparse_assembly_wta_aot():
    torch.manual_seed(19)
    a, w = torch.randn(4, 6), torch.randn(6, 11)
    p = capture(SparseK(), a, w)
    result = compile_numpy_aot(p).run(a.numpy(), w.numpy())
    np.testing.assert_array_equal(p.graph_module(a, w).numpy(), result)


def test_aot_decomposed_program(small):
    p, x = small
    decomposed = decompose_linear(p, x)
    artifact = compile_numpy_aot(decomposed)
    assert sum(n["op"] == "matmul" for n in artifact.manifest["nodes"]) == 2
    assert verify_outputs(p.graph_module(x), artifact.run(x.numpy())).passed


def test_verified_pass_pipeline_witnesses(small):
    p, x = small
    two = x * 0.5 - 0.2
    result = apply_verified_passes(
        p, [(x,), (two,)], ["decompose_linear", "fuse_linear_pattern"]
    )
    assert len(result.receipts) == 2
    assert all(r.witnesses_checked == 2 and r.passed for r in result.receipts)
    assert verify_outputs(p.graph_module(x), result.captured.graph_module(x)).passed
    assert len(result.as_dict()["passes"]) == 2


def test_verified_pass_rejects_wrong_rewrite(small):
    p, x = small

    def poison(captured, *witness):
        gm = copy.deepcopy(captured.graph_module)
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.relu:
                node.target = torch.sigmoid
        gm.recompile()
        return CapturedProgram(gm, inspect_fx(gm))

    with pytest.raises(VerificationFailure, match="changed behavior"):
        apply_verified_passes(p, [(x,)], ["wrong"], registry={"wrong": poison})


def test_nested_comparison_catches_mismatch():
    a = {"r": (np.array([1.0], dtype=np.float32), np.array([2], dtype=np.int64))}
    assert verify_outputs(a, copy.deepcopy(a)).passed
    with pytest.raises(VerificationFailure, match="dtype mismatch"):
        verify_outputs(
            a, {"r": (np.array([1.0], dtype=np.float64), np.array([2], dtype=np.int64))}
        )
    with pytest.raises(VerificationFailure, match="sequence mismatch"):
        verify_outputs(a, {"r": (np.array([1.0], dtype=np.float32),)})
    assert not verify_outputs(
        a, {"r": (np.array([5.0], dtype=np.float32), np.array([2], dtype=np.int64))}
    ).passed


def test_graph_digest_changes_with_weights_and_is_stable(small):
    p, x = small
    d1 = graph_digest(p.graph_module)
    d2 = graph_digest(copy.deepcopy(p.graph_module))
    assert d1 == d2
    p.graph_module.a.weight.data[0, 0] += 0.1
    assert graph_digest(p.graph_module) != d1


def test_cost_report_contract(small):
    p, x = small
    result = cost_report(p)
    assert result.peak_live_activation_bytes > 0
    assert result.parameter_bytes > 0
    assert result.known_flops > 0
    assert result.graph_digest == graph_digest(p.graph_module)
    json.dumps(result.as_dict())


def test_backend_registry_and_capability_contracts(small):
    p, x = small
    registry = default_registry()
    assert registry.list() == ("numpy-aot",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(NumpyAOTBackend())
    out = registry.compile(p, "numpy-aot")
    assert not out.placement.unsupported
    assert verify_outputs(p.graph_module(x), out.artifact.run(x.numpy())).passed
    with pytest.raises(ValueError, match="disjoint"):
        BackendCapabilities(
            name="bad",
            revision="1",
            native_ops=frozenset({"a"}),
            fallback_ops=frozenset({"a"}),
            supported_dtypes=frozenset({"torch.float32"}),
        )


def test_partition_reports_unsupported_and_cuts(small):
    p, x = small
    capabilities = BackendCapabilities(
        "hypothetical-photonic",
        "test-v1",
        frozenset({"linear"}),
        frozenset({"relu"}),
        frozenset({"torch.float32"}),
    )
    result = partition(p.ir, capabilities)
    assert not result.unsupported
    assert result.estimated_boundary_bytes > 0
    assert any(r.assigned == "host" and r.op == "relu" for r in result.placements)
    assert result.backend == "hypothetical-photonic"


def test_torch_export_frontend_normalizes_aten():
    torch.manual_seed(7)
    x = torch.randn(3, 4)
    exported = capture_export(Tiny(), x)
    kinds = {n.op for n in exported.graph.ir.nodes}
    assert {"linear", "relu"}.issubset(kinds)
    artifact = exported.compile_numpy()
    assert verify_outputs(exported.run_reference(x), artifact.run(x.numpy())).passed
    with pytest.raises(UnsupportedOperation, match="violates"):
        exported.run_reference(x[:2])


def test_torch_export_unsupported_operator_fails_closed():
    class Weird(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    with pytest.raises(UnsupportedOperation, match="needs a lowering rule"):
        capture_export(Weird(), torch.randn(2, 3))


def test_attention_decomposition_executes_cross_backend():
    class Attention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0
            )

    torch.manual_seed(123)
    inputs = tuple(torch.randn(2, 2, 5, 8) for _ in range(3))
    p = capture(Attention(), *inputs)
    with pytest.raises(UnsupportedOperation, match="No portable NumPy semantics"):
        compile_numpy_aot(p)
    out = apply_verified_passes(p, [inputs], ["decompose_sdpa", "decompose_softmax"])
    operations = [n.op for n in out.captured.ir.nodes]
    assert "sdpa" not in operations and "softmax" not in operations
    assert {"amax", "exp", "sum", "div"}.issubset(set(operations))
    artifact = compile_numpy_aot(out.captured)
    assert verify_outputs(
        p.graph_module(*inputs), artifact.run(*(v.numpy() for v in inputs))
    ).passed


def test_attention_rejects_causal_mode():
    class Causal(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

    from visuaml.compiler_lab.passes import decompose_sdpa

    xs = (torch.randn(1, 1, 3, 4),) * 3
    with pytest.raises(UnsupportedOperation, match="unmasked"):
        decompose_sdpa(capture(Causal(), *xs), *xs)


def test_stable_softmax_extreme_values():
    class S(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    x = torch.tensor([[1000.0, 999.0, -900.0], [-400.0, -402.0, -401.0]])
    p = capture(S(), x)
    out = apply_verified_passes(p, [(x,)], ["decompose_softmax"])
    artifact = compile_numpy_aot(out.captured)
    got = artifact.run(x.numpy())
    assert np.isfinite(got).all()
    assert verify_outputs(p.graph_module(x), got).passed


def test_layernorm_and_gelu_cross_backend():
    class N(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU())

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(28)
    x = torch.randn(3, 8)
    p = capture(N(), x)
    a = compile_numpy_aot(p)
    assert verify_outputs(p.graph_module(x), a.run(x.numpy())).passed


def test_inplace_relu_rejected():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.r(x)

    x = torch.randn(2, 4)
    p = capture(Model(), x)
    with pytest.raises(UnsupportedOperation, match="In-place ReLU"):
        compile_numpy_aot(p)


def test_topk_ties_rejected_by_generic_portable_runtime():
    class TopK(nn.Module):
        def forward(self, x):
            return torch.topk(x, 2)[1]

    x = torch.tensor([[1.0, 1.0, 2.0, 0.0]])
    p = capture(TopK(), x)
    with pytest.raises(BundleError, match="contains ties"):
        compile_numpy_aot(p).run(x.numpy())


def test_assembly_bridge_lowest_id_ties():
    from visuaml.compiler_lab.assembly_bridge import (
        CanonicalAssemblySelection,
        numpy_selection_reference,
    )

    x = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    w = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    b = torch.zeros(2, 4)
    k = 2
    p = capture(CanonicalAssemblySelection(k), x, w, b)
    a = compile_numpy_aot(p)
    reference = numpy_selection_reference(x.numpy(), w.numpy(), b.numpy(), k)
    np.testing.assert_array_equal(a.run(x.numpy(), w.numpy(), b.numpy()), reference)
    np.testing.assert_array_equal(p.graph_module(x, w, b).numpy(), reference)
    # Both rows have ties. By contract the smaller neuron IDs win.
    np.testing.assert_array_equal(
        reference, np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    )


def test_assembly_bridge_many_random_connectomes():
    from visuaml.compiler_lab.assembly_bridge import (
        CanonicalAssemblySelection,
        numpy_selection_reference,
    )

    for seed in range(6):
        rng = np.random.default_rng(seed)
        states = rng.integers(0, 2, (3, 8)).astype(np.float32)
        weights = rng.integers(0, 2, (8, 20)).astype(np.float32)
        bias = rng.integers(0, 3, (3, 20)).astype(np.float32)
        xs = [torch.from_numpy(v) for v in (states, weights, bias)]
        program = capture(CanonicalAssemblySelection(5), *xs)
        actual = compile_numpy_aot(program).run(states, weights, bias)
        np.testing.assert_array_equal(
            actual, numpy_selection_reference(states, weights, bias, 5)
        )


def test_large_int64_comparison_requires_exact_match():
    a = np.array([2**60], dtype=np.int64)
    b = np.array([2**60 + 1], dtype=np.int64)
    result = verify_outputs(a, b)
    assert not result.passed


def test_runtime_revalidates_in_memory_artifact(small):
    p, x = small
    target = compile_numpy_aot(p)
    key = next(iter(target.constants))
    target.constants[key][0, 0] += 0.25
    with pytest.raises(BundleError, match="digest"):
        target.run(x.numpy())


def test_attention_demo_cli_produces_canvas_and_artifact(tmp_path):
    import importlib.util

    demo = Path(__file__).resolve().parents[1] / "examples" / "compiler_lab_demo.py"
    spec = importlib.util.spec_from_file_location("compiler_lab_demo_for_test", demo)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run("attention", tmp_path, seed=31)
    assert report["scenario"] == "attention"
    assert report["witness"]["passed"]
    assert report["canvas"]["nodes"]
    assert (tmp_path / "numpy-aot" / "manifest.json").is_file()
    assert (tmp_path / "analysis.json").is_file()


def test_second_aot_backend_torch_export_roundtrip(small, tmp_path):
    from visuaml.compiler_lab.torch_target import (
        TorchExportBackend,
        TorchExportArtifact,
    )

    p, x = small
    compiler = TorchExportBackend((x,))
    artifact = compiler.compile(p)
    location = artifact.write(tmp_path)
    loaded = TorchExportArtifact.load(location, artifact.input_specs)
    assert verify_outputs(p.graph_module(x), loaded.run(x)).passed
    with pytest.raises(UnsupportedOperation, match="fixed export contract"):
        loaded.run(x[:2])


def test_torch_export_backend_capability_wildcard(small):
    from visuaml.compiler_lab.torch_target import TorchExportBackend

    p, x = small
    registry = BackendRegistry()
    registry.register(TorchExportBackend((x,)))
    result = registry.compile(p, "torch-export")
    assert not result.placement.unsupported
    assert result.placement.backend == "torch-export"
    assert verify_outputs(p.graph_module(x), result.artifact.run(x)).passed


def test_pipeline_exposes_individual_stage_graphs(small):
    p, x = small
    r = apply_verified_passes(p, [(x,)], ["decompose_linear", "fuse_linear_pattern"])
    assert len(r.stages) == 3
    assert len(list(r.stages[1].graph_module.graph.nodes)) > len(
        list(r.stages[0].graph_module.graph.nodes)
    )
    assert len(list(r.stages[2].graph_module.graph.nodes)) < len(
        list(r.stages[1].graph_module.graph.nodes)
    )
    assert len({graph_digest(z.graph_module) for z in r.stages}) >= 2


def test_inspector_generator_is_self_contained(tmp_path):
    import importlib.util

    base = Path(__file__).resolve().parents[1] / "examples"
    ins = base / "compiler_lab_inspector.py"
    demo = base / "compiler_lab_demo.py"
    mspec = importlib.util.spec_from_file_location("compiler_lab_demo_2", demo)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    module.run("assembly", tmp_path, seed=2)
    spec = importlib.util.spec_from_file_location("inspector_generator_for_test", ins)
    visual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visual)
    target = visual.generate(tmp_path / "analysis.json", tmp_path / "inspector.html")
    content = target.read_text()
    assert "application/json" in content and "stageControls" in content
    assert "<script src=" not in content and "<link href=" not in content
    assert "visuaml.compiler-lab.report.v2" in content


def test_content_fingerprint_supports_bfloat16():
    class F(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    x = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    p = capture(F(), x)
    assert len(graph_digest(p.graph_module)) == 64
