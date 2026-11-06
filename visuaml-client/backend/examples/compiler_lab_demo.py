#!/usr/bin/env python3
"""Produce inspectable VisuaML compiler-lab artifacts for three concrete workloads.

Run with PYTHONPATH=visuaml-client/backend python
visuaml-client/backend/examples/compiler_lab_demo.py --scenario attention --out /tmp/visuaml-attention

This runs entirely on local, deterministic fixture models; no network or
untrusted remote model import. For custom workloads, call the Python API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from visuaml.compiler_spike import capture, to_visuaml_graph, LoweringPlan, Placement
from visuaml.compiler_lab.analysis import cost_report, graph_digest
from visuaml.compiler_lab.backends import default_registry
from visuaml.compiler_lab.verification import apply_verified_passes, verify_outputs
from visuaml.compiler_lab.numpy_runtime import run_bundle
from visuaml.compiler_lab.torch_target import TorchExportBackend, TorchExportArtifact
from visuaml.compiler_lab.backends import BackendCapabilities
from visuaml.compiler_lab.placement_optimizer import HypotheticalCostModel, optimize_binary_placement


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.l2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.l2(torch.nn.functional.gelu(self.norm(self.l1(x))))


class AssemblyWinnerStep(nn.Module):
    """Sparse assembly-inspired top-k probe, NOT the full assemblies runtime."""
    def forward(self, current, connectivity):
        drive = current @ connectivity
        _, ix = torch.topk(drive, 5, dim=-1)
        return torch.zeros_like(drive).scatter(-1, ix, 1.0)


class Convolution(nn.Module):
    """Grouped spatial convolution, a useful alternative accelerator workload."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, kernel_size=(3, 2), padding=(2, 1), groups=2)

    def forward(self, x):
        return torch.relu(self.conv(x))


class Attention(nn.Module):
    """One non-causal, unmasked attention kernel; no training/dropout."""
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def run(scenario: str, out: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    if scenario == "mlp":
        model = MLP()
        inputs = (torch.randn(3, 8),)
        passes = ["decompose_linear", "fuse_linear_pattern"]
    elif scenario == "assembly":
        model = AssemblyWinnerStep()
        inputs = (torch.randn(4, 12), torch.randn(12, 32))
        passes = []
    elif scenario == "convolution":
        model = Convolution()
        inputs = (torch.randn(1, 4, 9, 8),)
        passes = []
    elif scenario == "attention":
        model = Attention()
        inputs = tuple(torch.randn(2, 2, 5, 8) for _ in range(3))
        passes = ["decompose_sdpa", "decompose_softmax"]
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    source = capture(model, *inputs)
    witnesses = [inputs, tuple(t * 0.57 + 0.13 for t in inputs)]
    pipeline = apply_verified_passes(source, witnesses, passes)
    lowered = default_registry().compile(pipeline.captured, "numpy-aot")
    out.mkdir(parents=True, exist_ok=True)
    bundle = lowered.artifact.write(out / "numpy-aot")
    candidate = run_bundle(bundle, *(t.numpy() for t in inputs))
    with torch.no_grad():
        expected = source.graph_module(*inputs)
    report = verify_outputs(expected, candidate)
    torch_backend = TorchExportBackend(inputs)
    torch_artifact = torch_backend.compile(pipeline.captured)
    torch_path = torch_artifact.write(out / "torch-export")
    roundtrip = TorchExportArtifact.load(torch_path, torch_artifact.input_specs)
    torch_report = verify_outputs(expected, roundtrip.run(*inputs))
    if not torch_report.passed:
        raise RuntimeError("Torch AOT roundtrip failed")
    if not report.passed:
        raise RuntimeError("Cross-backend witness failed")
    def graph_for_stage(stage, *, with_placement=False):
        if with_placement:
            placements = tuple(Placement(p.name, p.op, p.assigned, p.reason)
                               for p in lowered.placement.placements)
            overlay = LoweringPlan(lowered.placement.backend, placements,
                                   lowered.placement.unsupported, 0)
            return to_visuaml_graph(stage.ir, overlay)
        return to_visuaml_graph(stage.ir)

    stages = [{"name": "source" if i == 0 else passes[i-1],
               "graph_sha256": graph_digest(stage.graph_module),
               "cost": cost_report(stage).as_dict(),
               "canvas": graph_for_stage(stage, with_placement=(i == len(pipeline.stages)-1))}
              for i, stage in enumerate(pipeline.stages)]

    # Exact binary cut under declared illustrative costs; NOT a real photonic kernel.
    device_capabilities = BackendCapabilities(
        name="hypothetical-linear-accelerator", revision="v3-proposal",
        native_ops=frozenset({"linear", "matmul", "conv2d"}),
        fallback_ops=frozenset(), supported_dtypes=frozenset({"torch.float32"}),
        description="Research-only device capability hypothesis; no device code generation",
    )
    computational_ops = {node.op for node in pipeline.captured.ir.nodes
                         if node.op not in {"input", "constant", "output"}}
    host_costs = {op: (100.0 if op in device_capabilities.native_ops else 10.0)
                  for op in computational_ops}
    device_costs = {op: 2.0 for op in computational_ops if op in device_capabilities.native_ops}
    placement_sweep = []
    for transfer_overhead in (0., 2., 10., 40., 100.):
        profile = HypotheticalCostModel(
            name=f"illustrative-boundary-{transfer_overhead:g}",
            host_op_costs=host_costs, device_op_costs=device_costs,
            transfer_overhead=transfer_overhead,
            transfer_bytes_per_unit=512.0,
        )
        placement_sweep.append({"transfer_overhead": transfer_overhead,
                                "partition": optimize_binary_placement(
                                    pipeline.captured.ir, device_capabilities, profile).as_dict()})

    summary = {
        "schema": "visuaml.compiler-lab.report.v2",
        "scenario": scenario,
        "seed": seed,
        "source_sha256": graph_digest(source.graph_module),
        "final_sha256": graph_digest(pipeline.captured.graph_module),
        "source_cost": cost_report(source).as_dict(),
        "final_cost": cost_report(pipeline.captured).as_dict(),
        "passes": [r.as_dict() for r in pipeline.receipts],
        "stages": stages,
        "backend": lowered.placement.as_dict(),
        "hypothetical_partition_sweep": placement_sweep,
        "witness": report.as_dict(),
        "torch_export_roundtrip": torch_report.as_dict(),
        "canvas": graph_for_stage(pipeline.captured, with_placement=True),
        "ir": pipeline.captured.ir.as_dict(),
        "limitations": [
            "FX captures concrete tensor programs, not arbitrary Python control flow.",
            "Static FLOPs/liveness are estimates, not device performance measurements.",
            "Witness parity on sampled data is NOT a proof of universal equivalence.",
            "This NumPy AOT target is CPU software, not physical photonic hardware.",
        ],
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["mlp", "attention", "assembly", "convolution"], default="attention")
    parser.add_argument("--out", type=Path, default=Path("compiler-lab-output"))
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    report = run(args.scenario, args.out, args.seed)
    print(json.dumps({"scenario": args.scenario, "output": str(args.out),
                      "nodes": report["final_cost"]["node_count"],
                      "backend": report["backend"]["backend"],
                      "max_abs_error": max(v["max_abs_error"] for v in report["witness"]["leaves"]),
                      "passes": [p["pass_name"] for p in report["passes"]]}, indent=2))


if __name__ == "__main__":
    main()
