"""Allowlisted v10 integration demonstration for the existing VisuaML UI.

Both paths share the semantic envelope, but retain distinct legal execution
dialects. Hardware economics are explicit hypothetical inputs. No arbitrary
model uploads, untrusted code, or device connection is accepted here.
"""
from __future__ import annotations


import numpy as np
import torch
from torch import nn

from visuaml.compiler_spike import capture

from .assembly_schedule import Area, AssemblySchedule, Fiber, Project, lower_assembly_schedule
from .hybrid_execution import execute_hybrid
from .optical_dialect import OpticalProfile, lower_linear_to_optical_surrogate, run_optical_surrogate
from .schedule_sim import estimate_schedule
from .semantic_ir import canonicalize_effect_ir, canonicalize_tensor_ir


class _FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.activation = nn.ReLU()
        self.second = nn.Linear(6, 3)

    def forward(self, x):
        return self.second(self.activation(self.first(x)))


def run_v10_lab(*, seed: int = 19) -> dict:
    """Produce reproducible demonstrated execution and estimated schedule reports."""
    if type(seed) is not int or not 0 <= seed <= 100_000:
        raise ValueError("Demo seed must be a bounded integer")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _FixedMLP().eval()
    x = torch.tensor(rng.normal(size=(2, 4)).astype(np.float32))
    captured = capture(model, x)
    tensor_semantic = canonicalize_tensor_ir(captured.ir)
    labels = {n.id: ("device" if n.op == "linear" else "host") for n in captured.ir.nodes}
    executable_assignments = {n.id: labels[n.id] for n in captured.ir.nodes if n.op not in {"input", "output", "constant"}}
    hybrid, witness = execute_hybrid(captured, x, placements=executable_assignments)
    op_durations = {
        node.id: 1.0 + (node.estimated_flops or 0) / 1024.0
        for node in captured.ir.nodes if node.op not in {"input", "output", "constant"}
    }
    schedule = estimate_schedule(captured.ir, labels, op_durations, transfer_us=2.0,
                                 us_per_byte=1 / 4096, cache_transfers=True)

    optical_plan = lower_linear_to_optical_surrogate(model.first)
    optical = run_optical_surrogate(optical_plan, x.numpy(), OpticalProfile(
        attenuation_db=0.5, phase_bits=8, gaussian_field_noise=0.001,
        gaussian_readout_noise=0.001, provenance="v10-known-synthetic-surrogate"), seed=seed)

    assembly = AssemblySchedule(
        areas=(Area("A", 7, 2), Area("B", 8, 3), Area("C", 9, 2)),
        fibers=(Fiber("A", "C", .07), Fiber("B", "C", .11), Fiber("C", "A", .05)),
        projections=(Project("C", ("A", "B")), Project("A", ("C",))),
        rounds=3,
    )
    lowering = lower_assembly_schedule(assembly)
    state = {f"{a.name}.winners": np.arange(a.winners, dtype=np.int64) for a in assembly.areas}
    sizes = {a.name: a.population for a in assembly.areas}
    for f in assembly.fibers:
        state[f"{f.source}.{f.target}.weights"] = rng.integers(
            0, 4, size=(sizes[f.source], sizes[f.target])).astype(np.float32)
    result = lowering.execute(state)
    effect_semantic = canonicalize_effect_ir(lowering.program)
    return {
        "schema": "visuaml.compiler-lab.v10-integration.v1",
        "source_kind": "fixed-trusted-fixtures",
        "seed": seed,
        "semantic": {
            "tensor": {"sha256": tensor_semantic.digest(), "source_sha256": tensor_semantic.source_sha256,
                       "ops": len(tensor_semantic.regions[0].operations)},
            "effect": {"sha256": effect_semantic.digest(), "source_sha256": effect_semantic.source_sha256,
                       "ops": len(effect_semantic.regions[0].operations), "rounds": assembly.rounds},
        },
        "hybrid": {**hybrid.as_dict(), "witness": witness},
        "scheduled": schedule.as_dict(),
        "optical": optical.report(),
        "stateful": {**result.report(), "state_cells": sorted(result.final_state),
                     "effect_program_digest": lowering.program.digest()},
        "limitations": [
            "Stateful and tensor lowering now share an inspection envelope; they do not yet share a universal executable dialect.",
            "The hybrid device is a NumPy software adapter on CPU; no real hardware is connected.",
            "Schedule microseconds use explicit illustrative assumptions, not observed device measurements.",
            "The optical coherent-field surrogate does not model Ayo hardware or claim any power/latency advantage.",
            "The service executes fixed locally trusted fixtures only; arbitrary Python import is excluded.",
        ],
    }
