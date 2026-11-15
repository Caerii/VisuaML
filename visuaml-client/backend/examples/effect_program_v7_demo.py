"""Demonstrate two connected neural areas in VisuaML's v7 effect dialect.

Run with: PYTHONPATH=backend python backend/examples/effect_program_v7_demo.py
This is a deterministic NumPy research simulator, not a photonic backend.
"""

from __future__ import annotations

import json

import numpy as np
from visuaml.compiler_lab.effect_ir import (
    EffectProgram,
    Instruction,
    Repeat,
    StateSlot,
    TensorType,
    validate_effect_program,
)
from visuaml.compiler_lab.effect_runtime import execute_effect_program


def build_two_area_program(steps: int = 3) -> EffectProgram:
    n_a, n_b, k = 5, 4, 2
    op = Instruction
    e = (
        op(
            "read_winners_a",
            "read",
            output="src_a",
            cell="a.winners",
            token_in="e0",
            token_out="e1",
            revision=0,
        ),
        op(
            "read_weights_a",
            "read",
            output="wa",
            cell="a.weights",
            token_in="e1",
            token_out="e2",
            revision=0,
        ),
        op("drive_a", "gather_sum", args=("wa", "src_a"), output="raw_a"),
        op(
            "read_bias_a",
            "read",
            output="ba",
            cell="a.bias",
            token_in="e2",
            token_out="e3",
            revision=0,
        ),
        op("net_a", "subtract", args=("raw_a", "ba"), output="net_a"),
        op(
            "select_a",
            "topk_indices",
            args=("net_a",),
            output="winner_a",
            params=(("k", k),),
        ),
        op(
            "learn_a",
            "hebbian_update",
            args=("wa", "src_a", "winner_a"),
            output="next_wa",
            params=(("beta", 0.1), ("ceiling", 20.0)),
        ),
        op(
            "inhibit_a",
            "refraction_update",
            args=("ba", "raw_a", "winner_a"),
            output="next_ba",
            params=(("strength", 0.05),),
        ),
        op(
            "write_wa",
            "write",
            args=("next_wa",),
            cell="a.weights",
            token_in="e3",
            token_out="e4",
            revision=0,
        ),
        op(
            "write_ba",
            "write",
            args=("next_ba",),
            cell="a.bias",
            token_in="e4",
            token_out="e5",
            revision=0,
        ),
        op(
            "write_win_a",
            "write",
            args=("winner_a",),
            cell="a.winners",
            token_in="e5",
            token_out="e6",
            revision=0,
        ),
        op(
            "read_weights_b",
            "read",
            output="wb",
            cell="b.weights",
            token_in="e6",
            token_out="e7",
            revision=0,
        ),
        op("drive_b", "gather_sum", args=("wb", "winner_a"), output="raw_b"),
        op(
            "read_bias_b",
            "read",
            output="bb",
            cell="b.bias",
            token_in="e7",
            token_out="e8",
            revision=0,
        ),
        op("net_b", "subtract", args=("raw_b", "bb"), output="net_b"),
        op(
            "select_b",
            "topk_indices",
            args=("net_b",),
            output="winner_b",
            params=(("k", k),),
        ),
        op(
            "learn_b",
            "hebbian_update",
            args=("wb", "winner_a", "winner_b"),
            output="next_wb",
            params=(("beta", 0.2), ("ceiling", 20.0)),
        ),
        op(
            "inhibit_b",
            "refraction_update",
            args=("bb", "raw_b", "winner_b"),
            output="next_bb",
            params=(("strength", 0.1),),
        ),
        op(
            "write_wb",
            "write",
            args=("next_wb",),
            cell="b.weights",
            token_in="e8",
            token_out="e9",
            revision=0,
        ),
        op(
            "write_bb",
            "write",
            args=("next_bb",),
            cell="b.bias",
            token_in="e9",
            token_out="e10",
            revision=0,
        ),
        op(
            "write_win_b",
            "write",
            args=("winner_b",),
            cell="b.winners",
            token_in="e10",
            token_out="e11",
            revision=0,
        ),
    )
    t = TensorType
    return EffectProgram(
        (
            StateSlot("a.weights", t((n_a, n_a), "float32")),
            StateSlot("a.bias", t((n_a,), "float32")),
            StateSlot("a.winners", t((k,), "int64")),
            StateSlot("b.weights", t((n_a, n_b), "float32")),
            StateSlot("b.bias", t((n_b,), "float32")),
            StateSlot("b.winners", t((k,), "int64")),
        ),
        (),
        (Repeat("area_a_then_b", steps, e),),
    )


def demo() -> dict:
    rng = np.random.default_rng(17)
    initial = {
        "a.weights": rng.uniform(0.2, 2.0, (5, 5)).astype(np.float32),
        "a.bias": np.zeros(5, dtype=np.float32),
        "a.winners": np.array([0, 2], dtype=np.int64),
        "b.weights": rng.uniform(0.2, 2.0, (5, 4)).astype(np.float32),
        "b.bias": np.zeros(4, dtype=np.float32),
        "b.winners": np.array([0, 2], dtype=np.int64),
    }
    program = build_two_area_program()
    admitted = validate_effect_program(program)
    result = execute_effect_program(program, initial)
    return {
        "schema": "visuaml.effect-demo.v7",
        "disclaimer": "NumPy simulation only; not Ayo photonic hardware execution",
        "admitted_operations": admitted.expanded_operations,
        "program_sha256": result.program_sha256,
        "initial_state_sha256": result.initial_sha256,
        "final_state_sha256": result.final_sha256,
        "a_winners": result.final_state["a.winners"].tolist(),
        "b_winners": result.final_state["b.winners"].tolist(),
        "receipts": [vars(r) for r in result.receipts],
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
