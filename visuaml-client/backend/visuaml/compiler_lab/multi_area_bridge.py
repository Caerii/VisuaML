"""Two-area reciprocal projection as a typed v7 effect program.

This deliberately supports one narrow regime from Caerii/assemblies:
- explicitly materialized finite float32 A->B and B->A weights;
- k-WTA with lowest-ID tie policy; no inhibition, noise, normalization,
  refraction, column scaling, recurrence within one area, or random growth;
- multiplicative Hebbian updates capped at a fixed positive weight ceiling.
It does NOT claim full neural-assemblies parity. Reference-engine integration
is tested separately when the pinned upstream package is installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .effect_ir import (
    EffectIRError,
    EffectProgram,
    Instruction,
    Repeat,
    StateSlot,
    TensorType,
    validate_effect_program,
)
from .effect_runtime import EffectExecution, execute_effect_program


@dataclass(frozen=True)
class ReciprocalConfig:
    n_a: int
    n_b: int
    k_a: int
    k_b: int
    beta_a: float
    beta_b: float
    ceiling: float = 20.0

    def __post_init__(self) -> None:
        if any(type(x) is not int for x in (self.n_a, self.n_b, self.k_a, self.k_b)):
            raise EffectIRError("Population and assembly sizes must be integers")
        if not (1 <= self.k_a <= self.n_a <= 1024 and 1 <= self.k_b <= self.n_b <= 1024):
            raise EffectIRError("Invalid population or winner dimensions")
        for field in ("beta_a", "beta_b", "ceiling"):
            value = getattr(self, field)
            if type(value) not in (float, int) or not isfinite(value):
                raise EffectIRError("Plasticity parameters must be finite numbers")
        if self.beta_a < 0 or self.beta_b < 0 or self.ceiling <= 0:
            raise EffectIRError("Invalid plasticity or ceiling")


def reciprocal_program(config: ReciprocalConfig, *, steps: int = 1) -> EffectProgram:
    """A->B projection then B->A projection, with ordered writes between them."""
    if type(steps) is not int or not 1 <= steps <= 128:
        raise EffectIRError("Reciprocal iteration count must be in [1, 128]")
    op = Instruction
    instructions = (
        op("read_a", "read", output="a0", cell="a.winners", token_in="e0", token_out="e1", revision=0),
        op("read_ab", "read", output="ab0", cell="ab.weights", token_in="e1", token_out="e2", revision=0),
        op("b_drive", "gather_sum", args=("ab0", "a0"), output="bd"),
        op("choose_b", "topk_indices", args=("bd",), output="b1", params=(("k", config.k_b),)),
        op("learn_ab", "hebbian_update", args=("ab0", "a0", "b1"), output="ab1", params=(("beta", config.beta_b), ("ceiling", config.ceiling))),
        op("write_ab", "write", args=("ab1",), cell="ab.weights", token_in="e2", token_out="e3", revision=0),
        op("write_b", "write", args=("b1",), cell="b.winners", token_in="e3", token_out="e4", revision=0),
        op("read_b", "read", output="b_now", cell="b.winners", token_in="e4", token_out="e5", revision=1),
        op("read_ba", "read", output="ba0", cell="ba.weights", token_in="e5", token_out="e6", revision=0),
        op("a_drive", "gather_sum", args=("ba0", "b_now"), output="ad"),
        op("choose_a", "topk_indices", args=("ad",), output="a1", params=(("k", config.k_a),)),
        op("learn_ba", "hebbian_update", args=("ba0", "b_now", "a1"), output="ba1", params=(("beta", config.beta_a), ("ceiling", config.ceiling))),
        op("write_ba", "write", args=("ba1",), cell="ba.weights", token_in="e6", token_out="e7", revision=0),
        op("write_a", "write", args=("a1",), cell="a.winners", token_in="e7", token_out="e8", revision=0),
    )
    t = TensorType
    program = EffectProgram(
        (
            StateSlot("a.winners", t((config.k_a,), "int64")),
            StateSlot("b.winners", t((config.k_b,), "int64")),
            StateSlot("ab.weights", t((config.n_a, config.n_b), "float32")),
            StateSlot("ba.weights", t((config.n_b, config.n_a), "float32")),
        ),
        (),
        (Repeat("reciprocal", steps, instructions),),
    )
    validate_effect_program(program)
    return program


def execute_reciprocal(
    config: ReciprocalConfig,
    initial_state: Mapping[str, np.ndarray],
    *,
    steps: int = 1,
) -> EffectExecution:
    """Evaluate a restricted reciprocal schedule; never mutate original arrays."""
    expected = {"a.winners", "b.winners", "ab.weights", "ba.weights"}
    if set(initial_state) != expected:
        raise EffectIRError("Missing or unexpected reciprocal state slots")
    for name in ("ab.weights", "ba.weights"):
        weights = np.asarray(initial_state[name])
        if weights.dtype != np.float32 or not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > config.ceiling):
            raise EffectIRError("Reference admission requires finite nonnegative weights <= ceiling")
    for name, bound in (("a.winners", config.n_a), ("b.winners", config.n_b)):
        values = np.asarray(initial_state[name])
        if values.dtype != np.int64 or values.ndim != 1 or np.any(values < 0) or np.any(values >= bound) or len(np.unique(values)) != len(values):
            raise EffectIRError("Reference admission requires distinct valid neuron IDs")
    return execute_effect_program(reciprocal_program(config, steps=steps), initial_state)
