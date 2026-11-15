"""Closed-world multi-fiber assembly projection, lowered to the effect IR.

One destination C receives simultaneous afferents from independent areas A and
B. Every source has its own explicit weight matrix and learning coefficient.
This models only the admitted NumpyExplicitEngine subset: float32, positive
bounded connectomes, deterministic k-WTA, no stimuli, recurrence, inhibition,
refraction, normalization, scaling, stochasticity, or recruitment.

A projection is a transaction: read both old fibers, sum their drives in the
specified A-then-B order, select one C assembly, update each fiber from its
own active source, then publish the three resulting state cells. The source
winner cells are read-only in this projection. This is NOT a general neural
assembly calculus lowering or a claim of GPU/photonic equivalence.
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
class MultiFiberConfig:
    n_a: int
    n_b: int
    n_c: int
    k_a: int
    k_b: int
    k_c: int
    beta_a: float
    beta_b: float
    ceiling: float = 20.0

    def __post_init__(self) -> None:
        for population, active in ((self.n_a, self.k_a), (self.n_b, self.k_b), (self.n_c, self.k_c)):
            if type(population) is not int or type(active) is not int or not 1 <= active <= population <= 1024:
                raise EffectIRError("Multi-fiber populations must satisfy 1 <= k <= n <= 1024")
        for value in (self.beta_a, self.beta_b, self.ceiling):
            if type(value) not in (int, float) or not isfinite(value):
                raise EffectIRError("Multi-fiber learning coefficients must be finite numbers")
        if self.beta_a < 0 or self.beta_b < 0 or self.ceiling <= 0:
            raise EffectIRError("Multi-fiber coefficients must be nonnegative with a positive ceiling")


def multifiber_program(config: MultiFiberConfig, *, steps: int = 1) -> EffectProgram:
    """Lower fixed A+B->C projection with distinct per-fiber learning rates."""
    if type(steps) is not int or not 1 <= steps <= 128:
        raise EffectIRError("Multi-fiber iteration count must be in [1, 128]")
    i = Instruction
    instructions = (
        i("read_a", "read", output="source_a", cell="a.winners", token_in="e0", token_out="e1", revision=0),
        i("read_b", "read", output="source_b", cell="b.winners", token_in="e1", token_out="e2", revision=0),
        i("read_ac", "read", output="ac0", cell="ac.weights", token_in="e2", token_out="e3", revision=0),
        i("read_bc", "read", output="bc0", cell="bc.weights", token_in="e3", token_out="e4", revision=0),
        i("drive_a", "gather_sum", args=("ac0", "source_a"), output="da"),
        i("drive_b", "gather_sum", args=("bc0", "source_b"), output="db"),
        i("drive_total", "add", args=("da", "db"), output="drive"),
        i("choose_c", "topk_indices", args=("drive",), output="selected", params=(("k", config.k_c),)),
        i("learn_ac", "hebbian_update", args=("ac0", "source_a", "selected"), output="ac1", params=(("beta", config.beta_a), ("ceiling", config.ceiling))),
        i("learn_bc", "hebbian_update", args=("bc0", "source_b", "selected"), output="bc1", params=(("beta", config.beta_b), ("ceiling", config.ceiling))),
        i("write_ac", "write", args=("ac1",), cell="ac.weights", token_in="e4", token_out="e5", revision=0),
        i("write_bc", "write", args=("bc1",), cell="bc.weights", token_in="e5", token_out="e6", revision=0),
        i("publish_c", "write", args=("selected",), cell="c.winners", token_in="e6", token_out="e7", revision=0),
    )
    t = TensorType
    program = EffectProgram(
        (
            StateSlot("a.winners", t((config.k_a,), "int64")),
            StateSlot("b.winners", t((config.k_b,), "int64")),
            StateSlot("c.winners", t((config.k_c,), "int64")),
            StateSlot("ac.weights", t((config.n_a, config.n_c), "float32")),
            StateSlot("bc.weights", t((config.n_b, config.n_c), "float32")),
        ),
        (),
        (Repeat("multifiber", steps, instructions),),
    )
    validate_effect_program(program)
    return program


def execute_multifiber(
    config: MultiFiberConfig,
    initial_state: Mapping[str, np.ndarray],
    *,
    steps: int = 1,
) -> EffectExecution:
    """Validate reference admission, then evaluate without mutating inputs."""
    expected = {"a.winners", "b.winners", "c.winners", "ac.weights", "bc.weights"}
    if set(initial_state) != expected:
        raise EffectIRError("Unexpected or missing multi-fiber state cells")
    for name in ("ac.weights", "bc.weights"):
        w = np.asarray(initial_state[name])
        if w.dtype != np.float32 or not np.all(np.isfinite(w)) or np.any(w < 0) or np.any(w > config.ceiling):
            raise EffectIRError("Multi-fiber weights must be float32, finite, nonnegative, <= ceiling")
    for name, n in (("a.winners", config.n_a), ("b.winners", config.n_b), ("c.winners", config.n_c)):
        ix = np.asarray(initial_state[name])
        if (ix.dtype != np.int64 or ix.ndim != 1 or np.any(ix < 0)
            or np.any(ix >= n) or np.unique(ix).size != ix.size):
            raise EffectIRError("Multi-fiber source/target winners must be unique, in-bounds int64 neuron IDs")
    return execute_effect_program(multifiber_program(config, steps=steps), initial_state)
