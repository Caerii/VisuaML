"""Regression and optional actual-engine parity for reciprocal two-area projections."""
from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.effect_ir import EffectIRError, assert_effect_trace_preserved, validate_effect_program
from visuaml.compiler_lab.multi_area_bridge import (
    ReciprocalConfig,
    execute_reciprocal,
    reciprocal_program,
)


def example(seed: int = 47):
    rng = np.random.default_rng(seed)
    config = ReciprocalConfig(7, 5, 3, 2, 0.125, 0.25)
    initial = {
        "a.winners": np.array([0, 2, 5], dtype=np.int64),
        "b.winners": np.array([1, 3], dtype=np.int64),
        "ab.weights": rng.integers(0, 4, (7, 5)).astype(np.float32),
        "ba.weights": rng.integers(0, 4, (5, 7)).astype(np.float32),
    }
    return config, initial


def independent_step(config, initial):
    """Independent scalar-loop oracle, not using the effect interpreter."""
    state = {k: v.copy() for k, v in initial.items()}
    def project(source, wname, target, k, beta):
        weights = state[wname]
        scores = np.zeros(weights.shape[1], dtype=np.float32)
        for i in state[source]:
            scores += weights[int(i)]
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        selected = np.array(sorted(ranked[:k]), dtype=np.int64)
        for i in state[source]:
            for j in selected:
                weights[i, j] = min(float(config.ceiling), float(np.float32(weights[i, j] * np.float32(1 + beta))))
        state[target] = selected
    project("a.winners", "ab.weights", "b.winners", config.k_b, config.beta_b)
    project("b.winners", "ba.weights", "a.winners", config.k_a, config.beta_a)
    return state


@pytest.mark.parametrize("seed", [0, 1, 47, 99])
def test_reciprocal_five_step_scalar_parity(seed):
    cfg, state = example(seed)
    original = {k: v.copy() for k, v in state.items()}
    for _ in range(5):
        expected = independent_step(cfg, state)
        got = execute_reciprocal(cfg, state)
        for key in state:
            np.testing.assert_allclose(got.final_state[key], expected[key], atol=1e-6, rtol=0)
        state = {k: v.copy() for k,v in got.final_state.items()}
    for k in original:
        np.testing.assert_array_equal(original[k], example(seed)[1][k])
        assert not np.shares_memory(state[k], original[k])
    assert len(execute_reciprocal(cfg, original, steps=5).receipts) == 5


def test_effect_schedule_reads_b_after_write():
    cfg, _ = example()
    prog = reciprocal_program(cfg)
    accesses = [(x.op, x.cell, x.revision) for x in prog.loops[0].instructions if x.op in {"read", "write"}]
    assert ("read", "b.winners", 1) in accesses
    assert accesses.index(("write", "b.winners", 0)) < accesses.index(("read", "b.winners", 1))
    mutated = list(prog.loops[0].instructions)
    at = next(i for i,x in enumerate(mutated) if x.name == "read_b")
    mutated[at] = replace(mutated[at], revision=0)
    with pytest.raises(EffectIRError, match="revision mismatch"):
        validate_effect_program(
            replace(prog, loops=(replace(prog.loops[0], instructions=tuple(mutated)),))
        )


def test_effect_guard_rejects_dropped_learning_write():
    cfg, _ = example()
    prog = reciprocal_program(cfg)
    ops = tuple(x for x in prog.loops[0].instructions if x.name != "write_ab")
    # The effect token chain must be repaired for an otherwise structurally valid rewrite.
    patched = []
    for op in ops:
        if op.name == "write_b":
            op = replace(op, token_in="e2")
        patched.append(op)
    rewritten = replace(prog, loops=(replace(prog.loops[0], instructions=tuple(patched)),))
    with pytest.raises(EffectIRError, match="rewired"):
        assert_effect_trace_preserved(prog, rewritten)


def test_illegal_source_and_weights_fail_without_mutation():
    cfg, original = example()
    broken = {k:v.copy() for k,v in original.items()}
    broken["a.winners"][1] = broken["a.winners"][0]
    with pytest.raises(EffectIRError, match="distinct"):
        execute_reciprocal(cfg, broken)
    broken = {k:v.copy() for k,v in original.items()}
    broken["ab.weights"][0,0] = 21
    with pytest.raises(EffectIRError, match="ceiling"):
        execute_reciprocal(cfg, broken)
    for k in original:
        np.testing.assert_array_equal(original[k], example()[1][k])


@pytest.mark.skipif(importlib.util.find_spec("neural_assemblies") is None, reason="Requires pinned Caerii/assemblies to test actual NumpyExplicitEngine")
def test_pinned_explicit_engine_multi_area_parity():
    from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine

    cfg, state = example(22)
    engine = NumpyExplicitEngine(p=0.2, seed=31, w_max=cfg.ceiling)
    engine.add_area("A", cfg.n_a, cfg.k_a, cfg.beta_a)
    engine.add_area("B", cfg.n_b, cfg.k_b, cfg.beta_b)
    engine._area_conns["A"]["B"].weights[:] = state["ab.weights"]
    engine._area_conns["B"]["A"].weights[:] = state["ba.weights"]
    engine.set_winners("A", state["a.winners"])
    engine.set_winners("B", state["b.winners"])
    for _ in range(5):
        engine.project_into("B", [], ["A"])
        engine.project_into("A", [], ["B"])
        got = execute_reciprocal(cfg, state)
        np.testing.assert_array_equal(np.sort(engine.get_winners("A")), got.final_state["a.winners"])
        np.testing.assert_array_equal(np.sort(engine.get_winners("B")), got.final_state["b.winners"])
        np.testing.assert_allclose(engine._area_conns["A"]["B"].weights, got.final_state["ab.weights"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(engine._area_conns["B"]["A"].weights, got.final_state["ba.weights"], atol=1e-6, rtol=0)
        state = {k: v.copy() for k, v in got.final_state.items()}
