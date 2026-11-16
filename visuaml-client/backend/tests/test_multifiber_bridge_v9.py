"""V9 multiple independently learned fibers feeding a common target area."""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.effect_ir import EffectIRError, assert_effect_trace_preserved
from visuaml.compiler_lab.multifiber_bridge import (
    MultiFiberConfig,
    execute_multifiber,
    multifiber_program,
)


def example(seed: int = 28):
    rng = np.random.default_rng(seed)
    config = MultiFiberConfig(8, 6, 9, 3, 2, 4, 0.08, 0.23, 20.0)
    state = {
        "a.winners": np.array([0, 3, 6], dtype=np.int64),
        "b.winners": np.array([1, 4], dtype=np.int64),
        "c.winners": np.array([0, 1, 3, 5], dtype=np.int64),
        "ac.weights": rng.integers(0, 5, (8, 9)).astype(np.float32),
        "bc.weights": rng.integers(0, 5, (6, 9)).astype(np.float32),
    }
    return config, state


def independent_step(config: MultiFiberConfig, initial: dict[str, np.ndarray]):
    """Numerical oracle: explicit scalar sums, integer tie ranking, independent writes."""
    out = {k: v.copy() for k, v in initial.items()}
    drive = np.zeros(config.n_c, dtype=np.float32)
    for src, weights in (("a.winners", "ac.weights"), ("b.winners", "bc.weights")):
        one = np.zeros(config.n_c, dtype=np.float32)
        for index in out[src]:
            one += out[weights][int(index)]
        drive += one
    winners = np.array(
        sorted(
            sorted(range(config.n_c), key=lambda j: (-float(drive[j]), j))[: config.k_c]
        ),
        dtype=np.int64,
    )
    for name, src, rate in (
        ("ac.weights", "a.winners", config.beta_a),
        ("bc.weights", "b.winners", config.beta_b),
    ):
        for i in out[src]:
            for j in winners:
                out[name][i, j] = np.float32(
                    min(
                        config.ceiling,
                        np.float32(out[name][i, j] * np.float32(1 + rate)),
                    )
                )
    out["c.winners"] = winners
    return out


@pytest.mark.parametrize("seed", [0, 1, 11, 40])
def test_five_step_independent_multifiber_reference(seed):
    config, state = example(seed)
    original = {key: value.copy() for key, value in state.items()}
    for _ in range(5):
        expected = independent_step(config, state)
        run = execute_multifiber(config, state)
        assert len(run.receipts) == 1
        for name in state:
            np.testing.assert_allclose(
                run.final_state[name], expected[name], atol=1e-6, rtol=0
            )
        state = {name: value.copy() for name, value in run.final_state.items()}
    batch = execute_multifiber(config, original, steps=5)
    assert len(batch.receipts) == 5
    for name in state:
        np.testing.assert_array_equal(batch.final_state[name], state[name])
        np.testing.assert_array_equal(original[name], example(seed)[1][name])
        assert not np.shares_memory(original[name], batch.final_state[name])


def test_fiber_rates_are_distinct_and_learning_has_both_effects():
    cfg, state = example()
    out = execute_multifiber(cfg, state).final_state
    assert not np.array_equal(out["ac.weights"], state["ac.weights"])
    assert not np.array_equal(out["bc.weights"], state["bc.weights"])
    assert np.array_equal(out["a.winners"], state["a.winners"])
    assert np.array_equal(out["b.winners"], state["b.winners"])


def test_effect_guard_rejects_removed_fiber_learning():
    config, _ = example()
    before = multifiber_program(config)
    ops = tuple(op for op in before.loops[0].instructions if op.name != "write_bc")
    repaired = tuple(
        replace(op, token_in="e5") if op.name == "publish_c" else op for op in ops
    )
    after = replace(before, loops=(replace(before.loops[0], instructions=repaired),))
    with pytest.raises(EffectIRError, match="rewired"):
        assert_effect_trace_preserved(before, after)


def test_fail_closed_on_invalid_inputs():
    cfg, state = example()
    cases = [
        ("a.winners", np.array([0, 0, 6], dtype=np.int64)),
        ("b.winners", np.array([1, 12], dtype=np.int64)),
        ("ac.weights", np.full((8, 9), 21, dtype=np.float32)),
        ("bc.weights", np.zeros((6, 9), dtype=np.float64)),
    ]
    for name, replacement in cases:
        broken = dict(state)
        broken[name] = replacement
        with pytest.raises(EffectIRError):
            execute_multifiber(cfg, broken)
    with pytest.raises(EffectIRError):
        MultiFiberConfig(8, 6, 9, 3, 2, 4, True, 0.2)
    for name, value in state.items():
        np.testing.assert_array_equal(value, example()[1][name])


def test_digests_include_learning_rates():
    cfg, _ = example()
    a = multifiber_program(cfg)
    b = multifiber_program(replace(cfg, beta_b=0.3))
    assert a.digest() != b.digest()


@pytest.mark.skipif(
    importlib.util.find_spec("neural_assemblies") is None,
    reason="Pinned Caerii/assemblies required for independent three-area engine parity",
)
def test_pinned_explicit_engine_multi_fiber_parity():
    from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine

    cfg, state = example(49)
    engine = NumpyExplicitEngine(p=0.2, seed=24, w_max=cfg.ceiling)
    engine.add_area("A", cfg.n_a, cfg.k_a, 0.05)
    engine.add_area("B", cfg.n_b, cfg.k_b, 0.05)
    engine.add_area("C", cfg.n_c, cfg.k_c, 0.1)
    engine.set_beta("C", "A", cfg.beta_a)
    engine.set_beta("C", "B", cfg.beta_b)
    engine._area_conns["A"]["C"].weights[:] = state["ac.weights"]
    engine._area_conns["B"]["C"].weights[:] = state["bc.weights"]
    engine.set_winners("A", state["a.winners"])
    engine.set_winners("B", state["b.winners"])
    engine.set_winners("C", state["c.winners"])
    for _ in range(5):
        engine.project_into("C", [], ["A", "B"])
        got = execute_multifiber(cfg, state)
        np.testing.assert_array_equal(
            np.sort(engine.get_winners("C")), got.final_state["c.winners"]
        )
        np.testing.assert_allclose(
            engine._area_conns["A"]["C"].weights,
            got.final_state["ac.weights"],
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(
            engine._area_conns["B"]["C"].weights,
            got.final_state["bc.weights"],
            atol=1e-6,
            rtol=0,
        )
        state = {name: value.copy() for name, value in got.final_state.items()}
