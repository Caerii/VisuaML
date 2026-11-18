"""v53 shadow-frame replay over original v52 WASM execution traces."""
from __future__ import annotations

import numpy as np
import pytest
from visuaml.compiler_lab.neural_frame_circuit_v53 import (
    FrameCircuitError,
    FrameCircuitWeights,
    fixed_frame_weights,
)
from visuaml.compiler_lab.neural_frame_replay_v53 import replay_routed_frames
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlOperation,
    ControlProgram,
    encode_control,
    execute_control,
)


def op(name, immediate=None):
    return ControlOperation(name, immediate)


def countdown():
    return ControlProgram(1, (
        op("block"), op("loop"), op("local.get", 0),
        op("i32.eqz"), op("br_if", 1),
        op("local.get", 1), op("local.get", 0), op("i32.add"),
        op("local.set", 1), op("local.get", 0),
        op("i32.const", 1), op("i32.sub"),
        op("local.set", 0), op("br", 0),
        op("end"), op("end"), op("local.get", 1), op("end"),
    ))


def ifelse():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.lt_s"), op("if"),
        op("i32.const", 11), op("local.set", 1),
        op("else"), op("i32.const", 22),
        op("local.set", 1), op("end"),
        op("local.get", 1), op("end"),
    ))


def conditional_no_else():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.eqz"),
        op("if"), op("i32.const", 7),
        op("local.set", 1), op("end"),
        op("local.get", 1), op("end"),
    ))


def branch_block():
    return ControlProgram(1, (
        op("block"), op("local.get", 0),
        op("i32.const", 0), op("i32.gt_s"),
        op("br_if", 0),
        op("i32.const", 7), op("local.set", 1),
        op("end"), op("local.get", 1), op("end"),
    ))


def return_early():
    return ControlProgram(0, (
        op("local.get", 0), op("i32.const", 0),
        op("i32.gt_s"), op("if"),
        op("i32.const", 9), op("return"), op("end"),
        op("i32.const", -3), op("end"),
    ))


def infinite():
    return ControlProgram(0, (
        op("loop"), op("br", 0), op("end"),
        op("i32.const", 1), op("end"),
    ))


def gcd_program():
    """Self-contained canonical GCD fixture; no Wasmtime needed at import."""
    return ControlProgram(1, (
        op("local.get", 0), op("i32.eqz"),
        op("if"), op("i32.const", 12), op("return"), op("end"),
        op("local.get", 0), op("i32.const", 0), op("i32.lt_s"),
        op("if"), op("i32.const", 0), op("return"), op("end"),
        op("i32.const", 12), op("local.set", 1),
        op("block"), op("loop"),
        op("local.get", 0), op("local.get", 1),
        op("i32.sub"), op("i32.eqz"), op("br_if", 1),
        op("local.get", 0), op("local.get", 1), op("i32.gt_s"),
        op("if"),
        op("local.get", 0), op("local.get", 1),
        op("i32.sub"), op("local.set", 0),
        op("else"),
        op("local.get", 1), op("local.get", 0),
        op("i32.sub"), op("local.set", 1),
        op("end"), op("br", 0),
        op("end"), op("end"), op("local.get", 0), op("end"),
    ))


def programs():
    return {
        "countdown": (countdown(), tuple(range(10))),
        "ifelse": (ifelse(), (-7, -1, 0, 1, 7)),
        "no_else": (conditional_no_else(), (-3, -1, 0, 1, 3)),
        "br_block": (branch_block(), (-3, -1, 0, 1, 3)),
        "early_return": (return_early(), (-7, -1, 0, 1, 7)),
        "gcd_12": (gcd_program(), (-1, 0, 1, 2, 3, 5, 7, 9, 12, 17, 24)),
    }




@pytest.mark.parametrize("name", list(programs()))
def test_constructed_frame_replay_matches_v51_at_every_wasm_step(name):
    program, samples = programs()[name]
    for value in samples:
        replay = replay_routed_frames(program, value)
        reference = execute_control(program, value, fuel=1500)
        assert replay.frame_transitions == reference.transitions
        assert replay.routed.result_i32 == reference.value
        assert replay.fixed_frame_weights == (1., 1., 1., 1., 1.)
        for a, b in zip(replay.routed.trace, reference.trace, strict=True):
            for field in ("pc", "next_pc", "stack_i32", "locals_i32",
                          "control_labels", "halted"):
                assert a[field] == b[field], (name, value, field)


def test_gcd_exercises_nested_branch_restorations():
    replay = replay_routed_frames(gcd_program(), 17)
    assert replay.deepest_frame >= 2
    assert replay.branch_restorations_checked > 0


def test_all_programs_are_independently_valid_original_wasm():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for name, (program, samples) in programs().items():
        source = encode_control(program)
        wasmtime.Module.validate(engine, source)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(
            store, wasmtime.Module(engine, source), []
        ).exports(store)["run"]
        for value in samples:
            assert replay_routed_frames(
                program, value
            ).routed.result_i32 == run(store, value), (name, value)


def test_ablation_of_frame_transition_fails_closed():
    fixed = fixed_frame_weights()
    changed = fixed.action_weights.copy()
    changed[1] = 0.
    with pytest.raises(FrameCircuitError):
        replay_routed_frames(
            countdown(), 4, weights=FrameCircuitWeights(changed)
        )
    assert np.array_equal(fixed.action_weights, np.ones(5))
