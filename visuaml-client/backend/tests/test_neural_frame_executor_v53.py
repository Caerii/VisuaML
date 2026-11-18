"""v53 frame circuit actively drives WASM control-depth and restoration."""
from __future__ import annotations

import numpy as np
import pytest
from visuaml.compiler_lab.neural_control_routing_v52 import execute_routed_control
from visuaml.compiler_lab.neural_frame_circuit_v53 import (
    FrameCircuitError,
    FrameCircuitWeights,
    fixed_frame_weights,
)
from visuaml.compiler_lab.neural_frame_executor_v53 import (
    execute_frame_routed_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
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
def test_frame_authoritative_execution_matches_v51_v52_each_step(name):
    program, samples = programs()[name]
    for argument in samples:
        active = execute_frame_routed_control(program, argument, fuel=1500)
        old = execute_routed_control(program, argument, fuel=1500)
        oracle = execute_control(program, argument, fuel=1500)
        assert active.label_depth_driven_by_constructed_gates
        assert active.stack_restoration_driven_by_constructed_gates
        assert active.fixed_frame_weights == (1., 1., 1., 1., 1.)
        assert active.result_i32 == oracle.value == old.result_i32
        assert active.steps == oracle.transitions == old.steps
        for a, b, c in zip(active.trace, oracle.trace, old.trace, strict=True):
            for key in ("step", "pc", "opcode", "next_pc", "stack_i32",
                        "locals_i32", "control_labels", "halted"):
                assert a[key] == b[key] == c[key], (
                    name, argument, key, a, b, c
                )


def test_authoritative_frame_gate_corruption_fails_closed():
    weights = fixed_frame_weights()
    changed = weights.action_weights.copy()
    changed[3] = 0.
    with pytest.raises(FrameCircuitError):
        execute_frame_routed_control(
            countdown(), 4, fuel=500, frame_fixed=FrameCircuitWeights(changed)
        )
    assert np.all(weights.action_weights == 1.)


def test_authoritative_executor_original_wasm_parity():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for name, (program, samples) in programs().items():
        wasm = encode_control(program)
        wasmtime.Module.validate(engine, wasm)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(
            store, wasmtime.Module(engine, wasm), []
        ).exports(store)["run"]
        for argument in samples:
            assert execute_frame_routed_control(
                program, argument, fuel=1500
            ).result_i32 == run(store, argument), (name, argument)


def test_bounded_fuel_remains_separate_from_wasm_trap():
    with pytest.raises(ControlFuelExhausted):
        execute_frame_routed_control(infinite(), 0, fuel=70)
