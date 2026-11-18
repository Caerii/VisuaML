"""v54 gate-decoded frame events are authoritative during WASM execution."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.neural_control_routing_v52 import execute_routed_control
from visuaml.compiler_lab.neural_event_executor_v54 import (
    execute_event_routed_control,
)
from visuaml.compiler_lab.neural_frame_events_v54 import (
    EventWeights,
    fixed_event_weights,
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
def test_v54_gate_events_drive_all_six_real_programs(name):
    program, samples = programs()[name]
    for arg in samples:
        actual = execute_event_routed_control(program, arg, fuel=1500)
        v53 = execute_frame_routed_control(program, arg, fuel=1500)
        v52 = execute_routed_control(program, arg, fuel=1500)
        v51 = execute_control(program, arg, fuel=1500)
        assert actual.event_selection_gate_driven
        assert actual.result_i32 == v51.value == v52.result_i32 == v53.result_i32
        assert actual.steps == v51.transitions
        for row, old, control, source in zip(
            actual.trace, v53.trace, v52.trace, v51.trace, strict=True
        ):
            for field in (
                "step", "pc", "opcode", "next_pc", "stack_i32",
                "locals_i32", "control_labels", "halted",
            ):
                assert row[field] == old[field] == control[field] == source[field], (
                    name, arg, field
                )


def test_original_wasm_executed_by_independent_wasmtime():
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
            assert execute_event_routed_control(
                program, value, fuel=1500
            ).result_i32 == run(store, value), (name, value)


def test_event_weight_ablation_alters_execution_or_fails_closed():
    from visuaml.compiler_lab.neural_frame_events_v54 import EventCircuitError

    fixed = fixed_event_weights()
    broken = fixed.conditional_gains.copy()
    broken[2] = 0.
    # br_if no longer branches; changed outcome or explicit trap is required.
    baseline = execute_event_routed_control(branch_block(), 4)
    try:
        altered = execute_event_routed_control(
            branch_block(), 4, event_fixed=replace(
                fixed, conditional_gains=broken
            )
        )
    except (EventCircuitError, RuntimeError):
        return
    assert altered.result_i32 != baseline.result_i32


def test_reject_noncanonical_event_matrix():
    from visuaml.compiler_lab.neural_frame_events_v54 import EventCircuitError

    fixed = fixed_event_weights()
    default = fixed.default.copy()
    default[0, :] = 0.
    with pytest.raises(EventCircuitError):
        execute_event_routed_control(
            countdown(), 3,
            event_fixed=EventWeights(default, fixed.conditional_gains),
        )


def test_fuel_is_not_a_wasm_language_trap():
    with pytest.raises(ControlFuelExhausted):
        execute_event_routed_control(infinite(), 0, fuel=70)
