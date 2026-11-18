"""v55 actual tensor-operand WASM execution differential verification."""
from __future__ import annotations

from dataclasses import replace

import pytest
from visuaml.compiler_lab.neural_event_executor_v54 import execute_event_routed_control
from visuaml.compiler_lab.neural_operand_executor_v55 import (
    execute_operand_routed_control,
)
from visuaml.compiler_lab.neural_operand_storage_v55 import (
    OperandCircuitError,
    fixed_operand_weights,
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
def test_live_tensor_operands_match_v54_and_original_v51_each_step(name):
    program, inputs = programs()[name]
    for value in inputs:
        actual = execute_operand_routed_control(program, value, fuel=1500)
        older = execute_event_routed_control(program, value, fuel=1500)
        independent = execute_control(program, value, fuel=1500)
        assert actual.tensor_operand_storage_drives_execution
        assert actual.result_i32 == older.result_i32 == independent.value
        assert actual.steps == older.steps == independent.transitions
        for row, prior, oracle in zip(
            actual.trace, older.trace, independent.trace, strict=True
        ):
            for key in (
                "step", "pc", "opcode", "next_pc", "stack_i32",
                "locals_i32", "control_labels", "halted",
                "neural_frame_event",
            ):
                if key == "neural_frame_event":
                    assert row[key] == prior[key]
                else:
                    assert row[key] == prior[key] == oracle[key], (
                        name, value, key
                    )


def test_real_wasm_bytes_match_wasmtime():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for name, (program, samples) in programs().items():
        source = encode_control(program)
        wasmtime.Module.validate(engine, source)
        store = wasmtime.Store(engine)
        original = wasmtime.Instance(
            store, wasmtime.Module(engine, source), []
        ).exports(store)["run"]
        for value in samples:
            assert execute_operand_routed_control(
                program, value, fuel=1500
            ).result_i32 == original(store, value), (name, value)


def test_changing_fixed_operand_gate_fails_closed():
    weights = fixed_operand_weights()
    gains = weights.action_gain.copy()
    gains[1] = 0.
    with pytest.raises(OperandCircuitError):
        execute_operand_routed_control(
            countdown(), 4,
            operand_fixed=replace(weights, action_gain=gains),
        )


def test_nontermination_is_fuel_policy_not_wasm_trap():
    with pytest.raises(ControlFuelExhausted):
        execute_operand_routed_control(infinite(), 0, fuel=70)
