"""v56 actual gate-selected ALU execution against legacy VMs and Wasmtime."""
from __future__ import annotations

from dataclasses import replace

import pytest
from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_alu_v56 import (
    NeuralALUError,
    construct_alu_weights,
)
from visuaml.compiler_lab.neural_operand_executor_v55 import (
    execute_operand_routed_control,
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
def test_six_original_wasm_families_match_v51_v55_per_instruction(name):
    program, inputs = programs()[name]
    for argument in inputs:
        candidate = execute_alu_routed_control(program, argument, fuel=1500)
        previous = execute_operand_routed_control(program, argument, fuel=1500)
        oracle = execute_control(program, argument, fuel=1500)
        assert candidate.neural_alu_selection_drives_execution
        assert candidate.result_i32 == previous.result_i32 == oracle.value
        assert candidate.steps == previous.steps == oracle.transitions
        assert candidate.fixed_alu_weights_sha256 == construct_alu_weights().digest()
        for new, prior, ref in zip(
            candidate.trace, previous.trace, oracle.trace, strict=True
        ):
            for key in (
                "step", "pc", "opcode", "next_pc", "stack_i32",
                "locals_i32", "control_labels", "halted",
            ):
                assert new[key] == prior[key] == ref[key], (
                    name, argument, key
                )
            assert new["neural_frame_event"] == prior["neural_frame_event"]


def test_real_original_wasm_validates_and_executes_in_wasmtime():
    wasmtime = pytest.importorskip("wasmtime")
    engine = wasmtime.Engine()
    for name, (program, inputs) in programs().items():
        raw = encode_control(program)
        wasmtime.Module.validate(engine, raw)
        store = wasmtime.Store(engine)
        run = wasmtime.Instance(
            store, wasmtime.Module(engine, raw), []
        ).exports(store)["run"]
        for x in inputs:
            assert execute_alu_routed_control(
                program, x, fuel=1500
            ).result_i32 == run(store, x), (name, x)


def test_corrupted_alu_gate_is_used_by_real_execution():
    fixed = construct_alu_weights()
    gains = fixed.output_gains.copy()
    gains[0] = 0.
    with pytest.raises(NeuralALUError):
        execute_alu_routed_control(
            countdown(), 3,
            alu_fixed=replace(fixed, output_gains=gains),
        )


def test_fuel_bound_separate_from_original_wasm_trap():
    with pytest.raises(ControlFuelExhausted):
        execute_alu_routed_control(infinite(), 0, fuel=70)
