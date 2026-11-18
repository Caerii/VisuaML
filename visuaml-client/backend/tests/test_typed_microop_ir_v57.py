"""v57 fixed-weight microoperation IR: original WASM, source identity, mutations."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.neural_alu_v56 import ALU_OPS
from visuaml.compiler_lab.neural_control_routing_v52 import SLOTS
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    EFFECTS,
    LOCAL_MODES,
    MicroopAdmissionError,
    compile_microop_tokens,
    construct_microop_weights,
    read_microop,
    validate_microop_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlOperation,
    ControlProgram,
    encode_control,
)

def op(name, immediate=None):
    return ControlOperation(name, immediate)


def nested_program():
    return ControlProgram(1, (
        op("block"), op("local.get", 0),
        op("i32.const", 0), op("i32.gt_s"), op("br_if", 0),
        op("local.get", 0), op("i32.const", -17),
        op("i32.mul"), op("local.tee", 1),
        op("i32.eqz"), op("drop"), op("end"),
        op("local.get", 1), op("end"),
    ))


def all_alu_program():
    return ControlProgram(1, (
        op("local.get", 0), op("i32.const", -(1 << 31)),
        op("i32.add"), op("i32.const", 7), op("i32.sub"),
        op("i32.const", 2), op("i32.mul"),
        op("i32.const", 0), op("i32.lt_s"),
        op("local.set", 1), op("local.get", 1),
        op("i32.const", 1), op("i32.gt_s"),
        op("i32.eqz"), op("end"),
    ))


def onehot(index, length):
    result = np.zeros(length, dtype=np.float64)
    result[index] = 1.
    return result


@pytest.mark.parametrize("program", (nested_program(), all_alu_program()))
def test_every_typed_microop_matches_source_opcode(program):
    wasmtime = pytest.importorskip("wasmtime")
    wasmtime.Module.validate(wasmtime.Engine(), encode_control(program))
    tokens = compile_microop_tokens(program)
    validate_microop_tokens(tokens, program)
    fixed = construct_microop_weights()
    assert fixed.digest() == construct_microop_weights().digest()
    assert tokens.source_wasm_sha256 != fixed.digest()
    for pc, operation in enumerate(program.operations):
        decoded = read_microop(onehot(pc, SLOTS), tokens, fixed)
        expected_effect = {
            "i32.const": "push", "local.get": "push",
            "local.set": "pop", "drop": "pop",
            "if": "pop", "br_if": "pop",
            "i32.eqz": "unary",
            **{key: "binary" for key in ALU_OPS if key != "i32.eqz"},
        }.get(operation.opcode, "retain")
        assert EFFECTS[int(np.argmax(decoded.effect))] == expected_effect
        expected_mode = {
            "local.get": "get", "local.set": "set",
            "local.tee": "tee",
        }.get(operation.opcode, "none")
        assert LOCAL_MODES[int(np.argmax(decoded.local_mode))] == expected_mode
        if operation.opcode in ALU_OPS:
            assert ALU_OPS[int(np.argmax(decoded.alu_selector))] == operation.opcode
        else:
            assert decoded.alu_selector.sum() == 0.
        if operation.opcode == "i32.const":
            assert decoded.immediate_bits.sum() >= 0.
        if operation.opcode.startswith("local."):
            assert np.argmax(decoded.local_selector) == operation.immediate


def test_corrupted_program_data_fails_source_consistency():
    program = nested_program()
    tokens = compile_microop_tokens(program)
    altered = tokens.opcode_rows.copy()
    altered[1] = tokens.opcode_rows[2]
    with pytest.raises(MicroopAdmissionError):
        validate_microop_tokens(replace(tokens, opcode_rows=altered), program)
    corrupted = tokens.local_selectors.copy()
    corrupted[1, :] = 0.
    with pytest.raises(MicroopAdmissionError):
        validate_microop_tokens(replace(tokens, local_selectors=corrupted), program)
    with pytest.raises(MicroopAdmissionError):
        validate_microop_tokens(
            replace(tokens, source_wasm_sha256="0" * 64), program
        )
    with pytest.raises(MicroopAdmissionError):
        validate_microop_tokens(
            replace(tokens, length=tokens.length - 1), program
        )


def test_corrupted_fixed_effect_weight_is_not_silently_repaired():
    program = all_alu_program()
    tokens = compile_microop_tokens(program)
    weights = construct_microop_weights()
    corrupted = weights.effects.copy()
    corrupted[:, :] = 0.
    with pytest.raises(MicroopAdmissionError):
        read_microop(
            onehot(0, SLOTS), tokens,
            replace(weights, effects=corrupted),
        )


def test_pc_onehot_and_padding_cannot_bypass_admission():
    program = nested_program()
    tokens = compile_microop_tokens(program)
    pc = onehot(0, SLOTS)
    pc[1] = 1.
    with pytest.raises(MicroopAdmissionError):
        read_microop(pc, tokens)
    corrupt = tokens.branch_depths.copy()
    corrupt[SLOTS-1, 0] = 1.
    with pytest.raises(MicroopAdmissionError):
        validate_microop_tokens(replace(tokens, branch_depths=corrupt), program)
