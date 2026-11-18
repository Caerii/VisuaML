"""v57 source-bound typed microop execution checks.

Runs locally; no GitHub Actions dependency.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import (
    execute_microop_routed_control,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    MicroopAdmissionError,
    compile_microop_tokens,
    construct_microop_weights,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    encode_control,
    execute_control,
)

from test_neural_alu_executor_v56 import programs, infinite


@pytest.mark.parametrize("name", list(programs()))
def test_live_microop_decoding_matches_v51_and_v56_per_instruction(name):
    program, samples = programs()[name]
    for argument in samples:
        actual = execute_microop_routed_control(program, argument, fuel=1500)
        previous = execute_alu_routed_control(program, argument, fuel=1500)
        reference = execute_control(program, argument, fuel=1500)
        assert actual.microop_token_selection_drives_execution
        assert actual.result_i32 == previous.result_i32 == reference.value
        assert actual.steps == previous.steps == reference.transitions
        assert actual.fixed_microop_weights_sha256 == (
            construct_microop_weights().digest()
        )
        for curr, prior, oracle in zip(
            actual.trace, previous.trace, reference.trace, strict=True
        ):
            for field in (
                "step", "pc", "opcode", "next_pc", "stack_i32",
                "locals_i32", "control_labels", "halted",
            ):
                assert curr[field] == prior[field] == oracle[field], (
                    name, argument, field
                )
            assert curr["neural_frame_event"] == prior["neural_frame_event"]
            assert curr["neural_alu_opcode"] == prior["neural_alu_opcode"]


@pytest.mark.parametrize("name", list(programs()))
def test_v57_original_wasm_independently_executed(name):
    wasmtime = pytest.importorskip("wasmtime")
    program, samples = programs()[name]
    raw = encode_control(program)
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw)
    store = wasmtime.Store(engine)
    run = wasmtime.Instance(
        store, wasmtime.Module(engine, raw), []
    ).exports(store)["run"]
    for x in samples:
        assert execute_microop_routed_control(
            program, x, fuel=1500
        ).result_i32 == run(store, x)


def test_altering_alu_token_selector_changes_execution_or_fails_closed():
    program, _ = programs()["countdown"]
    weights = construct_microop_weights()
    corrupted = weights.alu_selectors.copy()
    corrupted[:, :] = 0.
    with pytest.raises(MicroopAdmissionError):
        execute_microop_routed_control(
            program, 4, microop_fixed=replace(
                weights, alu_selectors=corrupted
            ),
        )


def test_invalid_source_token_ownership_rejected_before_execution():
    program, _ = programs()["ifelse"]
    tokens = compile_microop_tokens(program)
    bad = tokens.local_selectors.copy()
    bad[0, 0] = 1.
    with pytest.raises(MicroopAdmissionError):
        execute_microop_routed_control(
            program, 3,
            microop_tokens=replace(tokens, local_selectors=bad),
        )
    bad = tokens.i32_immediates.copy()
    bad[2, 4] = 1. - bad[2, 4]
    with pytest.raises(MicroopAdmissionError):
        execute_microop_routed_control(
            program, 3,
            microop_tokens=replace(tokens, i32_immediates=bad),
        )


def test_empty_loop_fuel_exhaustion_is_not_language_trap():
    with pytest.raises(ControlFuelExhausted):
        execute_microop_routed_control(infinite(), 0, fuel=70)


def test_microop_local_write_gate_corruption_fail_closed():
    from visuaml.compiler_lab.neural_local_bank_v57 import (
        LocalBankWeights,
        construct_local_bank_weights,
    )

    program, _ = programs()["countdown"]
    default = construct_local_bank_weights()
    changed = default.write_gain.copy()
    changed[1] = 0.
    with pytest.raises(MicroopAdmissionError):
        execute_microop_routed_control(
            program, 4,
            local_fixed=LocalBankWeights(default.read_gain, changed),
        )
    assert np.array_equal(default.write_gain, np.ones_like(default.write_gain))
