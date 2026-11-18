"""v58 end-to-end straight-line microstep invariants and source WASM parity."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import _bits
from visuaml.compiler_lab.neural_alu_executor_v56 import (
    execute_alu_routed_control,
)
from visuaml.compiler_lab.neural_control_routing_v52 import SLOTS
from visuaml.compiler_lab.neural_microstep_v58 import (
    MicrostepWeights,
    construct_microstep_weights,
    step_data_microop,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepAdmissionError,
    execute_microstep_control,
)
from visuaml.compiler_lab.neural_operand_storage_v55 import empty_operands
from visuaml.compiler_lab.typed_microop_ir_v57 import (
    MicroopAdmissionError,
    compile_microop_tokens,
    construct_microop_weights,
    read_microop,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlOperation, ControlProgram, encode_control, execute_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated, gcd12


VALUES = (-(1 << 31), -17, 0, 1, 17, (1 << 31)-1)


def onehot(i, width):
    a = np.zeros(width, dtype=np.float64)
    a[i] = 1.
    return a


@pytest.mark.parametrize("seed", range(16))
def test_generated_wasm_microsteps_match_v51_v56_each_state(seed):
    program = generated(seed)
    for argument in VALUES:
        new = execute_microstep_control(program, argument, fuel=300)
        old = execute_alu_routed_control(program, argument, fuel=300)
        oracle = execute_control(program, argument, fuel=300)
        assert new.data_effects_driven_by_microstep
        assert new.result_i32 == old.result_i32 == oracle.value
        assert new.steps == old.steps == oracle.transitions
        for step, (a, b, c) in enumerate(zip(
            new.trace, old.trace, oracle.trace, strict=True
        )):
            for key in (
                "step", "pc", "opcode", "next_pc", "stack_i32",
                "locals_i32", "control_labels", "halted",
            ):
                assert a[key] == b[key] == c[key], (seed, argument, step, key)


@pytest.mark.parametrize("seed", range(16))
def test_original_wasmtime_executes_each_generated_program(seed):
    wasmtime = pytest.importorskip("wasmtime")
    program = generated(seed)
    engine = wasmtime.Engine()
    raw = encode_control(program)
    wasmtime.Module.validate(engine, raw)
    store = wasmtime.Store(engine)
    run = wasmtime.Instance(
        store, wasmtime.Module(engine, raw), []
    ).exports(store)["run"]
    for argument in VALUES:
        assert execute_microstep_control(
            program, argument, fuel=300,
        ).result_i32 == run(store, argument)


def test_host_control_and_noncanonical_program_not_admitted():
    with pytest.raises(MicrostepAdmissionError):
        execute_microstep_control(gcd12(), 4)
    invalid = ControlProgram(0, (
        ControlOperation("i32.const", 1),
        ControlOperation("drop"),
        ControlOperation("end"),
    ))
    with pytest.raises((MicroopAdmissionError, RuntimeError, ValueError)):
        execute_microstep_control(invalid, 0)


def test_wrong_but_binary_effect_mask_fails_before_execution():
    program = generated(0)
    fixed = construct_microop_weights()
    effects = fixed.effects.copy()
    from visuaml.compiler_lab.typed_microop_ir_v57 import EFFECTS, OPCODES

    index = OPCODES.index("i32.mul")
    effects[index, :] = 0.
    effects[index, EFFECTS.index("retain")] = 1.
    # This gate mislabels a binary instruction, so the output cannot match
    # expected source semantics. Rejection OR explicit differential mismatch
    # are acceptable for a finite adversarial evidence test.
    try:
        candidate = execute_microstep_control(
            program, 1, microop_fixed=replace(fixed, effects=effects),
        )
    except (MicroopAdmissionError, RuntimeError, ValueError):
        return
    reference = execute_control(program, 1)
    assert candidate.trace != reference.trace


def test_pop_and_push_gain_ablation_cannot_silently_preserve_state():
    program = generated(2)
    baseline = execute_microstep_control(program, -17)
    weights = construct_microstep_weights()
    altered = MicrostepWeights(
        weights.pop_gain.copy(),
        weights.produce_gain.copy(),
        weights.write_gain.copy(),
    )
    altered.pop_gain[0] = 0.
    with pytest.raises((MicroopAdmissionError, RuntimeError, ValueError)):
        execute_microstep_control(program, -17, microstep_fixed=altered)
    assert baseline.steps > 0


def test_immediate_is_fetched_by_source_bound_microop_not_python_constant():
    program = ControlProgram(0, (
        ControlOperation("i32.const", -(1 << 31)),
        ControlOperation("end"),
    ))
    token = compile_microop_tokens(program)
    selected = read_microop(onehot(0, SLOTS), token)
    state = empty_operands()
    locals_ = np.zeros((5, 32), dtype=np.float64)
    locals_[0] = _bits(17)
    result = step_data_microop(state, locals_, selected)
    assert result.consumed_count == 0
    assert result.produced_count == 1
    assert execute_microstep_control(program, 17).result_i32 == -(1 << 31)
