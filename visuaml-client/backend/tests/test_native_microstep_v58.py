from dataclasses import replace
import shutil

import numpy as np
import pytest

from visuaml.compiler_lab.bit_neural_vm_v49 import BitVMAdmissionError
from visuaml.compiler_lab.neural_control_routing_v52 import NeuralControlError, construct_router_weights
from visuaml.compiler_lab.neural_local_bank_v57 import construct_local_bank_weights
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram
from visuaml.compiler_lab.neural_microstep_v58 import construct_microstep_weights
from visuaml.compiler_lab.neural_operand_storage_v55 import MAX_STACK, OperandCircuitError, fixed_operand_weights
from visuaml.compiler_lab.typed_microop_ir_v57 import MicroopAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlExecutionTrap, ControlFuelExhausted, ControlOperation, ControlProgram,
)

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")


def _program(*opcodes):
    return ControlProgram(1, tuple(ControlOperation(*op) for op in (*opcodes, ("end",))))


@pytest.mark.parametrize("loop", (False, True))
def test_native_microstep_observes_live_local_write_mask(loop):
    program = _program(("i32.const", 23), ("local.tee", 1), ("drop",), ("local.get", 1))
    altered = replace(construct_microstep_weights(), write_gain=np.zeros(1))
    reference = PreparedMicrostepProgram(program, microstep_fixed=altered).run(0)
    native = PreparedMicrostepProgram(program, microstep_fixed=altered, use_native_microstep=True, use_native_loop=loop)
    assert native.run(0) == reference
    assert reference.result_i32 == 0
    assert PreparedMicrostepProgram(program, use_native_microstep=True, use_native_loop=loop).run(0).result_i32 == 23


@pytest.mark.parametrize("mode", ("python", "microstep", "loop"))
def test_microstep_traps_at_both_operand_bounds(mode):
    for program, reason in (
        (_program(("drop",)), "underflow"),
        (_program(*(("i32.const", 1),) * (MAX_STACK + 1)), "overflow"),
    ):
        with pytest.raises(OperandCircuitError, match=reason):
            PreparedMicrostepProgram(program, use_native_microstep=mode == "microstep", use_native_loop=mode == "loop").run(0)


@pytest.mark.parametrize("mode", ("python", "microstep", "loop"))
def test_unused_candidate_gates_still_trap_under_ablations(mode):
    program = _program(("i32.const", 4))
    local = construct_local_bank_weights()
    local.write_gain[0] = 0.
    with pytest.raises(MicroopAdmissionError, match="Local write gate ablated"):
        PreparedMicrostepProgram(program, local_fixed=local, use_native_microstep=mode == "microstep", use_native_loop=mode == "loop").run(0)
    operand = fixed_operand_weights()
    operand.action_gain[0] = 0.
    with pytest.raises(OperandCircuitError, match="Noncanonical one-hot operand token"):
        PreparedMicrostepProgram(program, operand_fixed=operand, use_native_microstep=mode == "microstep", use_native_loop=mode == "loop").run(0)


@pytest.mark.parametrize("capture", (False, True))
def test_native_loop_uses_live_router_and_exact_fuel_boundary(capture):
    program = _program(("local.get", 0), ("i32.eqz",))
    weights = construct_router_weights()
    reference = PreparedMicrostepProgram(program, pc_fixed=weights, use_native_microstep=True)
    native = PreparedMicrostepProgram(program, pc_fixed=weights, use_native_loop=True)
    steps = len(program.operations)
    assert native.run(-1, fuel=steps, capture_trace=capture) == reference.run(
        -1, fuel=steps, capture_trace=capture,
    )
    if capture:
        assert native.run(-1).trace[1]["neural_route"]["nonzero_condition_gate"] == 1
    with pytest.raises(ControlFuelExhausted, match="after 2 v58 microsteps"):
        native.run(0, fuel=steps - 1, capture_trace=capture)
    weights.path_weights[0] = 0.
    with pytest.raises(NeuralControlError, match="lost unique next PC"):
        PreparedMicrostepProgram(program, pc_fixed=weights, use_native_loop=True).run(0)
    with pytest.raises(BitVMAdmissionError, match="signed i32"):
        native.run(1 << 31)
    with pytest.raises(ControlExecutionTrap, match="exactly one i32"):
        PreparedMicrostepProgram(_program(("nop",)), use_native_loop=True).run(0)
