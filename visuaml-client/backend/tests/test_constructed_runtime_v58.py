from dataclasses import replace

import numpy as np
import pytest

from scripts.benchmark_faults_v58 import ATTACKS, mutate
from visuaml.compiler_lab import (
    bit_neural_vm_v49,
    neural_alu_v56,
    neural_control_routing_v52,
    neural_local_bank_v57,
    neural_microstep_v58,
    neural_operand_storage_v55,
    typed_microop_ir_v57,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    execute_microstep_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated


@pytest.mark.parametrize(("validator", "error"), (
    (bit_neural_vm_v49._admit_binary, bit_neural_vm_v49.BitVMAdmissionError),
    (neural_alu_v56._admit_binary, neural_alu_v56.NeuralALUError),
    (neural_control_routing_v52._admit_onehot, neural_control_routing_v52.NeuralControlError),
    (neural_local_bank_v57._binary, typed_microop_ir_v57.MicroopAdmissionError),
    (neural_microstep_v58._binary, typed_microop_ir_v57.MicroopAdmissionError),
    (neural_operand_storage_v55._binary, neural_operand_storage_v55.OperandCircuitError),
    (typed_microop_ir_v57._binary, typed_microop_ir_v57.MicroopAdmissionError),
))
def test_exact_binary_gate_boundaries(validator, error):
    binary = np.array([-0., 1., 0., 1.])[::2]
    validator(binary, (2,))
    validator(np.array([1., 0.]), (2,))
    for bad in (np.nan, np.inf, -np.inf, -1., np.nextafter(1., 2.), np.nextafter(0., 1.)):
        with pytest.raises(error):
            validator(np.array([0., bad]), (2,))
    with pytest.raises(error):
        validator(binary.astype(np.float32), (2,))
    with pytest.raises(error):
        validator(binary.reshape(2, 1), (2,))


@pytest.mark.parametrize("argument", (-(1 << 31), -17, 0, (1 << 31) - 1))
def test_trace_capture_does_not_change_execution(argument):
    program = generated(2)
    traced = execute_microstep_control(program, argument)
    bare = execute_microstep_control(program, argument, capture_trace=False)
    assert traced.steps == len(traced.trace)
    assert bare == replace(traced, trace=())


@pytest.mark.parametrize("capture_trace", (True, False))
def test_no_trace_keeps_fault_and_fuel_checks(capture_trace):
    program = generated(2)
    with pytest.raises(ControlFuelExhausted):
        execute_microstep_control(program, -17, capture_trace=capture_trace, fuel=1)
    for attack in ATTACKS:
        if attack.engine == "v58":
            with pytest.raises(typed_microop_ir_v57.MicroopAdmissionError):
                execute_microstep_control(
                    program, -17, capture_trace=capture_trace, **mutate(attack, program),
                )
