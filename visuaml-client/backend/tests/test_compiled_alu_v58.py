import shutil
from dataclasses import replace

import numpy as np
import pytest

from scripts.benchmark_faults_v58 import ATTACKS, mutate
from visuaml.compiler_lab.bit_neural_vm_v49 import BitVMAdmissionError, _bits, construct_bit_weights
from visuaml.compiler_lab.constructed_alu_v58 import evaluate_candidates
from visuaml.compiler_lab.neural_alu_v56 import (
    ALU_OPS,
    NeuralALUError,
    construct_alu_weights,
    execute_alu,
    onehot_alu,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.typed_microop_ir_v57 import MicroopAdmissionError
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="requires a C++17 compiler")


def test_all_compiled_alu_candidates_match_the_numpy_gate_equations():
    edge = [-(1 << 31), -1, 0, 1, (1 << 31) - 1]
    rng = np.random.default_rng(58)
    cases = [(a, b) for a in edge for b in edge]
    cases.extend(rng.integers(-(1 << 31), 1 << 31, size=(32, 2)).tolist())
    bit_weights = construct_bit_weights()
    for a, b in cases:
        left, right = _bits(a), _bits(b)
        actual = evaluate_candidates(left, right, bit_weights)
        expected = np.stack([execute_alu(onehot_alu(op), left, right, bit_fixed=bit_weights)
                             for op in ALU_OPS])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", (True, False))
def test_compiled_datapath_uses_live_gains_and_keeps_v58_fault_checks(mode):
    program = generated(2)
    expected = execute_microstep_control(program, -17, capture_trace=mode)
    assert execute_microstep_control(program, -17, capture_trace=mode, use_compiled_alu=True) == expected
    for attack in ATTACKS:
        if attack.engine == "v58":
            with pytest.raises(MicroopAdmissionError):
                execute_microstep_control(program, -17, capture_trace=mode, use_compiled_alu=True,
                                          **mutate(attack, program))

    bits = _bits(1)
    alu = construct_alu_weights()
    gains = alu.output_gains.copy()
    gains[0] = 0.
    with pytest.raises(NeuralALUError, match="unique selection"):
        execute_alu(onehot_alu("i32.add"), bits, bits, fixed=replace(alu, output_gains=gains),
                    use_compiled_candidates=True)
    weights = construct_bit_weights()
    xor = weights.xor_kernel.copy()
    xor[0, 0] = 0.
    with pytest.raises(BitVMAdmissionError, match="canonical"):
        execute_alu(onehot_alu("i32.lt_s"), bits, bits, bit_fixed=replace(weights, xor_kernel=xor),
                    use_compiled_candidates=True)
