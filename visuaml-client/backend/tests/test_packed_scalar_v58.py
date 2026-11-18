import random
import shutil

import pytest

from visuaml.compiler_lab.neural_alu_v56 import ALU_OPS
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")


@pytest.mark.parametrize("opcode", ALU_OPS)
def test_packed_alu_matches_float64_at_i32_boundaries(opcode):
    generator = random.Random(58)
    edges = (-(1 << 31), -1, 0, 1, (1 << 31) - 1)
    values = (*edges, *(generator.randrange(-(1 << 31), 1 << 31) for _ in range(25)))
    for right in edges:
        ops = [ControlOperation("local.get", 0)]
        if opcode != "i32.eqz":
            ops.append(ControlOperation("i32.const", right))
        program = ControlProgram(0, (*ops, ControlOperation(opcode), ControlOperation("end")))
        float64 = PreparedMicrostepProgram(program, use_native_loop=True)
        packed = PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True)
        assert packed.run_many(values) == float64.run_many(values)
