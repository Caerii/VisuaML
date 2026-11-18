from dataclasses import replace
from random import Random
from shutil import which

import numpy as np
import pytest

from visuaml.compiler_lab.neural_alu_v56 import ALU_OPS, construct_alu_weights
from visuaml.compiler_lab.neural_control_routing_v52 import construct_router_weights
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepBatchFailure, PreparedMicrostepProgram,
)
from visuaml.compiler_lab.neural_microstep_v58 import construct_microstep_weights
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


def _prepare(program, simd, **weights):
    return PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True,
                                    use_packed_simd=simd, **weights)


@pytest.mark.parametrize("opcode", ALU_OPS)
def test_simd_alu_matches_packed_scalar_for_full_groups_and_tail(opcode):
    random = Random(58)
    edge = (-(1 << 31), -1, 0, 1, (1 << 31) - 1)
    values = (*edge, *(random.randrange(-(1 << 31), 1 << 31) for _ in range(28)))
    for right in edge:
        ops = [ControlOperation("local.get", 0)]
        if opcode != "i32.eqz":
            ops.append(ControlOperation("i32.const", right))
        program = ControlProgram(0, (*ops, ControlOperation(opcode), ControlOperation("end")))
        simd = _prepare(program, True)
        scalar = _prepare(program, False)
        assert simd.run_many(values, capture_trace=False) == scalar.run_many(values, capture_trace=False)


def _normalized(outcomes):
    return tuple((type(row.error), str(row.error)) if isinstance(row, MicrostepBatchFailure)
                 else row for row in outcomes)


@pytest.mark.parametrize("capture_trace", (False, True))
def test_simd_preserves_mixed_fuel_errors_and_order(capture_trace):
    program = generated(2)
    arguments = tuple(range(-16, 17))
    fuels = [len(program.operations)] * len(arguments)
    fuels[9] = 2
    fuels[-1] = 1
    result = _prepare(program, True).run_many(arguments, fuel=fuels, capture_trace=capture_trace)
    expected = _prepare(program, False).run_many(arguments, fuel=fuels, capture_trace=capture_trace)
    assert _normalized(result) == _normalized(expected)
    assert isinstance(result[9], MicrostepBatchFailure)
    assert isinstance(result[-1], MicrostepBatchFailure)


@pytest.mark.parametrize("fixed", (
    {"pc_fixed": replace(construct_router_weights(), path_weights=np.array([0., 1.]))},
    {"alu_fixed": replace(construct_alu_weights(), output_gains=np.zeros(6))},
    {"microstep_fixed": replace(construct_microstep_weights(), write_gain=np.zeros(1))},
))
def test_simd_preserves_routing_arithmetic_and_state_gain_interventions(fixed):
    program = ControlProgram(1, tuple(ControlOperation(*op) for op in (
        ("local.get", 0), ("local.set", 1), ("local.get", 1), ("end",),
    )))
    arguments = tuple(range(-8, 8))
    result = _prepare(program, True, **fixed).run_many(arguments, capture_trace=False)
    expected = _prepare(program, False, **fixed).run_many(arguments, capture_trace=False)
    assert _normalized(result) == _normalized(expected)
    assert _normalized(result) != _prepare(program, False).run_many(arguments, capture_trace=False)
