from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import shutil

import pytest

from visuaml.compiler_lab.neural_alu_v56 import NeuralALUError
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    PreparedMicrostepProgram, compile_microop_tokens, compile_route_tokens,
    construct_alu_weights, construct_router_weights, execute_microstep_control,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import MicroopAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, ControlOperation, ControlProgram
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")
_OPTIONS = dict(use_compact_routing=True, use_indexed_pc=True,
                use_fused_state=True, use_compiled_alu=True)


@pytest.fixture(params=(False, True), ids=("python", "native"))
def prepared_factory(request):
    def prepare(program, **options):
        return PreparedMicrostepProgram(program, use_native_microstep=request.param, **options)

    return prepare


def test_prepared_calls_reset_machine_state_after_trap_and_under_concurrency(prepared_factory):
    program = generated(2)
    prepared = prepared_factory(program)
    inputs = (-17, 0, 1, -(1 << 31), (1 << 31) - 1, -17)
    expected = [execute_microstep_control(program, x, **_OPTIONS) for x in inputs]
    with pytest.raises(ControlFuelExhausted):
        prepared.run(inputs[0], fuel=2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(prepared.run, inputs)) == expected
    assert prepared.run(inputs[0], capture_trace=False) == replace(expected[0], trace=())


def test_preparation_captures_source_tokens_and_live_coefficients(prepared_factory):
    program = ControlProgram(0, (
        ControlOperation("local.get", 0), ControlOperation("i32.const", 3),
        ControlOperation("i32.add"), ControlOperation("end"),
    ))
    tokens = compile_route_tokens(program)
    microops = compile_microop_tokens(program)
    route = replace(tokens, sequential=tokens.sequential.copy())
    decoded = replace(microops, i32_immediates=microops.i32_immediates.copy())
    pc = construct_router_weights()
    alu = construct_alu_weights()
    options = dict(tokens=route, program_tokens=decoded, pc_fixed=pc, alu_fixed=alu)
    expected = execute_microstep_control(program, 7, **_OPTIONS, **options)
    assert expected.result_i32 == 10
    prepared = prepared_factory(program, **options)
    route.sequential[:] = 0.
    decoded.i32_immediates[:] = 0.
    pc.path_weights[:] = 0.
    alu.output_gains[0] = 0.
    with pytest.raises(NeuralALUError, match="fails unique selection"):
        prepared_factory(program, alu_fixed=alu).run(7)
    assert prepared.run(7) == expected


def test_preparation_rejects_noncanonical_tokens_before_running(prepared_factory):
    program = generated(2)
    tokens = compile_microop_tokens(program)
    corrupted = tokens.i32_immediates.copy()
    corrupted[0, 0] = 1. - corrupted[0, 0]
    with pytest.raises(MicroopAdmissionError, match="differ from canonical"):
        prepared_factory(program, program_tokens=replace(
            tokens, i32_immediates=corrupted,
        ))
