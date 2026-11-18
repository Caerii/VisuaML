from concurrent.futures import ThreadPoolExecutor
import shutil

import pytest

from visuaml.compiler_lab.neural_control_routing_v52 import construct_router_weights
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepAdmissionError, MicrostepBatchFailure, PreparedMicrostepProgram,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="requires C++17")


def _assert_outcomes(prepared, arguments, fuels, capture):
    batch = prepared.run_many(iter(arguments), fuel=iter(fuels), capture_trace=capture)
    assert len(batch) == len(arguments)
    for argument, fuel, outcome in zip(arguments, fuels, batch, strict=True):
        try:
            expected = prepared.run(argument, fuel=fuel, capture_trace=capture)
        except (ValueError, RuntimeError) as error:
            assert isinstance(outcome, MicrostepBatchFailure)
            assert type(outcome.error) is type(error)
            assert str(outcome.error) == str(error)
        else:
            assert outcome == expected


@pytest.mark.parametrize("capture", (False, True))
def test_batch_preserves_order_and_isolates_invalid_inputs_and_fuel_traps(capture):
    program = generated(2)
    prepared = PreparedMicrostepProgram(program, use_native_loop=True)
    sufficient = len(program.operations)
    _assert_outcomes(
        prepared, (-(1 << 31), 11, 1 << 31, (1 << 31) - 1, -3, -3, 4),
        (sufficient, 1, sufficient, sufficient, sufficient - 1, sufficient, False), capture,
    )
    assert prepared.run_many([], capture_trace=capture) == ()
    _assert_outcomes(prepared, (False, False), (1, 0), capture)


def test_batch_propagates_live_routing_and_structural_traps():
    program = ControlProgram(0, (ControlOperation("local.get", 0), ControlOperation("end")))
    weights = construct_router_weights()
    weights.path_weights[0] = 0.
    prepared = PreparedMicrostepProgram(program, pc_fixed=weights, use_native_loop=True)
    _assert_outcomes(prepared, (1, -1, 0), (2, 1, 2), True)
    malformed = ControlProgram(0, (ControlOperation("nop"), ControlOperation("end")))
    prepared = PreparedMicrostepProgram(malformed, use_native_loop=True)
    _assert_outcomes(prepared, (1, -1, 0), (2, 1, 2), True)


def test_shared_prepared_program_can_run_concurrent_batches():
    prepared = PreparedMicrostepProgram(generated(2), use_native_loop=True)
    inputs = [list(range(i, i + 16)) for i in (-200, -100, 10, 100)]
    expected = [tuple(prepared.run(x) for x in group) for group in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(prepared.run_many, inputs)) == expected


def test_batch_rejects_invalid_shared_options():
    program = generated(2)
    with pytest.raises(MicrostepAdmissionError, match="requires use_native_loop"):
        PreparedMicrostepProgram(program).run_many([1])
    prepared = PreparedMicrostepProgram(program, use_native_loop=True)
    with pytest.raises(MicrostepAdmissionError, match="one fuel value per input"):
        prepared.run_many([1, 2], fuel=[1])
    with pytest.raises(MicrostepAdmissionError, match="capture_trace"):
        prepared.run_many([1], capture_trace=0)
