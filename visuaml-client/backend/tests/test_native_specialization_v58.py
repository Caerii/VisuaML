import shutil

import pytest

from visuaml.compiler_lab.native_specialization_v58 import NativeSpecialization
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepAdmissionError,
    execute_microstep_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlExecutionTrap,
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    encode_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import gcd12, generated

pytestmark = pytest.mark.skipif(
    not any(shutil.which(name) for name in ("c++", "g++", "clang++")),
    reason="native C++ compiler unavailable",
)


def test_native_specialization_wraparound_and_full_state():
    program = generated(3)
    candidate = NativeSpecialization(encode_control(program))
    for value in (-(1 << 31), -1, 0, 1, (1 << 31) - 1):
        reference = execute_microstep_control(program, value)
        assert candidate.run(value) == reference.result_i32
        trace = candidate.trace(value)
        assert len(trace) == reference.steps
        for native, neural in zip(trace, reference.trace, strict=True):
            assert native == {key: neural[key] for key in native}

    with pytest.raises(ControlFuelExhausted):
        candidate.run(0, fuel=len(program.operations) - 1)
    with pytest.raises(ControlFuelExhausted):
        candidate.trace(0, fuel=len(program.operations) - 1)


def test_native_specialization_rejects_invalid_and_structured_programs():
    malformed = ControlProgram(0, (ControlOperation("drop"), ControlOperation("end")))
    with pytest.raises(ControlExecutionTrap, match="underflow"):
        NativeSpecialization(encode_control(malformed))
    with pytest.raises(MicrostepAdmissionError, match="straight-line"):
        NativeSpecialization(encode_control(gcd12()))
