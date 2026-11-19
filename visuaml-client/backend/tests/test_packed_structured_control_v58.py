from dataclasses import replace
from shutil import which

import pytest

from scripts.verify_packed_control_v58 import (
    FIELDS, first_divergence, frame_reference, intervention_evidence, prepare, witnesses,
)
from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    MicrostepAdmissionError, MicrostepBatchFailure, PreparedMicrostepProgram,
)
from visuaml.compiler_lab.typed_microop_ir_v57 import MicroopAdmissionError, compile_microop_tokens
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, execute_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import ifelse

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


@pytest.mark.parametrize("name", tuple(witnesses()))
def test_packed_control_preserves_reference_pc_data_and_frames(name):
    program, inputs = witnesses()[name]
    value = inputs[-1]
    engine = prepare(program)
    actual = engine.run(value)
    host = execute_control(program, value)
    gate = execute_microop_routed_control(program, value)
    assert actual.result_i32 == host.value == gate.result_i32
    assert first_divergence(host.trace, actual.trace, FIELDS) is None
    assert first_divergence(tuple(frame_reference(program, gate.trace)), actual.trace) is None
    assert engine.run(value, capture_trace=False) == replace(actual, trace=())


def test_branch_restore_and_loop_retention_keep_outer_stack():
    program, _ = witnesses()["nested_branches"]
    row = next(row for row in prepare(program).run(1).trace if row["opcode"] == "br_if")
    assert row["stack_i32"] == [5]
    assert row["control_frames"] == []

    loop, _ = witnesses()["nested_loop_target"]
    branches = [row for row in prepare(loop).run(2).trace if row["opcode"] == "br_if"]
    first, last = branches
    assert first["stack_i32"] == last["stack_i32"] == [101]
    assert first["next_pc"] == 2
    assert first["control_labels"] == ["loop"]
    assert last["control_labels"] == ["loop", "block"]


def test_mixed_structured_batch_fuel_and_default_admission():
    program = ifelse()
    with pytest.raises(MicrostepAdmissionError, match="straight-line"):
        PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True)
    engine = prepare(program)
    results = engine.run_many((-1, 0, 1), fuel=(30, 4, 30))
    assert results[0] == engine.run(-1)
    assert results[2] == engine.run(1)
    assert isinstance(results[1], MicrostepBatchFailure)
    assert isinstance(results[1].error, ControlFuelExhausted)


def test_live_route_event_and_frame_gains_leave_evidence():
    evidence = intervention_evidence()
    assert evidence["route"]["step"] == 3
    assert evidence["route"]["changes"]["next_pc"] == [7, 4]

def test_structured_admission_rejects_altered_branch_tokens():
    program, _ = witnesses()["nested_branches"]
    tokens = compile_microop_tokens(program)
    branches = tokens.branch_depths.copy()
    pc = next(i for i, op in enumerate(program.operations) if op.opcode == "br_if")
    branches[pc, :2] = (1., 0.)
    with pytest.raises(MicroopAdmissionError, match="differ from canonical"):
        prepare(program, program_tokens=replace(tokens, branch_depths=branches))
