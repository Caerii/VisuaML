from json import loads
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import terrain_verify as terrain
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.bit_neural_vm_v49 import BitVMAdmissionError
from visuaml.compiler_lab.neural_operand_adapter_v55 import _PreparedPopStack
from visuaml.compiler_lab.neural_operand_storage_v55 import OperandCircuitError, fixed_operand_weights
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlAdmissionError,
    decode_control,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scene", ["terrain", "terrain-omitted"])
def test_prepared_and_checked_causal_traces(scene):
    saved = loads((ROOT / "fixtures" / f"{scene}.proof.json").read_text())
    observed = []
    for prepared in (False, True):
        traces = []

        def trace(*args, **kwargs):
            output = vm.execute_microop_routed_control(*args, **{
                **kwargs, "capture_trace": True, "fuel": 2000, "prepare_operand_pop": prepared,
            })
            traces.append(output.trace)
            return output

        with patch.object(terrain, "execute_microop_routed_control", trace):
            proof = terrain.verify(saved["session"], saved)
        assert [r["steps"] for r in proof["ticks"]] == [r["steps"] for r in saved["ticks"]]
        assert any(row["opcode"] in ("br", "br_if") and row["next_pc"] < row["pc"]
                   for trace in traces for row in trace)
        observed.append(traces)
    assert observed[0] == observed[1]


def test_custom_coefficients_use_checked_stack():
    program = decode_control(terrain.GUEST.read_bytes(), scaled=True)
    session = terrain.scale_session(2, False, 4)
    page = terrain.initial_page(session["initial"])
    with patch.object(vm, "_PreparedPopStack", side_effect=AssertionError("Unexpected preparation")):
        result = vm.execute_microop_routed_control(
            program, terrain.BASE, array_memory=page, operand_fixed=fixed_operand_weights(),
        )
    assert result.steps > 0


@pytest.mark.parametrize("enabled", [True, False])
def test_invalid_public_inputs_still_fail(enabled):
    program = decode_control(terrain.GUEST.read_bytes(), scaled=True)
    session = terrain.scale_session(2, False, 4)
    page = terrain.initial_page(session["initial"])
    for argument, memory, fuel, flag, error in (
        (1 << 31, page, 500, enabled, BitVMAdmissionError),
        (terrain.BASE, bytearray(2), 500, enabled, ControlAdmissionError),
        (terrain.BASE, page, 0, enabled, ControlAdmissionError),
        (terrain.BASE, page, 500, 1, ControlAdmissionError),
    ):
        with pytest.raises(error):
            vm.execute_microop_routed_control(
                program, argument, array_memory=memory, fuel=fuel, prepare_operand_pop=flag,
            )

def test_prepared_pop_keeps_independent_buffers_and_underflow_guard():
    stack = _PreparedPopStack()
    word = np.ones(32, dtype=np.float64)
    stack.append(word)
    previous = stack.state
    returned = stack.pop()
    returned[0] = 0.
    assert previous.cells[0, 0] == 1.
    assert not np.shares_memory(previous.cells, stack.state.cells)
    assert not np.shares_memory(previous.height, stack.state.height)
    assert not np.shares_memory(returned, stack.state.cells)
    assert not stack.state.cells.any()
    with pytest.raises(OperandCircuitError, match="underflow"):
        stack.pop()
