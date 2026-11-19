from pathlib import Path

import pytest

from visuaml.compiler_lab.neural_microop_executor_v57 import execute_microop_routed_control
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    MEMORY_BYTES,
    ControlFuelExhausted,
    decode_control,
)


def test_array_store_survives_fuel_cutoff_without_native(monkeypatch):
    def reject_native(*args, **kwargs):
        pytest.fail("Array execution attempted to use a native executor")

    monkeypatch.setattr(
        "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep", reject_native,
    )
    monkeypatch.setattr("visuaml.compiler_lab.constructed_alu_v58.compiled_alu", reject_native)
    fixture = Path(__file__).resolve().parents[3] / "benchmarks/results/v58-array-memory/rollback.wasm"
    program = decode_control(fixture.read_bytes(), scaled=True)
    page = bytearray(MEMORY_BYTES)
    page[1:5] = page[7:11] = bytes.fromhex("01000080")
    expected = page.copy()
    expected[7:11] = bytes.fromhex("78563412")
    with pytest.raises(ControlFuelExhausted) as error:
        execute_microop_routed_control(program, 0, fuel=9, array_memory=page)
    assert page == expected
    assert error.value.trap_site == {"step": 9, "pc": 9, "opcode": None, "address": None}
    assert error.value.trace[-1]["memory_access"]["after_hex"] == "78563412"
