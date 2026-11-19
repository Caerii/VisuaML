from json import loads
from pathlib import Path
from unittest.mock import patch

import pytest

import terrain_verify as terrain
from operand_pairs import reference_transition
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab import neural_operand_adapter_v55 as adapter
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    LinearMemoryTrap,
    decode_control,
)

ROOT = Path(__file__).resolve().parents[1]
WITNESS = ROOT.parents[1] / "benchmarks/results/v58-array-memory/rollback.wasm"


@pytest.mark.parametrize("name", ["terrain", "terrain-omitted"])
def test_saved_causal_histories(name):
    expected = loads((ROOT / "fixtures" / f"{name}.proof.json").read_text())
    traces = []

    def traced(*args, **kwargs):
        result = vm.execute_microop_routed_control(*args, **{
            **kwargs, "fuel": 2000, "capture_trace": True,
        })
        traces[-1].append(result.trace)
        return result

    for step in (reference_transition(), adapter.step_operands):
        traces.append([])
        with patch.object(adapter, "step_operands", step), patch.object(
            terrain, "execute_microop_routed_control", traced,
        ):
            proof = terrain.verify(expected["session"], expected)
        assert [row["steps"] for row in proof["ticks"]] == [row["steps"] for row in expected["ticks"]]
    assert traces[0] == traces[1]



@pytest.mark.parametrize(("fuel", "argument", "active", "error"), [
    (7, 0, True, ControlFuelExhausted),
    (8, 0, True, ControlFuelExhausted),
    (9, 0, True, ControlFuelExhausted),
    (10, 0, True, ControlFuelExhausted),
    (100, 65526, False, LinearMemoryTrap),
])
def test_exact_fuel_and_trap_prefix(fuel, argument, active, error):
    program = decode_control(WITNESS.read_bytes(), scaled=True)
    initial = bytearray(terrain.PAGE_BYTES)
    if active:
        initial[1:5] = initial[7:11] = bytes.fromhex("01000080")
    else:
        initial[65527:65531] = bytes.fromhex("01000080")
        initial[65533:] = bytes.fromhex("aabbcc")
    observed = []
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for step in (reference_transition(), adapter.step_operands):
            page = initial.copy()
            with patch.object(adapter, "step_operands", step), pytest.raises(error) as failure:
                vm.execute_microop_routed_control(program, argument, array_memory=page, fuel=fuel)
            observed.append((failure.value.trace, failure.value.trap_site, page))
    assert observed[0] == observed[1]
    if error is LinearMemoryTrap:
        assert observed[0][2] == initial
    else:
        assert len(observed[0][0]) == fuel
