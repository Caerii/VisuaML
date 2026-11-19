from itertools import product
from json import loads
from pathlib import Path
from random import Random
from unittest.mock import patch

import numpy as np
import pytest

import terrain_verify as terrain
from test_operand_replay import WITNESS
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.bit_neural_vm_v49 import _add, _bits, _int, _unsigned_lt, construct_bit_weights
from visuaml.compiler_lab.typed_microop_ir_v57 import compile_microop_tokens
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlAdmissionError,
    ControlFuelExhausted,
    LinearMemoryTrap,
    decode_control,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("name", ["i32.add", "i32.lt_s"])
def test_circuit_matches_all_candidates_and_integer_reference(name):
    bits = construct_bit_weights()
    fixed = alu.construct_alu_weights()
    opcode = alu.onehot_alu(name)
    circuit = alu._prepare_selected_alu(name, opcode, fixed, bits)
    edge = [0, 1, -1, -2, 2**31 - 1, -2**31, 0x7fff, 0x8000, 0xffff, 0x10000]
    rng = Random(57)
    pairs = list(product(edge, repeat=2)) + [(rng.randrange(-2**31, 2**31),
                                             rng.randrange(-2**31, 2**31)) for _ in range(116)]
    for a, b in pairs:
        left, right = _bits(a), _bits(b)
        expected = alu.execute_alu(opcode, left, right, fixed=fixed, bit_fixed=bits)
        actual = circuit(left, right)
        np.testing.assert_array_equal(actual, expected)
        if name == "i32.add":
            assert _int(actual) == (a + b + 2**31) % 2**32 - 2**31
        else:
            assert _int(actual) == int(a < b)
            assert _unsigned_lt(left, right, bits) == int(a % 2**32 < b % 2**32)
    invalid = _bits(0)
    invalid[0] = 0.5
    with pytest.raises(alu.NeuralALUError, match="exact binary"):
        circuit(invalid, _bits(0))


def test_exhaustive_single_bit_relu_adder_carry():
    fixed = construct_bit_weights()
    for a, b, carry in product((0, 1), repeat=3):
        result = _add(_bits(a), _bits(b), cin=carry, fixed=fixed)
        # Higher source bits are zero, so bit 1 is the first cell's carry.
        assert (result[0], result[1]) == ((a + b + carry) % 2, (a + b + carry) // 2)


@pytest.mark.parametrize("name", ["terrain", "terrain-omitted"])
def test_selected_circuits_preserve_full_scene_traces(name):
    saved = loads((FIXTURES / f"{name}.proof.json").read_text())
    traces = []
    for enabled in (False, True):
        traces.append([])

        def execute(*args, **kwargs):
            result = vm.execute_microop_routed_control(*args, **{
                **kwargs, "prepare_alu": enabled, "capture_trace": True, "fuel": 2000,
            })
            traces[-1].append(result.trace)
            return result

        with patch.object(terrain, "execute_microop_routed_control", execute):
            verified = terrain.verify(saved["session"], saved)
        assert [t["steps"] for t in verified["ticks"]] == [t["steps"] for t in saved["ticks"]]
    assert traces[0] == traces[1]


@pytest.mark.parametrize("custom", ["alu_fixed", "bit_fixed", "microop_tokens"])
def test_custom_config_uses_generic_candidates(custom):
    raw = terrain.GUEST.read_bytes()
    program = decode_control(raw, scaled=True)
    scenario = terrain.scale_session(2, False, 4)
    page = terrain.initial_page(scenario["initial"])
    override = {"alu_fixed": alu.construct_alu_weights(), "bit_fixed": construct_bit_weights(),
                "microop_tokens": compile_microop_tokens(program)}
    with patch.object(vm, "_prepare_selected_alu", side_effect=AssertionError("Custom path prepared")), patch(
        "visuaml.compiler_lab.neural_microop_executor_v57.execute_alu", wraps=alu.execute_alu,
    ) as generic:
        vm.execute_microop_routed_control(program, terrain.BASE, array_memory=page,
                                          **{custom: override[custom]})
    assert generic.called


def test_preparation_rejects_selector_mismatch_and_nonboolean_flag():
    with pytest.raises(alu.NeuralALUError, match="does not match"):
        alu._prepare_selected_alu("i32.add", alu.onehot_alu("i32.lt_s"),
                                  alu.construct_alu_weights(), construct_bit_weights())
    program = decode_control(terrain.GUEST.read_bytes(), scaled=True)
    with pytest.raises(ControlAdmissionError, match="prepare_alu"):
        vm.execute_microop_routed_control(program, terrain.BASE, prepare_alu=1)


@pytest.mark.parametrize(("fuel", "argument", "error"), [
    (7, 0, ControlFuelExhausted), (8, 0, ControlFuelExhausted),
    (9, 0, ControlFuelExhausted), (10, 0, ControlFuelExhausted),
    (100, 65526, LinearMemoryTrap),
])
def test_fuel_and_memory_trap_prefix(fuel, argument, error):
    program = decode_control(WITNESS.read_bytes(), scaled=True)
    initial = bytearray(terrain.PAGE_BYTES)
    if argument:
        initial[65527:65531] = bytes.fromhex("01000080")
        initial[65533:] = bytes.fromhex("aabbcc")
    else:
        initial[1:5] = initial[7:11] = bytes.fromhex("01000080")
    outcomes = []
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for enabled in (False, True):
            page = initial.copy()
            with pytest.raises(error) as failure:
                vm.execute_microop_routed_control(program, argument, array_memory=page, fuel=fuel,
                                                  prepare_alu=enabled)
            outcomes.append((page, failure.value.trace, failure.value.trap_site))
    assert outcomes[0] == outcomes[1]
