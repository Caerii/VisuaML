from dataclasses import replace
from itertools import product
from json import loads
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wasmtime

import terrain_verify as terrain
from test_prepared_regions import SOURCE, WITNESS
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.compiled_frame_event_v58 import CompiledFrameEvents
from visuaml.compiler_lab.neural_frame_events_v54 import (
    OPCODES, EventCircuitError, decode_opcode_token, fixed_event_weights, select_frame_action,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap, decode_control

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


def test_complete_canonical_event_gate_truth_table():
    tokens = tuple(decode_opcode_token(name) for name in OPCODES)
    weights = fixed_event_weights()
    kernel = CompiledFrameEvents(tokens, weights)
    for pc, flags in product(range(len(tokens)), product((False, True), repeat=4)):
        sequential, has_else, active, taken = (np.array([float(flag)]) for flag in flags)
        expected = select_frame_action(tokens[pc], conditional_sequential=sequential,
                                       has_else=has_else, frame_active=active, branch_taken=taken,
                                       weights=weights)
        assert kernel.select(pc, *flags) == int(expected.argmax())
        np.testing.assert_array_equal(kernel.vector, expected)


def test_invalid_native_row_shape_rejected_before_binding():
    tokens = (decode_opcode_token("if"),)
    weights = fixed_event_weights()
    with pytest.raises(EventCircuitError, match="dimensions"):
        CompiledFrameEvents(tokens, replace(weights, default=weights.default[:, :-1]))


@pytest.mark.parametrize("scene", ["terrain", "terrain-omitted"])
def test_compiled_terrain_exact_pages_and_explicit_provenance(scene):
    expected = loads((FIXTURES / f"{scene}.proof.json").read_text())
    used = []

    def execute(*args, **kwargs):
        result = vm.execute_microop_routed_control(*args, **kwargs, use_compiled_frame_event=True)
        used.append(result.compiled_frame_event_sha256)
        return result

    with patch.object(terrain, "execute_microop_routed_control", execute):
        actual = terrain.verify(expected["session"], expected)
    assert len(used) == 2 and used[0] and used[0] == used[1]
    assert [row["steps"] for row in actual["ticks"]] == [row["steps"] for row in expected["ticks"]]


@pytest.mark.parametrize("source", [SOURCE, SOURCE.replace("i32.const 2147483647", "local.get 0")])
def test_fuel_boundaries_through_pair_and_triple(source):
    program = decode_control(bytes(wasmtime.wat2wasm(source)), scaled=True)
    page = bytearray(65536)
    traced = vm.execute_microop_routed_control(program, 0, array_memory=page)
    add_step = next(row["step"] for row in traced.trace if row["opcode"] == "i32.add")
    width = 3 if "2147483647" not in source else 2
    for cut in range(add_step - width + 1, add_step + 2):
        for capture in (True, False):
            observations = []
            with patch(PACKED, side_effect=AssertionError("Unexpected packed entry")):
                for enabled in (False, True):
                    page = bytearray(65536)
                    with pytest.raises(ControlFuelExhausted) as failure:
                        vm.execute_microop_routed_control(
                            program, 0, array_memory=page, fuel=cut, capture_trace=capture,
                            use_compiled_frame_event=enabled,
                        )
                    observations.append((page, failure.value.trace, failure.value.trap_site))
            assert observations[0] == observations[1]
            if capture:
                assert observations[0][1] == traced.trace[:cut]


def test_bypass_branch_and_page_straddling_trap():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed entry")):
        for argument, expected_steps in ((0, 1), (1, 0)):
            expected = bytearray(65536)
            baseline = vm.execute_microop_routed_control(program, argument, array_memory=expected)
            page = bytearray(65536)
            with patch.object(vm, "_run_const_add_region", wraps=vm._run_const_add_region) as region:
                result = vm.execute_microop_routed_control(
                    program, argument, array_memory=page, capture_trace=False,
                    use_compiled_frame_event=True,
                )
            assert region.call_count == expected_steps
            assert (result.result_i32, result.steps, page) == (baseline.result_i32, baseline.steps, expected)
        program = decode_control(WITNESS.read_bytes(), scaled=True)
        initial = bytearray(65536)
        initial[65527:65531] = bytes.fromhex("01000080")
        initial[65533:] = bytes.fromhex("aabbcc")
        for capture in (True, False):
            outcomes = []
            for enabled in (False, True):
                page = initial.copy()
                with pytest.raises(LinearMemoryTrap) as failure:
                    vm.execute_microop_routed_control(
                        program, 65526, array_memory=page, capture_trace=capture,
                        use_compiled_frame_event=enabled,
                    )
                outcomes.append((page, failure.value.trace, failure.value.trap_site))
            assert outcomes[0] == outcomes[1] and outcomes[0][0] == initial


def test_traced_and_custom_programs_keep_reference_path():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch.object(vm, "CompiledFrameEvents", side_effect=AssertionError("Unexpected compilation")):
        for traced, fixed in ((True, None), (False, fixed_event_weights())):
            result = vm.execute_microop_routed_control(
                program, 0, array_memory=bytearray(65536), capture_trace=traced, event_fixed=fixed,
                use_compiled_frame_event=True,
            )
            assert result.compiled_frame_event_sha256 is None
