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
from visuaml.compiler_lab.compiled_frame_event_v58 import _CompiledFrameRetain
from visuaml.compiler_lab.neural_frame_circuit_v53 import empty_frames
from visuaml.compiler_lab.neural_frame_events_v54 import (
    OPCODES, decode_opcode_token, fixed_event_weights, select_frame_action,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap, decode_control

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {"use_compiled_frame_event": True, "compiled_frame_retain": "copy"}
MODES = ((False, False), (True, False), (True, True))
PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


def test_static_rows_preserve_complete_truth_table_and_repeated_live_guards():
    entries = tuple((name, lexical) for name, lexical in product(OPCODES, (False, True)))
    tokens = tuple(decode_opcode_token(name) for name, _ in entries)
    weights = fixed_event_weights()
    flags = tuple(lexical for _, lexical in entries)
    kernel = _CompiledFrameRetain(tokens, weights, empty_frames(), resident=False, lexical_else=flags)
    other = _CompiledFrameRetain(tokens, weights, empty_frames(), resident=False,
                                 lexical_else=tuple(not flag for flag in flags))
    for pc, (_, lexical) in enumerate(entries):
        for sequential, active, taken in product((False, True), repeat=3):
            for bound, has_else in ((kernel, lexical), (other, not lexical)):
                expected = select_frame_action(
                    tokens[pc], conditional_sequential=np.array([float(sequential)]),
                    has_else=np.array([float(has_else)]), frame_active=np.array([float(active)]),
                    branch_taken=np.array([float(taken)]), weights=weights,
                )
                assert bound.select_live(pc, sequential, active, taken) == int(expected.argmax())
                np.testing.assert_array_equal(bound.vector, expected)
    assert not np.shares_memory(kernel.vector, other.vector)
    assert not kernel._rows.flags.writeable


@pytest.mark.parametrize("scene", ["terrain", "terrain-omitted"])
def test_all_modes_match_archived_terrain(scene):
    saved = loads((ROOT / "fixtures" / f"{scene}.proof.json").read_text())
    for prepare, compact in MODES:
        with patch.object(terrain, "execute_microop_routed_control", side_effect=lambda *a, **kw:
                          vm.execute_microop_routed_control(*a, **kw, **OPTIONS,
                                                           prepare_frame_inputs=prepare,
                                                           compact_frame_inputs=compact)):
            actual = terrain.verify(saved["session"], saved)
        assert [r["steps"] for r in actual["ticks"]] == [r["steps"] for r in saved["ticks"]]


def test_bypass_fuel_boundaries_and_trace_reference():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        expected = vm.execute_microop_routed_control(program, 0, array_memory=bytearray(65536))
        for cut in range(1, expected.steps):
            for trace in (True, False):
                observed = []
                for prepare, compact in MODES:
                    page = bytearray(65536)
                    with pytest.raises(ControlFuelExhausted) as err:
                        vm.execute_microop_routed_control(
                            program, 0, array_memory=page, fuel=cut, capture_trace=trace, **OPTIONS,
                            prepare_frame_inputs=prepare, compact_frame_inputs=compact,
                        )
                    observed.append((page, err.value.trace, err.value.trap_site))
                assert observed[0] == observed[1] == observed[2]
                if trace:
                    assert observed[0][1] == expected.trace[:cut]
        for prepare, compact in MODES:
            result = vm.execute_microop_routed_control(
                program, 1, array_memory=bytearray(65536), capture_trace=False, **OPTIONS,
                prepare_frame_inputs=prepare, compact_frame_inputs=compact,
            )
            assert result.result_i32 == 1


def test_custom_fallback_and_page_straddling_trap():
    program = decode_control(WITNESS.read_bytes(), scaled=True)
    start = bytearray(65536)
    start[65527:65531] = bytes.fromhex("01000080")
    start[65533:] = bytes.fromhex("aabbcc")
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for prepare, compact in MODES:
            page = start.copy()
            with pytest.raises(LinearMemoryTrap):
                vm.execute_microop_routed_control(
                    program, 65526, array_memory=page, capture_trace=False, **OPTIONS,
                    prepare_frame_inputs=prepare, compact_frame_inputs=compact,
                )
            assert page == start
        with patch.object(vm, "_CompiledFrameRetain", side_effect=AssertionError("Custom profile compiled")):
            with pytest.raises(LinearMemoryTrap):
                vm.execute_microop_routed_control(
                    program, 65526, array_memory=start.copy(), capture_trace=False, **OPTIONS,
                    prepare_frame_inputs=True, compact_frame_inputs=True, event_fixed=fixed_event_weights(),
                )


def test_same_live_backedge_and_if_row_replay_both_outcomes():
    raw = bytes(wasmtime.wat2wasm("""
    (module (func (export "run") (param i32) (result i32) (local i32)
      block loop
        local.get 0 i32.eqz br_if 1
        local.get 0 i32.const 1 i32.sub local.set 0
        local.get 1 i32.const 1 i32.add local.set 1 br 0
      end end
      local.get 1 if i32.const 7 local.set 1 else i32.const 9 local.set 1 end
      i32.const 0 local.get 1 i32.store
      i32.const 0 i32.load)
      (memory (export "memory") 1 1))
    """))
    program = decode_control(raw, scaled=True)
    store = wasmtime.Store()
    guest = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        for argument in (0, 3):
            expected = guest["run"](store, argument)
            observed = []
            for prepare, compact in MODES:
                page = bytearray(65536)
                result = vm.execute_microop_routed_control(
                    program, argument, array_memory=page, capture_trace=False, **OPTIONS,
                    prepare_frame_inputs=prepare, compact_frame_inputs=compact,
                )
                assert result.result_i32 == expected and page == guest["memory"].read(store)
                observed.append(result.steps)
            assert len(set(observed)) == 1
