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
from visuaml.compiler_lab.neural_frame_circuit_v53 import (
    ACTIONS, HEIGHTS, KINDS, MAX_NESTING, FrameTokens, empty_frames, onehot, step_frames,
)
from visuaml.compiler_lab.neural_frame_events_v54 import OPCODES, decode_opcode_token, fixed_event_weights
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap, decode_control

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PACKED = "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep"


def tokens(action, kind=2, branch=0):
    return FrameTokens(onehot(action, ACTIONS), onehot(kind, KINDS), onehot(3, HEIGHTS),
                       onehot(branch, MAX_NESTING), onehot(5, HEIGHTS))


def assert_state(left, right):
    for a, b in zip((left.depth, left.kinds, left.bases), (right.depth, right.kinds, right.bases), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("resident", [True, False])
def test_retain_identity_and_disjoint_runs(resident):
    entries = tuple(decode_opcode_token(op) for op in OPCODES)
    weights = fixed_event_weights()
    initial = empty_frames()
    other = _CompiledFrameRetain(entries, weights, initial, resident=resident)
    for depth in range(MAX_NESTING + 1):
        if depth:
            initial, _ = step_frames(initial, tokens(1, kind=depth % KINDS))
        kernel = _CompiledFrameRetain(entries, weights, initial, resident=resident)
        before = kernel.state
        for flags in product((True, False), repeat=4):
            assert kernel.select(OPCODES.index("other"), *flags) == 0
            expected, _ = step_frames(initial, tokens(0))
            assert_state(kernel.state, expected)
        # Copy mode alternates its two slabs and resident mode keeps identity.
        last = kernel.state
        assert kernel.select(OPCODES.index("other"), False, False, False, False) == 0
        assert (last is kernel.state) is resident
        assert not np.shares_memory(kernel.state.bases, other.state.bases)
        assert not np.shares_memory(before.depth, initial.depth)
    assert_state(other.state, empty_frames())


@pytest.mark.parametrize("resident", [True, False])
def test_switch_to_checked_frame_changes_and_back(resident):
    rows = tuple(decode_opcode_token(op) for op in OPCODES)
    # Enter a block and a loop, retain, branch to the loop backedge,
    # pop the active frame, and return.
    sequence = (("block", 1, 0), ("loop", 1, 1), ("other", 0, 0),
                ("br", 3, 0), ("end", 2, 0), ("return", 4, 0))
    expected = empty_frames()
    kernel = _CompiledFrameRetain(rows, fixed_event_weights(), expected, resident=resident)
    for opcode, action, kind in sequence:
        active = int(expected.depth.argmax()) > 0
        assert kernel.select(OPCODES.index(opcode), True, False, active, True) == action
        expected, _ = step_frames(expected, tokens(action, kind=kind))
        if action != 0:
            previous = expected.depth.copy()
            kernel.publish_checked(expected)
            assert not np.shares_memory(kernel.state.depth, expected.depth)
            np.testing.assert_array_equal(previous, expected.depth)
        assert_state(kernel.state, expected)


@pytest.mark.parametrize("scene", ["terrain", "terrain-omitted"])
def test_all_modes_verify_saved_terrain(scene):
    saved = loads((FIXTURES / f"{scene}.proof.json").read_text())
    for mode in ("off", "copy", "resident"):
        def execute(*args, **kwargs):
            result = vm.execute_microop_routed_control(
                *args, **kwargs, use_compiled_frame_event=True, compiled_frame_retain=mode,
            )
            assert result.compiled_frame_retain_mode == (None if mode == "off" else mode)
            return result

        with patch.object(terrain, "execute_microop_routed_control", execute):
            proof = terrain.verify(saved["session"], saved)
        assert [t["steps"] for t in proof["ticks"]] == [t["steps"] for t in saved["ticks"]]


def test_cutoffs_traps_and_trace_ownership():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        baseline = vm.execute_microop_routed_control(program, 0, array_memory=bytearray(65536))
        trace = repr(baseline.trace)
        for mode in ("off", "copy", "resident"):
            checked_page = bytearray(65536)
            # The bypass branch skips the region.
            result = vm.execute_microop_routed_control(
                program, 1, array_memory=checked_page, capture_trace=False,
                use_compiled_frame_event=True, compiled_frame_retain=mode,
            )
            assert result.result_i32 == 1
        for cut in range(1, baseline.steps):
            for capture in (True, False):
                outputs = []
                for mode in ("off", "copy", "resident"):
                    page = bytearray(65536)
                    with pytest.raises(ControlFuelExhausted) as err:
                        vm.execute_microop_routed_control(
                            program, 0, array_memory=page, fuel=cut, capture_trace=capture,
                            use_compiled_frame_event=True, compiled_frame_retain=mode,
                        )
                    outputs.append((page, err.value.trace, err.value.trap_site))
                assert outputs[0] == outputs[1] == outputs[2]
                if capture:
                    assert outputs[0][1] == baseline.trace[:cut]
        assert repr(baseline.trace) == trace

        trap_source = decode_control(WITNESS.read_bytes(), scaled=True)
        start = bytearray(65536)
        start[65527:65531] = bytes.fromhex("01000080")
        start[65533:] = bytes.fromhex("aabbcc")
        for mode in ("off", "copy", "resident"):
            page = start.copy()
            with pytest.raises(LinearMemoryTrap):
                vm.execute_microop_routed_control(
                    trap_source, 65526, array_memory=page, capture_trace=False,
                    use_compiled_frame_event=True, compiled_frame_retain=mode,
                )
            assert page == start


def test_custom_weight_and_trace_keep_original_selector():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch.object(vm, "_CompiledFrameRetain", side_effect=AssertionError("Illegal native entry")):
        for capture, fixed in ((True, None), (False, fixed_event_weights())):
            result = vm.execute_microop_routed_control(
                program, 0, array_memory=bytearray(65536), capture_trace=capture, event_fixed=fixed,
                use_compiled_frame_event=True, compiled_frame_retain="resident",
            )
            assert result.compiled_frame_event_sha256 is None
