from itertools import product
from json import loads
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wasmtime

import terrain_verify as terrain
from visuaml.compiler_lab import neural_alu_v56 as alu
from visuaml.compiler_lab import neural_microop_executor_v57 as vm
from visuaml.compiler_lab.bit_neural_vm_v49 import _bits, _int, construct_bit_weights
from visuaml.compiler_lab.neural_operand_storage_v55 import fixed_operand_weights
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlFuelExhausted, LinearMemoryTrap, decode_control

ROOT = Path(__file__).resolve().parents[1]
WITNESS = ROOT.parents[1] / "benchmarks/results/v58-array-memory/rollback.wasm"
SOURCE = """
(module
  (func (export "run") (param i32) (result i32)
    i32.const 0 local.get 0 i32.store
    block
      local.get 0 br_if 0
      i32.const 0
      i32.const 2147483647
      i32.const 2
      i32.add
      i32.store
    end
    i32.const 0 i32.load)
  (memory (export "memory") 1 1))
"""


def test_prepared_multiply_matches_dense_and_wrapping_reference():
    weights = construct_bit_weights()
    fixed = alu.construct_alu_weights()
    opcode = alu.onehot_alu("i32.mul")
    circuit = alu._prepare_selected_alu("i32.mul", opcode, fixed, weights)
    inputs = (0, 1, -1, -2, -2**31, 2**31 - 1, 0xffff, 0x10000)
    for a, b in product(inputs, repeat=2):
        left, right = _bits(a), _bits(b)
        result = circuit(left, right)
        np.testing.assert_array_equal(result, alu.execute_alu(opcode, left, right, fixed=fixed, bit_fixed=weights))
        assert _int(result) == (a * b + 2**31) % 2**32 - 2**31


@pytest.mark.parametrize("argument,expected,regions", [(0, -2147483647, 1), (1, 1, 0)])
def test_region_wrap_and_branch_bypass(argument, expected, regions):
    source = bytes(wasmtime.wat2wasm(SOURCE))
    program = decode_control(source, scaled=True)
    store = wasmtime.Store()
    guest = wasmtime.Instance(store, wasmtime.Module(store.engine, source), []).exports(store)
    assert guest["run"](store, argument) == expected
    baseline = bytearray(65536)
    checked = vm.execute_microop_routed_control(program, argument, array_memory=baseline,
                                                prepare_mul=False, prepare_regions=False)
    page = bytearray(65536)
    with patch.object(vm, "_run_const_add_region", wraps=vm._run_const_add_region) as region:
        result = vm.execute_microop_routed_control(program, argument, array_memory=page, capture_trace=False)
    assert region.call_count == regions
    assert result.result_i32 == checked.result_i32 == expected
    assert result.steps == checked.steps
    assert page == baseline == guest["memory"].read(store)


def test_exact_fuel_at_every_region_boundary_and_trace_fallback():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    page = bytearray(65536)
    baseline = vm.execute_microop_routed_control(program, 0, array_memory=page, prepare_regions=False)
    region_start = next(row["step"] for row in baseline.trace if row["opcode"] == "i32.add") - 1
    for cut in range(region_start, region_start + 3):
        for traced in (False, True):
            observed = []
            for enabled in (False, True):
                page = bytearray(65536)
                with patch.object(vm, "_run_const_add_region", wraps=vm._run_const_add_region) as region:
                    with pytest.raises(ControlFuelExhausted) as failure:
                        vm.execute_microop_routed_control(program, 0, array_memory=page, fuel=cut,
                                                          capture_trace=traced, prepare_regions=enabled)
                    assert region.call_count == int(enabled and not traced and cut == region_start + 2)
                observed.append((page, failure.value.trace, failure.value.trap_site))
            assert observed[0] == observed[1]
            if traced:
                assert observed[0][1] == baseline.trace[:cut]
                assert observed[0][2]["step"] == cut


def test_custom_profile_uses_generic_path():
    program = decode_control(bytes(wasmtime.wat2wasm(SOURCE)), scaled=True)
    with patch.object(vm, "_run_const_add_region", side_effect=AssertionError("Unexpected region")), patch(
        "visuaml.compiler_lab.neural_microop_executor_v57.execute_alu", wraps=alu.execute_alu,
    ) as generic:
        vm.execute_microop_routed_control(program, 0, array_memory=bytearray(65536),
                                          operand_fixed=fixed_operand_weights(), capture_trace=False)
    assert generic.call_count == 1


@pytest.mark.parametrize("name", ["terrain", "terrain-omitted"])
def test_saved_terrain_history_keeps_exact_world_and_trace(name):
    proof = loads((ROOT / "fixtures" / f"{name}.proof.json").read_text())
    actual = terrain.verify(proof["session"], proof)
    assert [tick["steps"] for tick in actual["ticks"]] == [tick["steps"] for tick in proof["ticks"]]
    observed = []
    for enabled in (False, True):
        traces = []

        def execute(*args, **kwargs):
            result = vm.execute_microop_routed_control(*args, **{
                **kwargs, "prepare_mul": enabled, "prepare_regions": enabled,
                "capture_trace": True, "fuel": 2000,
            })
            traces.append(result.trace)
            return result

        with patch.object(terrain, "execute_microop_routed_control", execute):
            terrain.verify(proof["session"], proof)
        observed.append(traces)
    assert observed[0] == observed[1]


def test_cross_page_store_trap_and_page_prefix():
    program = decode_control(WITNESS.read_bytes(), scaled=True)
    starting = bytearray(65536)
    starting[65527:65531] = bytes.fromhex("01000080")
    starting[65533:] = bytes.fromhex("aabbcc")
    for traced in (False, True):
        outcomes = []
        for enabled in (False, True):
            page = starting.copy()
            with pytest.raises(LinearMemoryTrap) as failure:
                vm.execute_microop_routed_control(program, 65526, array_memory=page,
                                                  capture_trace=traced, prepare_regions=enabled)
            outcomes.append((page, failure.value.trace, failure.value.trap_site))
        assert outcomes[0] == outcomes[1] and outcomes[0][0] == starting


def test_adjacent_local_read_region_keeps_all_fuel_boundaries():
    raw = wasmtime.wat2wasm(SOURCE.replace("i32.const 2147483647", "local.get 0"))
    program = decode_control(bytes(raw), scaled=True)
    page = bytearray(65536)
    expected = vm.execute_microop_routed_control(program, 0, array_memory=page, prepare_regions=False)
    start = next(row["step"] for row in expected.trace if row["opcode"] == "i32.add") - 2
    for cut in range(start, start + 4):
        for traced in (False, True):
            cases = []
            for enabled in (False, True):
                page = bytearray(65536)
                with patch.object(vm, "_run_local_const_add_region",
                                  wraps=vm._run_local_const_add_region) as region:
                    with pytest.raises(ControlFuelExhausted) as failed:
                        vm.execute_microop_routed_control(
                            program, 0, array_memory=page, fuel=cut, capture_trace=traced,
                            prepare_local_region=enabled,
                        )
                    assert region.call_count == int(enabled and not traced and cut == start + 3)
                trace = failed.value.trace
                if traced:
                    assert trace == expected.trace[:cut]
                cases.append((page, trace, failed.value.trap_site))
            assert cases[0] == cases[1]


def test_region_live_capacity_guard_preserves_operand_trap():
    from visuaml.compiler_lab.neural_operand_storage_v55 import OperandCircuitError

    source = SOURCE.replace("i32.const 2147483647", "i32.const 1 " * 23).replace(
        "i32.store\n    end", "i32.store " + "drop " * 22 + "\n    end",
    )
    wasmtime.Module(wasmtime.Engine(), source)
    program = decode_control(bytes(wasmtime.wat2wasm(source)), scaled=True)
    pages = []
    for enabled in (False, True):
        page = bytearray(65536)
        with patch.object(vm, "_run_const_add_region", side_effect=AssertionError("Bypassed overflow")):
            with pytest.raises(OperandCircuitError, match="overflow"):
                vm.execute_microop_routed_control(program, 0, array_memory=page,
                                                  capture_trace=False, prepare_regions=enabled)
        pages.append(page)
    assert pages[0] == pages[1]
