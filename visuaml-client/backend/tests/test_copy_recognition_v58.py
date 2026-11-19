from shutil import which

import pytest
import wasmtime

from scripts.verify_copy_recognition_v58 import fixture_page, independent_program, near_misses, trap_record, wasm_oracle
from scripts.verify_memory_region_v58 import Case
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.memory_region_contract_v58 import first_difference, observe
from visuaml.compiler_lab.wasm_control_flow_v51 import ControlOperation, ControlProgram, MemoryArgument, decode_control
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program, recognize_copy_region

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


@pytest.mark.parametrize("case", (Case(0, 0, 0), Case(3, 8, 12), Case(3, 12, 8), Case(4, 8, 9)))
def test_independent_copy_derives_roles_and_exact_fuel(case):
    program = independent_program()
    source, optimized = prepare(program), PreparedCopyRegion(program)
    assert optimized.plan != recognize_copy_region(copy_program())
    assert (optimized.plan.source_descriptor, optimized.plan.destination_descriptor) == (24, 28)
    initial = fixture_page(case)
    expected = observe(source.run_memory, case.count, initial)
    actual = optimized.run_memory(case.count, bytearray(initial), fuel=expected.source_steps, capture_trace=False)
    assert actual.fused and actual.steps == expected.source_steps
    assert first_difference(expected, observe(optimized.run_memory, case.count, initial), source_steps=True) is None

    budget = actual.steps - 1
    trap = observe(optimized.run_memory, case.count, initial, fuel=budget, capture_trace=True)
    assert trap.exit == "fuel" and trap.trap_site["step"] == budget
    ref_trap = observe(source.run_memory, case.count, initial, fuel=budget, capture_trace=True)
    assert first_difference(ref_trap, trap, access_order=True) is None


@pytest.mark.parametrize("name", tuple(near_misses()))
def test_lookalike_memory_programs_use_original_execution(name):
    program = decode_control(bytes(wasmtime.wat2wasm(near_misses()[name])))
    optimized = PreparedCopyRegion(program)
    assert optimized.plan is None
    case = Case(3, 8, 12)
    initial = fixture_page(case)
    observed = observe(optimized.run_memory, case.count, initial)
    assert first_difference(wasm_oracle(program)(case.count, initial), observed) is None
    expected = observe(PreparedCopyRegion(independent_program()).run_memory, case.count, initial)
    assert first_difference(expected, observed) is not None


def test_trap_prefix_explains_committed_write_and_next_invalid_read():
    case = Case(2, 65532, 8)
    result = trap_record(PreparedCopyRegion(independent_program()), case.count, fixture_page(case))
    assert result["site"]["address"] == 65536
    assert result["site"]["opcode"] == "i32.load"
    assert result["accesses"][-1]["after_hex"] == "6d000000"


def test_fault_site_keeps_widened_address_without_wraparound():
    source = ControlProgram(0, (
        ControlOperation("local.get", 0), ControlOperation("i32.load", MemoryArgument(1)),
        ControlOperation("end"),
    ))
    actual = observe(prepare(source).run_memory, -1, bytearray(65536), capture_trace=True)
    assert actual.exit == "memory_oob"
    assert actual.trap_site == {"step": 1, "pc": 1, "opcode": "i32.load", "address": 1 << 32}
    assert actual.accesses == ()
