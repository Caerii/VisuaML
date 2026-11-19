from dataclasses import replace
from shutil import which

import pytest

from scripts.verify_memory_region_v58 import Case, symbolic, wrong_forward_lowering
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.memory_region_contract_v58 import first_difference, observe
from visuaml.compiler_lab.neural_microstep_executor_v58 import MemoryWeights, MicrostepAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import MEMORY_BYTES
from visuaml.compiler_lab.wasm_copy_region_v58 import PreparedCopyRegion, copy_program
from visuaml.compiler_lab.wasm_research_corpus_v57 import ifelse

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


def test_shared_contract_rejects_both_wrong_overlap_proposals():
    case = Case(3, 8, 12)
    original = prepare(copy_program())
    expected = observe(original.run_memory, case.count, case.memory())
    wrong_recovery = first_difference(expected, symbolic(case, "forward"))
    wrong_lowering = first_difference(expected, wrong_forward_lowering(case))
    assert wrong_recovery == wrong_lowering == {"field": "memory", "address": 16, "expected": 2, "actual": 1}


def test_source_bound_fusion_preserves_exact_fuel_and_partial_traps():
    program = copy_program()
    original, optimized = prepare(program), PreparedCopyRegion(program)
    for case in (Case(3, 8, 9), Case(3, 12, 8), Case(2, 65532, 8)):
        expected = observe(original.run_memory, case.count, case.memory())
        actual = observe(optimized.run_memory, case.count, case.memory())
        assert first_difference(expected, actual, source_steps=case.fits()) is None
        if case.fits():
            exact = optimized.run_memory(case.count, case.memory(), fuel=expected.source_steps, capture_trace=False)
            assert exact.fused
            budget = expected.source_steps - 1
            expected = observe(original.run_memory, case.count, case.memory(), fuel=budget)
            actual = observe(optimized.run_memory, case.count, case.memory(), fuel=budget)
            assert actual.exit == "fuel" and first_difference(expected, actual) is None
        else:
            assert actual.exit == "memory_oob"
            assert actual.page[8:12] == b"\x6d\x00\x00\x00"
            assert len(actual.page) == MEMORY_BYTES


def test_access_order_is_a_stronger_and_explicit_claim():
    case = Case(2, 8, 24)
    expected = observe(prepare(copy_program()).run_memory, case.count, case.memory(), capture_trace=True)
    actual = symbolic(case, "forward")
    assert first_difference(expected, actual) is None
    assert first_difference(expected, actual, access_order=True)["index"] == 2
    with pytest.raises(ValueError, match="requires both traces"):
        first_difference(expected, replace(actual, accesses=None), access_order=True)


def test_gain_override_requests_source_replay_even_without_tracing():
    case = Case(3, 8, 12)
    optimized = PreparedCopyRegion(copy_program(), memory_fixed=MemoryWeights(store_gain=0))
    initial = case.memory()
    actual = optimized.run_memory(case.count, initial, capture_trace=False)
    assert not actual.fused and initial == case.memory()
    with pytest.raises(MicrostepAdmissionError, match="No certified copy-region"):
        PreparedCopyRegion(ifelse())
