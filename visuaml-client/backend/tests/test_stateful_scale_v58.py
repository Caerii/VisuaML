from shutil import which
from struct import unpack_from

import pytest
import wasmtime

from scripts.stateful_scale_workloads_v58 import PUZZLES, graph_case, matrix_case, matrix_wat, sudoku_page, sudoku_wat
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.neural_control_routing_v52 import TERMINAL, compile_route_coordinates, compile_route_tokens
from visuaml.compiler_lab.neural_microstep_executor_v58 import MicrostepAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    SCALED_MAX_OPERATIONS, SCALED_SCHEMA, ControlAdmissionError, ControlOperation, ControlProgram,
    decode_control, encode_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import gcd12

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


def test_scale_boundary_and_legacy_routing_compatibility():
    source = ControlProgram(0, (ControlOperation("local.get", 0),) +
                            (ControlOperation("nop"),) * (SCALED_MAX_OPERATIONS - 2) +
                            (ControlOperation("end"),), SCALED_SCHEMA)
    raw = encode_control(source)
    with pytest.raises(ControlAdmissionError, match="count exceeded"):
        decode_control(raw)
    admitted = decode_control(raw, scaled=True)
    assert admitted == source
    drv = prepare(admitted)
    assert drv.run(-7, fuel=SCALED_MAX_OPERATIONS, capture_trace=False).steps == SCALED_MAX_OPERATIONS
    with pytest.raises(MicrostepAdmissionError, match="fuel"):
        drv.run(-7, fuel=SCALED_MAX_OPERATIONS, capture_trace=True)
    reference = compile_route_tokens(gcd12())
    targets, guards = compile_route_coordinates(gcd12(), TERMINAL)
    count = len(gcd12().operations)
    assert targets[:count, 0].tolist() == reference.sequential[:count].argmax(axis=1).tolist()
    assert targets[:count, 1].tolist() == reference.alternate[:count].argmax(axis=1).tolist()
    assert (guards == reference.guard_kind).all()


def _oracle(wat, argument, page):
    raw = bytes(wasmtime.wat2wasm(wat))
    source = decode_control(raw, scaled=True)
    store = wasmtime.Store()
    wasm = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)
    wasm["memory"].write(store, page, 0)
    result = wasm["run"](store, argument)
    return source, result, wasm["memory"].read(store)


def test_hard_sudoku_executes_as_wasm_and_checks_all_page_bytes():
    board = tuple(map(int, PUZZLES["escargot"]))
    page, argument = sudoku_page(board)
    source, result, expected = _oracle(sudoku_wat(), argument, page)
    actual = prepare(source).run_memory(argument, page, fuel=5_000_000, capture_trace=False)
    assert actual.steps > 2_000_000 and actual.result_i32 == result == 1 and page == expected
    solution = unpack_from("<81i", page)
    assert all(solution[i] == x for i, x in enumerate(board) if x)
    assert all(set(solution[row * 9:row * 9 + 9]) == set(range(1, 10)) for row in range(9))


def test_large_static_matrix_program_matches_independent_product():
    size, page, result, expected = matrix_case(32, 3)
    source = decode_control(bytes(wasmtime.wat2wasm(matrix_wat(32))), scaled=True)
    assert len(source.operations) > 800
    actual = prepare(source).run_memory(size, page, fuel=1_000_000, capture_trace=False)
    assert actual.steps > 800_000 and actual.result_i32 == result and page == expected


def test_disconnected_graph_reference_exercises_unreachable_vertices():
    _, _, reached, _ = graph_case(0)
    assert reached == 32
