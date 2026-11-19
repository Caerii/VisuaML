from shutil import which

import pytest
import wasmtime

from scripts.verify_memory_recovery_v58 import (
    initial_page, prefix_program, replay_trace, symbolic, word_values,
)
from scripts.verify_packed_control_v58 import prepare
from visuaml.compiler_lab.neural_microstep_executor_v58 import MemoryWeights, MicrostepAdmissionError
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    MEMORY_BYTES, ControlAdmissionError, ControlOperation, ControlProgram, LinearMemoryTrap,
    MemoryArgument, decode_control, encode_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import op

pytestmark = pytest.mark.skipif(which("c++") is None, reason="requires C++17")


def _memory_op(name, offset=0, align=2):
    return ControlOperation(name, MemoryArgument(offset, align))


def _wasmtime(program, page):
    store = wasmtime.Store()
    functions = wasmtime.Instance(store, wasmtime.Module(store.engine, encode_control(program)), []).exports(store)
    functions["memory"].write(store, page, 0)
    return store, functions["run"], functions["memory"]


@pytest.mark.parametrize("offset, argument", ((1, 4), (0, MEMORY_BYTES - 4), (1, MEMORY_BYTES - 5)))
def test_memory_encoding_unaligned_store_and_load_match_wasmtime(offset, argument):
    program = ControlProgram(0, (
        op("local.get", 0), op("i32.const", -2), _memory_op("i32.store", offset),
        op("local.get", 0), _memory_op("i32.load", offset), op("end"),
    ))
    assert decode_control(encode_control(program)) == program
    page = initial_page(())
    store, run, memory = _wasmtime(program, page)
    result = prepare(program).run_memory(argument, page)
    assert result.result_i32 == run(store, argument) == -2
    assert page == memory.read(store)
    assert page[argument + offset:argument + offset + 4] == b"\xfe\xff\xff\xff"
    assert replay_trace(initial_page(()), result.trace) == (page, 2)


@pytest.mark.parametrize("offset, argument", ((0, MEMORY_BYTES - 3), (1, -1), ((1 << 32) - 1, 1)))
@pytest.mark.parametrize("instruction", ("i32.load", "i32.store"))
def test_memory_trap_preserves_completed_writes_and_failing_bytes(offset, argument, instruction):
    tail = (() if instruction == "i32.load" else (op("i32.const", -2),))
    result = (() if instruction == "i32.load" else (op("i32.const", 0),))
    program = ControlProgram(0, (
        op("i32.const", 0), op("i32.const", 42), _memory_op("i32.store"),
        op("local.get", 0), *tail, _memory_op(instruction, offset), *result, op("end"),
    ))
    page = initial_page(())
    store, run, memory = _wasmtime(program, page)
    with pytest.raises(LinearMemoryTrap, match="Out of bounds"):
        prepare(program).run_memory(argument, page)
    with pytest.raises(wasmtime.Trap, match="out of bounds"):
        run(store, argument)
    assert page == memory.read(store)
    assert page == initial_page((42,))


def test_memory_page_contract_and_load_gain():
    program = ControlProgram(0, (op("local.get", 0), _memory_op("i32.load"), op("end")))
    with pytest.raises(ControlAdmissionError, match="alignment or offset"):
        encode_control(ControlProgram(0, (
            op("local.get", 0), _memory_op("i32.load", align=3), op("end"),
        )))
    packed = prepare(program, memory_fixed=MemoryWeights(load_gain=0))
    with pytest.raises(MicrostepAdmissionError, match="one WASM memory page"):
        packed.run_memory(0, bytearray(10))
    with pytest.raises(MicrostepAdmissionError, match="require run_memory"):
        packed.run(0)
    page = initial_page((123,))
    assert packed.run_memory(0, page).result_i32 == 0
    assert page == initial_page((123,))


def test_prefix_sum_wraparound_and_store_ablation():
    inputs = ((1 << 31) - 1, 1, -2)
    source = prefix_program()
    page = initial_page(inputs)
    nominal = prepare(source).run_memory(len(inputs), page)
    assert (nominal.result_i32, page) == symbolic("sum", inputs)
    assert word_values(page, len(inputs)) == ((1 << 31) - 1, -(1 << 31), (1 << 31) - 2)
    untouched = initial_page(inputs)
    ablated = prepare(source, memory_fixed=MemoryWeights(store_gain=0)).run_memory(len(inputs), untouched)
    assert untouched == initial_page(inputs)
    assert ablated.memory_weights_sha256 != nominal.memory_weights_sha256
